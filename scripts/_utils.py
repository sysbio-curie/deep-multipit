import inspect
import os
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel
from lifelines.statistics import logrank_test
from tqdm.auto import tqdm

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import dmultipit.dataset.dataset as module_data
from dmultipit.base.base_dataset import CustomSubset, check_transform
from dmultipit.model import model as module_arch
import dmultipit.model.attentions as module_att
import dmultipit.model.embeddings as module_emb


def get_dataset(
    labels,
    list_raw_data,
    dataset_name,
    list_unimodal_processings,
    multimodal_processing,
    indexes,
    drop_modas=False,
    keep_unlabelled=False,
    radiomics=None,
    rad_transform=None,
):
    """
    Create multimodal dataset from raw data

    Parameters
    ----------
    labels: 1D numpy array or pandas serie (n_samples,)
        Label for each sample. It should be ordered in the same way as the raw data sets provied in *list_raw_data*.

    list_raw_data: list of numpy arrays or pandas dataframes of shape (n_samples, n_features_1),
        (n_samples, n_features_2)... The rows of the different datasets should be ordered in the same way (i.e., same
        order of samples each time).

    dataset_name: str
        Name of the Dataset object to consider. It should refer to an object from dmultipit.dataset.dataset and
        inheritating from dmultipit.base.base_dataset.MultiModalDataset.

    list_unimodal_processings: list of dictionaries, sklearn.base.TransformerMixin and None
        List of processing operations to apply to each modality separately.
        * If None no operation is performed for the corresponding modality
        * If dictionary it should define a processing strategy to be fitted on the data
         (e.g., {'scaling': {'with_std': True}, 'PCA': {'n_components': 0.9}} could define a standard scaling
         operations followed by a PCA (to be defined in the fit_process method !))
        * If sklearn.base.TransformerMixin it should correspond to a fitted transformer !

    multimodal_processing: dict, sklearn.base.TransformerMixin and None
        Processing operations to apply to the multimodal data set.
        * If None no operation is performed
        * If dictionary it should define a processing strategy to be fitted on the data (to be defined in the
         fit_multimodal_process method !)
        * If sklearn.base.TransformerMixin it should correspond to a fitted transformer !

    indexes: list of string or None
        List of sample indexes to consider in the data set. If None the function does not return a Dataset (only None
        value).

    drop_modas: bool
        If True, random droping of modalities will be applied to each sample (i.e., data augmentation).
        See dmultipit.dataset.dataset.DropModalities. The default is False

    keep_unlabelled: bool
        If True, keep unlabelled samples in the dataset. Discard them otherwise.
        See dmultipit.base.base_dataset.MultiModalDataset. The default is False.

    radiomics: int or None
        Position index of radiomics data within the list of raw data sets. If None, it is assumed no radiomic data
        were included in the list of raw data. This argument is only taken into accound when dataset_name is
        'MSKCCDataSet'. The default is None.

    rad_transform: dict or _transformers.MSKCCRadiomicsTransform object, or None
        Transformer for radiomic data. This argument is only taken into accound when dataset_name is 'MSKCCDataSet'.
        The default is None.

    Returns
    -------
    data_set: MultiModalDataset object or None
        A None value is returned when no indexes were passed as input.

    bool_mask: boolean array of size (n_samples,) or None
        A None value is returned when no indexes were passed as input. Otherwise, indicate samples with only NaN values.
    """

    if indexes is not None:

        # select indexes in raw data sets
        data_sets = []
        for raw_data in list_raw_data:
            if isinstance(raw_data, np.ndarray):
                data_sets.append(raw_data.copy()[indexes, :])
            elif isinstance(raw_data, pd.DataFrame):
                data_sets.append(raw_data.copy().iloc[indexes, :].values)
            elif isinstance(raw_data, tuple):
                new_ind = raw_data[1][indexes]
                new_data = raw_data[0].loc[new_ind.intersection(raw_data[0].index)]
                new_data.index.names = ["main_index"]
                data_sets.append((new_data, new_ind))

        # discard samples with only missing values
        all_data = np.hstack(data_sets)
        bool_mask = np.sum(np.isnan(all_data), axis=1) == all_data.shape[1]
        data_sets = [data[~bool_mask] for data in data_sets]

        # select indexes in labels
        if isinstance(labels, np.ndarray):
            labels = labels.copy()[indexes][~bool_mask]
        else:
            labels = labels.copy().iloc[indexes].values[~bool_mask]

        # Create MultiModalDataset
        if dataset_name == "MSKCCDataset":
            dataset = getattr(module_data, dataset_name)(
                list_raw_data=data_sets,
                labels=labels,
                list_unimodal_processings=list_unimodal_processings,
                multimodal_processing=multimodal_processing,
                keep_unlabelled=keep_unlabelled,
                radiomics=radiomics,
                rad_transform=rad_transform,
            )
        else:
            dataset = getattr(module_data, dataset_name)(
                list_raw_data=data_sets,
                labels=labels,
                list_unimodal_processings=list_unimodal_processings,
                multimodal_processing=multimodal_processing,
                keep_unlabelled=keep_unlabelled,
            )

        #
        if drop_modas:
            setattr(dataset, "transform", module_data.DropModalities())

    else:
        dataset, bool_mask = None, None

    return dataset, bool_mask


def train_test_split(
    train_index,
    test_index,
    labels,
    list_raw_data,
    dataset_name,
    list_unimodal_processings,
    multimodal_processing,
    drop_modas,
    keep_unlabelled,
    radiomics=None,
    rad_transform=None,
):
    """
    Create training and test data for a given train-test split. Unimodal and multimodal processings are fitted to the
    training data and subsequently applied to the test data.

    Parameters
    ----------
    train_index: list of int.
        Indexes for training data.

    test_index: list of int.
        Indexes for test data.

    Returns
    -------
    dataset_train: MultiModalDataset object
        Training dataset with labelled data.

    dataset_train_unlabelled: MultiModalDataset object or None
        Training dataset with unlabelled data. None if self.keep_unlabelled is False.

    dataset_test: MultiModalDataset object
        Test dataset.

    bool_mask_test: boolean array of shape (n_test_samples,)
        Indicate samples with only NaN values.
    """

    # create training dataset
    dataset_train, _ = get_dataset(
        labels,
        list_raw_data,
        dataset_name,
        list_unimodal_processings,
        multimodal_processing,
        train_index,
        drop_modas,
        keep_unlabelled,
        radiomics,
        rad_transform,
    )

    # create training dataset with unlabelled data if needed (for semi-supervised strategy)
    dataset_train_unlabelled = None

    if keep_unlabelled:
        if (dataset_train.unlabelled_data is not None) and (
            len(dataset_train.unlabelled_data) > 0
        ):
            dataset_train_unlabelled = CustomSubset(
                dataset_train, dataset_train.unlabelled_data
            )
            dataset_train = CustomSubset(
                dataset_train,
                list(
                    set(range(len(dataset_train))) - set(dataset_train.unlabelled_data)
                ),
            )
        else:
            warnings.warn(
                "Training data contains no unlabelled data. dataset_train_unlabelled is set to None"
            )

    # create test dataset with fitted unimodal and multimodal processings from training data
    dataset_test, bool_mask_test = get_dataset(
        labels,
        list_raw_data,
        dataset_name,
        dataset_train.list_unimodal_processings,
        dataset_train.multimodal_processing,
        test_index,
        keep_unlabelled=False,
        radiomics=radiomics,
        rad_transform=dataset_train.rad_transform
        if rad_transform is not None
        else None,
    )

    return dataset_train, dataset_train_unlabelled, dataset_test, bool_mask_test


def train_val_test_split(
    train_index,
    test_index,
    val_index,
    labels,
    list_raw_data,
    dataset_name,
    list_unimodal_processings,
    multimodal_processing,
    drop_modas,
    keep_unlabelled,
    radiomics=None,
    rad_transform=None,
):
    """
    Create training, validation and test data for a given train-val-test split. Unimodal and multimodal processings are
    fitted to the training data and subsequently applied to the validation and test data.

    Parameters
    ----------
    train_index: list of int
        Indexes for training data.

    test_index: list of int
        Indexes for test data.

    val_index: list of int
        Indexes for validation data.

    Returns
    -------
    dataset_train: MultiModalDataset object
        Training dataset with labelled data.

    dataset_train_unlabelled: MultiModalDataset object or None
        Training dataset with unlabelled data.

    dataset_val: MultiModalDataset object
        Validation dataset.

    dataset_test: MultiModalDataset object
        Test dataset.

    bool_mask_test: boolean array of shape (n_test_samples,)
        Indicate samples with only NaN values.

    bool_mask_train: boolean array of shape (n_val_samples,)
        Indicate samples with only NaN values.
    """

    # create training dataset
    dataset_train, bool_mask_train = get_dataset(
        labels,
        list_raw_data,
        dataset_name,
        list_unimodal_processings,
        multimodal_processing,
        train_index,
        drop_modas,
        keep_unlabelled,
        radiomics,
        rad_transform,
    )

    # create training dataset with unlabelled data if needed (for semi-supervised strategy)
    dataset_train_unlabelled = None

    if keep_unlabelled:
        if (dataset_train.unlabelled_data is not None) and (
            len(dataset_train.unlabelled_data) > 0
        ):
            dataset_train_unlabelled = CustomSubset(
                dataset_train, dataset_train.unlabelled_data
            )
            dataset_train = CustomSubset(
                dataset_train,
                list(
                    set(range(len(dataset_train))) - set(dataset_train.unlabelled_data)
                ),
            )
        else:
            warnings.warn(
                "Training data contains no unlabelled data. dataset_train_unlabelled is set to None"
            )

    # create validation dataset with fitted unimodal and multimodal processings from training data
    dataset_val, _ = get_dataset(
        labels,
        list_raw_data,
        dataset_name,
        dataset_train.list_unimodal_processings,
        dataset_train.multimodal_processing,
        val_index,
        keep_unlabelled=False,
        radiomics=radiomics,
        rad_transform=dataset_train.rad_transform
        if rad_transform is not None
        else None,
    )

    # create test dataset with fitted unimodal and multimodal processings from training data
    dataset_test, bool_mask_test = get_dataset(
        labels,
        list_raw_data,
        dataset_name,
        dataset_train.list_unimodal_processings,
        dataset_train.multimodal_processing,
        test_index,
        keep_unlabelled=False,
        radiomics=radiomics,
        rad_transform=dataset_train.rad_transform
        if rad_transform is not None
        else None,
    )

    return (
        dataset_train,
        dataset_train_unlabelled,
        dataset_val,
        dataset_test,
        bool_mask_test,
        bool_mask_train,
    )


def build_model(config_dict, device, training_data=None, logger=None):
    """
    Build multimodal predictive model from configuration dictionary

    Parameters
    ----------
    config_dict: dict

    device: str
        Torch.device on which to allocate model weights

    training_data: dmultipit.base.base_dataset.MultiModalDataset object or None
        Training data set. If None, the architecture of the model is not updated. The default is None.

    logger: logging device or None
        The default is None.

    Returns
    -------
    model: dmultipit.model.model.InterAttentionFusion or dmultipit.model.model.LateAttentionFusion

    """
    # update architecture (after training pre-processings)
    if training_data is not None:
        config_dict = _update_architecture(config_dict, training_data)

    # build embedding modules for each modality
    embeddings = [
        config_dict.init_obj(
            ("architecture", "modality_embeddings", modality), module_emb
        )
        for modality in config_dict["architecture"]["order"]
    ]

    if config_dict["architecture"]["intermediate_fusion"]:

        # build intermediate fusion model (see dmultipit.model.model)
        model = module_arch.InterAttentionFusion(
            modality_embeddings=embeddings,
            attention=config_dict.init_obj(["architecture", "attention"], module_att),
            predictor=config_dict.init_obj(["architecture", "predictor"], module_emb),
        )
    else:

        # build late fusion model (see dmultipit.model.model), where attention modules have the same dim input as
        # embedding modules for the different modalities
        model = module_arch.LateAttentionFusion(
            modality_embeddings=embeddings,
            multimodalattention=config_dict.init_obj(
                ["architecture", "attention"],
                module_att,
                dim_input=[
                    config_dict["architecture"]["modality_embeddings"][m]["args"][
                        "dim_input"
                    ]
                    for m in config_dict["architecture"]["order"]
                ],
            ),
        )

    if logger is not None:
        logger.info(model)

    model = model.to(device)

    return model


def _update_architecture(config_dict, training_data):
    """
    Update configuration dictionary (i.e., model architecture) to take into account pre-processing operations (e.g.,
    changes in the input dimensions)

    Parameters
    ----------
    config_dict: dict
        Configuration dictionary

    training_data: dmultipit.base.base_dataset.MultiModalDataset object
        Training data set

    Returns
    -------
    config_dict: dict
        Updated configuration dictionary
    """

    # If multimodal pre-processing is performed (last processing operations) use the get_multimodal_dimension method
    # of the dmultipit.base.base_transformer.MultimodalTransformer estimator to extract the updated dimension of each
    # modality
    if (len(config_dict["architecture"]["order"]) > 1) and (training_data.multimodal_processing is not None):
        list_new_dim = training_data.multimodal_processing.get_multimodal_dimension()
        for new, moda in zip(list_new_dim, config_dict["architecture"]["order"]):
            config_dict["architecture"]["modality_embeddings"][moda]["args"]["dim_input"] = new

    # Otherwise, use the get_dimension method of the last dmultimot.base.base_transformer.MultimodalTransformer applied
    # to each modality to obtain the updated input dimensions.
    else:
        for process, moda in zip(training_data.list_unimodal_processings, config_dict["architecture"]["order"]):
            if check_transform(process):
                classes = inspect.getmro(process.__class__)
                if classes[0].__name__ == "Pipeline":
                    process = process[-1]
                new = process.get_dimension()
                config_dict["architecture"]["modality_embeddings"][moda]["args"]["dim_input"] = new

            # special case for MKSCC dataset and radiomic transform (when no unimodal processing is applied)
            elif config_dict["MSKCC"] and (moda in {"radiomics_PL", "radiomics_PC", "radiomics_LN"}):
                new = len(training_data.rad_transform.selected_features_[moda.split("_")[1]])
                config_dict["architecture"]["modality_embeddings"][moda]["args"]["dim_input"] = new

    return config_dict


class ProgressParallel(Parallel):
    """Custom tqdm progress bar for parallel computing"""

    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def find_logrank_threshold(risk_score, labels_surv):
    """
    Find cutoff value that maximize logrank test (searching between 30th and 71st percentiles of provided risk scores)

    Parameters
    ----------
    risk_score: 1D array of shape (n_samples,)
        Predicted survival risk score.

    labels_surv: sksurv.util.Surv array of shape (n_samples,)
        Structured array containing event indicators (i.e., censored or not) and observed times.

    Returns
    -------
        float, optimal cutoff value
    """
    cutoffs, pvals = [], []
    for p in np.arange(30, 71):
        c = np.percentile(risk_score, p)
        group1 = risk_score <= c
        group2 = risk_score > c
        test = logrank_test(
            durations_A=labels_surv[group1]["time"],
            durations_B=labels_surv[group2]["time"],
            event_observed_A=1 * (labels_surv[group1]["event"]),
            event_observed_B=1 * (labels_surv[group2]["event"]),
        )
        cutoffs.append(c)
        pvals.append(test.summary["p"].values[0])
    return cutoffs[np.argmin(pvals)]
