import warnings

import torch
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from dmultipit.base import MultiModalDataset
from .transformers import *


class TIPITDataset(MultiModalDataset):
    """
    Data set for TIPIT project. Several preprocessing steps are available including imputation, scaling, feature
    selection and pca.
    """

    def fit_process(self, X, y, params):
        """
        Fit processing pipeline

        Parameters
        ----------
        X: 2D array of shape (n_samples, n_features)
            Training data associated to a given modality

        y: 1D array of shape (n_samples,)
            Label for each sample

        params: dict
            Keys refer to processing operations (as specified in the following) and values correspond to dictionaries of
            parameters for each specific operations (e.g., key = "pca", value = {"n_components": 0.9}).

        Returns
        -------
            Fitted transforming pipeline.
        """
        steps = []
        for key, value in params.items():
            if value is None:
                value = {}
            if key == "imputer":
                steps.append(("imputer", CustomImputer(**value)))
            elif key == "scaler":
                steps.append(("scaler", CustomScaler(**value)))
            elif key == "selection":
                steps.append(("selection", CustomSelection(**value)))
            elif key == "vif":
                steps.append(("vif", CustomVIF(**value)))
            elif key == "pca":
                steps.append(("pca", CustomPCA(**value)))
            elif key == "log_transform":
                steps.append(("log_transformer", CustomLogTransform(**value)))
            elif key == "omics_imputer":
                steps.append(("omics_imputer", CustomOmicsImputer(**value)))
            else:
                raise ValueError("only imputation, scaling, feature selection and pca are implemented")

        transformer = Pipeline(steps=steps).fit(X, y)

        # if (modalities is not None) and ('selection' in params.keys()):
        #     transformer = Pipeline(steps=steps).fit(X, y, selection__modalities=modalities)
        # else:
        #     transformer = Pipeline(steps=steps).fit(X, y)
        return transformer

    def fit_multimodal_process(self, X, y, params, modalities):
        """
        Fit processing pipeline for multimodal data

        Parameters
        ----------
        X: 2D array of shape (n_samples, n_features)
            Training mulitmodal data.

        y: 1D array of shape (n_samples,)
            Label for each sample.

        params: dict
            Keys refer to processing operations (as specified in the following) and values correspond to dictionaries of
            parameters for each specific operations (e.g., key = "selection", value = {"n_components": 0.9}).

        modalities: 1D array of shape (n_features,)
            Integer labels indicating the membership of each feature to a specific modality
            (e.g. [0, 1, 2, 0, 0, 1...]).

        Returns
        -------
            Fitted multimodal transforming pipeline.

        Note
        ----
        Here we only implement the multimodal selection (see _transformers.CustomSelection) but several steps could be
        added and pipelines could be built. Each transforming step should have a transform_multimodal method which
        outputs

        """
        if (len(params) > 1) or ("selection" not in params.keys()):
            raise ValueError("Only multimodal selection is currently implemented")
        multimodal_transformer = CustomSelection(**params["selection"]).fit(X, y, modalities=modalities
                                                                            )
        return multimodal_transformer


class MSKCCDataset(MultiModalDataset):
    """
    Data set for reproducing the experiments described in Vanguri et al. (https://doi.org/10.1038/s43018-022-00416-8)

    Parameters
    ----------
    list_raw_data: see base.base_dataset.MultiModalDataset

    labels: see base.base_dataset.MultiModalDataset

    list_unimodal_processings: see base.base_dataset.MultiModalDataset

    transform: callable or None
            Transformation to apply to each sample in the batch (e.g., data augmentation). If None, no transformation is
            performed. The default is None.

    radiomics: int or None
        Position index of radiomics data within the list of raw data sets. If None, it is assumed no radiomic data
        were included in the list of raw data. The default is None.

    rad_transform: dict or _transformers.MSKCCRadiomicsTransform object, or None
        Transformer for radiomic data. The default is None.

    References
    ----------
    1. Zwanenburg, A. et al. Assessing robustness of radiomic features by image perturbation. Sci. Rep. 9, 1–10 (2019).
    (https://doi.org/10.1038/s41598-018-36938-4)

    2. Vanguri, R.S. et al. Multimodal integration of radiology, pathology and genomics for prediction of response to
    PD-(L)1 blockade in patients with non-small cell lung cancer. Nat Cancer 3, 1151–1164 (2022).
    (https://doi.org/10.1038/s43018-022-00416-8)
    """

    def __init__(
            self,
            list_raw_data,
            labels,
            list_unimodal_processings,
            transform=None,
            radiomics=None,
            rad_transform=None,
    ):
        # Transform and add radiomics data to the list of raw modalities
        if radiomics is not None:
            assert isinstance(radiomics, int), ("radiomics shoud correspond to the position index of raw radiomic"
                                                " data within the provided list of raw datasets.")
            assert rad_transform is not None, ("if radiomics is not None a transformer for radiomics data should be"
                                               " provided")
            n_modalities = len(list_raw_data)
            radiomics_data, self.rad_transform = self.transform_radiomics(list_raw_data[radiomics],
                                                                          labels,
                                                                          rad_transform)
            if radiomics == 0:
                list_raw_data = list(radiomics_data) + list_raw_data[1:] if n_modalities > 1 else radiomics_data
            elif radiomics == n_modalities - 1:
                list_raw_data = list_raw_data[:-1] + list(radiomics_data)
            else:
                list_raw_data = list_raw_data[:radiomics] + list(radiomics_data) + list_raw_data[radiomics + 1:]
        else:
            warnings.warn("radiomics was set to None. We will therefore assume that no radiomic data was included in"
                          "the list of raw data (no specific radiomic transformation will be applied).")

        # Initialize MultiModalDataset with the updated list of raw data (i.e., transformed radiomics included)
        super(MSKCCDataset, self).__init__(
            list_raw_data,
            labels,
            keep_unlabelled=False,
            list_unimodal_processings=list_unimodal_processings,
            multimodal_processing=None,
            transform=transform,
        )

    @staticmethod
    def transform_radiomics(radiomics_data, labels, rad_transform):
        """Fit and transform (or just transform if already fitted) radiomics data

        Parameters
        ----------
        radiomics_data: tuple of pandas dataframe and pandas Index
            * dataframe containing radiomics features extracted target lesions and their perturbed segmentations (10
            times)
            * Index containing the indexes of all the samples

        labels: pandas dataframe
            Label for each sample.

        rad_transform: dict or _transformers.MSKCCRadiomicsTransform object
            * If dictionary, corresponds to the set of parameters for _transformers.MSKCCRadiomicsTransform object which
            will be fitted to the provided radiomics_data
            * If _transformers.MSKCCRadiomicsTransform object, it should be already fitted.

        Returns
        -------
        output: list of 2D pandas dataframes of shape (n_samples, radiomic_features)
            Transformed radiomic data. Could contain several transformations (e.g., extracted features for different
            types of lesion).

        mskcc_transform/rad_transform: MSKCCRadiomicsTransform object
            Fitted transformer.
        """
        if isinstance(rad_transform, dict):
            mskcc_transform = MSKCCRadiomicsTransform(**rad_transform).fit(radiomics_data, labels)
            output = mskcc_transform.transform(radiomics_data)
            return output, mskcc_transform
        elif _check_MSKCC(rad_transform):
            output = rad_transform.transform(radiomics_data)
            return output, rad_transform
        else:
            raise ValueError(
                "rad_transform should either be a dictionary of parameters to instantiate and fit a "
                "MSKCCRadiomicsTransform object or an already fitted MSKCCRadiomicsTransform object."
            )

    def fit_process(self, X, y, params):
        steps = []
        for key, value in params.items():
            if value is None:
                value = {}
            if key == "imputer":
                steps.append(("imputer", CustomImputer(**value)))
            elif key == "scaler":
                steps.append(("scaler", CustomScaler(**value)))
            elif key == "selection":
                steps.append(("selection", CustomSelection(**value)))
            elif key == "vif":
                steps.append(("vif", CustomVIF(**value)))
            elif key == "pca":
                steps.append(("pca", CustomPCA(**value)))
            elif key == "log_transform":
                steps.append(("log_transformer", CustomLogTransform(**value)))
            else:
                raise ValueError(
                    "only imputation, scaling, feature selection and pca are implemented"
                )

        transformer = Pipeline(steps=steps).fit(X, y)
        return transformer

    def fit_multimodal_process(self, X, y, params, modalities):
        return


def _check_MSKCC(test_object):
    """
    Check whether the input object belongs to the _transformers.MSKCCRadiomicsTransform class and whether it is fitted.
    """
    check = False
    if test_object.__class__.__name__ == "MSKCCRadiomicsTransform":
        try:
            check_is_fitted(test_object)
            check = True
        except NotFittedError:
            pass
    return check


class DropModalities(object):
    """
    Transformer for data augmentation. Randomly drop one or several modalities for each patient (keeping at least one
    available).
    """

    def __call__(self, sample):
        *list_data, mask, target = sample
        nonzeros_indices = torch.nonzero(mask).numpy().reshape(-1)
        # print(nonzeros_indices)
        if len(nonzeros_indices) > 0:  # deal with cases where all the modalities are masked
            n_dropped = np.random.randint(0, len(nonzeros_indices))
            if n_dropped > 0:
                dropped_indices = np.random.choice(nonzeros_indices, size=n_dropped, replace=False)
                mask[dropped_indices] = 0
        return tuple(list_data) + (mask, target)
