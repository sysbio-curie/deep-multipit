import inspect
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel
from lifelines.statistics import logrank_test
from tqdm.auto import tqdm

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import dmultipit.dataset.dataset as module_data
from dmultipit.base.base_dataset import CustomSubset
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

    Parameters
    ----------
    labels:

    list_raw_data:

    dataset_name:

    list_unimodal_processings:

    multimodal_processing:

    indexes:

    drop_modas:

    keep_unlabelled:

    radiomics:

    rad_transform:

    Returns
    -------
    data_set:

    bool_mask:

    """
    if indexes is not None:

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

        if isinstance(labels, np.ndarray):
            labels = labels.copy()[indexes][~bool_mask]
        else:
            labels = labels.copy().iloc[indexes].values[~bool_mask]

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

    Parameters
    ----------
    train_index:

    test_index:

    labels:

    list_raw_data:

    dataset_name:

    list_unimodal_processings:

    multimodal_processing:

    drop_modas:

    keep_unlabelled:

    radiomics:

    rad_transform:

    Returns
    -------
    dataset_train:

    dataset_train_unlabelled:

    dataset_test:

    bool_mask_test:

    """
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

    dataset_train_unlabelled = None

    if keep_unlabelled:
        assert dataset_train.unlabelled_data is not None, ""
        if len(dataset_train.unlabelled_data) > 0:
            dataset_train_unlabelled = CustomSubset(
                dataset_train, dataset_train.unlabelled_data
            )
            dataset_train = CustomSubset(
                dataset_train,
                list(
                    set(range(len(dataset_train))) - set(dataset_train.unlabelled_data)
                ),
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

    Parameters
    ----------
    train_index:

    test_index:

    val_index:

    labels:

    list_raw_data:

    dataset_name:

    list_unimodal_processings:

    multimodal_processing:

    drop_modas:

    keep_unlabelled:

    radiomics:

    rad_transform:

    Returns
    -------
    dataset_train:

    dataset_train_unlabelled:

    dataset_val:

    dataset_test:

    bool_mask_test:

    bool_mask_train:

    """
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

    dataset_train_unlabelled = None

    if keep_unlabelled:
        assert dataset_train.unlabelled_data is not None, ""
        if len(dataset_train.unlabelled_data) > 0:
            dataset_train_unlabelled = CustomSubset(dataset_train, dataset_train.unlabelled_data)
            dataset_train = CustomSubset(
                dataset_train,
                list(set(range(len(dataset_train))) - set(dataset_train.unlabelled_data)),
            )

    return dataset_train, dataset_train_unlabelled, dataset_val, dataset_test, bool_mask_test, bool_mask_train


def build_model(config_dict, device, logger=None):
    """

    Parameters
    ----------
    config_dict:

    device:

    logger:

    Returns
    -------
    model:

    """
    embeddings = [
        config_dict.init_obj(
            ("architecture", "modality_embeddings", modality), module_emb
        )
        for modality in config_dict["architecture"]["order"]
    ]

    if config_dict["architecture"]["intermediate_fusion"]:

        model = module_arch.InterAttentionFusion(
            modality_embeddings=embeddings,
            attention=config_dict.init_obj(["architecture", "attention"], module_att),
            predictor=config_dict.init_obj(["architecture", "predictor"], module_emb),
        )
    else:
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


class ProgressParallel(Parallel):
    """ """

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


def fing_logrank_threshold(risk_score, labels_surv):
    """

    Parameters
    ----------
    risk_score:

    labels_surv:

    Returns
    -------

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
