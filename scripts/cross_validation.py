import argparse
import copy
import inspect
import os
import sys
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from joblib import delayed
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from _utils import train_val_test_split, build_model, ProgressParallel

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import dmultipit.model.loss as module_loss
import dmultipit.model.metric as module_metric
import dmultipit.dataset.loader as module_data
from dmultipit.parse_config import ConfigParser
from dmultipit.testing import Testing
from dmultipit.trainer import Trainer
from dmultipit.utils import prepare_device

# filter RuntimeWarnings that appear when dealing with PowerTransformer within the pre-processing step for radiomic
# MSKCC data. We recommend not using this line at first as it may hide other issues.
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


def main(config_dict):
    # 0. fix random seeds for reproducibility
    seed = config_dict["cross_val"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    device, _ = prepare_device(config_dict["n_gpu"])

    logger = config_dict.get_logger("cross_validation", verbosity=0)

    # Define all possible combinations of modalities with the specified list of modalities
    # in config_dict["architecture"]["order"]
    list_modas = config_dict["architecture"]["order"]
    drop_modas = config_dict["cross_val_data"]["drop_modalities"]

    names, indices = [], []
    for i in range(1, len(list_modas) + 1):
        for comb in combinations(range(len(list_modas)), i):
            indices.append(comb)
            names.append("+".join([list_modas[c] for c in comb]))

    # 1. Initialize data set which will be used for cross validation
    dict_raw_data, labels = config_dict.init_ftn(["cross_val_data", "loader"],
                                                 module_data,
                                                 order=config_dict["architecture"]["order"],
                                                 keep_unlabelled=False,
                                                 )()

    # 2. Define function that will be parallelized
    def _fun_parallel(r, disable_infos):

        # Initialize cross-validation splits
        if config_dict["cross_val"]["strategy"] == "StratifiedKFold":
            cv = StratifiedKFold(n_splits=config_dict["cross_val"]["n_splits"],
                                 shuffle=True,
                                 random_state=r,
                                 )
        elif config_dict["cross_val"]["strategy"] == "KFold":
            cv = KFold(n_splits=config_dict["cross_val"]["n_splits"],
                       shuffle=True,
                       random_state=r,
                       )
        else:
            raise ValueError("Only 'StratifiedKFold' and 'KFold' cross-validation schemes are available")

        logger.info(config_dict["cross_val"]["strategy"] + " cross-validation scheme n_splits = "
                    + str(config_dict["cross_val"]["n_splits"]))

        # Define dataframes to save prediction results
        df_preds = pd.DataFrame(index=labels.index, columns=["fold_index"] + names + ["label"])
        df_preds["label"] = labels.copy()
        df_preds["repeat"] = r

        # Iterate over all possible combinations of modalities
        for m, ind in enumerate(indices):
            if not disable_infos:
                print("model: ", names[m])

            if drop_modas and len(ind) == 1:
                config_dict["cross_val_data"]["drop_modalities"] = False
            else:
                config_dict["cross_val_data"]["drop_modalities"] = drop_modas

            config_dict["architecture"]["order"] = [list_modas[d] for d in ind]
            if config_dict["MSKCC"]:
                list_data = []
                rad = False
                for mod in config_dict["architecture"]["order"]:
                    if mod.split("_")[0] == "radiomics":
                        if not rad:
                            list_data.append(dict_raw_data["radiomics"])
                            rad = True
                    else:
                        list_data.append(dict_raw_data[mod])
            else:
                list_data = [dict_raw_data[mod] for mod in config_dict["architecture"]["order"]]

            # Get function handles of loss and metrics for training and test
            training_criterion = config_dict.init_obj(["training", "loss"], module_loss)
            training_metrics = [getattr(module_metric, met) for met in config_dict["training"]["metrics"]]
            test_metrics = [getattr(module_metric, met) for met in config_dict["testing"]["metrics"]]

            # Main cross-validation scheme
            for fold_index, (train_index, test_index) in enumerate(
                    tqdm(cv.split(np.zeros(len(labels)), np.where(~np.isnan(labels), labels, 2)),
                         leave=False,
                         total=cv.get_n_splits(np.zeros(len(labels))),
                         disable=disable_infos,
                         )
            ):
                # Set aside a validation subset if specified
                if config_dict["cross_val_data"]["validation_split"] is not None:
                    cv_val = StratifiedShuffleSplit(n_splits=1,
                                                    test_size=config_dict["cross_val_data"]["validation_split"])
                    train, val = next(cv_val.split(np.zeros(len(labels[train_index])),
                                                   np.where(~np.isnan(labels[train_index]), labels[train_index], 2))
                                      )
                    train_train_index, train_val_index = (train_index[train], train_index[val])
                else:
                    train_train_index, train_val_index = train_index, None

                # Train and test model on the cross-validation fold
                testing, bool_mask_test = _train_test_cv(config_dict=copy.deepcopy(config_dict),
                                                         train_train_index=train_train_index,
                                                         train_val_index=train_val_index,
                                                         test_index=test_index,
                                                         list_raw_data=list_data,
                                                         labels=labels,
                                                         training_metrics=training_metrics,
                                                         test_metrics=test_metrics,
                                                         training_criterion=training_criterion,
                                                         device=device,
                                                         logger=logger)

                df_preds.loc[labels.index.values[test_index[~bool_mask_test]], names[m]] = (testing.outputs
                                                                                            .cpu()
                                                                                            .numpy()
                                                                                            .reshape(-1))
                df_preds.loc[labels.index.values[test_index[~bool_mask_test]], "fold_index"] = fold_index

        return df_preds

    # 3. Parallel loop
    temp = ((config_dict["parallelization"]["n_jobs_repeats"] is not None)
            and (config_dict["parallelization"]["n_jobs_repeats"] > 1))

    parallel = ProgressParallel(n_jobs=config_dict["parallelization"]["n_jobs_repeats"],
                                total=config_dict["cross_val"]["n_repeats"])

    list_preds = parallel(delayed(_fun_parallel)(r, disable_infos=temp)
                          for r in range(config_dict["cross_val"]["n_repeats"])
                          )

    # 4. Save results (i.e., collected predictions over the repeats of the cv scheme)
    pd.concat(list_preds, axis=0).to_csv(config_dict.save_dir / "predictions.csv")

    return


def _train_test_cv(
        config_dict,
        train_train_index,
        train_val_index,
        test_index,
        list_raw_data,
        labels,
        training_criterion,
        training_metrics,
        test_metrics,
        device,
        logger,
):
    # Deal with radiomics data for MSKCC
    radiomics, rad_transform = None, None
    if config_dict["MSKCC"]:
        rad_transform = config_dict["radiomics_transform"]
        radiomics_list = list(sorted(set(config_dict["architecture"]["order"])
                                     & {"radiomics_PL", "radiomics_LN", "radiomics_PC"},
                                     key=config_dict["architecture"]["order"].index))
        radiomics_list_ind = [config_dict["architecture"]["order"].index(item) for item in radiomics_list]
        radiomics_list = [item.split("_")[-1] for item in radiomics_list]

        radiomics = int(np.min(radiomics_list_ind)) if len(radiomics_list) > 0 else None
        rad_transform["lesion_type"] = radiomics_list

    # Initialize train, test and validation datasets
    (training_data,
     _,
     val_data,
     test_data,
     bool_mask_test,
     _,
     ) = train_val_test_split(train_index=train_train_index,
                              val_index=train_val_index,
                              test_index=test_index,
                              labels=labels,
                              list_raw_data=list_raw_data,
                              dataset_name=config_dict["cross_val_data"]["dataset"],
                              list_unimodal_processings=[config_dict["cross_val_data"]["processing"][modality]
                                                         for modality in config_dict["architecture"]["order"]
                                                         ],
                              multimodal_processing=(None if len(config_dict["architecture"]["order"]) == 1
                                                     else config_dict["cross_val_data"]["processing"]["multimodal"]
                                                     ),
                              drop_modas=config_dict["cross_val_data"]["drop_modalities"],
                              keep_unlabelled=False,
                              rad_transform=rad_transform,
                              radiomics=radiomics,
                              )

    # Load train, validation and test data loaders
    training_data_loader_kwargs = config_dict["cross_val_data"]["training_data_loader"].copy()

    if config_dict["cross_val_data"]["sampler"]:
        sample_weights = training_data.sample_weights
        assert sample_weights is not None, ("sampler is only available for binary classification setting, check your"
                                            " target or set sampler to False")
        assert len(sample_weights) == len(training_data), "sample_weights should be the same length as your data set"

        training_data_loader_kwargs["sampler"] = WeightedRandomSampler(weights=sample_weights,
                                                                       num_samples=len(training_data),
                                                                       replacement=True)

    train_data_loader = DataLoader(dataset=training_data, **training_data_loader_kwargs)

    valid_data_loader = DataLoader(dataset=val_data, batch_size=len(val_data)) if val_data is not None else None

    test_data_loader = DataLoader(dataset=test_data)

    # Build model, optimizer, learning rate scheduler
    model = build_model(config_dict, device=device, training_data=training_data)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config_dict.init_obj(["training", "optimizer"], torch.optim, trainable_params)

    lr_scheduler = None
    if config_dict["training"]["lr_scheduler"]["type"] is not None:
        logger.info("Learning rate scheduler is activated.")
        lr_scheduler = config_dict.init_obj(["training", "lr_scheduler"], torch.optim.lr_scheduler, optimizer)

    if config_dict["training"]["balanced_weights"] and (config_dict["training"]["loss"]["type"] == "BCELogitLoss"):
        weight = (training_data.labels == 0).sum() / (training_data.labels == 1).sum()
        setattr(training_criterion, "pos_weight", torch.tensor(weight))
        logger.info("Balanced weigths for BCE with logits loss (weight: " + str(np.round(weight, 4)) + ").")

    trainer = Trainer(
        model=model,
        criterion=training_criterion,
        metric_ftns=training_metrics,
        optimizer=optimizer,
        config=config_dict,
        device=device,
        data_loader=train_data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
        disable_checkpoint=True
    )

    trainer.train()

    # Test the trained model on the corresponding test set
    testing = Testing(model=trainer.model,
                      loss_ftn=training_criterion,
                      metric_ftns=test_metrics,
                      config=config_dict,
                      device=device,
                      data_loader=test_data_loader,
                      intermediate_fusion=config_dict["architecture"]["intermediate_fusion"],
                      disable_tqdm=True,
                      )

    testing.test(collect_a=False, collect_modalitypred=False)

    return testing, bool_mask_test


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Multimodal Fusion")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-e",
        "--experiment",
        default=None,
        type=str,
        help="experiment file path (default: None)",
    )

    config = ConfigParser.from_args(args, setting="cross_val")
    main(config_dict=config)
