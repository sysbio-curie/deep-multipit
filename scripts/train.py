import argparse
import collections
import inspect
import os
import sys
import warnings

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, WeightedRandomSampler

from _utils import train_test_split, build_model

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import dmultipit.dataset.loader as module_data
import dmultipit.model.loss as module_loss
import dmultipit.model.metric as module_metric
from dmultipit.parse_config import ConfigParser
from dmultipit.trainer import Trainer
from dmultipit.utils import prepare_device

# filter RuntimeWarnings that appear when dealing with PowerTransformer within the pre-processing step for radiomic
# MSKCC data. We recommend not using this line at first as it may hide other issues.
# warnings.simplefilter(action="ignore", category=RuntimeWarning)


def main(config_dict):
    """
    Train a multimodal prediction model (with optional pseudo-labelling)
    """

    # 0. fix random seeds for reproducibility
    seed = config_dict["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    logger = config_dict.get_logger("train")

    # 1. Load data

    # whether to perform pseudo-labelling or not
    keep_unlabelled = config_dict["training"]["pseudo_labelling"]

    dict_raw_data, labels = config_dict.init_ftn(
        ["training_data", "loader"],
        module_data,
        order=config_dict["architecture"]["order"],
        keep_unlabelled=keep_unlabelled,
    )()
    list_raw_data = tuple(dict_raw_data.values())

    # 2. Split data into training and validation
    val_index = config_dict["training_data"]["val_index"]  # specified validation indexes
    train_index = np.arange(len(labels))

    # If no validation indexes are specified, look for validation_split
    if val_index is None:
        split = config_dict["training_data"]["validation_split"]
        if split is not None:
            split_generator = StratifiedShuffleSplit(n_splits=1, test_size=split)
            train_index, val_index = next(
                split_generator.split(
                    np.zeros(len(labels)), np.where(~np.isnan(labels), labels, 2)
                )
            )
            config_dict["training_data"]["val_index"] = list(val_index)  # save validation index for checkpoint
    else:
        if len(val_index) > 0:
            train_index = np.delete(train_index, val_index)
        else:
            val_index = None

    # deal with radiomics data for MSKCC
    radiomics, rad_transform = None, None
    if config_dict["MSKCC"]:
        rad_transform = config_dict["radiomics_transform"]
        radiomics_list = []
        for item in ["radiomics_PL", "radiomics_LN", "radiomics_PC"]:
            try:
                radiomics_list.append(config_dict["architecture"]["order"].index(item))
            except ValueError:
                pass
        radiomics = int(np.min(radiomics_list)) if len(radiomics_list) > 0 else None
    if (rad_transform is not None) and (radiomics is not None):
        temp = [item.split('_')[-1] for item in config_dict["architecture"]["order"]
                if item.split('_')[0] == 'radiomics']
        if len(set(temp) ^ set(rad_transform["lesion_type"])) > 0:
            raise ValueError("Lesion types specified in rad_transform parameters and those specified in the"
                             " architecture/order parameter are different.")

    dataset_train, dataset_train_unlabelled, dataset_val, _ = train_test_split(
        train_index=train_index,
        test_index=val_index,
        labels=labels,
        list_raw_data=list_raw_data,
        dataset_name=config_dict["training_data"]["dataset"],
        list_unimodal_processings=[
            config_dict["training_data"]["processing"][modality]
            for modality in config_dict["architecture"]["order"]
        ],
        multimodal_processing=(None
                               if len(config_dict["architecture"]["order"]) == 1
                               else config_dict["training_data"]["processing"]["multimodal"]
                               ),
        drop_modas=config_dict["training_data"]["drop_modalities"],
        keep_unlabelled=keep_unlabelled,
        rad_transform=rad_transform,
        radiomics=radiomics
    )

    if dataset_train_unlabelled is not None:
        logger.info(str(len(dataset_train_unlabelled)) + " unlabelled data are kept for pseudo-labelling")

    if dataset_val is not None:
        logger.info(
            "Perform train-validation split with "
            + str(len(dataset_train))
            + " training samples"
              " and " + str(len(dataset_val)) + " validation samples."
        )
    else:
        logger.info("No train-validation split is performed.")

    # 3. load data loaders
    data_loader_kwargs = config_dict["training_data"]["data_loader"].copy()

    if config_dict["training_data"]["sampler"]:
        sample_weights = dataset_train.sample_weights
        assert sample_weights is not None, ("sampler is only available for binary classification "
                                            "setting, check your target or set sampler to False"
                                            )
        assert len(sample_weights) == len(dataset_train), ("sample_weights should be the same length"
                                                           " as your data set"
                                                           )
        data_loader_kwargs["sampler"] = WeightedRandomSampler(weights=sample_weights,
                                                              num_samples=len(dataset_train),
                                                              replacement=True
                                                              )

    data_loader = DataLoader(dataset=dataset_train, **data_loader_kwargs)

    unlabelled_data_loader = None
    if dataset_train_unlabelled is not None:
        unlabelled_data_loader = DataLoader(dataset=dataset_train_unlabelled,
                                            batch_size=len(dataset_train_unlabelled)
                                            )

    valid_data_loader = (DataLoader(dataset=dataset_val, batch_size=len(dataset_val))
                         if dataset_val is not None
                         else None
                         )

    # 4. build model architecture, then print to console
    device, device_ids = prepare_device(config_dict["n_gpu"])

    model = build_model(config_dict, device, training_data=dataset_train, logger=logger)

    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # 5. get function handles of loss and metrics
    criterion = config_dict.init_obj(["training", "loss"], module_loss)
    criterion_unlabelled = (config_dict.init_obj(["training", "unlabelled_loss"], module_loss)
                            if keep_unlabelled else None
                            )
    metrics = [getattr(module_metric, met) for met in config_dict["training"]["metrics"]]

    # 6. build optimizer, learning rate scheduler.
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config_dict.init_obj(["training", "optimizer"], torch.optim, trainable_params)

    weight_unlabelled = config_dict.init_obj(["training", "unlabelled_scheduler"], module_loss)

    lr_scheduler = None
    if config_dict["training"]["lr_scheduler"]["type"] is not None:
        logger.info("Learning rate scheduler is activated.")
        lr_scheduler = config_dict.init_obj(["training", "lr_scheduler"], torch.optim.lr_scheduler, optimizer)

    if config_dict["training"]["balanced_weights"] and config_dict["training"]["loss"]["type"] == "BCELogitLoss":
        weight = (dataset_train.labels == 0).sum() / (dataset_train.labels == 1).sum()
        setattr(criterion, 'pos_weight', torch.tensor(weight))
        logger.info("Balanced weigths for BCE with logits loss (weight: " + str(np.round(weight, 4)) + ").")

    # 7. Load trainer and train
    trainer = Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        config=config_dict,
        device=device,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        unlabelled_data_loader=unlabelled_data_loader,
        weight_unlabelled=weight_unlabelled,
        criterion_unlabelled=criterion_unlabelled,
        lr_scheduler=lr_scheduler,
    )

    trainer.train()


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
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-e",
        "--experiment",
        default=None,
        type=str,
        help="experiment file path (default: None)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="training;optimizer;args;lr"),
        CustomArgs(["--bs", "--batch_size"], type=int, target="data;train;dataset;args;batch_size"),
        CustomArgs(["--ri", "--run_id"], type=str, target="run_id"),
    ]

    config = ConfigParser.from_args(args, options=options, setting="train")
    main(config_dict=config)

