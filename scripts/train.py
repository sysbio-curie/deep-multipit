import argparse
import collections
import inspect
import os
import sys

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, WeightedRandomSampler

from _utils import train_test_split

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import dmultipit.dataset.loader as module_data
import dmultipit.model.loss as module_loss
import dmultipit.model.metric as module_metric
from dmultipit.model import model as module_arch
import dmultipit.model.attentions as module_att
import dmultipit.model.embeddings as module_emb
from dmultipit.parse_config import ConfigParser
from dmultipit.trainer import Trainer
from dmultipit.utils import prepare_device


def main(config_dict):
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

    *list_raw_data, labels = config_dict.init_ftn(
        ["training_data", "loader"],
        module_data,
        order=config_dict["architecture"]['order'],
        keep_unlabelled=keep_unlabelled
    )()

    # 2. Split data into training and validation
    val_index = config_dict["training_data"]["val_index"]
    train_index = np.arange(len(labels))
    if val_index is None:
        split = config_dict["training_data"]["validation_split"]
        if split is not None:
            split_generator = StratifiedShuffleSplit(n_splits=1, test_size=split)
            train_index, val_index = next(
                split_generator.split(np.zeros(len(labels)), np.where(~np.isnan(labels), labels, 2))
            )
    else:
        if len(val_index) > 0:
            train_index = np.delete(train_index, val_index)
        else:
            val_index = None

    # # Initialize data sets with or without independent predictions
    # list_predictors = None
    # if config_dict["architecture"]["independent_predictions"]:
    #     assert config_dict["training_data"]["prediction"] is not None, "for independent predictions one predictor per" \
    #                                                                    " modality should be specified"
    #     list_predictors = []
    #     for modality in config_dict["architecture"]["order"]:
    #         predictor = config_dict["training_data"]["prediction"][modality]
    #         assert predictor is not None, "predictor for " + modality + " is not specified"
    #         list_predictors.append(predictor)

    dataset_train, dataset_train_unlabelled, dataset_val = train_test_split(
        train_index=train_index,
        test_index=val_index,
        labels=labels,
        list_raw_data=list_raw_data,
        dataset_name=config_dict["training_data"]["dataset"],
        list_processings=[config_dict["training_data"]["processing"][modality]
                          for modality in config_dict["architecture"]["order"]],
        drop_modas=config_dict["training_data"]['drop_modalities'],
        keep_unlabelled=keep_unlabelled
    )
    #     list_predictors=list_predictors
    # )

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
        assert sample_weights is not None, "sampler is only available for binary classification " \
                                           "setting, check your target or set sampler to False"
        assert len(sample_weights) == len(dataset_train), "sample_weights should be the same length" \
                                                          " as your data set"

        data_loader_kwargs['sampler'] = WeightedRandomSampler(weights=sample_weights,
                                                              num_samples=len(dataset_train),
                                                              replacement=True)

    data_loader = DataLoader(dataset=dataset_train, **data_loader_kwargs)

    unlabelled_data_loader = None
    if dataset_train_unlabelled is not None:
        unlabelled_data_loader = DataLoader(dataset=dataset_train_unlabelled,
                                            batch_size=len(dataset_train_unlabelled))

    valid_data_loader = (
        DataLoader(dataset=dataset_val, batch_size=len(dataset_val))
        if dataset_val is not None
        else None
    )

    # 4. build model architecture, then print to console
    if config_dict["architecture"]["intermediate_fusion"]:

        model = module_arch.InterAttentionFusion(
            modality_embeddings=[config_dict.init_obj(("architecture", "modality_embeddings", modality), module_emb)
                                 for modality in config_dict["architecture"]["order"]
                                 ],
            attention=config_dict.init_obj(
                ["architecture", "attention"], module_att
            ),
            predictor=config_dict.init_obj(["architecture", "predictor"], module_emb),
        )
    # elif config_dict["architecture"]["independent_predictions"]:
    #     model = module_arch.LateAttentionFusionWithPreds(
    #         multimodalattention=config_dict.init_obj(
    #             ["architecture", "attention"], module_att
    #         )
    #     )
    else:
        model = module_arch.LateAttentionFusion(
            modality_embeddings=[config_dict.init_obj(("architecture", "modality_embeddings", modality), module_emb)
                                 for modality in config_dict["architecture"]["order"]
                                 ],
            multimodalattention=config_dict.init_obj(
                ["architecture", "attention"], module_att
            )
        )
    logger.info(model)

    # 5. prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config_dict["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # 6. get function handles of loss and metrics
    criterion = config_dict.init_obj(["training", "loss"],
                                     module_loss)  # getattr(module_loss, config_dict["training"]["loss"])
    criterion_unlabelled = config_dict.init_obj(["training", "unlabelled_loss"], module_loss)
    metrics = [
        getattr(module_metric, met) for met in config_dict["training"]["metrics"]
    ]

    # 7. build optimizer, learning rate scheduler.
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config_dict.init_obj(
        ["training", "optimizer"], torch.optim, trainable_params
    )

    weight_unlabelled = config_dict.init_obj(["training", "unlabelled_scheduler"], module_loss)

    lr_scheduler = None
    if config_dict["training"]["lr_scheduler"]["type"] is not None:
        logger.info("Learning rate scheduler is activated.")
        lr_scheduler = config_dict.init_obj(["training", "lr_scheduler"], torch.optim.lr_scheduler, optimizer)

    # 8. Load trainer and train
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
        CustomArgs(
            ["--lr", "--learning_rate"], type=float, target="training;optimizer;args;lr"
        ),
        CustomArgs(
            ["--bs", "--batch_size"],
            type=int,
            target="data;train;dataset;args;batch_size",
        ),
        CustomArgs(["--ri", "--run_id"], type=str, target="run_id"),
    ]
    config = ConfigParser.from_args(args, options=options, setting="train")
    main(config_dict=config)

# args.add_argument('-d', '--device', default=None, type=str,
#                   help='indices of GPUs to enable (default: all)')
