import argparse
import inspect
import os
import sys
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from _utils import train_val_test_split, build_model

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import dmultipit.model.loss as module_loss
import dmultipit.model.metric as module_metric

# import dmultipit.dataset.dataset as module_dataset
import dmultipit.dataset.loader as module_data
from dmultipit.trainer import Trainer
from dmultipit.parse_config import ConfigParser
from dmultipit.testing import Testing
from dmultipit.utils import prepare_device

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


def main(config_dict):
    # 0. fix random seeds for reproducibility
    seed = config_dict["cross_val"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    device, _ = prepare_device(config_dict["n_gpu"])
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    logger = config_dict.get_logger("cross_validation", verbosity=0)

    list_modas = config_dict["architecture"]['order']
    drop_modas = config_dict["cross_val_data"]["drop_modalities"]  # save drop_modas parameter

    names, indices = [], []
    for i in range(1, len(list_modas) + 1):
        for comb in combinations(range(len(list_modas)), i):
            indices.append(comb)
            names.append("+".join([list_modas[c] for c in comb]))

    # 1. Initialize data set which will be used for cross validation

    # whether to perform pseudo-labelling or not
    keep_unlabelled = config_dict["training"]["pseudo_labelling"]

    *list_raw_data, labels = config_dict.init_ftn(
        ["cross_val_data", "loader"],
        module_data,
        order=config_dict["architecture"]['order'],
        keep_unlabelled=keep_unlabelled
    )()

    list_outputs = []
    for r in range(config_dict["cross_val"]["n_repeats"]):
        print("repeat: ", r + 1)
        # 2. Initialize cross-validation splits
        if config_dict["cross_val"]["strategy"] == "StratifiedKFold":
            cv = StratifiedKFold(n_splits=config_dict["cross_val"]["n_splits"], shuffle=True)
        elif config_dict["cross_val"]["strategy"] == "KFold":
            cv = KFold(n_splits=config_dict["cross_val"]["n_splits"], shuffle=True)
        else:
            raise ValueError("Only 'StratifiedKFold' and 'KFold' cross-validation schemes are available")

        logger.info(
            config_dict["cross_val"]["strategy"] + " cross-validation scheme"
                                                   " n_splits = " + str(config_dict["cross_val"]["n_splits"])
        )

        df_preds = pd.DataFrame(index=labels.index, columns=["fold_index"] + names + ["label"])
        df_preds["label"] = labels.copy()
        df_preds["repeat"] = r

        for m, ind in enumerate(indices):
            print("model: ", names[m])

            if drop_modas and len(ind) == 1:
                config_dict["cross_val_data"]["drop_modalities"] = False
            else:
                config_dict["cross_val_data"]["drop_modalities"] = drop_modas

            config_dict["architecture"]['order'] = [list_modas[i] for i in ind]
            list_data = [list_raw_data[i] for i in ind]
            # # 3. build model architecture, then print to console
            # if config_dict["architecture"]["intermediate_fusion"]:
            #
            #     model = module_arch.InterAttentionFusion(
            #         modality_embeddings=[config_dict.init_obj(("architecture", "modality_embeddings", modality), module_emb)
            #                              for modality in config_dict["architecture"]["order"]
            #                              ],
            #         attention=config_dict.init_obj(
            #             ["architecture", "attention"], module_att
            #         ),
            #         predictor=config_dict.init_obj(["architecture", "predictor"], module_emb),
            #     )
            # else:
            #     model = module_arch.LateAttentionFusion(
            #         modality_embeddings=[config_dict.init_obj(("architecture", "modality_embeddings", modality), module_emb)
            #                              for modality in config_dict["architecture"]["order"]
            #                              ],
            #         multimodalattention=config_dict.init_obj(
            #             ["architecture", "attention"], module_att
            #         )
            #     )
            #
            # logger.info(model)

            # 5. get function handles of loss and metrics for training and test
            training_criterion = config_dict.init_obj(["training", "loss"], module_loss)
            training_criterion_unlabelled = config_dict.init_obj(["training", "unlabelled_loss"], module_loss)
            training_metrics = [getattr(module_metric, met) for met in config_dict["training"]["metrics"]]
            test_metrics = [getattr(module_metric, met) for met in config_dict["testing"]["metrics"]]

            # 6. Initialize output dictionary to save cv results
            outputs = {"test_" + met.__name__: [] for met in test_metrics}
            outputs.update({"train_" + met.__name__: [] for met in training_metrics})
            outputs["test_" + training_criterion.__class__.__name__] = []
            outputs["train_" + training_criterion.__class__.__name__] = []
            list_preds = []

            # 7. Main cross-validation scheme
            # for fold_index, (train_index, test_index) in enumerate(cv.split(np.zeros(len(dataset)), dataset.labels)):
            for fold_index, (train_index, test_index) in enumerate(
                    tqdm(cv.split(np.zeros(len(labels)), np.where(~np.isnan(labels), labels, 2)),
                         leave=False,
                         total=cv.get_n_splits(np.zeros(len(labels))))
            ):

                # # 7.1 create log subdirectory for tracking the training of each fold
                # new_log_dir = None
                # if config_dict["training"]["tensorboard"]:
                #     cv_directory = "cv_fold_" + str(fold_index)
                #     new_log_dir = config_dict.log_dir / cv_directory
                #     new_log_dir.mkdir(parents=True, exist_ok=True)

                if config_dict["cross_val_data"]["ensembling"]["strategy"] is None:

                    new_log_dir = None
                    if config_dict["training"]["tensorboard"]:
                        cv_directory = "cv_fold_" + str(fold_index)
                        new_log_dir = config_dict.log_dir / cv_directory
                        new_log_dir.mkdir(parents=True, exist_ok=True)

                    if config_dict["cross_val_data"]["validation_split"] is not None:
                        cv_val = StratifiedShuffleSplit(n_splits=1,
                                                        test_size=config_dict["cross_val_data"]["validation_split"])
                        train, val = next(cv_val.split(np.zeros(len(labels[train_index])),
                                                       np.where(~np.isnan(labels[train_index]), labels[train_index],
                                                                2)))
                        train_train_index, train_val_index = train_index[train], train_index[val]
                    else:
                        train_train_index, train_val_index = train_index, None

                    trainer, test_data_loader = _train_cv(config_dict,
                                                          train_train_index,
                                                          train_val_index,
                                                          test_index,
                                                          fold_index,
                                                          list_data,  # list_raw_data
                                                          labels,
                                                          # model,
                                                          outputs,
                                                          new_log_dir,
                                                          keep_unlabelled,
                                                          training_criterion,
                                                          training_metrics,
                                                          training_criterion_unlabelled,
                                                          device,
                                                          logger,
                                                          ensembling_index=None,
                                                          mskcc=config_dict["MSKCC"])

                    assert len(os.listdir(trainer.checkpoint_dir)) == 1, "there should be one unique checkpoint"
                    checkpoints = [os.path.join(trainer.checkpoint_dir, c_path) for c_path in
                                   os.listdir(trainer.checkpoint_dir) if c_path.endswith(".pth")][0]
                    test_data_loaders = test_data_loader
                    models = trainer.model

                else:

                    if config_dict["cross_val_data"]["ensembling"]["strategy"] == "StratifiedKFold":
                        cv_val = StratifiedKFold(n_splits=config_dict["cross_val_data"]["ensembling"]["n_splits"],
                                                 shuffle=True)
                    elif config_dict["cross_val_data"]["ensembling"]["strategy"] == "KFold":
                        cv_val = KFold(n_splits=config_dict["cross_val_data"]["ensembling"]["n_splits"], shuffle=True)
                    else:
                        raise ValueError("Only 'StratifiedKFold' and 'KFold' cross-validation schemes are available")

                    test_data_loaders = []
                    models = torch.nn.ModuleList()

                    for ensembling_index, (train, val) in enumerate(cv_val.split(np.zeros(len(labels[train_index])),
                                                                                 np.where(
                                                                                     ~np.isnan(labels[train_index]),
                                                                                     labels[train_index], 2))):
                        train_train_index, train_val_index = train_index[train], train_index[val]

                        # create log subdirectory for tracking the training of each fold
                        new_log_dir = None
                        if config_dict["training"]["tensorboard"]:
                            cv_directory = os.path.join("cv_fold_" + str(fold_index),
                                                        "ensembling_" + str(ensembling_index))
                            new_log_dir = config_dict.log_dir / cv_directory
                            new_log_dir.mkdir(parents=True, exist_ok=True)

                        trainer, test_data_loader = _train_cv(config_dict,
                                                              train_train_index,
                                                              train_val_index,
                                                              test_index,
                                                              fold_index,
                                                              list_data,  # list_raw_data
                                                              labels,
                                                              # model,
                                                              outputs,
                                                              new_log_dir,
                                                              keep_unlabelled,
                                                              training_criterion,
                                                              training_metrics,
                                                              training_criterion_unlabelled,
                                                              device,
                                                              logger,
                                                              ensembling_index,
                                                              mskcc=config_dict["MSKCC"])

                        test_data_loaders.append(test_data_loader)
                        models.append(trainer.model)

                    checkpoints = [os.path.join(trainer.checkpoint_dir, c_path) for c_path in
                                   os.listdir(trainer.checkpoint_dir) if c_path.endswith(".pth")]

                # 7.8 test and save test results

                # checkpoints = os.listdir(trainer.checkpoint_dir)
                # if len(checkpoints) > 1:
                #     test_model = []
                #     for i, checkpoint_path in enumerate(checkpoints):
                #         checkpoint = torch.load(os.path.join(trainer.checkpoint_dir, checkpoint_path))
                #         new_model = copy.deepcopy(model)
                #         logger.info("Loading best model for ensembling " + str(i) + " from epoch " + str(checkpoint["epoch"]))
                #         state_dict = checkpoint["state_dict"]
                #         test_model.append(new_model.load_state_dict(state_dict))
                # else:
                #     checkpoint = torch.load(os.path.join(trainer.checkpoint_dir, checkpoints[0]))
                #     new_model = copy.deepcopy(model)
                #     logger.info("Loading best model for ensembling " + str(0) + " from epoch " + str(checkpoint["epoch"]))
                #     state_dict = checkpoint["state_dict"]
                #     test_model = new_model.load_state_dict(state_dict)

                # if config_dict["training"]["save_best_only"]:
                #     checkpoint = torch.load(trainer.checkpoint_dir / "model_best_" + str(ensembling_index) + ".pth")
                #
                #     logger.info("Loading best model from epoch " + str(checkpoint["epoch"]))
                #     state_dict = checkpoint["state_dict"]
                #     model.load_state_dict(state_dict)

                testing = Testing(
                    model=models,  # model
                    loss_ftn=training_criterion,
                    metric_ftns=test_metrics,
                    config=config_dict,
                    device=device,
                    data_loader=test_data_loaders,
                    intermediate_fusion=config_dict["architecture"]["intermediate_fusion"],
                    checkpoints=checkpoints,
                    disable_tqdm=True
                )

                testing.test(collect_a=False,
                             # collect_modalityemb=False,
                             # collect_fusedemb=False,
                             collect_modalitypred=False)

                # save test results
                for key, value in testing.total_metrics.items():
                    outputs["test_" + key].append(value)
                outputs["test_" + training_criterion.__class__.__name__].append(testing.total_loss)

                df_preds.loc[labels.index.values[test_index], names[m]] = testing.outputs.cpu().numpy().reshape(-1)
                # df_preds = pd.DataFrame(np.hstack((testing.outputs.numpy().reshape(-1, 1),
                #                                    testing.targets.numpy().reshape(-1, 1))),
                #                         columns=['+'.join(config_dict["architecture"]['order']), 'label'],
                #                         index=labels.index.values[test_index])
                df_preds.loc[labels.index.values[test_index], "fold_index"] = fold_index
                # list_preds.append(df_preds)
        # df_preds_repeat = pd.concat(list_preds, axis=0).loc[labels.index]
        # df_preds_repeat[labels.index.values[test_index], "repeat"] = r
        list_outputs.append(df_preds)
    # print(config_dict.save_dir)
    #
    # with open(config_dict.save_dir / "results_cv.json", "w") as f:
    #     json.dump(outputs, f)
    #
    # df_cv = pd.DataFrame(
    #     {key: value for key, value in outputs.items() if key.split("_")[0] == 'test'},
    #     index=["cv_fold_" + str(i) for i in range(cv.get_n_splits())]
    # )
    # df_cv.to_csv(config_dict.save_dir / "results_cv.csv")
    pd.concat(list_outputs, axis=0).to_csv(config_dict.save_dir / "predictions.csv")


def _train_cv(config_dict,
              train_train_index,
              train_val_index,
              test_index,
              fold_index,
              list_raw_data,
              labels,
              outputs,
              new_log_dir,
              keep_unlabelled,
              training_criterion,
              training_metrics,
              training_criterion_unlabelled,
              device,
              logger,
              ensembling_index,
              mskcc=False):
    # deal with radiomics data for MSKCC
    radiomics, rad_transform = None, None
    if mskcc:
        rad_transform = config_dict["radiomics_transform"]
        radiomics_list = []
        for item in ["radiomics_PL", "radiomics_LN", "radiomics_PC"]:
            try:
                radiomics_list.append(config_dict["architecture"]["order"].index(item))
            except ValueError:
                pass
        radiomics = int(np.min(radiomics_list)) if len(radiomics_list) > 0 else None

    # Initialize train, test and validation datasets
    training_data, training_data_unlabelled, val_data, test_data = train_val_test_split(
        train_index=train_train_index,
        val_index=train_val_index,
        test_index=test_index,
        # split_test=test_index,
        # split_val=val_index, #config_dict["cross_val_data"]["validation_split"],
        labels=labels,
        list_raw_data=list_raw_data,
        dataset_name=config_dict["cross_val_data"]["dataset"],
        list_processings=[config_dict["cross_val_data"]["processing"][modality]
                          for modality in config_dict["architecture"]["order"]],
        drop_modas=config_dict["cross_val_data"]["drop_modalities"],
        keep_unlabelled=keep_unlabelled,
        rad_transform=rad_transform,
        radiomics=radiomics
    )

    # 7.3 load train, validation and test data loaders
    training_data_loader_kwargs = config_dict["cross_val_data"]["training_data_loader"].copy()

    if config_dict["cross_val_data"]["sampler"]:
        sample_weights = training_data.sample_weights
        assert sample_weights is not None, "sampler is only available for binary classification " \
                                           "setting, check your target or set sampler to False"
        assert len(sample_weights) == len(training_data), "sample_weights should be the same length" \
                                                          " as your data set"

        training_data_loader_kwargs['sampler'] = WeightedRandomSampler(weights=sample_weights,
                                                                       num_samples=len(training_data),
                                                                       replacement=True)
    train_data_loader = DataLoader(
        dataset=training_data, **training_data_loader_kwargs
    )

    unlabelled_data_loader = None
    if training_data_unlabelled is not None:
        unlabelled_data_loader = DataLoader(dataset=training_data_unlabelled,
                                            batch_size=len(training_data_unlabelled))

    valid_data_loader = (
        DataLoader(dataset=val_data, batch_size=len(val_data))
        if val_data is not None
        else None
    )

    # test_data_loaders.append(DataLoader(dataset=test_data))
    test_data_loader = DataLoader(dataset=test_data)

    # update architecture for mskcc data (using config dictionary)
    if mskcc and len({"radiomics_PL", "radiomics_PC", "radiomics_LN"} & set(config_dict["architecture"]['order'])) > 0:
        input_attentions = config_dict["architecture"]["attention"]["args"]["dim_input"]
        for modality in {"radiomics_PL", "radiomics_PC", "radiomics_LN"}:
            if modality in config_dict["architecture"]['order']:
                temp = len(training_data.rad_transform.selected_features_[modality.split('_')[1]])
                config_dict["architecture"]["modality_embeddings"][modality]["args"]["dim_input"] = temp
                input_attentions[config_dict["architecture"]['order'].index(modality)] = temp
        config_dict["architecture"]["attention"]["args"]["dim_input"] = input_attentions

    # update architecture for pathomics data (using config dictionary)
    if "pathomics" in config_dict["architecture"]["order"]:
        idx = config_dict["architecture"]['order'].index("pathomics")
        if 'selection' in [name for name, _ in training_data.list_processings[idx].steps]:
            input_attentions = config_dict["architecture"]["attention"]["args"]["dim_input"]
            temp = len(training_data.list_processings[idx]['selection'].features_)
            config_dict["architecture"]["modality_embeddings"]["pathomics"]["args"]["dim_input"] = temp
            input_attentions[idx] = temp
            config_dict["architecture"]["attention"]["args"]["dim_input"] = input_attentions

    # Reset model weights
    # new_model = copy.deepcopy(model)
    # model.reset_weights()
    model = build_model(config_dict, device)  # , logger=logger
    # Build optimizer, learning rate scheduler.
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config_dict.init_obj(["training", "optimizer"], torch.optim, trainable_params)
    weight_unlabelled = config_dict.init_obj(["training", "unlabelled_scheduler"], module_loss)

    lr_scheduler = None
    if config_dict["training"]["lr_scheduler"]["type"] is not None:
        logger.info("Learning rate scheduler is activated.")
        lr_scheduler = config_dict.init_obj(["training", "lr_scheduler"], torch.optim.lr_scheduler, optimizer)

    if config_dict["training"]["balanced_weights"] and config_dict["training"]["loss"]["type"] == "BCELogitLoss":
        weight = (training_data.labels == 0).sum() / (training_data.labels == 1).sum()
        setattr(training_criterion, 'pos_weight', torch.tensor(weight))
        logger.info("Balanced weigths for BCE with logits loss (weight: " + str(np.round(weight, 4)) + ").")

    trainer = Trainer(
        model,
        training_criterion,
        training_metrics,
        optimizer,
        config=config_dict,
        device=device,
        data_loader=train_data_loader,
        valid_data_loader=valid_data_loader,
        log_dir=new_log_dir,
        unlabelled_data_loader=unlabelled_data_loader,
        weight_unlabelled=weight_unlabelled,
        criterion_unlabelled=training_criterion_unlabelled,
        lr_scheduler=lr_scheduler,
        ensembling_index=ensembling_index,
        save_architecture=mskcc
    )

    # if config_dict["training"]["save_best_only"]:
    cv_directory = "cv_fold_" + str(fold_index)
    setattr(trainer, "checkpoint_dir", config_dict.model_dir / cv_directory)
    trainer.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if ensembling_index is not None:
        logger.info("Train and test fold " + str(fold_index) + ", ensembling " + str(ensembling_index))
    else:
        logger.info("Train and test fold " + str(fold_index))
    trainer.train()

    # 7.7 save training results
    for key, value in trainer.train_metrics.result().items():
        if key == "loss":
            outputs["train_" + training_criterion.__class__.__name__].append(value)
        else:
            try:
                outputs["train_" + key].append(value)
            except KeyError:
                pass
    return trainer, test_data_loader


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
