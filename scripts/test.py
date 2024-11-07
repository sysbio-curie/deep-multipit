import argparse
import inspect
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from _utils import train_test_split, get_dataset, build_model

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import dmultipit.dataset.loader as module_data
import dmultipit.model.loss as module_loss
import dmultipit.model.metric as module_metric
from dmultipit.parse_config import ConfigParser
from dmultipit.testing import Testing
from dmultipit.utils import prepare_device

# filter RuntimeWarnings that appear when dealing with PowerTransformer within the pre-processing step for radiomic
# MSKCC data. We recommend not using this line at first as it may hide other issues.
# warnings.simplefilter(action="ignore", category=RuntimeWarning)


def main(config_dict):
    logger = config_dict.get_logger("test")

    # 1. Load the test dataset (apply the preprocessing of the training dataset if any)
    *list_raw_data_train, labels_train = config_dict.init_ftn(
        ["training_data", "loader"],
        module_data,
        order=config_dict["architecture"]['order'],
        keep_unlabelled=config_dict["training"]["pseudo_labelling"]
    )()

    # deal with train-validation split if any
    train_index = np.arange(len(labels_train))
    val_index = config_dict["training_data"]["val_index"]
    if val_index is not None:
        train_index = np.delete(train_index, val_index)

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

    training_dataset, *_ = train_test_split(
        train_index=train_index,
        test_index=val_index,
        labels=labels_train,
        list_raw_data=list_raw_data_train,
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
        keep_unlabelled=config_dict["training"]["pseudo_labelling"],
        rad_transform=rad_transform,
        radiomics=radiomics
    )

    *list_raw_data, labels = config_dict.init_ftn(["test_data", "loader"],
                                                  module_data,
                                                  order=config_dict["architecture"]['order'],
                                                  keep_unlabelled=False)()
    dataset, bool_mask_missing_test = get_dataset(
        labels=labels,
        list_raw_data=list_raw_data,
        dataset_name=config_dict["test_data"]["dataset"],
        list_unimodal_processings=training_dataset.list_unimodal_processings,
        multimodal_processing=training_dataset.multimodal_processing,
        indexes=np.arange(len(labels)),
        drop_modas=False,
        keep_unlabelled=False,
        radiomics=radiomics,
        rad_transform=training_dataset.rad_transform if config_dict["MSKCC"] else None,
    )

    # 2. load data loader
    data_loader = DataLoader(dataset=dataset)

    # 3. build model architecture then print to console
    device, _ = prepare_device(config_dict["n_gpu"])
    model = build_model(config_dict, device)

    # 4. Load checkpoint
    assert config_dict.resume is not None, "No existing checkpoint"
    logger.info("Loading checkpoint: {} ...".format(config_dict.resume))
    checkpoint = torch.load(config_dict.resume)
    state_dict = checkpoint["state_dict"]

    if config_dict["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # 5. get function handles of loss and metrics
    loss_fn = config_dict.init_obj(["testing", "loss"], module_loss)
    metric_fns = [getattr(module_metric, met) for met in config_dict["testing"]["metrics"]]

    # 6. load tester, test and save results
    testing = Testing(
        model=model,
        loss_ftn=loss_fn,
        metric_ftns=metric_fns,
        config=config_dict,
        device=device,
        data_loader=data_loader,
        intermediate_fusion=config_dict["architecture"]["intermediate_fusion"],
    )

    # 6.1 test
    testing.test(collect_a=config_dict["testing"]["save_attentions"],
                 collect_modalitypred=config_dict["testing"]["save_modality_predictions"]
                 )

    # 6.2 save modality predictions (use NaN values for samples with only missing modalities (bool_mask_missing_test))
    if config_dict["testing"]["save_modality_predictions"]:
        df_modalitypreds = pd.DataFrame(index=labels.index, columns=config_dict["architecture"]["order"])
        df_modalitypreds.loc[labels[~bool_mask_missing_test].index] = torch.vstack(testing.modalitypreds).numpy()
        df_modalitypreds["label"] = labels.copy()
        df_modalitypreds.to_csv(config_dict.save_dir / "modality_predictions.csv")
        del df_modalitypreds

    # 6.3 save attentions (use NaN values for samples with only missing modalities (bool_mask_missing_test))
    if config_dict["testing"]["save_attentions"]:
        df_att = pd.DataFrame(index=labels.index, columns=config_dict["architecture"]["order"])
        df_att.loc[labels[~bool_mask_missing_test].index] = torch.vstack(testing.attentions).numpy()
        df_att["label"] = labels.copy()
        df_att.to_csv(config_dict.save_dir / "attentions.csv")
        del df_att

    # 6.4 save outputs (use NaN values for samples with only missing modalities (bool_mask_missing_test))
    # distinguish cases where the sigmoid function is included in the model or not (to compute "probas")
    df_out = pd.DataFrame(index=labels.index, columns=["outputs", "probas"])
    if config_dict["architecture"]["intermediate_fusion"]:
        if config_dict["architecture"]["predictor"]["args"]["final_activation"] == "sigmoid":
            temp = torch.hstack((testing.outputs.view(-1, 1),  testing.outputs.view(-1, 1)))
        else:
            temp = torch.hstack((testing.outputs.view(-1, 1), torch.sigmoid(testing.outputs.view(-1, 1))))
    else:
        only_sigmoid = True
        for moda in config_dict["architecture"]["order"]:
            if config_dict["architecture"]["modality_embeddings"][moda]["args"]["final_activation"] != "sigmoid":
                only_sigmoid = False
                break
        if only_sigmoid:
            temp = torch.hstack((testing.outputs.view(-1, 1), testing.outputs.view(-1, 1)))
        else:
            temp = torch.hstack((testing.outputs.view(-1, 1), torch.sigmoid(testing.outputs.view(-1, 1))))

    df_out.loc[labels[~bool_mask_missing_test].index] = temp
    df_out["label"] = labels.copy()
    df_out.to_csv(config_dict.save_dir / "predictions.csv")
    del df_out


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Multimodal Fusion")
    args.add_argument(
        "-e",
        "--experiment",
        default=None,
        type=str,
        help="experiment file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )

    config = ConfigParser.from_args(args, setting="test")
    main(config_dict=config)
