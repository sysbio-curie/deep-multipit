import argparse
import inspect
import os
import sys
import itertools
from itertools import product

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from _utils import train_test_split, get_dataset

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import latefusatt.dataset.loader as module_data
import latefusatt.model.loss as module_loss
import latefusatt.model.metric as module_metric
from latefusatt.model import model as module_arch
import latefusatt.model.attentions as module_att
import latefusatt.model.embeddings as module_emb
from latefusatt.parse_config import ConfigParser
from latefusatt.testing import Testing
from latefusatt.utils import prepare_device


def main(config_dict):
    logger = config_dict.get_logger("test")

    # 1. Load the test dataset (apply the preprocessing of the training dataset if any)
    *list_raw_data_train, labels_train = config_dict.init_ftn(
        ["training_data", "loader"],
        module_data,
        order=config_dict["architecture"]['order'],
        keep_unlabelled=config_dict["training"]["pseudo_labelling"]
    )()
    training_dataset, *_ = train_test_split(
        split=config_dict["training_data"]["val_index"],
        labels=labels_train,
        list_raw_data=list_raw_data_train,
        dataset_name=config_dict["training_data"]["dataset"],
        list_processings=[config_dict["training_data"]["processing"][modality]
                          for modality in config_dict["architecture"]["order"]],
        drop_modas=config_dict["training_data"]["drop_modalities"],
        keep_unlabelled=config_dict["training"]["pseudo_labelling"]
    )

    *list_raw_data, labels = config_dict.init_ftn(
        ["test_data", "loader"], module_data,  order=config_dict["architecture"]['order']
    )()
    dataset = get_dataset(
        labels=labels,
        list_raw_data=list_raw_data,
        dataset_name=config_dict["test_data"]["dataset"],
        list_processings=training_dataset.list_processings,
        indexes=np.arange(len(labels)),
    )

    # 2. load data loader
    data_loader = DataLoader(dataset=dataset)

    # 3. build model architecture then print to console
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

    # 4. Load checkpoint
    assert config_dict.resume is not None, "No existing checkpoint"
    logger.info("Loading checkpoint: {} ...".format(config_dict.resume))
    checkpoint = torch.load(config_dict.resume)
    state_dict = checkpoint["state_dict"]

    if config_dict["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # 5. prepare model for testing
    device, _ = prepare_device(config_dict["n_gpu"])
    model = model.to(device)

    # 6. get function handles of loss and metrics
    loss_fn = config_dict.init_obj(["testing", "loss"], module_loss)
    metric_fns = [
        getattr(module_metric, met) for met in config_dict["testing"]["metrics"]
    ]

    # 7. load tester, test and save results
    testing = Testing(
        model=model,
        loss_ftn=loss_fn,
        metric_ftns=metric_fns,
        config=config_dict,
        device=device,
        data_loader=data_loader,
        intermediate_fusion=config_dict["architecture"]["intermediate_fusion"],
    )

    # 7.1 test
    testing.test(
        collect_a=config_dict["testing"]["save_attentions"],
        collect_modalityemb=config_dict["testing"]["save_modality_embeddings"],
        collect_fusedemb=config_dict["testing"]["save_fused_embeddings"],
        collect_modalitypred=config_dict["testing"]["save_modality_predictions"]
    )

    # 7.2 save embeddings
    if config_dict["architecture"]["intermediate_fusion"]:
        if config_dict["testing"]["save_modality_embeddings"]:

            indexes = [(i, j) for i in dataset.sample_names for j in config_dict["architecture"]["order"]]
            df_modalityembs = pd.DataFrame(
                torch.vstack(testing.modality_emb).numpy(),
                index=pd.MultiIndex.from_tuples(indexes, names=["sample", "modality"]),
            )
            df_modalityembs.to_csv(config_dict.save_dir / "modality_embeddings.csv")
            del df_modalityembs

        if config_dict["testing"]["save_fused_embeddings"]:

            df_fusedembs = pd.DataFrame(
                torch.vstack(testing.fused_emb).numpy(),
                index=dataset.sample_names,
            )
            df_fusedembs.to_csv(config_dict.save_dir / "fused_embeddings.csv")
            del df_fusedembs

    # 7.3 save modality predictions:
    if config_dict["testing"]["save_modality_predictions"]:

        df_modalitypreds = pd.DataFrame(torch.vstack(testing.modalitypreds).numpy(),
                                        index=dataset.sample_names,
                                        columns=config_dict["architecture"]["order"])
        df_modalitypreds.to_csv(config_dict.save_dir / "modality_predictions.csv")
        del df_modalitypreds

    # 7.4 save attentions
    if config_dict["testing"]["save_attentions"]:
        df_att = pd.DataFrame(
            torch.vstack(testing.attentions).numpy(),
            index=dataset.sample_names,
            columns=config_dict["architecture"]["order"],
        )
        df_att.to_csv(config_dict.save_dir / "attentions.csv")
        del df_att

    # 7.4 save outputs
    df_out = pd.DataFrame(
        torch.hstack(
            (
                testing.outputs.view(-1, 1),
                torch.sigmoid(testing.outputs.view(-1, 1)),
            )
        ),
        columns=["outputs", "probas"],
        index=dataset.sample_names,
    )
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

# args.add_argument('-d', '--device', default=None, type=str,
#                   help='indices of GPUs to enable (default: all)')
