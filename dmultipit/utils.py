import json
from collections import OrderedDict
from itertools import repeat
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


def set_device(data, device):
    """Set torch.device to tensor or list of tensors"""
    if isinstance(data, list):
        data = [d.to(device) for d in data]
    else:
        data = data.to(device)
    return data


def collate_variable_size(batch):
    """
    Custom collate function to deal with batches with variable number of instances (e.g., Multiple Instance Learning)
    """
    data, target = [], []
    for item in batch:
        if isinstance(item[0], list):
            data.append([el.unsqueeze(0) for el in item[0]])
        else:
            data.append(item[0].unsqueeze(0))
        target.append(torch.FloatTensor([item[1]]))
    return [data, target]


def ensure_dir(dirname):
    """ Test whether a directory exists and create it if it is not the case."""
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_yaml(fname):
    """Read .yaml file"""
    with open(fname) as yaml_file:
        return yaml.safe_load(yaml_file)


def read_json(fname):
    """Read .json file"""
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_yaml(content, fname):
    """Write .yaml file"""
    content = _clean_nested_dict(content)
    with open(fname, "w") as yaml_file:
        yaml.safe_dump(
            content, yaml_file, default_flow_style=None
        )  # , default_flow_style=False)


def write_json(content, fname):
    """Write .json file"""
    fname = Path(fname)
    content = _clean_nested_dict(content)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are "
            "available on this machine."
        )
        n_gpu_use = n_gpu
    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    """ Custom metric tracker"""

    def __init__(self, keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            if col != "ema":
                self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        # if self.writer is not None:
        #     self.writer.add_scalar(key, value)
        self._data.total[key] += value * n if not np.isnan(value) else 0
        self._data.counts[key] += n if not np.isnan(value) else 0
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def update_writer(self, key):
        if self.writer is not None:
            self.writer.add_scalar(key, self.avg(key))

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def _clean_nested_dict(d):
    """
    Clean nested dictionary, converting numpy types into python types for saving dictionary in .json format"
    """
    if isinstance(d, (np.int8, np.int16, np.int32, np.int64)):
        return int(d)
    if isinstance(d, (np.float16, np.float32, np.float64)):
        return float(d)
    if isinstance(d, list):
        return [_clean_nested_dict(x) for x in d]
    if isinstance(d, dict):
        for key, value in d.items():
            d.update({key: _clean_nested_dict(value)})
    return d


def masked_softmax(input_tensor, input_mask, dim):
    """ Custom softmax function to deal with masked values """
    max_values = torch.max(
        torch.where(input_mask, input_tensor, torch.min(input_tensor)),
        dim=dim,
        keepdim=True,
    )[0]
    input_exp = torch.exp(input_tensor - max_values)
    input_exp_masked = torch.where(input_mask, input_exp, torch.tensor(0.0))
    output = input_exp_masked / torch.sum(input_exp, dim=dim, keepdim=True)

    norm = input_exp_masked.norm(p=2, dim=dim).mean()
    return output, norm
