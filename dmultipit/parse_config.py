import logging
from datetime import datetime
from functools import reduce, partial
from operator import getitem
from pathlib import Path

import torch

from dmultipit.logger import setup_logging
from dmultipit.utils import read_yaml, write_yaml


class ConfigParser:
    """
    class to parse configuration json file. Handles hyperparameters for training, initializations of modules,
    checkpoint saving and logging module.

    Parameters
    ----------
    config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.

    resume: String, path to the checkpoint being loaded.

    modification: Dict keychain:value, specifying position values to be replaced from config dict.

    setting: string in ['train', 'test']

    """

    def __init__(self, config, resume=None, modification=None, setting="train"):
        # load config file and apply modification
        self._config = _update_config(config, modification)
        self.resume = resume
        save_dir = Path(self.config["save_dir"])

        exper_name = self.config["name"]

        # set save_dir where trained model and log will be saved.
        if setting == "train":

            if self.config["run_id"] is None:  # use timestamp as default run-id
                self.config["run_id"] = datetime.now().strftime(r"%m%d_%H%M%S")

            self._save_dir = save_dir / "models" / exper_name / self.config["run_id"]
            self._log_dir = save_dir / "log" / exper_name / self.config["run_id"]

            # make directory for saving checkpoints and log.
            exist_ok = self.config["run_id"] == ""
            self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
            self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

            setup_logging(self.log_dir)

        elif setting == "test":

            if self.config["exp_id"] is None:  # use timestamp as default run-id
                self.config["exp_id"] = "exp_" + datetime.now().strftime(r"%m%d_%H%M%S")

            self._save_dir = (
                save_dir
                / "results"
                / exper_name
                / self.config["run_id"]
                / self.config["exp_id"]
            )
            self.save_dir.mkdir(parents=True, exist_ok=True)

            logging.basicConfig(level=logging.INFO)

        elif setting == "cross_val":

            if self.config["exp_id"] is None:  # use timestamp as default run-id
                self.config["exp_id"] = "exp_" + datetime.now().strftime(r"%m%d_%H%M%S")

            self._log_dir = save_dir / "log" / exper_name / self.config["exp_id"]
            self._save_dir = save_dir / "results" / exper_name / self.config["exp_id"]
            self.model_dir = save_dir / "models" / exper_name / self.config["exp_id"]

            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.model_dir.mkdir(parents=True, exist_ok=True)

            setup_logging(self.log_dir)

        else:
            raise ValueError("setting should be either 'train', 'test' or 'cross_val'")

        # save updated config file to the checkpoint dir
        write_yaml(self.config, self.save_dir / "config.yaml")

        # set log_levels
        self.log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}

    @classmethod
    def from_args(cls, args, setting, options=""):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        if setting == "train":
            msg = (
                "Configuration file and resume file cannot be specified at the same time. For training please "
                "specify one of the two. For testing only resume file should be specified. "
            )
            assert (args.config is None) or (args.resume is None), msg

        if (setting != "cross_val") and args.resume is not None:
            resume = Path(args.resume)
            checkpoint = torch.load(resume)
            config = checkpoint["config"].config

            # config = read_json(resume.parent / 'config.json')
            # config = read_yaml(resume.parent / 'config_late.yaml')

            # update configuration file with specific configurations for the experiment
            if args.experiment is not None:
                # config.update(read_json(args.experiment))
                config.update(read_yaml(args.experiment))
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config_late.yaml', for example."
            assert args.config is not None, msg_no_cfg
            resume = None

            # update configuration file with specific configurations for the experiment
            msg_no_exp = "Experiment file need to be specified. Add '-e config_exp.yaml', for example."
            assert args.experiment is not None, msg_no_exp
            # config = read_json(args.config)
            # config.update(read_json(args.experiment))
            config = read_yaml(args.config)
            config.update(read_yaml(args.experiment))

        # parse custom cli options into dictionary
        modification = {
            opt.target: getattr(args, _get_opt_name(opt.flags)) for opt in options
        }
        return cls(config, resume, modification, setting)

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        obj_config = _get_by_path(self, name)
        try:
            module_name = obj_config["type"]
        except KeyError:
            return None
        if module_name is None:
            return None

        module_args = {} if obj_config["args"] is None else dict(obj_config["args"])
        assert all(
            [k not in module_args for k in kwargs]
        ), "Overwriting kwargs given in config file is not allowed"
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        obj_config = _get_by_path(self, name)
        module_name = obj_config["type"]
        module_args = {} if obj_config["args"] is None else dict(obj_config["args"])
        # module_name = self[name]['type']
        # module_args = dict(self[name]['args'])
        assert all(
            [k not in module_args for k in kwargs]
        ), "Overwriting kwargs given in config file is not allowed"
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = "verbosity option {} is invalid. Valid options are {}.".format(
            verbosity, self.log_levels.keys()
        )
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir


# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith("--"):
            return flg.replace("--", "")
    return flags[0].replace("--", "")


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(";")
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
