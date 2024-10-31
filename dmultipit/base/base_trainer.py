from abc import abstractmethod

import torch
from numpy import inf

from dmultipit.logger import TensorboardWriter
from dmultipit.utils import write_yaml


class BaseTrainer:
    """
    Base class for all trainers

    Parameters
    ----------
    model: base_model.BaseModel object
        Model to train.

    criterion: callable with output and target inputs
        Training criterion (i.e., loss)

    metric_ftns: callable with output and target inputs
        Training and validation metrics to monitor training

    optimizer: torch.optim.Optimizer object
        Optimization algorithms (e.g., torch.optim.Adam, torch.optim.SGD)

    config: dict
        Configuration dictionnary

    log_dir: string
        Path to the directory to save Tensorboard logs in

    ensembling_index: int or None
        Index to associate the trained/saved model to an ensemble of other trained models for further ensembling
        strategies

    save_architecture: bool
        If true, save model architecture as .yaml file. The default is False.

    disable_checkpoint: bool
        If true, no checkpoint is saved. The default is False.
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, log_dir, ensembling_index,
                 save_architecture=False, disable_checkpoint=False):

        self.config = config
        self.logger = config.get_logger("trainer", config["training"]["verbosity"])

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config["training"]
        self.epochs = cfg_trainer["epochs"]
        self.save_period = cfg_trainer["save_period"]
        self.save_best_only = cfg_trainer["save_best_only"]
        self.print_period = cfg_trainer["print_period"]
        self.monitor = cfg_trainer.get("monitor", "off")

        # configuration to monitor model performance and save best model
        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
            self.mnt_ema = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max", "ema_min", "ema_max"]

            self.mnt_best = inf if self.mnt_mode in ["ema_min", "min"] else -inf

            self.mnt_ema_alpha = cfg_trainer.get("ema_alpha", 0)
            self.mnt_ema = 0

            self.early_stop = cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        self.writer = TensorboardWriter(
            log_dir, self.logger, cfg_trainer["tensorboard"]
        )

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

        self.ensembling_index = ensembling_index

        if save_architecture:
            if self.ensembling_index is not None:
                write_yaml(config["architecture"],
                           self.checkpoint_dir / "config_architecture_{}.yaml".format(self.ensembling_index))
            else:
                write_yaml(config["architecture"], self.checkpoint_dir / "config_architecture.yaml")
        self.disable_checkpoint = disable_checkpoint

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        Parameters
        ----------
        epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {"epoch": epoch}
            log.update(result)

            # print logged information to the screen
            if (self.print_period is not None) and (epoch % self.print_period == 0):
                for key, value in log.items():
                    self.logger.info("    {:15s}: {}".format(str(key), value))

            # evaluate model performance according to configured metric, save the best checkpoint as model_best
            best = False
            if self.mnt_mode != "off":
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    if self.mnt_mode in ["ema_min", "ema_max"]:
                        self.mnt_ema = (1-self.mnt_ema_alpha)*log[self.mnt_metric] + self.mnt_ema_alpha*self.mnt_ema
                        # debias EMA (see tensorboard)
                        debias_weight = (1 - self.mnt_ema_alpha**epoch) if self.mnt_ema_alpha < 1 else 1
                        improved = (
                                           self.mnt_mode == "ema_min" and self.mnt_ema/debias_weight <= self.mnt_best
                                   ) or (
                                           self.mnt_mode == "ema_max" and self.mnt_ema/debias_weight >= self.mnt_best
                                   )
                    else:
                        improved = (
                            self.mnt_mode == "min" and log[self.mnt_metric] <= self.mnt_best
                        ) or (
                            self.mnt_mode == "max" and log[self.mnt_metric] >= self.mnt_best
                        )
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. "
                        "Model performance monitoring is disabled.".format(
                            self.mnt_metric
                        )
                    )
                    self.mnt_mode = "off"
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric] if self.mnt_mode in ["min", "max"] else self.mnt_ema/debias_weight
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(
                        "Validation performance didn't improve for {} epochs. "
                        "Training stops.".format(self.early_stop)
                    )
                    break

            if not self.disable_checkpoint:
                if self.save_best_only:
                    if best:
                        self._save_checkpoint(epoch, save_best=best, save_best_only=True)
                elif (self.save_period is not None) and (epoch % self.save_period == 0):
                    self._save_checkpoint(epoch, save_best=best)

        # save last model (if save_best_only and save_period are disabled)
        if not self.disable_checkpoint:
            if (not self.save_best_only) and (self.save_period is None):
                self.logger.info("Save checkpoint from last epoch (i.e epoch " + str(self.epochs) + ")")
                self._save_checkpoint(self.epochs)

    def _save_checkpoint(self, epoch, save_best=False, save_best_only=False):
        """
        Saving checkpoints

        Parameters
        ----------
        epoch: current epoch number

        save_best: if True, rename the saved checkpoint to 'model_best.pth'

        save_best_only: if True, save only best model according to the monitoring metric
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        if not save_best_only:
            if self.ensembling_index is not None:
                filename = str(self.checkpoint_dir / "checkpoint-epoch{}_{}.pth".format(epoch, self.ensembling_index))
            else:
                filename = str(self.checkpoint_dir / "checkpoint-epoch{}.pth".format(epoch))
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))

        if save_best:
            if self.ensembling_index is not None:
                best_path = str(self.checkpoint_dir / "model_best_{}.pth".format(self.ensembling_index))
            else:
                best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            # self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        Parameters
        ----------
        resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["architecture"] != self.config["architecture"]:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is different from that of "
                "checkpoint. This may yield an exception while state_dict is being loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
            checkpoint["config"]["training"]["optimizer"]["type"]
            != self.config["training"]["optimizer"]["type"]
        ):
            self.logger.warning(
                "Warning: Optimizer type given in config file is different from that of checkpoint. "
                "Optimizer parameters not being resumed."
            )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
        )
