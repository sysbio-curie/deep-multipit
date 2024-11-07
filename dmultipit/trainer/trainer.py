import warnings

import numpy as np
import torch

from dmultipit.base import BaseTrainer
from dmultipit.utils import inf_loop, MetricTracker, set_device


class Trainer(BaseTrainer):
    """
    Trainer class (with optionnal semi-supervised / pseudo-labelling strategy)

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

    device: str
        Torch.device on which to allocate tensors

    data_loader: DataLoader
        Training data

    valid_data_loader: DataLoader or None
        Validation data. If None, no validation is performed. The default is None.

    unlabelled_data_loader: Dataloader or None.
        Unlabelled data for semi-supervised strategy (i.e., pseudo-labelling). If None, no unlabelled data is added to
        the training. The default is None.

    weight_unlabelled: Callable or None
        Function to specify the weight of the unlabelled step depending on the training epoch (e.g., 0 until epoch 50
        and 1 after). See dmultipit.model.loss.StepScheduler. The default is None.

    criterion_unlabelled: Callable or None
        Criterion for pseudo-labelling strategy. See dmultipit.model.loss.UnlabelledBCELoss. The default is None.

    lr_scheduler: torch.optim.lr_scheduler object or None
        Learning rate scheduler. If None the learning rate is kept constant throughout training. The default is None.

    len_epoch: Int or None.
        Number of batches within an epoch. If None, len_epoch will be set to the length of the provided data_loader.
        The default is None.

    log_dir: string or None.
        Path to the directory to save Tensorboard logs in. The default is None.

    ensembling_index: int or None
        Index to associate the trained/saved model to an ensemble of other trained models for further ensembling
        strategies. The default is None.

    save_architecture: bool
        If true, save model architecture as .yaml file. The default is False.

    disable_checkpoint: bool
        If true, no checkpoint is saved. The default is False.
    """

    def __init__(
            self,
            model,
            criterion,
            metric_ftns,
            optimizer,
            config,
            device,
            data_loader,
            valid_data_loader=None,
            unlabelled_data_loader=None,
            weight_unlabelled=None,
            criterion_unlabelled=None,
            lr_scheduler=None,
            len_epoch=None,
            log_dir=None,
            ensembling_index=None,
            save_architecture=False,
            disable_checkpoint=False,
    ):

        if log_dir is None:
            log_dir = config.log_dir

        super().__init__(
            model,
            criterion,
            metric_ftns,
            optimizer,
            config,
            log_dir,
            ensembling_index,
            save_architecture,
            disable_checkpoint,
        )
        self.config = config
        self.device = device
        self.data_loader = data_loader

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        # initiate pseudo-labelling
        self.unlabelled_data_loader = unlabelled_data_loader
        self.pseudo_labelling = self.unlabelled_data_loader is not None
        self.criterion_unlabelled = criterion_unlabelled
        self.weigth_unlabelled = weight_unlabelled
        if (self.pseudo_labelling
                and ((self.criterion_unlabelled is None) or (self.weigth_unlabelled is None))):
            raise ValueError("When unlabelled data are passed for semi-supervised learning criterion_unlabelled and"
                             "weight_unlabelled must be specified (i.e., not set to None).")
        elif ((not self.pseudo_labelling)
              and ((self.criterion_unlabelled is not None) or (self.weigth_unlabelled is not None))):
            warnings.warn("criterion_unlabelled and/or weight_unlabelled are specified but no unlabelled_data_loader"
                          "is passed. No semi-supervised strategy will be performed and criterion_unlabelled and"
                          "weight_unlabelled will be reset to None.")
            self.criterion_unlabelled = None
            self.weigth_unlabelled = None

        # initiate training metrics tracking
        metric_keys = ["loss"] + [m.__name__ for m in self.metric_ftns]
        self.valid_metrics = MetricTracker(metric_keys, writer=self.writer)
        if self.pseudo_labelling:
            metric_keys.append("unlabelled_loss")
        self.train_metrics = MetricTracker(metric_keys, writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        Parameters
        ----------
        epoch: Int
            Current training epoch.

        Returns
        -------
        log: A log that contains average loss and metric in this epoch.
        """

        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (*list_modas, mask, target) in enumerate(self.data_loader):
            list_modas, mask, target = (
                set_device(list_modas, self.device),
                set_device(mask, self.device),
                set_device(target, self.device),
            )
            self.optimizer.zero_grad()

            output = self.model(list_modas, mask)
            loss = self.criterion(output, target)

            # Add optional L2 penalties (for model weights and/or attention weights)
            if self.config["training"]["l2_penalty"] is not None:
                l2_penalty = torch.stack(
                    [p.norm(p=2) for n, p in self.model.named_parameters() if "weight" in n]
                ).sum()  # **2
                loss += self.config["training"]["l2_penalty"] * l2_penalty
            if self.config["training"]["attention_penalty"] is not None:
                att_penalty = self.model.multimodalattention.attention_norm
                loss += self.config["training"]["attention_penalty"] * att_penalty

            loss.backward()
            self.optimizer.step()

            # self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update("loss", loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), loss.item()
                    )
                )

            if batch_idx == self.len_epoch:
                break

        # Semi-supervised / pseudo-labelling step
        if self.pseudo_labelling:
            self._train_unlabelled_epoch(epoch)

        log = self.train_metrics.result()

        # add training metrics to tensorboard
        self.writer.set_step(epoch)
        self.train_metrics.update_writer("loss")
        for met in self.metric_ftns:
            self.train_metrics.update_writer(met.__name__)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")

        # perform validation
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        # adjust learning rate
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _train_unlabelled_epoch(self, epoch):
        """
        Train on unlabelled data

        Parameters
        ----------
        epoch: Int
            Current epoch
        """

        for batch_idx, (*list_modas, mask, _) in enumerate(self.unlabelled_data_loader):
            list_modas, mask = set_device(list_modas, self.device), set_device(mask, self.device)

            # compute the pseudo_labels
            self.model.eval()
            output_unlabelled = self.model(list_modas, mask)

            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(list_modas, mask)
            loss = self.weigth_unlabelled(epoch) * self.criterion_unlabelled(output, output_unlabelled)
            loss.backward()
            self.optimizer.step()

            self.train_metrics.update("unlabelled_loss", loss.item())
        return

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        Parameters
        ----------
        epoch: Int
            Current training epoch.

        Returns
        -------
        log: A log that contains information about validation
        """

        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (*list_modas, mask, target) in enumerate(self.valid_data_loader):
                list_modas, mask, target = (
                    set_device(list_modas, self.device),
                    set_device(mask, self.device),
                    set_device(target, self.device),
                )
                output = self.model(list_modas, mask)
                loss = self.criterion(output, target)

                # self.writer.set_step(
                #     (epoch - 1) * len(self.valid_data_loader) + batch_idx, "valid"
                # )
                self.valid_metrics.update("loss", loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))

        # add validation metrics to tensorboard
        self.writer.set_step(epoch, "valid")
        self.valid_metrics.update_writer("loss")
        for met in self.metric_ftns:
            self.valid_metrics.update_writer(met.__name__)

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
