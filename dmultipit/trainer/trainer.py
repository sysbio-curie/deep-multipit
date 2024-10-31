import numpy as np
import torch

from dmultipit.base import BaseTrainer
from dmultipit.utils import inf_loop, MetricTracker, set_device


class Trainer(BaseTrainer):
    """
    Trainer class
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
            disable_checkpoint=False
    ):

        if log_dir is None:
            log_dir = config.log_dir

        super().__init__(model, criterion, metric_ftns, optimizer, config, log_dir, ensembling_index, save_architecture,
                         disable_checkpoint)
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

        self.unlabelled_data_loader = unlabelled_data_loader
        self.pseudo_labelling = self.unlabelled_data_loader is not None
        self.criterion_unlabelled = criterion_unlabelled
        self.weigth_unlabelled = weight_unlabelled

        metric_keys = ['loss'] + [m.__name__ for m in self.metric_ftns]
        self.valid_metrics = MetricTracker(metric_keys, writer=self.writer)
        if self.pseudo_labelling:
            metric_keys.append("unlabelled_loss")
        self.train_metrics = MetricTracker(metric_keys, writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (*list_modas, mask, target) in enumerate(self.data_loader):
            list_modas, mask, target = (set_device(list_modas, self.device), set_device(mask, self.device),
                                        set_device(target, self.device)
                                        )
            self.optimizer.zero_grad()

            output = self.model(list_modas, mask)
            loss = self.criterion(output, target)

            if self.config["training"]["l2_penalty"] is not None:
                l2_penalty = torch.stack(
                    [p.norm(p=2) for n, p in self.model.named_parameters() if 'weight' in n]).sum()  # **2
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

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _train_unlabelled_epoch(self, epoch):
        """
        Train on unlabelled data
        :param epoch:
        :return:
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

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (*list_modas, mask, target) in enumerate(
                    self.valid_data_loader
            ):
                list_modas, mask, target = (set_device(list_modas, self.device), set_device(mask, self.device),
                                            set_device(target, self.device)
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


class FastTrainer(BaseTrainer):
    def __init__(
            self,
            model,
            criterion,
            metric_ftns,
            optimizer,
            config,
            device,
            data_loader,
            log_dir=None,
            ensembling_index=None,
            save_architecture=False,
            disable_checkpoint=False
    ):

        if log_dir is None:
            log_dir = config.log_dir

        super().__init__(model, criterion, metric_ftns, optimizer, config, log_dir, ensembling_index, save_architecture,
                         disable_checkpoint)
        self.config = config
        self.device = device
        self.data_loader = data_loader

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        log = {}

        for batch_idx, (*list_modas, mask, target) in enumerate(self.data_loader):
            list_modas, mask, target = (set_device(list_modas, self.device), set_device(mask, self.device),
                                        set_device(target, self.device)
                                        )
            self.optimizer.zero_grad()

            output = self.model(list_modas, mask)
            loss = self.criterion(output, target)

            if self.config["training"]["l2_penalty"] is not None:
                l2_penalty = torch.stack(
                    [p.norm(p=2) for n, p in self.model.named_parameters() if 'weight' in n]).sum()  # **2
                loss += self.config["training"]["l2_penalty"] * l2_penalty
            if self.config["training"]["attention_penalty"] is not None:
                att_penalty = self.model.multimodalattention.attention_norm
                loss += self.config["training"]["attention_penalty"] * att_penalty

            loss.backward()
            self.optimizer.step()

        return log
