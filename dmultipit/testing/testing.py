import torch
from tqdm import tqdm

from dmultipit.utils import set_device


class Testing:
    """
    Testing class

    Parameters
    ----------
    model:

    loss_ftn:

    metric_ftns:

    config:

    device:

    data_loader:

    intermediate_fusion:

    checkpoints:

    disable_tqdm:

    """

    def __init__(
            self,
            model,
            loss_ftn,
            metric_ftns,
            config,
            device,
            data_loader,
            intermediate_fusion=False,
            checkpoints=None,
            disable_tqdm=False
    ):
        self.config = config
        self.device = device
        self.loss_ftn = loss_ftn
        self.metric_ftns = metric_ftns
        self.model = model
        self.intermediate_fusion = intermediate_fusion

        if isinstance(data_loader, list):
            for loader in data_loader:
                assert loader.batch_size == 1, "Batch_size for test data should be 1"
        else:
            assert data_loader.batch_size == 1, "Batch_size for test data should be 1"

        self.data_loader = data_loader

        self.logger = self.config.get_logger("test", verbosity=0)

        self.targets = None
        self.outputs = None
        self.total_metrics = None
        self.attentions = []
        self.modalitypreds = []
        self.fused_emb = []
        self.modality_emb = []

        self.checkpoints = checkpoints
        self.disable_tqdm = disable_tqdm

    def test(self, collect_a=True, collect_modalitypred=True):
        if self.checkpoints is None:
            self.targets, self.outputs, self.attentions, self.modalitypreds = _predict(self.model,
                                                                                       self.data_loader,
                                                                                       self.device,
                                                                                       collect_a,
                                                                                       collect_modalitypred,
                                                                                       self.intermediate_fusion,
                                                                                       disable_tqdm=self.disable_tqdm)
        elif isinstance(self.checkpoints, str):
            checkpoint = torch.load(self.checkpoints)
            self.logger.info("Loading best model from epoch " + str(checkpoint["epoch"]))
            state_dict = checkpoint["state_dict"]
            self.model.load_state_dict(state_dict)
            self.targets, self.outputs, self.attentions, self.modalitypreds = _predict(self.model,
                                                                                       self.data_loader,
                                                                                       self.device,
                                                                                       collect_a,
                                                                                       collect_modalitypred,
                                                                                       self.intermediate_fusion,
                                                                                       disable_tqdm=self.disable_tqdm)
        elif isinstance(self.checkpoints, list):
            assert isinstance(self.data_loader, list), ""
            outputs = []
            i = 0
            if isinstance(self.model, torch.nn.ModuleList):
                for checkpoint_path, data_loader in zip(self.checkpoints, self.data_loader):
                    checkpoint = torch.load(checkpoint_path)
                    self.logger.info(
                        "Loading best model for ensembling " + str(i) + " from epoch " + str(checkpoint["epoch"]))
                    state_dict = checkpoint["state_dict"]
                    self.model[i].load_state_dict(state_dict)
                    self.targets, outputs_ensembing, *_ = _predict(self.model[i],
                                                                   data_loader,
                                                                   self.device,
                                                                   collect_attentions=False,
                                                                   collect_modalitypred=False,
                                                                   intermediate_fusion=self.intermediate_fusion,
                                                                   disable_tqdm=self.disable_tqdm)
                    outputs.append(outputs_ensembing)
                    i += 1
            else:
                for checkpoint_path, data_loader in zip(self.checkpoints, self.data_loader):
                    checkpoint = torch.load(checkpoint_path)
                    self.logger.info(
                        "Loading best model for ensembling " + str(i) + " from epoch " + str(checkpoint["epoch"]))
                    state_dict = checkpoint["state_dict"]
                    self.model.load_state_dict(state_dict)
                    self.targets, outputs_ensembing, *_ = _predict(self.model,
                                                                   data_loader,
                                                                   self.device,
                                                                   collect_attentions=False,
                                                                   collect_modalitypred=False,
                                                                   intermediate_fusion=self.intermediate_fusion,
                                                                   disable_tqdm=self.disable_tqdm)
                    outputs.append(outputs_ensembing)
                    i += 1
            self.outputs = torch.mean(torch.stack(outputs, dim=1), dim=1)
        else:
            raise ValueError("checkpoints should be None, path or list of paths")
        self.total_loss = self.loss_ftn(self.outputs, self.targets).item()
        self.total_metrics = {
            met.__name__: met(self.outputs.squeeze(), self.targets.squeeze())
            for met in self.metric_ftns
        }
        log = {"loss": self.total_loss}
        log.update(self.total_metrics)
        self.logger.info(log)
        return


def _predict(model, data_loader, device, collect_attentions, collect_modalitypred, intermediate_fusion=False,
             disable_tqdm=False):
    targets, outputs, attentions, modalitypreds = [], [], [], []
    model.eval()
    with torch.no_grad():
        for *list_modas, mask, target in tqdm(data_loader, disable=disable_tqdm):
            list_modas, mask, target = (set_device(list_modas, device), set_device(mask, device),
                                        set_device(target, device)
                                        )
            targets.append(target)
            outputs.append(model(list_modas, mask))
            if collect_attentions:
                attentions.append(model.attention_weights_.view(1, -1))
            if collect_modalitypred:
                if intermediate_fusion:
                    modalitypreds.append(model.predictor(model.modality_embs_).view(1, -1))
                else:
                    modalitypreds.append(model.modality_preds_.view(1, -1))
        targets = torch.cat(targets).to(device)
        outputs = torch.cat(outputs)
    return targets, outputs, attentions, modalitypreds
