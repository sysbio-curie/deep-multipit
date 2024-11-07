import warnings
import torch
from tqdm import tqdm

from dmultipit.utils import set_device


class Testing:
    """
    Testing class

    Parameters
    ----------
    model: base_model.BaseModel object
        Model to test.

    loss_ftn: callable with output and target inputs
        Loss function.

    metric_ftns: list callable with output and target inputs
        Metric functions.

    config: dict
        Configuration dictionnary

    device: str
        Specify the torch.device on which to allocate tensors.

    data_loader: list of DataLoaders or DataLoader
        * If DataLoader, it defines how to load the test data to test the input model (resumed from a single checkpoint
        or not).
        * If list of DataLoaders, it defines several versions of the test data (e.g., with different pre-processing
        learned from different models) for an ensembling strategy (i.e., average predictions of everal models, resumed
        from different checkpoints).

    intermediate_fusion: bool
        Specify whether the model is from the model.model.InterAttentionFusion class (True) or from the
        model.model.LateAttentionFusion class (False). The default is False.

    checkpoints: path, list of pathes, or None.
        * If None, no checkpoint is used and the input model is tested
        * If path, the model to test is resumed from a checkpoint (accessible via the indicated path)
        * If list of pathes, several models are resumed from different checkpoints (accessible via the different pathes)
        and an ensembling strategy (i.e., average predictions from the different models) is used.
        The defaut is None.

    disable_tqdm: bool
        If True, no tqdm progression bar is displayed. The default is False.

    Notes
    -----
    For ensembling strategies the user is asked to specify a different DataLoader for each model of the ensemble,
    although each loader should load the same test data. This is to take into account potential differences in the
    pre-processing steps applied to the test data that may come with the different models and that are specified for
    each dataset/data loader (see base.base_dataset.MultimodalDataset).
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
            disable_tqdm=False,
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

        self.logger = self.config.get_logger("test", config["testing"]["verbosity"])

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
        """
        Testing logic.

        * For a single model, collect the multimodal predictions, the attention weights and the unimodal predictions
        from a test dataset.
        * For an ensemble of models, collect the multimodal predictions from the different models and average them
        (ensembling strategy).

        Parameters
        ----------
        collect_a: boolean
            If True collect attention weights. The default is True.

        collect_modalitypred: boolean
            If True collect unimodal predictions (before aggregation with attention weights). The default is True.
        """

        if self.checkpoints is None:
            assert not isinstance(self.data_loader, list), ("When no checkpoint is passed (i.e., checkpoints = None), "
                                                            "no ensembling strategy is possible and data_loader should "
                                                            "refer to a single Dataloader, not a list of DataLoaders.")
            self.targets, self.outputs, self.attentions, self.modalitypreds = _predict(
                self.model,
                self.data_loader,
                self.device,
                collect_a,
                collect_modalitypred,
                self.intermediate_fusion,
                disable_tqdm=self.disable_tqdm,
            )
        elif isinstance(self.checkpoints, str):
            assert not isinstance(self.data_loader, list), ("When a single checkpoint is passed, no ensembling strategy"
                                                            " is possible and data_loader should refer to a single"
                                                            " Dataloader, not a list of DataLoaders.")
            checkpoint = torch.load(self.checkpoints)
            self.logger.info("Loading best model from epoch " + str(checkpoint["epoch"]))
            state_dict = checkpoint["state_dict"]
            self.model.load_state_dict(state_dict)
            self.targets, self.outputs, self.attentions, self.modalitypreds = _predict(
                self.model,
                self.data_loader,
                self.device,
                collect_a,
                collect_modalitypred,
                self.intermediate_fusion,
                disable_tqdm=self.disable_tqdm,
            )
        elif isinstance(self.checkpoints, list):
            assert isinstance(self.data_loader, list), ("When several checkpoints are passed, data_loader should be a"
                                                        " list of DataLoaders for ensembling strategy")
            assert len(self.checkpoints) == len(self.data_loader), ("When several checkpoints are passed, the same"
                                                                    " numer of DataLoaders (each one corresponding to"
                                                                    " one checkpoint should be stored in the list"
                                                                    " data_loaders.")
            if collect_a or collect_modalitypred:
                warnings.warn("For ensembling strategy the collection of attention weights and unimodal predictions"
                              " is not implemented. Only the aggregated predictions, across the different models of"
                              " the ensemble are available.")
            outputs = []
            i = 0
            if isinstance(self.model, torch.nn.ModuleList):
                for checkpoint_path, data_loader in zip(self.checkpoints, self.data_loader):
                    checkpoint = torch.load(checkpoint_path)
                    self.logger.info(
                        "Loading best model for ensembling "
                        + str(i)
                        + " from epoch "
                        + str(checkpoint["epoch"])
                    )
                    state_dict = checkpoint["state_dict"]
                    self.model[i].load_state_dict(state_dict)
                    self.targets, outputs_ensembing, *_ = _predict(
                        self.model[i],
                        data_loader,
                        self.device,
                        collect_attentions=False,
                        collect_modalitypred=False,
                        intermediate_fusion=self.intermediate_fusion,
                        disable_tqdm=self.disable_tqdm,
                    )
                    outputs.append(outputs_ensembing)
                    i += 1
            else:
                for checkpoint_path, data_loader in zip(self.checkpoints, self.data_loader):
                    checkpoint = torch.load(checkpoint_path)
                    self.logger.info(
                        "Loading best model for ensembling "
                        + str(i)
                        + " from epoch "
                        + str(checkpoint["epoch"])
                    )
                    state_dict = checkpoint["state_dict"]
                    self.model.load_state_dict(state_dict)
                    self.targets, outputs_ensembing, *_ = _predict(
                        self.model,
                        data_loader,
                        self.device,
                        collect_attentions=False,
                        collect_modalitypred=False,
                        intermediate_fusion=self.intermediate_fusion,
                        disable_tqdm=self.disable_tqdm,
                    )
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


def _predict(
        model,
        data_loader,
        device,
        collect_attentions,
        collect_modalitypred,
        intermediate_fusion=False,
        disable_tqdm=False,
):
    """
    Prediction function.

    Parameters
    ----------
    model: base_model.BaseModel object
        Model to test.

    data_loader: DataLoader

    device: str
        Torch.device on which to allocate tensors

    collect_attentions: bool
        If True, attention weights are collected

    collect_modalitypred: bool
        If True, unimodal predictions are collected

    intermediate_fusion: bool
        Specify whether the model is from the model.model.InterAttentionFusion class (True) or from the
        model.model.LateAttentionFusion class (False). The default is False.

    disable_tqdm: bool
        If True, no tqdm progression bar is displayed. The default is False.

    Returns
    -------
    targets: tensor of size (n_samples_test,)
        Labels of test samples

    outputs: tensor of size (n_samples_test,)
        Output of the predictive model (warning: depending on the model architecture an additional sigmoid step could be
        required to transfrom the raw outputs)

    attentions: list of tensors of length n_samples_test
        Tensors of attention weights for each test sample (size (1, n_modalities))

    modalitypreds: list of tensors of length n_samples_test
        Tensors of unimodal predictions/outputs for each test sample (size (1, n_modalities))
    """

    targets, outputs, attentions, modalitypreds = [], [], [], []
    model.eval()
    with torch.no_grad():
        for *list_modas, mask, target in tqdm(data_loader, disable=disable_tqdm):
            list_modas, mask, target = (
                set_device(list_modas, device),
                set_device(mask, device),
                set_device(target, device),
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
