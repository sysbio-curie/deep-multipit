import torch
import torch.nn as nn

from dmultipit.base import BaseModel
from dmultipit.utils import masked_softmax


class MultiModalAttention(BaseModel):
    """
    Attention mechanism for multimodal framework. Attention weights are computed with one specific network per modality
    (with the possibility of sharing some layers) and normalized over all the modalities. Missing modalities are handled
    by masking them.

    Parameters
    ----------
    dim_input: list of int
        Input dimension for each modality

    h1: int or list of int
        Size of hidden layers.
        * If shared_weights is True, h1 shoud be an integer refering to the same latent dimension for each modality
        * If shared_weights is False, if h1 is an integer it refers to the same latent dimension for each modality,
        otherwise it should be a list of integers of the same length as *dim_input* specifying the size of the latent
        dimension for each input modality.

    shared_weights: boolean
        * If True, each modality is embedded in the same latent dimension and then passed through a single fully
        connected layer, shared between all modalities.
        * If False, each modality passes through its own fully connected layer after embedding.
        The default is True.

    p_dropout: float in [0, 1]
        Dropout probability. The default is 0 (i.e., no dropout).
    """

    def __init__(self, dim_input, h1, shared_weights=True, p_dropout=0):
        super(MultiModalAttention, self).__init__()
        self.dim_input = dim_input
        self.h1 = h1
        self.shared_weights = shared_weights
        self.p_dropout = p_dropout
        self.attention_norm = None

        self._build_layers()
        self.reset_weights()

    def _build_layers(self):
        assert type(self.h1) == int or float or list, "h1 should be a list or a number"
        if self.shared_weights:
            assert (type(self.h1) == int or float), "If shared_weights is true h1 should refer to a single latent size"
            self.embeddings = nn.ModuleList(
                [nn.Linear(d, self.h1, bias=False) for d in self.dim_input]
            )
            self.fc = nn.Sequential(
                nn.Dropout(p=self.p_dropout), nn.Linear(self.h1, 1, bias=False)
            )
        else:
            if type(self.h1) == list:
                assert len(self.dim_input) == len(self.h1), "If h1 is a list it should be of the same size as dim_input"
                self.embeddings = nn.ModuleList(
                    [nn.Linear(self.dim_input[i], self.h1[i], bias=False) for i in range(len(self.dim_input))]
                )
                self.fc = nn.ModuleList(
                    [nn.Sequential(nn.Dropout(p=self.p_dropout), nn.Linear(h, 1)) for h in self.h1]
                )
            else:
                self.embeddings = nn.ModuleList(
                    [nn.Linear(d, self.h1, bias=False) for d in self.dim_input]
                )
                self.fc = nn.ModuleList(
                    [nn.Sequential(nn.Dropout(p=self.p_dropout), nn.Linear(self.h1, 1)) for _ in range(self.dim_input)]
                )

    @staticmethod
    def _init_weights(m):
        nn.init.xavier_normal_(m.weight)
        return

    def reset_weights(self):
        if self.shared_weights:
            for layer in self.embeddings:
                self._init_weights(layer)
            self._init_weights(self.fc[1])
        else:
            for i in range(len(self.embeddings)):
                self._init_weights(self.embeddings[i])
                self._init_weights(self.fc[i][1])
        return self

    def forward(self, x_list, mask):
        """
        Forward function

        Parameters
        ----------
        x_list: list of tensors
            Tensor for each modality of size (batch_size, modality_dimension)

        mask: Boolean tensor of size (batch_size, n_modalities)
            Indicate whether a modality is missing (i.e., 0) for each sample.

        Returns
        -------
        attentions: tensor of size (batch_size, n_modalities)
            Attention weights for each modality and for each sample of the batch
        """
        if self.shared_weights:
            emb = []
            for i, t in enumerate(x_list):
                emb.append(self.embeddings[i](t))
            emb = torch.tanh(torch.stack(emb, dim=1))
            att = self.fc(emb).squeeze(dim=-1)  # torch.where(mask, self.fc(emb).squeeze(), torch.tensor(0.))
        else:
            att = []
            for i, t in enumerate(x_list):
                att.append(self.fc[i](torch.tanh(self.embeddings[i](t))))
            att = torch.cat(att, dim=-1)
            # att = torch.where(mask, torch.cat(att, dim=-1), torch.tensor(0.))
        attentions, self.attention_norm = masked_softmax(att, mask, dim=-1)
        return attentions  # F.softmax(att, dim=-1)


class MultiModalGatedAttention(BaseModel):
    """
    Gated attention mechanism for multimodal framework. Attention weights are computed with one specific network per
    modality (with the possibility of sharing some layers) and normalized over all the modalities. Missing modalities
    are handled by masking them.

    Parameters
    ----------
    dim_input: list of int
        Input dimension for each modality

    h1: int or list of int
        Size of hidden layers.
        * If shared_weights is True, h1 shoud be an integer refering to the same latent dimension for each modality
        * If shared_weights is False, if h1 is an integer it refers to the same latent dimension for each modality,
        otherwise it should be a list of integers of the same length as *dim_input* specifying the size of the latent
        dimension for each input modality.

    shared_weights: boolean
        * If True, each modality is embedded in the same latent dimension and then passed through a single fully
        connected layer, shared between all modalities.
        * If False, each modality passes through its own fully connected layer after embedding.
        The default is True.

    p_dropout: float in [0, 1]
        Dropout probability. The default is 0 (i.e., no dropout).
    """

    def __init__(self, dim_input, h1, shared_weights=True, p_dropout=0):
        super(MultiModalGatedAttention, self).__init__()
        self.dim_input = dim_input
        self.h1 = h1
        self.shared_weights = shared_weights
        self.p_dropout = p_dropout
        self.attention_norm = None

        self._build_layers()
        self.reset_weights()

    def _build_layers(self):
        assert type(self.h1) == int or float or list, "h1 should be a list or a number"
        if self.shared_weights:
            assert (type(self.h1) == int or float), "If shared_weights is true h1 should refer to a single latent size"
            self.embeddings_1 = nn.ModuleList(
                [nn.Linear(d, self.h1, bias=False) for d in self.dim_input]
            )
            self.embeddings_2 = nn.ModuleList(
                [nn.Linear(d, self.h1, bias=False) for d in self.dim_input]
            )
            self.fc = nn.Sequential(
                nn.Dropout(p=self.p_dropout), nn.Linear(self.h1, 1, bias=False)
            )  # nn.Linear(self.h1, 1, bias=False)

        else:
            if type(self.h1) == list:
                assert len(self.dim_input) == len(self.h1), "If h1 is a list it should be of the same size as dim_input"
                self.embeddings_1 = nn.ModuleList(
                    [nn.Linear(self.dim_input[i], self.h1[i], bias=False) for i in range(len(self.dim_input))]
                )
                self.embeddings_2 = nn.ModuleList(
                    [nn.Linear(self.dim_input[i], self.h1[i], bias=False) for i in range(len(self.dim_input))]
                )
                self.fc = nn.ModuleList(
                    [nn.Sequential(nn.Dropout(p=self.p_dropout), nn.Linear(h, 1)) for h in self.h1]
                )
            else:
                self.embeddings_1 = nn.ModuleList(
                    [nn.Linear(d, self.h1, bias=False) for d in self.dim_input]
                )
                self.embeddings_2 = nn.ModuleList(
                    [nn.Linear(d, self.h1, bias=False) for d in self.dim_input]
                )
                self.fc = nn.ModuleList(
                    [nn.Sequential(nn.Dropout(p=self.p_dropout), nn.Linear(self.h1, 1)) for _ in range(self.dim_input)]
                )
        return

    @staticmethod
    def _init_weights(m):
        nn.init.xavier_normal_(m.weight)
        return

    def reset_weights(self):
        for i in range(len(self.embeddings_1)):
            self._init_weights(self.embeddings_1[i])
            self._init_weights(self.embeddings_2[i])

        if self.shared_weights:
            self._init_weights(self.fc[1])
        else:
            for i in range(len(self.embeddings_1)):
                self._init_weights(self.fc[i][1])
        return self

    def forward(self, x_list, mask):
        """
        Forward function

        Parameters
        ----------
        x_list: list of tensors
            Tensor for each modality of size (batch_size, modality_dimension)

        mask: Boolean tensor of size (batch_size, n_modalities)
            Indicate whether a modality is missing (i.e., 0) for each sample.

        Returns
        -------
        attentions: tensor of size (batch_size, n_modalities)
            Attention weights for each modality and for each sample of the batch
        """
        if self.shared_weights:
            emb = []
            for i, t in enumerate(x_list):
                emb.append(
                    torch.tanh(self.embeddings_1[i](t))
                    * torch.sigmoid(self.embeddings_2[i](t))
                )
            emb = torch.stack(emb, dim=1)
            att = self.fc(emb).squeeze(dim=-1)  # torch.where(mask, self.fc(emb).squeeze(), torch.tensor(0.))
        else:
            att = []
            for i, t in enumerate(x_list):
                att.append(
                    self.fc[i](
                        torch.tanh(self.embeddings_1[i](t))
                        * torch.sigmoid(self.embeddings_2[i](t))
                    )
                )
            att = torch.cat(att, dim=-1)  # torch.where(mask, torch.cat(att, dim=-1), torch.tensor(0.))
        attentions, self.attention_norm = masked_softmax(att, mask, dim=-1)
        return attentions  # F.softmax(att, dim=-1)


class Attention(BaseModel):
    """
    Define a neural network to compute attention weights (see [1] for more details)

    Parameters
    ----------
    dim_input: int
        Input size.

    h1: int
        Size of hidden layer.

    dim_output: int
        Output size. The default is 1.

    References
    ----------
    [1] Attention-based Deep Multiple Instance Learning - Ilse et al. 2018

    Notes
    -----
    This attention mechanisms could be seen as a special case of the MultimodalAttention mechanism where each modality
    would have the same input dimension and a single embedding layer would be shared across all modalities (i.e.,
    self.fc1).

    """

    def __init__(self, dim_input, h1, dim_output=1):
        super(Attention, self).__init__()

        self.fc1 = nn.Linear(dim_input, h1, bias=False)
        nn.init.xavier_normal_(self.fc1.weight)
        # nn.init.zeros_(self.fc1.bias)

        self.fc2 = nn.Linear(h1, dim_output, bias=False)
        nn.init.xavier_normal_(self.fc2.weight)
        # nn.init.zeros_(self.fc2.bias)

        self.attention_norm = None

    def reset_weights(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        return self

    def forward(self, x, mask):
        """
        Forward function

        Parameters
        ----------
        x: tensor of size (batch_size, n_modalities, dim_input)

        mask: boolean tensor of size (batch_size, n_modalities)
            Indicate whether a modality is missing (i.e., 0) for each sample.

        Returns
        -------
        attentions: tensor of size (batch_size, n_modalities)
            Attention weights for each modality and for each sample of the batch
        """
        out = torch.tanh(self.fc1(x))
        out = self.fc2(out).squeeze()
        # out = torch.where(mask, torch.transpose(out, -1, -2), torch.tensor(0.))  # to deal with missing modalities
        attentions, self.attention_norm = masked_softmax(out, mask, dim=-1)
        return attentions  # F.softmax(out, dim=-1)  torch.transpose(out, -1, -2)


class GatedAttention(BaseModel):
    """
    Define a neural network to compute attention weights with a gating mechanism
    (see [1] for more details)

    Parameters
    ----------
    dim_input: int
        Input size.

    h1: int
        Size of hidden layer.

    dim_output: int
        Output size.

    References
    ----------
    [1] Attention-based Deep Multiple Instance Learning - Ilse et al. 2018

    Notes
    -----
    This attention mechanisms could be seen as a special case of the MultimodalGatedAttention mechanism where each
    modality would have the same input dimension and a single embedding layer would be shared across all modalities
    (i.e., self.fc1).


    """

    def __init__(self, dim_input, h1, dim_output=1):
        super(GatedAttention, self).__init__()

        self.fc1 = nn.Linear(dim_input, h1, bias=False)
        nn.init.xavier_normal_(self.fc1.weight)
        # nn.init.zeros_(self.fc1.bias)

        self.fc2 = nn.Linear(dim_input, h1, bias=False)
        nn.init.xavier_normal_(self.fc2.weight)
        # nn.init.zeros_(self.fc2.bias)

        self.fc3 = nn.Linear(h1, dim_output, bias=False)
        nn.init.xavier_normal_(self.fc3.weight)
        # nn.init.zeros_(self.fc3.bias)

        self.attention_norm = None

    def reset_weights(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        return self

    def forward(self, x, mask):
        """
        Forward function

        Parameters
        ----------
        x: tensor of size (batch_size, n_modalities, dim_input)

        mask: boolean tensor of size (batch_size, n_modalities)
            Indicate whether a modality is missing (i.e., 0) for each sample.

        Returns
        -------
        attentions: tensor of size (batch_size, n_modalities)
            Attention weights for each modality and for each sample of the batch
        """
        x_v = torch.tanh(self.fc1(x))
        x_u = torch.sigmoid(self.fc2(x))
        out = self.fc3(x_v * x_u).squeeze()
        # out = torch.where(mask, torch.transpose(out, -1, -2), torch.tensor(0.))
        attentions, self.attention_norm = masked_softmax(out, mask, dim=-1)
        return attentions  # F.softmax(out, dim=-1) torch.transpose(out, -1, -2)


class MSKCCAttention(BaseModel):
    """
    Attention mechanism to reproduce experiments from Vanguri et al. (https://doi.org/10.1038/s43018-022-00416-8).

    Parameters
    ----------
    dim_input: list of int
        Input dimension for each modality.

    References
    ----------
    1. Vanguri, R.S. et al. Multimodal integration of radiology, pathology and genomics for prediction of response to
    PD-(L)1 blockade in patients with non-small cell lung cancer. Nat Cancer 3, 1151–1164 (2022).
    (https://doi.org/10.1038/s43018-022-00416-8)

    """

    def __init__(self, dim_input):
        super(MSKCCAttention, self).__init__()
        self.dim_input = dim_input
        self.attention_layers = nn.ModuleList([nn.Linear(d, 1) for d in self.dim_input])
        self.softplus = torch.nn.Softplus()
        self.attention_norm = None

    def reset_weights(self):
        for layer in self.attention_layers:
            # nn.init.kaiming_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
        return self

    def forward(self, x_list, mask):
        """
        Forward function

        Parameters
        ----------
        x_list: list of tensors
            Tensor for each modality of size (batch_size, modality_dimension)

        mask: Boolean tensor of size (batch_size, n_modalities)
            Indicate whether a modality is missing (i.e., 0) for each sample.

        Returns
        -------
        attentions: tensor of size (batch_size, n_modalities)
            Attention weights for each modality and for each sample of the batch
        """
        attentions = []
        for i, t in enumerate(x_list):
            attentions.append(self.attention_layers[i](t) / self.dim_input[i])
        attentions = torch.stack(attentions, dim=1).squeeze(dim=-1)
        attentions_softplus = self.softplus(attentions)
        attentions_scores = torch.where(mask, attentions_softplus, torch.tensor(0.0))
        self.attention_norm = attentions_scores.norm(p=2, dim=-1).mean()
        return torch.nn.functional.normalize(attentions_scores, p=1, dim=-1)
