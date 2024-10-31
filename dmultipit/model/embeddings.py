import torch.nn as nn

from dmultipit.base import BaseModel


class ModalityEmbedding(BaseModel):
    """
    Unimodal embedding network

    Parameters
    ---------
    dim_input: int
        Input dimension for the modality

    h_sizes: list of int
        Sizes of hidden layers. The network is composed of several hidden layers (of dimension specified by h_sizes) and
        with ReLU activation. If empty the network consists of a single linear layer (dim_input -> dim_output).

    p_dropout: float in [0, 1]
        Probability dropout

    dim_output: int
        Output dimension

    final_activation: string in ["sigmdoid", "tanh"]
    """

    def __init__(self, dim_input, h_sizes, p_dropout=0.5, dim_output=1, final_activation="sigmoid"):
        super(ModalityEmbedding, self).__init__()

        self.layers = nn.ModuleList()
        input_size = dim_input
        if len(h_sizes) > 0:
            for h in h_sizes:
                self.layers.append(nn.Linear(input_size, h))
                self.layers.extend(nn.ModuleList([nn.ReLU(), nn.Dropout(p=p_dropout)]))
                input_size = h

        if final_activation == "sigmoid":
            assert dim_output == 1, "Sigmoid should only be used with a 1-dimensional output"
            self.layers.extend(
                nn.ModuleList(
                    [
                        nn.Linear(input_size, dim_output),
                        nn.Sigmoid(),
                    ]
                )
            )
        elif final_activation == "tanh":
            assert dim_output == 1, "Tanh should only be used with a 1-dimensional output"
            self.layers.extend(
                nn.ModuleList(
                    [
                        nn.Linear(input_size, dim_output),
                        nn.Tanh(),
                    ]
                )
            )

        else:
            self.layers.extend(
                nn.ModuleList(
                    [
                        nn.Linear(input_size, dim_output),
                        nn.ReLU(),
                        nn.Dropout(p=p_dropout),
                    ]
                )
            )

        self.reset_weights()

    def reset_weights(self):
        for i, l in enumerate(self.layers):
            if isinstance(l, nn.Linear):
                # nn.init.kaiming_normal_(l.weight)
                nn.init.zeros_(l.bias)
        return self

    def forward(self, x):
        """
        Forward function

        Parameters
        ----------
        x: tensor of size (batch_size, dim_input)

        Returns
        -------
            Tensor of size (batch_size, dim_output)
        """
        for i, l in enumerate(self.layers):
            x = l(x)
        return x
