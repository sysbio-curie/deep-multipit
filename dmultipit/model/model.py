import torch
import torch.nn as nn

from dmultipit.base import BaseModel


class LateAttentionFusion(BaseModel):
    """
    Late fusion model with attention weights. Each modality is passed through a predictor and the final prediction
    is computed as the sum of unimodal predictions weighted by attention weights.

    Parameters
    ----------
    modality_embeddings: list of models
        Embedding models/predictors for the different modalities. Each model shoud have dim_output=1.

    multimodalattention: Attention mechanism

    Attributes
    ----------
    attention_weights_: tensor of size (batch_size, n_modalities)
        Attention weight for each modality and each sample.

    modality_preds_: tensor of size (batch_size, n_modalities)
        Prediction for each modality and each sample (before the application of the attention mechanism)
    """

    def __init__(self, modality_embeddings, multimodalattention):
        super(LateAttentionFusion, self).__init__()
        self.modality_embeddings = nn.ModuleList(modality_embeddings)
        self.multimodalattention = multimodalattention

        self.attention_weights_ = None
        self.modality_preds_ = None

    def reset_weights(self):
        self.multimodalattention.reset_weights()
        for emb in self.modality_embeddings:
            emb.reset_weights()
        return self

    def forward(self, list_x, mask):
        """
        Forward function

        Parameters
        ----------
        list_x: list of tensors
            Tensor for each modality of size (batch_size, modality_dimension)

        mask: Boolean tensor of size (batch_size, n_modalities)
            Indicate whether a modality is missing (i.e., 0) for each sample.

        Returns
        -------
            Tensor of size (batch_size,)
                Aggregated multimodal predictions for each sample of the batch (weighted sum of unimodal predictions
                with attention weights).
        """
        assert (len(list_x) == mask.shape[-1]), "mask should be of shape (batch size x n_modalities)"
        logits = []
        for x, embedding in zip(list_x, self.modality_embeddings):
            logits.append(embedding(x))
        logits = torch.where(mask, torch.cat(logits, dim=-1), torch.tensor(0.0))
        attention_weights = self.multimodalattention(list_x, mask)
        assert (logits.shape == attention_weights.shape), "attention weights and logits shoudd be of same shape"
        weighted_logits = attention_weights * logits
        self.attention_weights_ = attention_weights
        self.modality_preds_ = logits
        return weighted_logits.sum(1)


class InterAttentionFusion(BaseModel):
    """
    Intermediate fusion with attention weights. Each modality is embedded into a common latent space. The sample
    embedding is computed as the sum of the unimodal embeddings weighted by attention weights. The fused embedding is
    then passed through a simple predictor.

    Parameters
    ----------
    modality_embeddings: list of models
        Embedding models for the different modalities. All models should have the same output dimension (typically >1)

    attention: attention mechanisms

    predictor: model
        Predictive model to apply to the aggregated embedding (weighted sum of unimodal embeddings with attention
        weights).
    """

    def __init__(self, modality_embeddings, attention, predictor):
        super(InterAttentionFusion, self).__init__()
        self.modality_embeddings = nn.ModuleList(modality_embeddings)
        self.attention = attention
        self.predictor = predictor

        self.attention_weights_ = None
        self.modality_embs_ = None
        self.multimodal_embs_ = None

    def reset_weights(self):
        self.attention.reset_weights()
        self.predictor.reset_weights()
        for emb in self.modality_embeddings:
            emb.reset_weights()
        return self

    def forward(self, list_x, mask):
        """
        Forward function

        Parameters
        ----------
        list_x: list of tensors
            Tensor for each modality of size (batch_size, modality_dimension)

        mask: Boolean tensor of size (batch_size, n_modalities)
            Indicate whether a modality is missing (i.e., 0) for each sample.

        Returns
        -------
            Tensor of size (batch_size,)
                Predictions from the aggregated embeddings (weighted sum of unimodal embeddings with attention weights).
        """
        embeddings = []
        for x, modality_embedding in zip(list_x, self.modality_embeddings):
            embeddings.append(modality_embedding(x))
        embeddings = torch.stack(embeddings, dim=1)
        attentions = self.attention(embeddings, mask)
        fused_embedding = torch.matmul(torch.unsqueeze(attentions, 1), embeddings)

        self.attention_weights_ = attentions
        self.modality_embs_ = embeddings
        self.multimodal_embs_ = fused_embedding

        return self.predictor(fused_embedding.squeeze())
