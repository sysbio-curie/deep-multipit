import torch
import torch.nn.functional as F


class BCELoss(object):
    """
    Binary Cross-Entropy loss function
    """

    def __call__(self, output, target):
        criterion = torch.nn.BCELoss()
        output = criterion(output.squeeze(), target.to(dtype=torch.float32).squeeze())
        return output


class BCELogitLoss(object):
    """
    Binary Cross-Entropy loss function, applying a sigmoid before computing the loss (more stable)
    """

    def __init__(self, pos_weight=None):
        self.pos_weight = pos_weight

    def __call__(self, output, target):
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        output = criterion(output.squeeze(), target.to(dtype=torch.float32).squeeze())
        return output


# warning: nll_loss needs a log_softmax !
class NllLoss(object):
    """
    Negative Log-Likelihood loss
    """

    def __call__(self, output, target, model):
        return F.nll_loss(output.squeeze(), target.squeeze())


# Tools for semi-supervised learning (regularization with unlabelled data)
class StepScheduler(object):
    """
    Custom step scheduler (wait before considering unlabelled data)
    """

    def __init__(self, wait):
        self.wait = wait

    def __call__(self, epoch):
        if epoch >= self.wait:
            return 1
        return 0


class UnlabelledBCELoss(object):
    """
    Binary cros-entropy loss with pseudo binary labels generated from the predictions of the model for unlabelled
    data, considering only "confident" ones (i.e., predictions sufficiently far from 0.5 as determined by a predefined
    threshod).
    """

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, output, output_labels):
        criterion = torch.nn.BCELoss()
        mask = torch.abs(torch.tensor(0.5) - output_labels).ge(self.threshold)
        output = torch.masked_select(output, mask)
        target = torch.torch.masked_select(output_labels, mask).ge(0.5)
        return criterion(output.squeeze(), target.to(dtype=torch.float32).squeeze())
