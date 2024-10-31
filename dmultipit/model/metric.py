import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def roc_auc(output, target):
    with torch.no_grad():
        try:
            auc = roc_auc_score(target.cpu().numpy(), output.cpu().numpy())
        # deal with batch containing only one class
        except ValueError:
            auc = np.nan
    return auc


def accuracy(output, target):
    with torch.no_grad():
        pred = output > 0.5
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def balanced_accuracy(output, target):
    with torch.no_grad():
        pred = (output > 0.5).squeeze()
        assert pred.shape[0] == len(target)
        n_items_0 = torch.sum(target == 0).item()
        n_items_1 = torch.sum(target == 1).item()
        # deal with batch containing only one class
        if n_items_1 > 0 and n_items_0 > 0:
            recall_0 = torch.sum((pred + target) == 0).item() / n_items_0
            recall_1 = torch.sum((pred + target) == 2).item() / n_items_1
            baccuracy = 0.5 * (recall_0 + recall_1)
        else:
            baccuracy = np.nan
    return baccuracy


def accuracy_with_logit(output, target):
    with torch.no_grad():
        pred = torch.sigmoid(output) > 0.5
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def balanced_accuracy_with_logit(output, target):
    with torch.no_grad():
        pred = (torch.sigmoid(output) > 0.5).squeeze()
        assert pred.shape[0] == len(target)
        n_items_0 = torch.sum(target == 0).item()
        n_items_1 = torch.sum(target == 1).item()
        # deal with batch containing only one class
        if n_items_1 > 0 and n_items_0 > 0:
            recall_0 = torch.sum((pred + target) == 0).item() / n_items_0
            recall_1 = torch.sum((pred + target) == 2).item() / n_items_1
            baccuracy = 0.5 * (recall_0 + recall_1)
        else:
            baccuracy = np.nan
    return baccuracy


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
