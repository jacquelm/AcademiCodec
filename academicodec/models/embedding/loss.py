import torchmetrics
import torch
from torch import nn

accuracy_metric = torchmetrics.classification.Accuracy(
    task="binary", threshold=0.5
).cuda()


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6, reduction="mean"):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


def compute_accuracy(y_pred, y):
    return accuracy_metric(y_pred, y)


def get_accuracy(project):
    if project == "classifier":
        accuracy = compute_accuracy
    elif project == "predictor":
        accuracy = nn.functional.mse_loss
    return accuracy


def get_loss(project, **kwargs):
    if project == "classifier":
        loss = nn.BCELoss(reduction="mean")
    elif project == "predictor":
        loss = RMSELoss(reduction="mean")
    return loss
