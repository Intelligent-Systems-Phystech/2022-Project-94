from torch import nn
import torch


class EVL(nn.Module):
    r"""
        EVL loss for classification with imbalanced classes
    """
    def __init__(self, gamma, beta0,  beta1):
        super().__init__()
        self.gamma = gamma
        self.beta0 = beta0
        self.beta1 = beta1

    def forward(self, y_pred, y_true):
        loss = - self.beta0 * (1 - y_pred/self.gamma) ** self.gamma * y_true * torch.log(y_pred) \
               - self.beta1 * (1 - (1 - y_pred)/self.gamma) ** self.gamma * (1 - y_true) * torch.log(1 - y_pred)
        loss = loss.mean()
        return loss
