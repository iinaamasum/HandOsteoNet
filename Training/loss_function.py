import torch
import torch.nn as nn


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.6, gamma=0.1):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.mae = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, outputs, targets):
        smooth_l1_loss = self.smooth_l1(outputs, targets)
        mae_loss = self.mae(outputs, targets)
        mse_loss = self.mse(outputs, targets)
        return (
            self.alpha * smooth_l1_loss + self.beta * mae_loss + self.gamma * mse_loss
        )
