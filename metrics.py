"""Pixel-level height metrics, accumulated across a dataset."""
import math

import torch


class RegressionMeter:
    def __init__(self):
        self.count = 0
        self.absolute_error = 0.0
        self.squared_error = 0.0
        self.signed_error = 0.0

    def update(self, prediction, target, mask):
        mask = mask.bool() & torch.isfinite(prediction) & torch.isfinite(target)
        error = (prediction - target)[mask]
        self.count += error.numel()
        self.absolute_error += error.abs().sum().item()
        self.squared_error += error.square().sum().item()
        self.signed_error += error.sum().item()

    def result(self):
        if self.count == 0:
            return {'pixels': 0, 'mae_m': None, 'rmse_m': None, 'bias_m': None}
        return {'pixels': self.count,
                'mae_m': self.absolute_error / self.count,
                'rmse_m': math.sqrt(self.squared_error / self.count),
                'bias_m': self.signed_error / self.count}
