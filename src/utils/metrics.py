"""Metrics for time-series forecasting."""

import torch
import numpy as np
from typing import Union


def mse(y_true: Union[torch.Tensor, np.ndarray], y_pred: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Mean Squared Error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        MSE value
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: Union[torch.Tensor, np.ndarray], y_pred: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Mean Absolute Error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        MAE value
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true: Union[torch.Tensor, np.ndarray], y_pred: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Symmetric Mean Absolute Percentage Error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        sMAPE value (in percentage)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)

    # Avoid division by zero
    mask = denominator != 0
    smape_val = np.zeros_like(diff)
    smape_val[mask] = diff[mask] / denominator[mask]

    return float(np.mean(smape_val) * 100)


def rmse(y_true: Union[torch.Tensor, np.ndarray], y_pred: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Root Mean Squared Error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        RMSE value
    """
    return float(np.sqrt(mse(y_true, y_pred)))
