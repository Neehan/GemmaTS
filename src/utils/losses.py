"""Loss functions for time series forecasting."""

import torch


def mse_loss(predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error loss.

    Args:
        predictions: Predicted values, shape (batch, pred_len)
        target: Target values, shape (batch, pred_len)

    Returns:
        Scalar loss value
    """
    return torch.nn.functional.mse_loss(predictions, target)


def quantile_loss(
    quantile_preds: torch.Tensor, target: torch.Tensor, quantiles: list[float]
) -> torch.Tensor:
    """Quantile loss (pinball loss) for probabilistic forecasting.

    Same loss used in Chronos models. Trains all quantiles simultaneously.

    Args:
        quantile_preds: Predicted quantiles, shape (batch, num_quantiles, pred_len)
        target: Target values, shape (batch, pred_len)
        quantiles: List of quantile levels (e.g., [0.1, 0.2, ..., 0.9])

    Returns:
        Scalar loss value
    """
    target_expanded = target.unsqueeze(1)

    quantiles_tensor = torch.tensor(quantiles, device=quantile_preds.device).view(
        1, -1, 1
    )

    loss = 2 * torch.abs(
        (target_expanded - quantile_preds)
        * ((target_expanded <= quantile_preds).float() - quantiles_tensor)
    )

    loss = loss.mean(dim=-1)
    loss = loss.sum(dim=-1)
    loss = loss.mean()

    return loss
