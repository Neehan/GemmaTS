"""Loss functions for time series forecasting."""

import torch
import torch.nn.functional as F


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


def alignment_loss(
    projected_embeddings: torch.Tensor, target_embedding: torch.Tensor
) -> torch.Tensor:
    """Cosine alignment loss for embedding space alignment.

    Encourages projected time series embeddings to align with target embeddings
    (e.g., numeric token embeddings from language models).

    Args:
        projected_embeddings: Projected embeddings, shape (batch, seq_len, dim)
        target_embedding: Target embedding to align to, shape (dim,)

    Returns:
        Scalar loss value (1 - mean cosine similarity)
    """
    projected_embeddings = projected_embeddings.view(-1, projected_embeddings.shape[-1])
    target_expanded = target_embedding.unsqueeze(0).expand_as(projected_embeddings)
    cosine_sim = F.cosine_similarity(projected_embeddings, target_expanded, dim=-1)
    loss = 1 - cosine_sim.mean()

    return loss
