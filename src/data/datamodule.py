"""Data loading and preprocessing for time-series forecasting."""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Dict, Tuple, List
import numpy as np


class TimeSeriesDataset(Dataset):
    """
    Dataset for time-series forecasting with sliding windows.

    Args:
        data: Time-series data of shape (T, C)
        context_len: Length of context window
        patch_len: Length of each patch
        stride: Stride for sliding window
    """

    def __init__(self, data: np.ndarray, context_len: int, patch_len: int, stride: int):
        # Store as 1D tensor (Chronos expects (B, T) not (B, T, C))
        if len(data.shape) > 1:
            # If multivariate, take first channel for now
            data = data[:, 0]
        self.data = torch.FloatTensor(data)
        self.context_len = context_len
        self.patch_len = patch_len
        self.stride = stride

        # Calculate valid starting indices
        self.indices = self._get_valid_indices()

    def _get_valid_indices(self) -> List[int]:
        """Get valid starting indices for windows."""
        max_idx = len(self.data) - self.context_len - self.patch_len
        return list(range(0, max_idx, self.stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start_idx = self.indices[idx]
        end_idx = start_idx + self.context_len

        # Context window - 1D tensor
        series = self.data[start_idx:end_idx]

        # Target: next prediction_length values after context
        target = self.data[end_idx : end_idx + self.patch_len]

        # Masks (1 = observed, NaN values get 0)
        series_mask = (~torch.isnan(series)).float()
        target_mask = (~torch.isnan(target)).float()

        return {
            "context": series,
            "mask": series_mask,
            "target": target,
            "target_mask": target_mask,
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching.

    Args:
        batch: List of samples from dataset

    Returns:
        Batched dictionary with context, mask, target, target_mask
    """
    context = torch.stack([item["context"] for item in batch])
    mask = torch.stack([item["mask"] for item in batch])
    target = torch.stack([item["target"] for item in batch])
    target_mask = torch.stack([item["target_mask"] for item in batch])

    return {
        "context": context,
        "mask": mask,
        "target": target,
        "target_mask": target_mask,
    }


def build_datasets(
    dataset_name: str = "Salesforce/electricity_hourly",
    context_len: int = 360,
    patch_len: int = 30,
    stride: int = 15,
    train_split: float = 0.7,
    val_split: float = 0.15,
) -> Tuple[TimeSeriesDataset, TimeSeriesDataset, TimeSeriesDataset]:
    """
    Build train/val/test datasets from HuggingFace.

    Args:
        dataset_name: Name of dataset on HuggingFace
        context_len: Length of context window
        patch_len: Length of prediction patch
        stride: Stride for sliding window
        train_split: Fraction for training
        val_split: Fraction for validation

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Load dataset from HuggingFace
    # Note: Using electricity_hourly as an alternative since ETTh1 format varies
    # You can replace this with your preferred dataset
    dataset = load_dataset(dataset_name, split="train")

    # Extract time series data
    # Try 'target' field first, then 'value', then first dict key
    try:
        data = np.array([item["target"] for item in dataset])  # type: ignore[call-overload,index]
    except (KeyError, TypeError):
        try:
            data = np.array([item["value"] for item in dataset])  # type: ignore[call-overload,index]
        except (KeyError, TypeError):
            first_item = dataset[0]  # type: ignore[index]
            key = list(first_item.keys())[0]  # type: ignore[attr-defined]
            data = np.array([item[key] for item in dataset])  # type: ignore[call-overload,index]

    # Ensure data is 2D (T, C)
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    # Split into train/val/test
    n = len(data)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # Create datasets
    train_dataset = TimeSeriesDataset(train_data, context_len, patch_len, stride)
    val_dataset = TimeSeriesDataset(val_data, context_len, patch_len, stride)
    test_dataset = TimeSeriesDataset(test_data, context_len, patch_len, stride)

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(
    dataset_name: str = "Salesforce/electricity_hourly",
    context_len: int = 360,
    patch_len: int = 30,
    stride: int = 15,
    batch_size: int = 4,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get train/val/test dataloaders.

    Args:
        dataset_name: Name of dataset on HuggingFace
        context_len: Length of context window
        patch_len: Length of prediction patch
        stride: Stride for sliding window
        batch_size: Batch size
        num_workers: Number of dataloader workers

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset, val_dataset, test_dataset = build_datasets(
        dataset_name=dataset_name,
        context_len=context_len,
        patch_len=patch_len,
        stride=stride,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
