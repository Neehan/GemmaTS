"""PatchTST baseline model for comparison."""

import sys
import os

# Add PatchTST_supervised to path
patchtst_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "PatchTST_supervised"
)
sys.path.insert(0, patchtst_path)

import torch
import torch.nn as nn
from models.PatchTST import Model as PatchTST_Model


class PatchTSTWrapper(nn.Module):
    """Wrapper around PatchTST to match our interface."""

    def __init__(self, config):
        super().__init__()
        self.model = PatchTST_Model(config)
        self.pred_len = config.pred_len

    def forward(self, context, mask=None, target=None, target_mask=None):
        """
        Forward pass matching Chronos interface.

        Args:
            context: (batch_size, seq_len) for univariate
            Others ignored for compatibility

        Returns:
            Output dict with 'predictions' key
        """
        # PatchTST expects (batch_size, seq_len, n_features)
        if context.dim() == 2:
            x = context.unsqueeze(-1)  # (batch, seq_len, 1)
        else:
            x = context

        # Forward through PatchTST
        output = self.model(x)  # (batch, pred_len, n_features)

        # Return dict matching our interface
        return PatchTSTOutput(predictions=output.squeeze(-1))


class PatchTSTOutput:
    """Output object matching Chronos interface."""

    def __init__(self, predictions):
        self.predictions = predictions


class PatchTSTConfig:
    """Config object for PatchTST matching expected interface."""

    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in=1,
        e_layers=3,
        n_heads=4,
        d_model=16,
        d_ff=128,
        dropout=0.3,
        fc_dropout=0.3,
        head_dropout=0.0,
        patch_len=16,
        stride=8,
        individual=False,
        revin=True,
        affine=True,
        subtract_last=False,
        decomposition=False,
        kernel_size=25,
        padding_patch="end",
    ):
        self.seq_len = seq_len
        self.label_len = 0  # Not used but expected by some code
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.e_layers = e_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.fc_dropout = fc_dropout
        self.head_dropout = head_dropout
        self.patch_len = patch_len
        self.stride = stride
        self.individual = individual
        self.revin = revin
        self.affine = affine
        self.subtract_last = subtract_last
        self.decomposition = decomposition
        self.kernel_size = kernel_size
        self.padding_patch = padding_patch


def create_patchtst(
    seq_len=512,
    pred_len=64,
    enc_in=1,
    e_layers=3,
    n_heads=4,
    d_model=16,
    d_ff=128,
    dropout=0.3,
    fc_dropout=0.3,
    head_dropout=0.0,
    patch_len=16,
    stride=8,
    **kwargs
):
    """Create PatchTST model with standard config."""
    config = PatchTSTConfig(
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=enc_in,
        e_layers=e_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_ff=d_ff,
        dropout=dropout,
        fc_dropout=fc_dropout,
        head_dropout=head_dropout,
        patch_len=patch_len,
        stride=stride,
        **kwargs
    )
    return PatchTSTWrapper(config)
