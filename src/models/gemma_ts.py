"""GemmaTS: Chronos Bolt with Gemma decoder."""

import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from chronos.chronos_bolt import ChronosBoltModelForForecasting
from typing import Optional


class GemmaTS(ChronosBoltModelForForecasting):
    """
    Inherits ChronosBolt, only replaces the T5 decoder with Gemma.
    Everything else (encoder, output head, loss) is unchanged.
    """

    def __init__(self, config, gemma_model_name: str = "google/gemma-2-2b"):
        # Initialize parent (this creates encoder, output head, etc.)
        super().__init__(config)

        # Now replace the decoder
        self._init_gemma_decoder(gemma_model_name)

    def _init_gemma_decoder(self, model_name: str):
        """Replace T5 decoder with Gemma."""
        # Remove T5 decoder
        del self.decoder

        # Load Gemma with HF token if available
        hf_token = os.getenv("HF_TOKEN")
        gemma = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, token=hf_token
        )
        self.gemma = gemma.model
        gemma_dim = gemma.config.hidden_size

        # Store BOS token ID to use its embedding directly
        self.bos_token_id = gemma.config.bos_token_id
        if self.bos_token_id is None:
            raise ValueError("BOS token ID is not set for Gemma model")

        # Projection if dimensions don't match
        if gemma_dim != self.model_dim:
            self.enc_to_gemma = nn.Linear(self.model_dim, gemma_dim)
            self.gemma_to_dec = nn.Linear(gemma_dim, self.model_dim)
        else:
            self.enc_to_gemma = nn.Identity()
            self.gemma_to_dec = nn.Identity()

    def decode(
        self, input_embeds, attention_mask, hidden_states, output_attentions=False
    ):
        """
        Replace T5 decoder with Gemma.

        Returns: (B, 1, d_model) to match Chronos Bolt's expectation
        """
        B = hidden_states.shape[0]

        # Project encoder outputs to Gemma dimension
        h = self.enc_to_gemma(hidden_states)

        # Get BOS token embedding (always up-to-date with fine-tuning)
        bos_embed = self.gemma.embed_tokens.weight[self.bos_token_id]
        dec_start = bos_embed.unsqueeze(0).unsqueeze(0).expand(B, -1, -1)
        combined = torch.cat([h, dec_start], dim=1)

        # Extend attention mask
        dec_mask = torch.ones(
            B, 1, device=attention_mask.device, dtype=attention_mask.dtype
        )
        full_mask = torch.cat([attention_mask, dec_mask], dim=1)

        # Pass through Gemma
        out = self.gemma(
            inputs_embeds=combined, attention_mask=full_mask, return_dict=True
        )

        # Get last token's hidden state
        last = out.last_hidden_state[:, -1:, :]

        # Project back to decoder dimension
        return self.gemma_to_dec(last)

    @torch.no_grad()
    def generate(
        self, context: torch.Tensor, steps: int, mask: Optional[torch.Tensor] = None
    ):
        """Autoregressive generation."""
        self.eval()

        curr_ctx = context.clone()
        curr_mask = mask.clone() if mask is not None else None
        generated = context.clone()

        # Find median quantile index
        quantiles = self.chronos_config.quantiles
        med_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2

        for _ in range(steps):
            outputs = self.forward(curr_ctx, curr_mask)
            assert outputs.quantile_preds is not None
            next_vals = outputs.quantile_preds[:, med_idx, :]

            generated = torch.cat([generated, next_vals], dim=-1)

            curr_ctx = torch.cat([curr_ctx, next_vals], dim=-1)
            curr_ctx = curr_ctx[:, -self.chronos_config.context_length :]

            if curr_mask is not None:
                new_mask = torch.ones_like(next_vals)
                curr_mask = torch.cat([curr_mask, new_mask], dim=-1)
                curr_mask = curr_mask[:, -self.chronos_config.context_length :]

        return generated


def create_gemma_ts(
    chronos_base: str = "amazon/chronos-bolt-tiny",
    gemma_model: str = "google/gemma-2-2b",
    context_length: int = 512,
    prediction_length: int = 64,
    patch_size: int = 16,
    patch_stride: int = 8,
):
    """Create GemmaTS from Chronos config."""
    from transformers import AutoConfig

    # Load Chronos config
    config = AutoConfig.from_pretrained(chronos_base)

    # Override settings
    config.chronos_config["context_length"] = context_length
    config.chronos_config["prediction_length"] = prediction_length
    config.chronos_config["input_patch_size"] = patch_size
    config.chronos_config["input_patch_stride"] = patch_stride

    return GemmaTS(config, gemma_model)
