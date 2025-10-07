"""GemmaTS: Chronos Bolt with Gemma decoder."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from chronos.chronos_bolt import ChronosBoltModelForForecasting, ChronosBoltOutput
from typing import Optional


@dataclass
class GemmaTSOutput(ChronosBoltOutput):
    """Extended output with encoder hidden states."""

    encoder_hidden_states: Optional[torch.Tensor] = None
    projected_hidden_states: Optional[torch.Tensor] = None


class GemmaTS(ChronosBoltModelForForecasting):
    """
    Inherits ChronosBolt, only replaces the T5 decoder with Gemma.
    Everything else (encoder, output head, loss) is unchanged.
    """

    def __init__(
        self,
        config,
        gemma_model_name: str,
        text_prompt: Optional[str] = None,
        use_bfloat16: bool = True,
    ):
        # Initialize parent (this creates encoder, output head, etc.)
        super().__init__(config)

        # Now replace the decoder
        self._init_gemma_decoder(gemma_model_name, text_prompt, use_bfloat16)

    def _init_gemma_decoder(
        self,
        model_name: str,
        text_prompt: Optional[str] = None,
        use_bfloat16: bool = True,
    ):
        """Replace T5 decoder with Gemma."""
        # Remove T5 decoder
        del self.decoder

        # Load Gemma with HF token if available
        hf_token = os.getenv("HF_TOKEN")
        dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        gemma = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, token=hf_token
        )

        # Detect model type and extract the text model
        if hasattr(gemma, "language_model"):
            # Gemma3ForConditionalGeneration (4B+) - multimodal
            # language_model is already Gemma3TextModel (has embed_tokens, layers, etc)
            self.gemma = gemma.language_model
            gemma_dim = gemma.config.text_config.hidden_size
        else:
            # Gemma3ForCausalLM (1B or less)
            self.gemma = gemma.model
            gemma_dim = gemma.config.hidden_size

        # Get BOS token from generation_config (most reliable)
        self.bos_token_id = gemma.generation_config.bos_token_id
        if self.bos_token_id is None:
            self.bos_token_id = gemma.config.bos_token_id
        if self.bos_token_id is None:
            raise ValueError("BOS token ID not found in generation_config")

        # Load tokenizer for text prompts and numeric embeddings
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

        # Optional: Set up text prompt if provided
        if text_prompt is not None:
            prompt_tokens = tokenizer(
                text_prompt, return_tensors="pt", add_special_tokens=False
            )
            self.prompt_ids = prompt_tokens.input_ids[0]
        else:
            self.prompt_ids = None

        # Precompute average numeric token embedding
        numeric_embedding = self._compute_numeric_embedding(tokenizer)
        self.register_buffer("numeric_embedding", numeric_embedding)

        # Projection layers (trainable)
        self.enc_to_gemma = nn.Sequential(
            nn.Linear(self.model_dim, gemma_dim),
            nn.RMSNorm(gemma_dim),
        )
        self.gemma_to_dec = nn.Sequential(
            nn.Linear(gemma_dim, self.model_dim),
            nn.RMSNorm(self.model_dim),
        )

    def _compute_numeric_embedding(self, tokenizer):
        """Compute average embedding of numeric tokens from Gemma."""
        numeric_tokens = tokenizer(
            [
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                ".",
                "-",
                "e",
                "time series",
            ],
            return_tensors="pt",
            add_special_tokens=False,
        )
        with torch.no_grad():
            num_embeds = self.gemma.embed_tokens(numeric_tokens.input_ids)
            numeric_embedding = num_embeds.mean(dim=0).mean(dim=0)
        return numeric_embedding

    def forward(self, context, mask=None, target=None, target_mask=None):
        """Forward pass that returns both ChronosBoltOutput and encoder hidden states."""
        batch_size = context.size(0)

        hidden_states, loc_scale, input_embeds, attention_mask = self.encode(
            context=context, mask=mask
        )
        sequence_output = self.decode(input_embeds, attention_mask, hidden_states)

        quantile_preds_shape = (
            batch_size,
            self.num_quantiles,
            self.chronos_config.prediction_length,
        )
        quantile_preds = self.output_patch_embedding(sequence_output).view(
            *quantile_preds_shape
        )

        loss = None
        if target is not None:
            target, _ = self.instance_norm(target, loc_scale)
            target = target.unsqueeze(1)
            assert self.chronos_config.prediction_length >= target.shape[-1]

            target = target.to(quantile_preds.device)
            target_mask = (
                target_mask.unsqueeze(1).to(quantile_preds.device)
                if target_mask is not None
                else ~torch.isnan(target)
            )
            target[~target_mask] = 0.0

            if self.chronos_config.prediction_length > target.shape[-1]:
                padding_shape = (
                    *target.shape[:-1],
                    self.chronos_config.prediction_length - target.shape[-1],
                )
                target = torch.cat(
                    [target, torch.zeros(padding_shape).to(target)], dim=-1
                )
                target_mask = torch.cat(
                    [target_mask, torch.zeros(padding_shape).to(target_mask)], dim=-1
                )

            loss = (
                2
                * torch.abs(
                    (target - quantile_preds)
                    * (
                        (target <= quantile_preds).float()
                        - self.quantiles.view(1, self.num_quantiles, 1)  # type: ignore
                    )
                )
                * target_mask.float()
            )
            loss = loss.mean(dim=-2)
            loss = loss.sum(dim=-1)
            loss = loss.mean()

        quantile_preds = self.instance_norm.inverse(
            quantile_preds.view(batch_size, -1),
            loc_scale,
        ).view(*quantile_preds_shape)

        return GemmaTSOutput(
            loss=loss,
            quantile_preds=quantile_preds,
            projected_hidden_states=self._projected_hidden_states,
        )

    def decode(
        self, input_embeds, attention_mask, hidden_states, output_attentions=False
    ):
        """
        Replace T5 decoder with Gemma, optionally using text prompt.

        Returns: (B, 1, d_model) to match Chronos Bolt's expectation
        """
        B = hidden_states.shape[0]
        device = hidden_states.device

        # Project encoder outputs (time series) to Gemma dimension
        h = self.enc_to_gemma(hidden_states)  # (B, seq_len, dim)
        self._projected_hidden_states = h

        # Get BOS token embedding
        bos_ids = torch.tensor([self.bos_token_id], device=device)
        bos_embed = self.gemma.embed_tokens(bos_ids)  # (1, dim)
        bos_embed = bos_embed.unsqueeze(0).expand(B, -1, -1)  # (B, 1, dim)

        # Conditionally add text prompt if configured
        if self.prompt_ids is not None:
            # Get prompt embeddings
            prompt_ids = self.prompt_ids.to(device)
            prompt_embeds = self.gemma.embed_tokens(
                prompt_ids
            )  # (num_prompt_tokens, dim)
            prompt_embeds = prompt_embeds.unsqueeze(0).expand(
                B, -1, -1
            )  # (B, num_prompt_tokens, dim)

            # Concatenate: [bos, prompt, time_series]
            combined = torch.cat([bos_embed, prompt_embeds, h], dim=1)
        else:
            # No prompt: just [bos, time_series]
            combined = torch.cat([bos_embed, h], dim=1)

        # Pass through Gemma
        out = self.gemma(inputs_embeds=combined, return_dict=True)

        # Get last token's hidden state
        last = out.last_hidden_state[:, -1:, :]

        # Project back to decoder dimension
        return self.gemma_to_dec(last)


def create_gemma_ts(
    chronos_base: str,
    gemma_model: str,
    context_length: int,
    prediction_length: int,
    patch_size: int,
    patch_stride: int,
    text_prompt: Optional[str],
    freeze: bool,
    use_bfloat16: bool,
):
    """Create GemmaTS from Chronos config.

    Note: This creates the model from scratch using the Chronos config as a template,
    so we CAN change prediction_length here (unlike loading pretrained Chronos Bolt).
    The output layer will be initialized with the correct size for the specified prediction_length.
    """

    # Load Chronos config as template
    config = AutoConfig.from_pretrained(chronos_base)

    # Override settings - this is safe because we're creating from scratch
    config.chronos_config["context_length"] = context_length
    config.chronos_config["prediction_length"] = prediction_length
    config.chronos_config["input_patch_size"] = patch_size
    config.chronos_config["input_patch_stride"] = patch_stride

    model = GemmaTS(config, gemma_model, text_prompt, use_bfloat16)

    # Convert entire model to bfloat16 if configured
    if use_bfloat16:
        model = model.to(torch.bfloat16)  # type: ignore

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

        for param in model.enc_to_gemma.parameters():
            param.requires_grad = True

        for param in model.gemma_to_dec.parameters():
            param.requires_grad = True

        # output layer params
        for param in model.output_patch_embedding.parameters():
            param.requires_grad = True

        # unfreeze encoder's last layer
        for param in model.encoder.block[-1].parameters():
            param.requires_grad = True

        # unfreeze last Gemma MLP
        # for name, param in model.gemma.layers[-1].self_attn.named_parameters():
        #     if any(k in name for k in ["q_proj", "v_proj"]):
        #         param.requires_grad = True

    return model
