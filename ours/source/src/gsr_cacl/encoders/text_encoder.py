"""Trainable Text Encoder with gradient flow through the transformer backbone.

Unlike the LangChain HuggingFaceEmbeddings wrapper (which returns numpy and breaks
the computation graph), this module keeps full gradient flow from the transformer
backbone through to the final loss.

Supports three fine-tuning strategies:
    - "full":   all transformer parameters are trainable
    - "lora":   LoRA adapters on attention layers (requires peft)
    - "frozen": legacy mode — backbone frozen, no gradient (not recommended)

Compatible models:
    - BAAI/bge-base-en-v1.5          (110M, d=768)
    - BAAI/bge-large-en-v1.5         (335M, d=1024)
    - intfloat/e5-large-v2           (335M, d=1024)
    - intfloat/multilingual-e5-large-instruct  (560M, d=1024)
    - McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp  (8B, d=4096)

The output dimension is auto-detected from the model config.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class TextEncoder(nn.Module):
    """Differentiable text encoder wrapping a HuggingFace transformer.

    Produces normalized embeddings with full gradient flow for contrastive training.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        finetune: Literal["full", "lora", "frozen"] = "full",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        max_length: int = 512,
        normalize: bool = True,
        pooling: Literal["cls", "mean"] = "cls",
    ):
        super().__init__()
        self.model_name = model_name
        self.finetune = finetune
        self.max_length = max_length
        self.normalize = normalize
        self.pooling = pooling

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.embed_dim = self.backbone.config.hidden_size

        if finetune == "frozen":
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info(f"TextEncoder({model_name}): FROZEN — {self.embed_dim}-dim")

        elif finetune == "lora":
            try:
                from peft import LoraConfig, get_peft_model, TaskType
            except ImportError as e:
                raise ImportError("pip install peft  is required for LoRA fine-tuning") from e

            for p in self.backbone.parameters():
                p.requires_grad = False

            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "v_proj", "query", "value"],
                modules_to_save=[],
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
            trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.backbone.parameters())
            logger.info(
                f"TextEncoder({model_name}): LoRA r={lora_r} — "
                f"{trainable:,}/{total:,} trainable ({100*trainable/total:.1f}%)"
            )

        else:  # full
            trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
            logger.info(f"TextEncoder({model_name}): FULL fine-tune — {trainable:,} params, {self.embed_dim}-dim")

    def forward(self, texts: list[str]) -> torch.Tensor:
        """Encode a list of texts → [B, embed_dim] with gradient.

        Args:
            texts: list of B strings
        Returns:
            embeddings: [B, embed_dim] normalized if self.normalize
        """
        device = next(self.backbone.parameters()).device
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)

        outputs = self.backbone(**encoded)

        if self.pooling == "cls":
            embeds = outputs.last_hidden_state[:, 0, :]
        else:  # mean pooling
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            embeds = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        if self.normalize:
            embeds = nn.functional.normalize(embeds, p=2, dim=-1)

        return embeds

    def encode_single(self, text: str) -> torch.Tensor:
        """Encode one text → [embed_dim] with gradient."""
        return self.forward([text]).squeeze(0)

    @torch.no_grad()
    def encode_no_grad(self, texts: list[str], batch_size: int = 32) -> torch.Tensor:
        """Batch encode without gradient (for indexing / inference)."""
        all_embeds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embeds.append(self.forward(batch))
        return torch.cat(all_embeds, dim=0)

    def get_trainable_params(self) -> list[nn.Parameter]:
        """Return only the parameters that require gradients."""
        return [p for p in self.backbone.parameters() if p.requires_grad]
