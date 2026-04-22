"""EntityEncoder: Bi-encoder for entity understanding with gradient flow through BGE backbone.

This module implements §4.4 of the GSR-CACL proposal:
    - Encodes (company, year, sector) metadata → entity embeddings
    - Shares BGE backbone with TextEncoder (gradient flows to shared backbone)
    - Enables s_ent = cos(e_Q, e_D) learned matching (not exact match)

Why this matters:
    Exact match ("Apple" vs "Apple Inc." → score=0) produces no gradient.
    EntityEncoder learns continuous representations where entity variants
    cluster together in embedding space, enabling gradient-based learning.

Architecture (Eq.12 in proposal):
    e = LayerNorm(BGE(company) ⊕ BGE(year) ⊕ BGE(sector)) ∈ R^{d_e}

Training:
    - EntitySupConLoss (Khosla et al., NeurIPS 2020)
    - Positive pairs: same entity ("Apple", "Apple Inc.", "AAPL")
    - Gradient flows through shared BGE backbone → improves both entity AND text embeddings
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class EntityEncoder(nn.Module):
    """
    Bi-encoder for entity understanding — company/year/sector → embedding space.

    Shares the same BGE backbone as TextEncoder so that:
        (1) Gradient from entity loss flows back into the shared transformer
        (2) Entity understanding improves text embeddings (joint benefit)

    The encoder processes three metadata fields independently, concatenates
    their [CLS] embeddings, and projects to entity space.

    Architecture:
        company → BGE → [CLS] ──┐
        year   → BGE → [CLS] ──┼── concat ──→ proj ──→ LayerNorm → e ∈ R^{d_e}
        sector → BGE → [CLS] ──┘

    Training signal: EntitySupConLoss (see losses/entity_supcon_loss.py)
    """

    def __init__(
        self,
        backbone: nn.Module,
        entity_dim: int = 256,
        dropout: float = 0.1,
        share_tokenizer: bool = True,
    ):
        """
        Args:
            backbone: The transformer backbone (e.g., BAAI/bge-large-en-v1.5).
                      Must have .config.hidden_size and share tokenizer with this encoder.
            entity_dim: Output embedding dimension d_e (default 256).
            dropout: Dropout rate before projection.
            share_tokenizer: If True, reuses the backbone's tokenizer.
        """
        super().__init__()
        self.backbone = backbone
        self.embed_dim = backbone.config.hidden_size  # d = 1024 (bge-large)
        self.entity_dim = entity_dim  # d_e = 256
        self.share_tokenizer = share_tokenizer

        # FIX BUG 3: Tokenizer must come from a Tokenizer object, NOT embed_tokens
        # (embed_tokens is the embedding layer, not a tokenizer)
        # Priority: (1) passed tokenizer, (2) backbone's own tokenizer, (3) AutoTokenizer
        if hasattr(backbone, "tokenizer"):
            self.tokenizer = backbone.tokenizer
        elif hasattr(backbone, "get_input_embeddings"):
            # For models loaded via AutoModel, try AutoTokenizer with the model name
            model_name = getattr(backbone, "name_or_path", None) or getattr(
                backbone, "config", None
            )
            if isinstance(model_name, str):
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            else:
                self.tokenizer = None
        else:
            self.tokenizer = None

        # Three-branch projection: company ⊕ year ⊕ sector → entity_dim
        # Each branch: [3*embed_dim] → [embed_dim] → project to [entity_dim]
        self.proj = nn.Sequential(
            nn.Linear(3 * self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, entity_dim),
        )
        self.norm = nn.LayerNorm(entity_dim)

        logger.info(
            f"EntityEncoder: embed_dim={self.embed_dim}, entity_dim={self.entity_dim}, "
            f"tokenizer={'set' if self.tokenizer else 'NONE'}, "
            f"share_backbone={share_tokenizer}"
        )

    @property
    def temperature(self) -> float:
        """Temperature for contrastive loss (exposed for SupConLoss)."""
        return 0.07

    def _encode_field(self, text: str, device: torch.device) -> torch.Tensor:
        """
        Encode a single text field (company / year / sector) → [embed_dim].

        Uses [CLS] token from the backbone.
        FIX BUG 2: tokenizer returns [1, seq_len] → squeeze to [seq_len]
                    backbone returns [1, 1, embed_dim] → squeeze to [embed_dim]
        """
        if not text or not text.strip():
            return torch.zeros(self.embed_dim, device=device)

        if self.tokenizer is None:
            raise ValueError("Tokenizer not available. Set backbone with embed_tokens or use AutoTokenizer.")

        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=32,  # Short inputs — entity fields are short
            return_tensors="pt",
        ).to(device)

        with torch.cuda.amp.autocast(enabled=False):
            # Cast to float32 to avoid AMP issues with backbone
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = self.backbone(**inputs)

        # [CLS] token embedding: [1, seq, embed_dim] → [embed_dim]
        cls_embed = outputs.last_hidden_state[:, 0, :].squeeze(0)
        return cls_embed

    def encode(
        self,
        company: str,
        year: str,
        sector: str,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Encode entity metadata → normalized entity embedding.

        Args:
            company: Company name (e.g., "Apple Inc.", "AAPL", "Apple")
            year: Report year (e.g., "2023")
            sector: Company sector (e.g., "Technology")
            device: torch device
        Returns:
            entity embedding: [entity_dim], L2-normalized
        """
        c_emb = self._encode_field(company, device)
        y_emb = self._encode_field(year, device)
        s_emb = self._encode_field(sector, device)

        # Concatenate: [3*embed_dim]
        combined = torch.cat([c_emb, y_emb, s_emb], dim=-1)

        # Project to entity space
        entity = self.proj(combined)
        entity = self.norm(entity)

        # L2 normalize (required for cosine similarity to equal dot product)
        entity = F.normalize(entity, p=2, dim=-1)
        return entity

    def forward(
        self,
        companies: list[str],
        years: list[str],
        sectors: list[str],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Encode a batch of entities.

        Args:
            companies: List of B company names
            years: List of B years
            sectors: List of B sectors
            device: torch device
        Returns:
            embeddings: [B, entity_dim], L2-normalized
        """
        B = len(companies)
        embeddings = []

        for i in range(B):
            emb = self.encode(
                company=companies[i],
                year=years[i],
                sector=sectors[i],
                device=device,
            )
            embeddings.append(emb)

        return torch.stack(embeddings)  # [B, entity_dim]

    def encode_with_bge_text(
        self,
        text: str,
        company: str,
        year: str,
        sector: str,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode both text AND entity metadata for joint training.

        Returns:
            (text_embed, entity_embed) — both [1, embed_dim] or [1, entity_dim]
            This is used when we want the BGE text embedding alongside entity embedding.
        """
        # BGE text encoding (from shared backbone)
        text_inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        with torch.cuda.amp.autocast(enabled=False):
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            text_outputs = self.backbone(**text_inputs)
        text_emb = text_outputs.last_hidden_state[:, 0, :]
        text_emb = F.normalize(text_emb, p=2, dim=-1)

        # Entity encoding
        entity_emb = self.encode(company, year, sector, device)

        return text_emb, entity_emb


class SharedEncoder(nn.Module):
    """
    Unified encoder that wraps both TextEncoder and EntityEncoder with a SHARED backbone.

    This is the recommended way to use both encoders during training:
        - text_emb = encoder.text_encode(texts)        → gradient flows to shared backbone
        - entity_emb = encoder.entity_encode(metas)    → gradient flows to shared backbone
        - Loss = text_loss + lambda * entity_loss      → shared backbone gets combined gradient

    The shared backbone learns both:
        (1) Text semantics from retrieval tasks
        (2) Entity equivalence from SupCon task
    This is the key insight of §4.4 — entity understanding improves text representations.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        finetune: Literal["full", "lora", "frozen"] = "lora",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        entity_dim: int = 256,
        max_length: int = 512,
    ):
        """
        Args:
            model_name: HuggingFace model name
            finetune: Fine-tuning strategy
            lora_r/alpha/dropout: LoRA hyperparameters
            entity_dim: Output dimension for entity embeddings
            max_length: Max sequence length for text encoding
        """
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.finetune = finetune
        self.max_length = max_length

        # Shared backbone
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.embed_dim = self.backbone.config.hidden_size

        # Apply fine-tuning strategy
        if finetune == "frozen":
            for p in self.backbone.parameters():
                p.requires_grad = False

        elif finetune == "lora":
            try:
                from peft import LoraConfig, get_peft_model, TaskType
            except ImportError as e:
                raise ImportError("pip install peft is required for LoRA") from e

            for p in self.backbone.parameters():
                p.requires_grad = False

            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "v_proj", "query", "value"],
            )
            self.backbone = get_peft_model(self.backbone, lora_config)

        # Text encoder branch
        self.text_norm = nn.LayerNorm(self.embed_dim)

        # Entity encoder branch — shares the SAME backbone
        self.entity_encoder = EntityEncoder(
            backbone=self.backbone,
            entity_dim=entity_dim,
            share_tokenizer=True,
        )

        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.backbone.parameters())
        logger.info(
            f"SharedEncoder({model_name}): finetune={finetune}, "
            f"trainable={trainable:,}/{total:,} ({100*trainable/total:.1f}%)"
        )

    def text_encode(self, texts: list[str], normalize: bool = True) -> torch.Tensor:
        """
        Encode texts → text embeddings [B, embed_dim].

        Uses shared backbone with gradient flow.
        """
        device = next(self.backbone.parameters()).device
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)

        outputs = self.backbone(**inputs)
        embeds = outputs.last_hidden_state[:, 0, :]
        embeds = self.text_norm(embeds)

        if normalize:
            embeds = F.normalize(embeds, p=2, dim=-1)
        return embeds

    def entity_encode(
        self,
        companies: list[str],
        years: list[str],
        sectors: list[str],
    ) -> torch.Tensor:
        """
        Encode entity metadata → entity embeddings [B, entity_dim].

        Uses shared backbone with gradient flow.
        """
        device = next(self.backbone.parameters()).device
        return self.entity_encoder.forward(companies, years, sectors, device)

    def forward(
        self,
        texts: list[str] | None = None,
        companies: list[str] | None = None,
        years: list[str] | None = None,
        sectors: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Unified forward pass.

        If texts provided: returns text embeddings
        If entity fields provided: returns entity embeddings
        If both provided: returns (text_emb, entity_emb)
        """
        if texts is not None and companies is not None:
            return self.text_encode(texts), self.entity_encode(companies, years, sectors)
        elif texts is not None:
            return self.text_encode(texts)
        elif companies is not None:
            return self.entity_encode(companies, years, sectors)
        else:
            raise ValueError("Must provide at least one of: texts or (companies, years, sectors)")

    def get_trainable_params(self) -> list[nn.Parameter]:
        """Return only parameters requiring gradients."""
        return [p for p in self.backbone.parameters() if p.requires_grad]
