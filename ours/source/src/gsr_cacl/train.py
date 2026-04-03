#!/usr/bin/env python3
"""
GSR + CACL Training Script.

Three-stage training (overall_idea.md §4.7, architecture.md §5):
  Stage 1 — Identity Pretraining:   learn (Company, Year) discrimination via entity scoring
  Stage 2 — Structural Pretraining: KG encoding + constraint scoring calibration
  Stage 3 — Joint Finetuning:       full CACL objective with CHAP negatives

All stages support end-to-end training with gradient flow through the text encoder.
Fine-tuning strategy is configurable: "full", "lora", or "frozen".

Usage:
    # Full fine-tuning on Kaggle/Colab T4 GPU (recommended)
    python -m gsr_cacl.train --dataset finqa --stage all --epochs 5 --batch-size 8 \
        --encoder BAAI/bge-large-en-v1.5 --finetune lora

    # Full fine-tuning on A100 (no resource limit)
    python -m gsr_cacl.train --dataset finqa --stage all --epochs 5 --batch-size 16 \
        --encoder BAAI/bge-large-en-v1.5 --finetune full

    # Legacy frozen mode (not recommended for benchmark)
    python -m gsr_cacl.train --dataset finqa --stage all --finetune frozen
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

from gsr_cacl.training import (
    RetrievalDataset,
    RetrievalSample,
    CACLLoss,
    TripletLoss,
)
from gsr_cacl.negative_sampler import CHAPNegativeSampler
from gsr_cacl.kg import build_constraint_kg
from gsr_cacl.scoring import JointScorer, compute_constraint_score
from gsr_cacl.encoders import GATEncoder, TextEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Dataset configs (mirrors benchmark_gsr.py)
DATASET_CONFIGS = {
    "finqa": {"config_name": "FinQA", "train_split": "train", "eval_split": "test"},
    "convfinqa": {"config_name": "ConvFinQA", "train_split": "turn_0", "eval_split": "turn_0"},
    "tatqa": {"config_name": "TAT-DQA", "train_split": "train", "eval_split": "test"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_trainable_params(*modules: torch.nn.Module) -> list[torch.nn.Parameter]:
    """Collect all parameters requiring gradients from multiple modules."""
    params = []
    seen = set()
    for m in modules:
        for p in m.parameters():
            if p.requires_grad and id(p) not in seen:
                params.append(p)
                seen.add(id(p))
    return params


# ---------------------------------------------------------------------------
# Data loading (no g4k — load from HuggingFace directly)
# ---------------------------------------------------------------------------

def load_training_data(dataset: str, split: str = "train") -> list[RetrievalSample]:
    """Load training samples from T²-RAGBench via HuggingFace datasets."""
    ds_cfg = DATASET_CONFIGS.get(dataset)
    if ds_cfg is None:
        raise ValueError(f"Unknown dataset: {dataset}")

    hf_split = ds_cfg["train_split"] if split == "train" else ds_cfg["eval_split"]
    logger.info(f"Loading {hf_split} split for {dataset}...")
    df = load_dataset(
        "G4KMU/t2-ragbench",
        ds_cfg["config_name"],
        split=hf_split,
    ).to_pandas()

    samples = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing training samples"):
        query = f"{row.get('company_name', '')}: {row.get('question', '')}"
        context = str(row.get("context", ""))
        sample = RetrievalSample(
            query=query,
            positive_context=context,
            negative_contexts=[],
            company_name=str(row.get("company_name", "")),
            report_year=str(row.get("report_year", "")),
            company_sector=str(row.get("company_sector", "")),
        )
        samples.append(sample)

    logger.info(f"Loaded {len(samples)} training samples")
    return samples


def _meta_to_tensor(sample: RetrievalSample, device: torch.device) -> torch.Tensor:
    """Convert sample metadata to float tensor [3] for entity matching.

    Encodes company_name, report_year, company_sector as hashed floats for comparison.
    """
    def _hash_str(s: str) -> float:
        h = 0
        for c in s.lower().strip():
            h = (h * 31 + ord(c)) % 10000
        return h / 10000.0

    return torch.tensor(
        [_hash_str(sample.company_name), _hash_str(sample.report_year), _hash_str(sample.company_sector)],
        dtype=torch.float32, device=device,
    )


# ---------------------------------------------------------------------------
# Stage 1 — Identity Pretraining (§4.7 / architecture.md §5.1)
# ---------------------------------------------------------------------------

def stage_identity(
    text_encoder: TextEncoder,
    scorer: JointScorer,
    gat_encoder: GATEncoder,
    dataset: list[RetrievalSample],
    device: torch.device,
    epochs: int = 3,
    lr: float = 1e-4,
    batch_size: int = 16,
) -> None:
    """Train entity matching: learn to distinguish (Company, Year) pairs.

    End-to-end: texts → TextEncoder → JointScorer → TripletLoss → backprop
    through both scorer AND text encoder.
    """
    logger.info("=== Stage 1: Identity Pretraining ===")
    params = _collect_trainable_params(text_encoder, scorer)
    optimizer = torch.optim.Adam(params, lr=lr)
    triplet = TripletLoss(margin=0.2)

    dataset_obj = RetrievalDataset(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset_obj, batch_size=batch_size, shuffle=True, collate_fn=lambda b: b,
    )

    text_encoder.train()
    scorer.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for batch in tqdm(dataloader, desc=f"Identity Epoch {epoch+1}"):
            B = len(batch)

            # Batch encode queries and documents through differentiable encoder
            queries = [s.query for s in batch]
            docs = [s.positive_context[:512] for s in batch]

            q_t = text_encoder(queries)      # [B, embed_dim] with grad
            d_t = text_encoder(docs)          # [B, embed_dim] with grad

            q_metas = []
            d_metas_pos = []
            d_metas_neg = []
            for sample in batch:
                q_metas.append(_meta_to_tensor(sample, device))
                d_metas_pos.append(_meta_to_tensor(sample, device))

            # Create negative: swap entity metadata within batch (circular shift)
            for i in range(B):
                neg_idx = (i + 1) % B
                d_metas_neg.append(_meta_to_tensor(batch[neg_idx], device))

            kg_dummy = torch.zeros(B, gat_encoder.hidden_dim, device=device)
            q_meta_t = torch.stack(q_metas)
            d_meta_pos_t = torch.stack(d_metas_pos)
            d_meta_neg_t = torch.stack(d_metas_neg)

            # Constraint features: neutral (all 1.0, 0.0, 0.0) — not trained in Stage 1
            cs_feats = torch.tensor([[1.0, 0.0, 0.0]] * B, device=device)

            pos_scores = scorer(q_t, d_t, kg_dummy, q_meta_t, d_meta_pos_t, cs_feats)
            neg_scores = scorer(q_t, d_t, kg_dummy, q_meta_t, d_meta_neg_t, cs_feats)

            loss = triplet(pos_scores, neg_scores)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        logger.info(f"Epoch {epoch+1} — Loss: {total_loss/max(n_batches,1):.4f}")


# ---------------------------------------------------------------------------
# Stage 2 — Structural Pretraining (§4.7 / architecture.md §5.1)
# ---------------------------------------------------------------------------

def stage_structural(
    text_encoder: TextEncoder,
    scorer: JointScorer,
    gat_encoder: GATEncoder,
    dataset: list[RetrievalSample],
    device: torch.device,
    epochs: int = 3,
    lr: float = 1e-4,
    batch_size: int = 8,
) -> None:
    """Train KG encoding + constraint scoring calibration.

    Objective: MSE(CS(G_D), 1.0) — push well-formed KGs to score ≈ 1.
    End-to-end: TextEncoder + GATEncoder + JointScorer all receive gradients.
    """
    logger.info("=== Stage 2: Structural Pretraining ===")
    params = _collect_trainable_params(text_encoder, scorer, gat_encoder)
    optimizer = torch.optim.Adam(params, lr=lr)

    dataset_obj = RetrievalDataset(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset_obj, batch_size=batch_size, shuffle=True, collate_fn=lambda b: b,
    )

    text_encoder.train()
    scorer.train()
    gat_encoder.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for batch in tqdm(dataloader, desc=f"Structural Epoch {epoch+1}"):
            B = len(batch)

            # Build KGs and encode them (gradient flows through GATEncoder)
            kg_embeds = []
            cs_results = []
            for sample in batch:
                kg = build_constraint_kg(sample.positive_context)
                kg_embed = gat_encoder.encode_graph(kg)
                kg_embeds.append(kg_embed)
                cs_results.append(compute_constraint_score(kg))

            kg_embed_t = torch.stack(kg_embeds)  # [B, hidden_dim]

            # Constraint features through scorer
            cs_feats = scorer.build_constraint_features(cs_results, device)
            predicted_cs = scorer.forward_constraint(cs_feats)  # [B]
            target_cs = torch.ones(B, device=device)  # well-formed KGs → score 1.0

            # MSE loss: push constraint refinement towards 1.0
            loss_cs = F.mse_loss(predicted_cs, target_cs)

            # Regularization: encourage KG embeddings to be non-degenerate
            loss_reg = -torch.clamp(kg_embed_t.var(dim=0).mean(), max=1.0) * 0.1

            loss = loss_cs + loss_reg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        logger.info(f"Epoch {epoch+1} — Loss: {total_loss/max(n_batches,1):.4f}")


# ---------------------------------------------------------------------------
# Stage 3 — Joint Finetuning with CACL (§4.6 + §4.7 / architecture.md §5)
# ---------------------------------------------------------------------------

def stage_joint(
    text_encoder: TextEncoder,
    scorer: JointScorer,
    gat_encoder: GATEncoder,
    dataset: list[RetrievalSample],
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 5e-5,
    margin: float = 0.2,
    lambda_constraint: float = 0.5,
    save_path: Path | None = None,
) -> None:
    """Full CACL objective with CHAP negatives — end-to-end.

    For each (Q, C+, C-_CHAP):
        q = TextEncoder(Q)                    ← gradient flows
        d+ = TextEncoder(C+)                  ← gradient flows
        d- = TextEncoder(C-_CHAP)             ← gradient flows
        kg+ = GATEncoder(build_kg(C+))        ← gradient flows
        kg- = GATEncoder(build_kg(C-))        ← gradient flows
        s+ = JointScorer(q, d+, kg+, ...)     ← gradient flows
        s- = JointScorer(q, d-, kg-, ...)     ← gradient flows
        loss = CACL(s+, s-, violates)
        loss.backward()                       ← updates TextEncoder + GATEncoder + JointScorer
    """
    logger.info("=== Stage 3: Joint Finetuning (CACL) ===")

    params = _collect_trainable_params(text_encoder, scorer, gat_encoder)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = CACLLoss(margin=margin, lambda_constraint=lambda_constraint)
    sampler = CHAPNegativeSampler(chap_a_prob=0.5, chap_s_prob=0.3, chap_e_prob=0.2)

    dataset_obj = RetrievalDataset(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset_obj, batch_size=batch_size, shuffle=True, collate_fn=lambda b: b,
    )

    text_encoder.train()
    scorer.train()
    gat_encoder.train()
    for epoch in range(epochs):
        total = triplet_total = const_total = 0.0
        n_batches = 0

        for batch in tqdm(dataloader, desc=f"Joint Epoch {epoch+1}/{epochs}"):
            B = len(batch)

            pos_scores_list = []
            neg_scores_list = []
            violates_list = []

            for sample in batch:
                # Query embedding — differentiable
                q_emb = text_encoder.encode_single(sample.query).unsqueeze(0)  # [1, d]

                # Positive: KG + text embedding — differentiable
                pos_kg = build_constraint_kg(sample.positive_context)
                pos_kg_embed = gat_encoder.encode_graph(pos_kg).unsqueeze(0)  # [1, h]
                pos_d_emb = text_encoder.encode_single(
                    sample.positive_context[:512]
                ).unsqueeze(0)  # [1, d]

                q_meta = _meta_to_tensor(sample, device).unsqueeze(0)
                d_meta = _meta_to_tensor(sample, device).unsqueeze(0)
                pos_cs = compute_constraint_score(pos_kg)
                pos_cs_feats = scorer.build_constraint_features([pos_cs], device)

                pos_score = scorer(q_emb, pos_d_emb, pos_kg_embed, q_meta, d_meta, pos_cs_feats)
                pos_scores_list.append(pos_score.squeeze(0))

                # Negatives: CHAP perturbation
                negs = sampler.sample(pos_kg, n_negatives=1)
                if not negs:
                    from gsr_cacl.negative_sampler import apply_chap_e
                    negs = [apply_chap_e(pos_kg)]

                for neg in negs:
                    neg_kg = build_constraint_kg(neg.table_md)
                    neg_kg_embed = gat_encoder.encode_graph(neg_kg).unsqueeze(0)
                    neg_d_emb = text_encoder.encode_single(
                        neg.table_md[:512]
                    ).unsqueeze(0)

                    neg_cs = compute_constraint_score(neg_kg)
                    neg_cs_feats = scorer.build_constraint_features([neg_cs], device)

                    neg_score = scorer(q_emb, neg_d_emb, neg_kg_embed, q_meta, d_meta, neg_cs_feats)
                    neg_scores_list.append(neg_score.squeeze(0))
                    violates_list.append(1.0 if neg.is_violated else 0.0)

            pos_scores_t = torch.stack(pos_scores_list)
            neg_scores_t = torch.stack(neg_scores_list)
            violates_t = torch.tensor(violates_list, dtype=torch.float32, device=device)

            # Align dimensions for triplet loss
            neg_for_triplet = neg_scores_t[:B] if len(neg_scores_t) >= B else neg_scores_t
            if len(neg_for_triplet) < B:
                pad = torch.zeros(B - len(neg_for_triplet), device=device)
                neg_for_triplet = torch.cat([neg_for_triplet, pad])

            losses = criterion(pos_scores_t, neg_for_triplet, violates_t)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            total += losses["total"].item()
            triplet_total += losses["triplet"].item()
            const_total += losses["constraint"].item()
            n_batches += 1

        scheduler.step()
        logger.info(
            f"Epoch {epoch+1} — "
            f"Loss: {total/max(n_batches,1):.4f} "
            f"(triplet={triplet_total/max(n_batches,1):.4f}, "
            f"constraint={const_total/max(n_batches,1):.4f})"
        )

        # Checkpoint
        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)
            ckpt = save_path / f"ckpt_epoch_{epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "text_encoder_state": text_encoder.state_dict(),
                "scorer_state": scorer.state_dict(),
                "gat_encoder_state": gat_encoder.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "loss": total / max(n_batches, 1),
            }, ckpt)
            logger.info(f"Checkpoint saved: {ckpt}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# Recommended encoder presets for different hardware
ENCODER_PRESETS = {
    "t4":   {"encoder": "BAAI/bge-large-en-v1.5",  "finetune": "lora",   "batch_size": 8},
    "a100": {"encoder": "BAAI/bge-large-en-v1.5",  "finetune": "full",   "batch_size": 16},
    "v100": {"encoder": "BAAI/bge-base-en-v1.5",   "finetune": "full",   "batch_size": 16},
    "cpu":  {"encoder": "BAAI/bge-base-en-v1.5",   "finetune": "frozen", "batch_size": 4},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="GSR + CACL Training (end-to-end)")
    parser.add_argument("--dataset", type=str, default="finqa",
                        choices=["finqa", "convfinqa", "tatqa"])
    parser.add_argument("--stage", type=str, default="joint",
                        choices=["identity", "structural", "joint", "all"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--lambda", dest="lambda_constraint", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    parser.add_argument("--save", type=str, default=None)

    # Text encoder arguments
    parser.add_argument("--encoder", type=str, default="BAAI/bge-large-en-v1.5",
                        help="HuggingFace model name for text encoder")
    parser.add_argument("--finetune", type=str, default="lora",
                        choices=["full", "lora", "frozen"],
                        help="Fine-tuning strategy for text encoder")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank (only used when --finetune=lora)")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha (only used when --finetune=lora)")
    parser.add_argument("--preset", type=str, default=None,
                        choices=list(ENCODER_PRESETS.keys()),
                        help="Hardware preset overriding encoder/finetune/batch-size")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save VRAM")

    args = parser.parse_args()

    # Apply hardware preset if specified
    if args.preset:
        preset = ENCODER_PRESETS[args.preset]
        args.encoder = preset["encoder"]
        args.finetune = preset["finetune"]
        args.batch_size = preset["batch_size"]
        logger.info(f"Applied preset '{args.preset}': {preset}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Device: {device}")

    samples = load_training_data(args.dataset, split="train")

    # Initialize text encoder with gradient flow
    text_encoder = TextEncoder(
        model_name=args.encoder,
        finetune=args.finetune,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_length=512,
    ).to(device)

    if args.gradient_checkpointing and hasattr(text_encoder.backbone, "gradient_checkpointing_enable"):
        text_encoder.backbone.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    embed_dim = text_encoder.embed_dim

    # Initialize models — dimensions adapt to chosen encoder
    gat_encoder = GATEncoder(
        embed_dim=embed_dim, hidden_dim=256, num_heads=4, num_layers=2,
    ).to(device)

    scorer = JointScorer(
        text_embed_dim=embed_dim, kg_embed_dim=256, entity_feat_dim=3, hidden_dim=64,
    ).to(device)

    save_path = Path(args.save) if args.save else Path(f"outputs/gsr_training/{args.dataset}")

    # Log trainable parameter summary
    enc_params = sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)
    gat_params = sum(p.numel() for p in gat_encoder.parameters() if p.requires_grad)
    sc_params = sum(p.numel() for p in scorer.parameters() if p.requires_grad)
    total_params = enc_params + gat_params + sc_params
    logger.info(
        f"Trainable params: TextEncoder={enc_params:,} + GAT={gat_params:,} + "
        f"Scorer={sc_params:,} = {total_params:,} total"
    )

    if args.stage in ("identity", "all"):
        stage_identity(text_encoder, scorer, gat_encoder, samples, device,
                       epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)

    if args.stage in ("structural", "all"):
        stage_structural(text_encoder, scorer, gat_encoder, samples, device,
                         epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)

    if args.stage in ("joint", "all"):
        stage_joint(
            text_encoder, scorer, gat_encoder, samples, device,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, margin=args.margin,
            lambda_constraint=args.lambda_constraint,
            save_path=save_path,
        )

    save_path.mkdir(parents=True, exist_ok=True)
    torch.save({
        "text_encoder_state": text_encoder.state_dict(),
        "scorer_state": scorer.state_dict(),
        "gat_encoder_state": gat_encoder.state_dict(),
        "encoder_model_name": args.encoder,
        "finetune_strategy": args.finetune,
        "embed_dim": embed_dim,
    }, save_path / "final_model.pt")
    logger.info(f"Model saved to {save_path / 'final_model.pt'}")


if __name__ == "__main__":
    main()
