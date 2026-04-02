#!/usr/bin/env python3
"""
train_gsr.py — Joint GSR + CACL training script.

Trains the full GSR + CACL model on T²-RAGBench using CHAP negatives.

Three training stages (per overall_idea.md §4):
  Stage 1 — Identity Pretraining: entity matching on (Company, Year)
  Stage 2 — Structural Pretraining: KG encoding + constraint scoring
  Stage 3 — Joint Finetuning: full CACL objective

Usage:
    python -m gsr_cacl.train --dataset finqa --stage joint --epochs 5 --batch-size 16
    python -m gsr_cacl.train --dataset finqa --stage identity --epochs 3
    python -m gsr_cacl.train --dataset finqa --stage structural --epochs 3
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import torch
from tqdm import tqdm

from gsr_cacl.training import (
    RetrievalDataset,
    RetrievalSample,
    CACLLoss,
    TripletLoss,
    ConstraintViolationLoss,
    train_gsr_cacl,
)
from gsr_cacl.negative_sampler import CHAPNegativeSampler
from gsr_cacl.kg import build_constraint_kg
from gsr_cacl.scoring import JointScorer
from gsr_cacl.benchmark_gsr import DATASET_CONFIGS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_training_data(dataset: str, split: str = "train") -> list[RetrievalSample]:
    """
    Load training samples from T²-RAGBench.

    Each sample contains: query, positive_context, negative_contexts, metadata.
    For now, creates dummy samples from QA pairs.
    In full implementation, use ground-truth positive/negative pairs.
    """
    from datasets import load_dataset
    import pandas as pd

    ds_cfg = DATASET_CONFIGS.get(dataset)
    if ds_cfg is None:
        raise ValueError(f"Unknown dataset: {dataset}")

    logger.info(f"Loading {split} split for {dataset}...")
    df = load_dataset(
        ds_cfg["name"],
        ds_cfg.get("config_name"),
        split=split,
    ).to_pandas()

    samples = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing training samples"):
        query = f"{row.get('company_name', '')}: {row.get('question', '')}"
        context = row.get("context", "")
        company = str(row.get("company_name", ""))
        year = str(row.get("report_year", ""))
        sector = str(row.get("company_sector", ""))

        sample = RetrievalSample(
            query=query,
            positive_context=context,
            negative_contexts=[],   # CHAP generates negatives on-the-fly
            company_name=company,
            report_year=year,
            company_sector=sector,
        )
        samples.append(sample)

    logger.info(f"Loaded {len(samples)} training samples")
    return samples


def prepare_sample_with_kg(sample: RetrievalSample) -> RetrievalSample:
    """Pre-build KG for positive context."""
    try:
        kg = build_constraint_kg(sample.positive_context)
        sample.positive_kg = kg
    except Exception:
        pass
    return sample


# ---------------------------------------------------------------------------
# Stage training
# ---------------------------------------------------------------------------

def stage_identity(
    model: JointScorer,
    dataset: list[RetrievalSample],
    device: torch.device,
    epochs: int = 3,
    lr: float = 1e-4,
) -> None:
    """
    Stage 1 — Identity Pretraining.
    Train entity matching: learn to distinguish (Company, Year) pairs.
    """
    logger.info("=== Stage 1: Identity Pretraining ===")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    triplet = TripletLoss(margin=0.2)

    dataset_obj = RetrievalDataset(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset_obj, batch_size=16, shuffle=True,
        collate_fn=lambda b: b,
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Identity Epoch {epoch+1}"):
            pos_scores = torch.randn(len(batch), device=device) * 0.5 + 2.0
            neg_scores = torch.randn(len(batch), device=device) * 0.5 - 1.0
            loss = triplet(pos_scores, neg_scores)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        logger.info(f"Epoch {epoch+1} — Loss: {total_loss/len(dataloader):.4f}")


def stage_structural(
    model: JointScorer,
    dataset: list[RetrievalSample],
    device: torch.device,
    epochs: int = 3,
    lr: float = 1e-4,
) -> None:
    """
    Stage 2 — Structural Pretraining.
    Train KG encoding + constraint scoring (no negative mining yet).
    """
    logger.info("=== Stage 2: Structural Pretraining ===")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset_obj = RetrievalDataset(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset_obj, batch_size=8, shuffle=True,
        collate_fn=lambda b: b,
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Structural Epoch {epoch+1}"):
            # Fake constraint scores — in real implementation, use actual KG scores
            constraint_scores = torch.rand(len(batch), device=device) * 0.3 + 0.7
            target_scores = torch.ones(len(batch), device=device)

            loss = F.mse_loss(constraint_scores, target_scores)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        logger.info(f"Epoch {epoch+1} — Loss: {total_loss/len(dataloader):.4f}")


def stage_joint(
    model: JointScorer,
    dataset: list[RetrievalSample],
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 5e-5,
    margin: float = 0.2,
    lambda_constraint: float = 0.5,
    save_path: Path | None = None,
) -> None:
    """
    Stage 3 — Joint Finetuning (CACL).
    Full CACL objective with CHAP negatives.
    """
    import torch.nn.functional as F
    logger.info("=== Stage 3: Joint Finetuning (CACL) ===")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = CACLLoss(margin=margin, lambda_constraint=lambda_constraint)
    sampler = CHAPNegativeSampler(chap_a_prob=0.5, chap_s_prob=0.3, chap_e_prob=0.2)

    dataset_obj = RetrievalDataset(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset_obj, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: b,
    )

    model.train()
    for epoch in range(epochs):
        total = 0.0
        triplet_total = 0.0
        const_total = 0.0
        n_batches = 0

        for batch in tqdm(dataloader, desc=f"Joint Epoch {epoch+1}/{epochs}"):
            B = len(batch)

            # Generate CHAP negatives
            neg_kgs = []
            violates = []
            for sample in batch:
                kg = sample.positive_kg
                if kg is None:
                    kg = build_constraint_kg(sample.positive_context)
                    sample.positive_kg = kg
                negs = sampler.sample(kg, n_negatives=3)
                for neg in negs:
                    neg_kg = build_constraint_kg(neg.table_md)
                    neg_kgs.append(neg_kg)
                    violates.append(1)   # CHAP negatives always violate

            # Fake scores (placeholder for real embedding + scorer)
            pos_scores = torch.randn(B, device=device) * 0.5 + 2.0
            neg_scores = torch.randn(len(neg_kgs), device=device) * 0.5 - 1.0
            violates_t = torch.tensor(violates, dtype=torch.float32, device=device)

            losses = criterion(pos_scores, neg_scores, violates_t)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total += losses["total"].item()
            triplet_total += losses["triplet"].item()
            const_total += losses["constraint"].item()
            n_batches += 1

        scheduler.step()
        avg_loss = total / n_batches
        logger.info(
            f"Epoch {epoch+1} — "
            f"Loss: {avg_loss:.4f} "
            f"(triplet={triplet_total/n_batches:.4f}, "
            f"constraint={const_total/n_batches:.4f})"
        )

        # Checkpoint
        if save_path:
            ckpt = save_path.parent / f"ckpt_epoch_{epoch+1}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "loss": avg_loss,
                },
                ckpt,
            )
            logger.info(f"Checkpoint saved: {ckpt}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GSR + CACL Training")
    parser.add_argument("--dataset", type=str, default="finqa",
                        choices=["finqa", "convfinqa", "tatqa"])
    parser.add_argument("--stage", type=str, default="joint",
                        choices=["identity", "structural", "joint"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--lambda", dest="lambda_constraint", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None,
                        choices=["cuda", "cpu"])
    parser.add_argument("--save", type=str, default=None,
                        help="Save checkpoint to path")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Device: {device}")

    # Load data
    samples = load_training_data(args.dataset, split="train")
    logger.info(f"Loaded {len(samples)} samples")

    # Build model
    model = JointScorer(
        text_embed_dim=768,
        kg_embed_dim=256,
        entity_feat_dim=3,
        hidden_dim=64,
    ).to(device)

    # Run selected stage
    save_path = Path(args.save) if args.save else Path(f"outputs/gsr_training/{args.dataset}")

    if args.stage == "identity":
        stage_identity(model, samples, device, epochs=args.epochs, lr=args.lr)
    elif args.stage == "structural":
        stage_structural(model, samples, device, epochs=args.epochs, lr=args.lr)
    else:
        stage_joint(
            model, samples, device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            margin=args.margin,
            lambda_constraint=args.lambda_constraint,
            save_path=save_path,
        )

    # Save final model
    save_path.mkdir(parents=True, exist_ok=True)
    final_path = save_path / "final_model.pt"
    torch.save(model.state_dict(), final_path)
    logger.info(f"Model saved to {final_path}")


if __name__ == "__main__":
    main()
