#!/usr/bin/env python3
"""
GSR + CACL Training Script (END-TO-END WITH ENTITY SUPERVISED CONTRASTIVE LEARNING).

Three-stage curriculum training (§4.7 of proposal):
    Stage 1 — Identity:     learn entity discrimination via SupCon + TripletLoss
    Stage 2 — Structural:   learn KG encoding + constraint score calibration
    Stage 3 — Joint CACL:  full CACL objective with CHAP hard negatives

Key improvements over legacy train.py:
    - Uses SharedEncoder (shared BGE backbone) for both text AND entity encoding
    - EntitySupConLoss from §4.4 of proposal
    - EntityEncoder produces learned s_entity = cos(e_Q, e_D)
    - JointScorer updated with entity_embed_dim
    - GAT encoder receives entity embeddings for EntitySim term

Usage:
    python -m gsr_cacl.train --dataset finqa --stage all --preset t4 --gradient-checkpointing
    python -m gsr_cacl.train --dataset finqa --stage identity --epochs 3
    python -m gsr_cacl.train --dataset finqa --stage all --add-entity-sim
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from gsr_cacl.training import (
    RetrievalDataset,
    CACLLoss,
    TripletLoss,
    EntitySupConLoss,
    EntityRegistry,
)
from gsr_cacl.negative_sampler import CHAPNegativeSampler
from gsr_cacl.kg import build_constraint_kg
from gsr_cacl.scoring import JointScorer, compute_constraint_score
from gsr_cacl.scoring.constraint_score import ConstraintScoringVersion
from gsr_cacl.encoders import GATEncoder, build_numeric_encoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET_CONFIGS = {
    "finqa": {"config_name": "FinQA", "train_split": "train", "eval_split": "test"},
    "convfinqa": {"config_name": "ConvFinQA", "train_split": "turn_0", "eval_split": "turn_0"},
    "tatqa": {"config_name": "TAT-DQA", "train_split": "train", "eval_split": "test"},
}


def _collect_trainable_params(*modules):
    params = []
    seen = set()
    for m in modules:
        for p in m.parameters():
            if p.requires_grad and id(p) not in seen:
                params.append(p)
                seen.add(id(p))
    return params


def _batch_meta_to_tensors(samples, device):
    companies = [s.company_name for s in samples]
    years = [s.report_year for s in samples]
    sectors = [s.company_sector for s in samples]
    return companies, years, sectors


def build_encoder(
    model_name, finetune, lora_r=16, lora_alpha=32, lora_dropout=0.05,
    entity_dim=256, max_length=512,
):
    from gsr_cacl.encoders.entity_encoder import SharedEncoder
    return SharedEncoder(
        model_name=model_name, finetune=finetune,
        lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        entity_dim=entity_dim, max_length=max_length,
    )


def load_training_data(dataset, split="train"):
    from gsr_cacl.datasets.wrappers import load_t2ragbench_train_samples

    ds_cfg = DATASET_CONFIGS[dataset]
    hf_split = ds_cfg["train_split"] if split == "train" else ds_cfg["eval_split"]
    logger.info(f"Loading local {hf_split} split for {dataset}...")
    return load_t2ragbench_train_samples(ds_cfg["config_name"], split=hf_split)


# ─── Stage 1 ───────────────────────────────────────────────────────────────────

def stage_identity(encoder, scorer, gat_encoder, dataset, device, entity_registry,
                   epochs=3, lr=1e-4, batch_size=16, entity_dim=256,
                   use_entity_signal: bool = True):
    logger.info("=== Stage 1: Identity Pretraining ===")
    params = _collect_trainable_params(encoder, scorer)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    triplet_loss = TripletLoss(margin=0.2)
    supcon_loss = EntitySupConLoss(temperature=0.07)

    dataset_obj = RetrievalDataset(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset_obj, batch_size=batch_size, shuffle=True, collate_fn=lambda b: b,
    )
    encoder.train()
    scorer.train()

    supcon_loss = EntitySupConLoss(temperature=0.07) if use_entity_signal else None
    if not use_entity_signal:
        logger.info("Entity signal disabled: using in-batch text negatives for triplet loss")

    for epoch in range(epochs):
        total_loss = trip_loss = supcon_loss_total = 0.0
        n_batches = 0

        for batch in tqdm(dataloader, desc=f"Identity Epoch {epoch+1}"):
            B = len(batch)
            queries = [s.query for s in batch]
            docs = [s.positive_context[:512] for s in batch]

            # Text encoding
            q_text_emb = encoder.text_encode(queries)
            d_text_emb = encoder.text_encode(docs)

            if use_entity_signal:
                # FIX ISSUE 5: Different entity representations for Q vs D
                # Query → original name, Doc → canonical name from registry
                q_companies = [s.company_name for s in batch]
                q_years = [s.report_year for s in batch]
                q_sectors = [s.company_sector for s in batch]
                d_companies = [
                    entity_registry.get_canonical_name(s.company_name) or s.company_name
                    for s in batch
                ]
                d_years = [s.report_year for s in batch]
                d_sectors = [s.company_sector for s in batch]

                q_entity_emb = encoder.entity_encode(q_companies, q_years, q_sectors)
                d_entity_emb = encoder.entity_encode(d_companies, d_years, d_sectors)

                # SupCon loss: learn that "Apple" and "Apple Inc." are the same entity
                entity_labels = entity_registry.build_entity_labels(q_companies)
                loss_supcon = supcon_loss(q_entity_emb, entity_labels)

                # Triplet loss: negatives swap entity within batch
                neg_companies = [batch[(i + 1) % B].company_name for i in range(B)]
                neg_years = [batch[(i + 1) % B].report_year for i in range(B)]
                neg_sectors = [batch[(i + 1) % B].company_sector for i in range(B)]
                neg_entity_emb = encoder.entity_encode(neg_companies, neg_years, neg_sectors)
            else:
                loss_supcon = torch.tensor(0.0, device=device)
                neg_docs = [batch[(i + 1) % B].positive_context[:512] for i in range(B)]
                neg_text_emb = encoder.text_encode(neg_docs)

            kg_dummy = torch.zeros(B, gat_encoder.hidden_dim, device=device)
            cs_feats = torch.tensor([[1.0, 0.0, 0.0]] * B, device=device)

            if use_entity_signal:
                pos_scores = scorer(q_text_emb, d_text_emb, kg_dummy, q_entity_emb, d_entity_emb, cs_feats)
                neg_scores = scorer(q_text_emb, d_text_emb, kg_dummy, q_entity_emb, neg_entity_emb, cs_feats)
            else:
                pos_scores = scorer(q_text_emb, d_text_emb, kg_dummy, None, None, cs_feats)
                neg_scores = scorer(q_text_emb, neg_text_emb, kg_dummy, None, None, cs_feats)
            loss_triplet = triplet_loss(pos_scores, neg_scores)
            loss_total = loss_triplet + loss_supcon

            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            total_loss += loss_total.item()
            trip_loss += loss_triplet.item()
            supcon_loss_total += loss_supcon.item()
            n_batches += 1

        logger.info(
            f"Epoch {epoch+1} — Loss: {total_loss/max(n_batches,1):.4f} "
            f"(triplet={trip_loss/max(n_batches,1):.4f}, supcon={supcon_loss_total/max(n_batches,1):.4f})"
        )


# ─── Stage 2 ───────────────────────────────────────────────────────────────────

def stage_structural(encoder, scorer, gat_encoder, dataset, device,
                    epochs=3, lr=1e-4, batch_size=8,
                    contr_version="v1", entity_dim=256):
    logger.info("=== Stage 2: Structural Pretraining ===")
    params = _collect_trainable_params(encoder, scorer, gat_encoder)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    dataset_obj = RetrievalDataset(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset_obj, batch_size=batch_size, shuffle=True, collate_fn=lambda b: b,
    )
    encoder.train()
    scorer.train()
    gat_encoder.train()

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(dataloader, desc=f"Structural Epoch {epoch+1}"):
            B = len(batch)
            companies, years, sectors = _batch_meta_to_tensors(batch, device)

            kg_embeds = []
            cs_results = []
            for sample in batch:
                kg = build_constraint_kg(sample.positive_context)
                kg_embed = gat_encoder.encode_graph(kg, entity_embeddings=None)
                kg_embeds.append(kg_embed)
                cs_results.append(compute_constraint_score(kg, version=contr_version))

            kg_embed_t = torch.stack(kg_embeds)
            cs_feats = scorer.build_constraint_features(cs_results, device)
            predicted_cs = scorer.forward_constraint(cs_feats)
            target_cs = torch.ones(B, device=device)

            loss = F.mse_loss(predicted_cs, target_cs)
            loss += -torch.clamp(kg_embed_t.var(dim=0).mean(), max=1.0) * 0.1

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        logger.info(f"Epoch {epoch+1} — Loss: {total_loss/max(n_batches,1):.4f}")


# ─── Stage 3 ───────────────────────────────────────────────────────────────────

def stage_joint(encoder, scorer, gat_encoder, dataset, device, entity_registry,
               epochs=5, batch_size=16, lr=5e-5, margin=0.2,
               lambda_constraint=0.5, lambda_entity=0.5, save_path=None,
               contr_version="v1", entity_dim=256, add_entity_sim=False,
               use_entity_signal: bool = True):
    logger.info("=== Stage 3: Joint Finetuning (CACL) ===")
    params = _collect_trainable_params(encoder, scorer, gat_encoder)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    triplet_loss = TripletLoss(margin=margin)
    supcon_loss = EntitySupConLoss(temperature=0.07)
    caclloss = CACLLoss(margin=margin, lambda_constraint=lambda_constraint)
    chap_sampler = CHAPNegativeSampler(chap_a_prob=0.5, chap_s_prob=0.3, chap_e_prob=0.2)

    dataset_obj = RetrievalDataset(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset_obj, batch_size=batch_size, shuffle=True, collate_fn=lambda b: b,
    )
    encoder.train()
    scorer.train()
    gat_encoder.train()

    supcon_loss = EntitySupConLoss(temperature=0.07) if use_entity_signal else None

    for epoch in range(epochs):
        total = trip_total = const_total = supcon_total = 0.0
        n_batches = 0

        for batch in tqdm(dataloader, desc=f"Joint Epoch {epoch+1}/{epochs}"):
            B = len(batch)
            companies, years, sectors = _batch_meta_to_tensors(batch, device)

            if use_entity_signal:
                # FIX ISSUE 5: Q uses original names, D uses canonical names
                d_companies = [
                    entity_registry.get_canonical_name(c) or c for c in companies
                ]
                q_entity_emb = encoder.entity_encode(companies, years, sectors)
                d_entity_emb = encoder.entity_encode(d_companies, years, sectors)

            pos_scores_list = []
            neg_scores_list = []
            violates_list = []

            for idx, sample in enumerate(batch):
                q_text_emb = encoder.text_encode([sample.query])
                d_text_emb = encoder.text_encode([sample.positive_context[:512]])

                if use_entity_signal:
                    # FIX ISSUE 6: Index into correct entity embeddings for this sample
                    q_e = q_entity_emb[idx:idx+1]   # [1, entity_dim]
                    d_e = d_entity_emb[idx:idx+1]   # [1, entity_dim]

                # KG + constraint for positive
                pos_kg = build_constraint_kg(sample.positive_context)
                # FIX ISSUE 9: Pass entity embeddings to GAT if EntitySim is enabled
                if use_entity_signal and add_entity_sim and gat_encoder.add_entity_sim:
                    ent_emb_for_kg = q_entity_emb[idx:idx+1].squeeze(0)      # [entity_dim]
                    pos_kg_emb = gat_encoder.encode_graph(
                        pos_kg, entity_embeddings=ent_emb_for_kg.unsqueeze(0)
                    ).unsqueeze(0)
                else:
                    pos_kg_emb = gat_encoder.encode_graph(pos_kg, entity_embeddings=None).unsqueeze(0)

                pos_cs = compute_constraint_score(pos_kg, version=contr_version)
                pos_cs_feats = scorer.build_constraint_features([pos_cs], device)
                if use_entity_signal:
                    pos_score = scorer(q_text_emb, d_text_emb, pos_kg_emb, q_e, d_e, pos_cs_feats)
                else:
                    pos_score = scorer(q_text_emb, d_text_emb, pos_kg_emb, None, None, pos_cs_feats)
                pos_scores_list.append(pos_score.squeeze(0))

                # CHAP negatives
                negs = chap_sampler.sample(pos_kg, n_negatives=1)
                if not negs:
                    from gsr_cacl.negative_sampler.chap import apply_chap_e
                    negs = [apply_chap_e(pos_kg)]

                for neg in negs:
                    neg_kg = build_constraint_kg(neg.table_md)
                    if use_entity_signal and add_entity_sim and gat_encoder.add_entity_sim:
                        neg_kg_emb = gat_encoder.encode_graph(
                            neg_kg, entity_embeddings=ent_emb_for_kg.unsqueeze(0)
                        ).unsqueeze(0)
                    else:
                        neg_kg_emb = gat_encoder.encode_graph(neg_kg, entity_embeddings=None).unsqueeze(0)

                    neg_text_emb = encoder.text_encode([neg.table_md[:512]])
                    neg_cs = compute_constraint_score(neg_kg, version=contr_version)
                    neg_cs_feats = scorer.build_constraint_features([neg_cs], device)
                    if use_entity_signal:
                        neg_score = scorer(q_text_emb, neg_text_emb, neg_kg_emb, q_e, d_e, neg_cs_feats)
                    else:
                        neg_score = scorer(q_text_emb, neg_text_emb, neg_kg_emb, None, None, neg_cs_feats)
                    neg_scores_list.append(neg_score.squeeze(0))
                    violates_list.append(1.0 if neg.is_violated else 0.0)

            # Stack scores
            pos_scores_t = torch.stack(pos_scores_list)
            neg_scores_t = torch.stack(neg_scores_list)
            violates_t = torch.tensor(violates_list, dtype=torch.float32, device=device)

            # Align dimensions (ensure the negative scores and violation mask match)
            neg_for_loss = neg_scores_t
            violates_for_loss = violates_t
            if len(neg_for_loss) >= B:
                neg_for_loss = neg_for_loss[:B]
                violates_for_loss = violates_for_loss[:B]
            else:
                pad_len = B - len(neg_for_loss)
                neg_pad = torch.zeros(pad_len, device=device)
                mask_pad = torch.zeros(pad_len, device=device)
                neg_for_loss = torch.cat([neg_for_loss, neg_pad])
                violates_for_loss = torch.cat([violates_for_loss, mask_pad])

            # CACL loss: triplet + constraint violation
            losses = caclloss(pos_scores_t, neg_for_loss, violates_for_loss)

            # EntitySupConLoss: entity clustering
            if use_entity_signal:
                entity_labels = entity_registry.build_entity_labels(companies)
                loss_supcon = supcon_loss(q_entity_emb, entity_labels)
            else:
                loss_supcon = torch.tensor(0.0, device=device)

            loss_total = losses["total"] + lambda_entity * loss_supcon

            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            total += losses["total"].item()
            trip_total += losses["triplet"].item()
            const_total += losses["constraint"].item()
            supcon_total += loss_supcon.item()
            n_batches += 1

        scheduler.step()
        logger.info(
            f"Epoch {epoch+1} — Loss: {total/max(n_batches,1):.4f} "
            f"(triplet={trip_total/max(n_batches,1):.4f}, "
            f"constraint={const_total/max(n_batches,1):.4f}, "
            f"supcon={supcon_total/max(n_batches,1):.4f})"
        )

        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)
            ckpt = save_path / f"ckpt_epoch_{epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1, "encoder_state": encoder.state_dict(),
                "scorer_state": scorer.state_dict(), "gat_encoder_state": gat_encoder.state_dict(),
                "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                "loss": total / max(n_batches, 1),
            }, ckpt)
            logger.info(f"Checkpoint saved: {ckpt}")


# ─── CLI ───────────────────────────────────────────────────────────────────────

ENCODER_PRESETS = {
    "t4":   {"encoder": "BAAI/bge-large-en-v1.5",  "finetune": "lora",   "batch_size": 8},
    "a100": {"encoder": "BAAI/bge-large-en-v1.5",  "finetune": "full",   "batch_size": 16},
    "v100": {"encoder": "BAAI/bge-base-en-v1.5",    "finetune": "full",   "batch_size": 16},
    "cpu":  {"encoder": "BAAI/bge-base-en-v1.5",    "finetune": "frozen", "batch_size": 4},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="finqa", choices=["finqa", "convfinqa", "tatqa"])
    parser.add_argument("--stage", default="joint", choices=["identity", "structural", "joint", "all"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--lambda-constraint", type=float, default=0.5)
    parser.add_argument("--lambda-entity", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save", default=None)
    parser.add_argument("--encoder", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--finetune", default="lora", choices=["full", "lora", "frozen"])
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--entity-dim", type=int, default=256)
    parser.add_argument("--preset", default=None, choices=list(ENCODER_PRESETS.keys()))
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--contr1", default="v1", choices=["v1", "v2"])
    parser.add_argument("--contr2", default="v1", choices=["v1", "v2"])
    parser.add_argument("--add-entity-sim", action="store_true",
                        help="Enable EntitySim in GAT attention")
    parser.add_argument("--disable-entity-signal", action="store_true",
                        help="Disable entity embeddings and EntitySupConLoss; keep text + KG only")
    args = parser.parse_args()

    use_entity_signal = not args.disable_entity_signal

    if args.preset:
        p = ENCODER_PRESETS[args.preset]
        args.encoder, args.finetune, args.batch_size = p["encoder"], p["finetune"], p["batch_size"]
        logger.info(f"Preset '{args.preset}': {p}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Device: {device}")

    samples = load_training_data(args.dataset, split="train")

    encoder = build_encoder(
        model_name=args.encoder, finetune=args.finetune,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, entity_dim=args.entity_dim,
    ).to(device)

    if args.gradient_checkpointing and hasattr(encoder.backbone, "gradient_checkpointing_enable"):
        encoder.backbone.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    embed_dim = encoder.embed_dim
    numeric_mod, _ = build_numeric_encoder(args.contr1, embed_dim)
    gat_encoder = GATEncoder(
        embed_dim=embed_dim, hidden_dim=256, num_heads=4, num_layers=2,
        numeric_encoder=numeric_mod, numeric_version=args.contr1,
        entity_embed_dim=args.entity_dim, add_entity_sim=(args.add_entity_sim and use_entity_signal),
    ).to(device)

    scorer = JointScorer(
        text_embed_dim=embed_dim, kg_embed_dim=gat_encoder.output_dim,
        entity_embed_dim=args.entity_dim, hidden_dim=64,
        use_entity_signal=use_entity_signal,
    ).to(device)

    entity_registry = EntityRegistry()
    save_path = Path(args.save) if args.save else Path(f"outputs/gsr_training/{args.dataset}")

    enc_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    gat_params = sum(p.numel() for p in gat_encoder.parameters() if p.requires_grad)
    sc_params = sum(p.numel() for p in scorer.parameters() if p.requires_grad)
    logger.info(
        f"Trainable params: SharedEncoder={enc_params:,} + GAT={gat_params:,} + "
        f"Scorer={sc_params:,} = {enc_params+gat_params+sc_params:,}"
    )

    if not use_entity_signal:
        logger.info("Entity signal disabled: training with text + KG only")

    if args.stage in ("identity", "all"):
        stage_identity(encoder, scorer, gat_encoder, samples, device, entity_registry,
                       epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                       entity_dim=args.entity_dim, use_entity_signal=use_entity_signal)

    if args.stage in ("structural", "all"):
        stage_structural(encoder, scorer, gat_encoder, samples, device,
                        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                        contr_version=args.contr2, entity_dim=args.entity_dim)

    if args.stage in ("joint", "all"):
        stage_joint(encoder, scorer, gat_encoder, samples, device, entity_registry,
                    epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, margin=args.margin,
                    lambda_constraint=args.lambda_constraint, lambda_entity=args.lambda_entity,
                    save_path=save_path, contr_version=args.contr2, entity_dim=args.entity_dim,
                    add_entity_sim=args.add_entity_sim, use_entity_signal=use_entity_signal)

    save_path.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder_state": encoder.state_dict(), "scorer_state": scorer.state_dict(),
        "gat_encoder_state": gat_encoder.state_dict(),
        "encoder_model_name": args.encoder, "finetune_strategy": args.finetune,
        "embed_dim": embed_dim, "entity_dim": args.entity_dim,
    }, save_path / "final_model.pt")
    logger.info(f"Model saved to {save_path / 'final_model.pt'}")


if __name__ == "__main__":
    main()
