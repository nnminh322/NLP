#!/usr/bin/env python3
"""Hybrid evaluation tool:

1) Run GSR retrieval (baseline), save per-query candidates/scores and baseline metrics.
2) Run BM25, fuse with GSR via RRF, evaluate fused metrics and save outputs.

Run as module from `source/`:
    python -m gsr_cacl.tools.hybrid_eval --dataset tatqa --sample 200
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from gsr_cacl.datasets.wrappers import load_t2ragbench_split
from gsr_cacl.methods.gsr_retrieval import GSRRetrieval
from gsr_cacl.support.bm25 import BM25Index

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def compute_cosine_scores(q_emb: np.ndarray, docs_emb: np.ndarray) -> np.ndarray:
    q = np.asarray(q_emb, dtype=float)
    docs = np.asarray(docs_emb, dtype=float)
    q_norm = np.linalg.norm(q) + 1e-12
    docs_norm = np.linalg.norm(docs, axis=1) + 1e-12
    return (docs @ q) / (docs_norm * q_norm)


def recall_at_k_from_lists(ranked_ids_list: List[List[str]], gt_ids: List[str], k: int) -> float:
    hits = 0
    n = len(gt_ids)
    for ranked, gt in zip(ranked_ids_list, gt_ids):
        if str(gt) in [str(x) for x in ranked[:k]]:
            hits += 1
    return hits / n if n > 0 else 0.0


def mrr_at_k_from_lists(ranked_ids_list: List[List[str]], gt_ids: List[str], k: int) -> float:
    total = 0.0
    n = len(gt_ids)
    for ranked, gt in zip(ranked_ids_list, gt_ids):
        for rank, doc in enumerate(ranked[:k], start=1):
            if str(doc) == str(gt):
                total += 1.0 / rank
                break
    return total / n if n > 0 else 0.0


def run(args: argparse.Namespace) -> None:
    t0 = datetime.now().strftime("%y%m%d_%H%M%S")
    outdir = Path(args.output_dir or f"outputs/hybrid_eval/{args.dataset}/{t0}")
    outdir.mkdir(parents=True, exist_ok=True)

    # Dataset
    cfg_map = {
        "finqa": "FinQA",
        "convfinqa": "ConvFinQA",
        "tatqa": "TAT-DQA",
    }
    config_name = cfg_map.get(args.dataset.lower())
    if config_name is None:
        raise ValueError("Unknown dataset; choose finqa|convfinqa|tatqa")

    ds = load_t2ragbench_split(config_name=config_name, split="test", sample_size=args.sample)
    logger.info(f"Loaded {len(ds.queries)} queries, {len(ds.corpus)} corpus docs")

    # Embeddings (same as benchmark)
    from langchain_huggingface import HuggingFaceEmbeddings

    emb_device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = HuggingFaceEmbeddings(model_name=args.embedding, model_kwargs={"device": emb_device})

    # Instantiate GSR (this builds FAISS, KGs and pre-encodes KG embeddings)
    logger.info("Building GSRRetrieval (this may take a while)...")
    gsr = GSRRetrieval(
        corpus=ds.corpus,
        embedding_function=embeddings,
        top_k=args.top_k,
        device=emb_device,
        checkpoint_path=args.checkpoint,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        gat_hidden_dim=args.gat_hidden_dim,
        gat_num_heads=args.gat_num_heads,
        gat_num_layers=args.gat_num_layers,
        contr1=args.contr1,
        contr_version=args.contr2,
        rel_tol=args.rel_tol,
    )

    # Phase 1: run baseline GSR joint scoring and save per-query candidates
    logger.info("Phase 1 — running GSR joint retrieval and saving candidates...")
    candidate_n = args.candidate_n

    gsr_all_ranked_ids = []
    gsr_results_j = []
    vec_all_ranked_ids = []

    # Precompute doc embeddings (numpy) for vector similarity
    doc_text_embs = np.array(gsr.doc_text_embeds)

    for qi, (query, qmeta, gt) in enumerate(tqdm(zip(ds.queries, ds.meta_data, ds.ground_truth_ids), total=len(ds.queries), desc="GSR eval")):
        # Get FAISS candidates
        candidates = gsr.vector_store.similarity_search(query, k=candidate_n)

        # Query embedding (vector-only)
        q_vec = np.array(embeddings.embed_query(query), dtype=float)
        cosine_scores = compute_cosine_scores(q_vec, doc_text_embs)

        # Collect GSR joint scores for FAISS candidates
        cand_list = []
        for cand in candidates:
            cand_id = cand.metadata.get("id", "")
            idx = gsr._id_to_idx.get(cand_id)
            if idx is None:
                continue

            doc_tensor = torch.tensor(gsr.doc_text_embeds[idx], dtype=torch.float32, device=gsr.device)
            kg_embed = gsr.kg_embeds[idx]
            cs_result = gsr_constr = None
            from gsr_cacl.scoring.constraint_score import compute_constraint_score

            cs_result = compute_constraint_score(
                gsr.doc_kgs[idx], epsilon=gsr.epsilon, version=gsr.contr_version, relative_tolerance=gsr.rel_tol
            )

            # entity embedding path if available
            q_entity_emb = gsr._encode_query_entity(qmeta)
            if q_entity_emb is not None and getattr(gsr, "doc_entity_embeds", None) is not None:
                doc_entity_emb = gsr.doc_entity_embeds[idx].to(gsr.device)
                with torch.no_grad():
                    final_score = float(
                        gsr.scorer.score_single_learned(
                            query_text_embed=torch.tensor(q_vec, dtype=torch.float32, device=gsr.device),
                            doc_text_embed=doc_tensor,
                            kg_embed=kg_embed,
                            query_entity_embed=q_entity_emb,
                            doc_entity_embed=doc_entity_emb,
                            constraint_result=cs_result,
                        )
                    )
            else:
                entity_score = gsr._compute_entity_score(qmeta, idx)
                with torch.no_grad():
                    final_score = float(
                        gsr.scorer.score_single(
                            query_text_embed=torch.tensor(q_vec, dtype=torch.float32, device=gsr.device),
                            doc_text_embed=doc_tensor,
                            kg_embed=kg_embed,
                            entity_score=entity_score,
                            constraint_result=cs_result,
                        )
                    )

            cand_list.append({"doc_id": str(gsr.corpus[idx].id), "idx": idx, "joint_score": final_score, "vec_score": float(cosine_scores[idx])})

        # Rank by joint_score
        cand_list_sorted = sorted(cand_list, key=lambda x: x["joint_score"], reverse=True)
        gsr_ranked_ids = [c["doc_id"] for c in cand_list_sorted[: args.top_k]]
        gsr_all_ranked_ids.append(gsr_ranked_ids)

        # Save vector-only ranking top candidate_n
        vec_idx_sorted = np.argsort(cosine_scores)[::-1][: candidate_n]
        vec_ranked_ids = [str(gsr.corpus[int(i)].id) for i in vec_idx_sorted]
        vec_all_ranked_ids.append(vec_ranked_ids[: args.top_k])

        # Save per-query JSON
        gsr_results_j.append({
            "query_id": qi,
            "query": query,
            "ground_truth": str(gt),
            "gsr_candidates": cand_list_sorted,
            "vec_top_ids": vec_ranked_ids[: candidate_n],
        })

    # Baseline metrics (GSR joint)
    baseline_recall3 = recall_at_k_from_lists(gsr_all_ranked_ids, ds.ground_truth_ids, k=args.top_k)
    baseline_mrr3 = mrr_at_k_from_lists(gsr_all_ranked_ids, ds.ground_truth_ids, k=args.top_k)
    logger.info(f"Baseline GSR — Recall@{args.top_k}: {baseline_recall3:.4f}, MRR@{args.top_k}: {baseline_mrr3:.4f}")

    # Save baseline outputs
    with open(outdir / "results_gsr.jsonl", "w") as f:
        for r in gsr_results_j:
            f.write(json.dumps(r) + "\n")

    with open(outdir / "metrics_gsr.json", "w") as f:
        json.dump({"recall@k": baseline_recall3, "mrr@k": baseline_mrr3}, f, indent=2)

    # Phase 2: BM25 + RRF fusion
    logger.info("Phase 2 — building BM25 index and fusing via RRF...")
    bm25 = BM25Index([doc.page_content for doc in gsr.corpus])

    bm25_per_query = []
    rrf_k = args.rrf_k
    fused_ranked_ids = []

    for qi, (query, gt) in enumerate(tqdm(zip(ds.queries, ds.ground_truth_ids), total=len(ds.queries), desc="BM25+RRF")):
        # BM25 top candidates
        bm25_top = bm25.top_n(query, args.candidate_n)
        bm25_top_ids = [str(gsr.corpus[idx].id) for idx, _ in bm25_top]
        bm25_per_query.append({"query_id": qi, "bm25_top": [{"doc_id": str(gsr.corpus[idx].id), "score": float(score)} for idx, score in bm25_top]})

        # Build union of candidate ids from GSR joint list and BM25
        gsr_ids_all = [c["doc_id"] for c in gsr_results_j[qi]["gsr_candidates"]]
        union_ids = list(dict.fromkeys(gsr_ids_all + bm25_top_ids))

        rrf_scores = {}
        for did in union_ids:
            # ranks (0-based)
            try:
                rg = gsr_ids_all.index(did)
            except ValueError:
                rg = args.candidate_n + 1000
            try:
                rb = bm25_top_ids.index(did)
            except ValueError:
                rb = args.candidate_n + 1000
            rrf_scores[did] = 1.0 / (rrf_k + rg + 1) + 1.0 / (rrf_k + rb + 1)

        fused_sorted = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        fused_ranked_ids.append([did for did, _ in fused_sorted[: args.top_k]])

    # Evaluate fused
    fused_recall3 = recall_at_k_from_lists(fused_ranked_ids, ds.ground_truth_ids, k=args.top_k)
    fused_mrr3 = mrr_at_k_from_lists(fused_ranked_ids, ds.ground_truth_ids, k=args.top_k)
    logger.info(f"Fused RRF — Recall@{args.top_k}: {fused_recall3:.4f}, MRR@{args.top_k}: {fused_mrr3:.4f}")

    with open(outdir / "results_bm25.jsonl", "w") as f:
        for r in bm25_per_query:
            f.write(json.dumps(r) + "\n")

    with open(outdir / "metrics_fused.json", "w") as f:
        json.dump({"recall@k": fused_recall3, "mrr@k": fused_mrr3}, f, indent=2)

    print("\n--- Summary ---")
    print(f"Baseline GSR Recall@{args.top_k}: {baseline_recall3:.4f}, MRR@{args.top_k}: {baseline_mrr3:.4f}")
    print(f"Fused (GSR+BM25 RRF) Recall@{args.top_k}: {fused_recall3:.4f}, MRR@{args.top_k}: {fused_mrr3:.4f}")
    print(f"Outputs saved to: {outdir}")


def cli():
    p = argparse.ArgumentParser(description="GSR + BM25 RRF evaluation script")
    p.add_argument("--dataset", type=str, default="tatqa", help="finqa|convfinqa|tatqa")
    p.add_argument("--sample", type=int, default=200, help="number of queries to sample (debug)")
    p.add_argument("--embedding", type=str, default="intfloat/multilingual-e5-large-instruct")
    p.add_argument("--candidate-n", type=int, default=50, dest="candidate_n")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--rrf-k", type=int, default=60, dest="rrf_k")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.3)
    p.add_argument("--gamma", type=float, default=0.2)
    p.add_argument("--gat-hidden-dim", type=int, default=256)
    p.add_argument("--gat-num-heads", type=int, default=4)
    p.add_argument("--gat-num-layers", type=int, default=2)
    p.add_argument("--contr1", type=str, default="v1")
    p.add_argument("--contr2", type=str, default="v1")
    p.add_argument("--rel-tol", type=float, default=1e-3)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    cli()
