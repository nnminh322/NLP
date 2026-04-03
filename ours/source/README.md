# GSR-CACL: Graph-Structured Retrieval + Constraint-Aware Contrastive Learning

> **Paper:** Structured Knowledge-Enhanced Retrieval for Financial Documents
> **Venue:** EMNLP / SIGIR | **Benchmark:** T²-RAGBench (EACL 2026)

---

## Overview

This is the implementation of **GSR + CACL** — two complementary contributions for
retrieving financial text+table documents.

### Contributions

| Contribution | Description | Addresses |
|---|---|---|
| **C1: GSR** | Graph-Structured Retrieval — constraint KG from IFRS/GAAP templates, GAT encoder, ε-tolerance scoring | Lexical Overlap Illusion, Mathematical Inconsistency |
| **C2: CACL** | Constraint-Aware Contrastive Learning — CHAP negative generation (additive/scale/entity violations) | Hard Negative Validity (H3), Numerical Density Paradox |

---

## Cloud Setup (Kaggle / Google Colab)

### Option A: Kaggle Notebook (T4 GPU, 16GB VRAM — Free)

```python
# Cell 1: Clone & Install
!git clone https://github.com/<YOUR_REPO>/gsr-cacl.git /kaggle/working/gsr-cacl
%cd /kaggle/working/gsr-cacl/ours/source
!pip install -e ".[gpu]" --quiet
!pip install peft accelerate transformers --quiet

# Cell 2: Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# Cell 3: Train (LoRA on T4 — recommended for 16GB VRAM)
!python -m gsr_cacl.train \
    --dataset finqa \
    --stage all \
    --epochs 5 \
    --preset t4 \
    --gradient-checkpointing \
    --save /kaggle/working/outputs

# Cell 4: Benchmark
!python -m gsr_cacl.benchmark_gsr --mode gsr --dataset finqa --top-k 3
```

### Option B: Google Colab (T4 free / A100 Pro)

```python
# Cell 1: Clone & Install
!git clone https://github.com/<YOUR_REPO>/gsr-cacl.git /content/gsr-cacl
%cd /content/gsr-cacl/ours/source
!pip install -e ".[gpu]" --quiet
!pip install peft accelerate transformers --quiet

# Cell 2: Train
# For free T4 (16GB VRAM):
!python -m gsr_cacl.train --dataset finqa --stage all --preset t4 --gradient-checkpointing

# For Colab Pro A100 (40GB VRAM):
!python -m gsr_cacl.train --dataset finqa --stage all --preset a100

# Cell 3: Benchmark
!python -m gsr_cacl.benchmark_gsr --mode gsr --dataset finqa --top-k 3
```

### Hardware Presets

| Preset | Encoder | Fine-tune | Batch | VRAM | Trainable Params |
|--------|---------|-----------|-------|------|-----------------|
| `t4`   | bge-large (335M) | LoRA r=16 | 8 | ~12 GB | ~5.4M |
| `a100` | bge-large (335M) | Full | 16 | ~20 GB | ~336M |
| `v100` | bge-base (110M)  | Full | 16 | ~14 GB | ~111M |
| `cpu`  | bge-base (110M)  | Frozen | 4 | — | ~1.3M |

### VRAM Estimation

| Config | Model | Fine-tune | Batch=8 | Batch=16 |
|--------|-------|-----------|---------|----------|
| bge-base + LoRA | 110M | LoRA r=16 | ~6 GB | ~8 GB |
| bge-large + LoRA | 335M | LoRA r=16 | ~10 GB | ~14 GB |
| bge-large + Full | 335M | Full | ~14 GB | ~20 GB |
| e5-large + LoRA | 560M | LoRA r=16 | ~14 GB | ~20 GB |

> **Tip:** Add `--gradient-checkpointing` to halve activation memory at ~20% speed cost.

---

## Local Installation

```bash
# Using conda
conda activate master

# Install package
cd /project/ours/source
pip install -e ".[dev]"

# For GPU support
pip install -e ".[gpu]"

# For LoRA fine-tuning
pip install peft accelerate transformers

# Verify
python -c "import gsr_cacl; print(gsr_cacl.__version__)"
```

### Dependencies

```
torch>=2.0.0
transformers>=4.36.0, peft>=0.7.0, accelerate>=0.25.0
langchain-core, langchain-huggingface, langchain-community
rank-bm25, faiss-cpu (or faiss-gpu)
datasets, pandas, tqdm, hydra-core, omegaconf
```

---

## Quick Start

### 1. Train GSR + CACL (end-to-end)

```bash
# Recommended: LoRA fine-tuning with bge-large (fits T4 16GB)
python -m gsr_cacl.train \
    --dataset finqa \
    --stage all \
    --encoder BAAI/bge-large-en-v1.5 \
    --finetune lora \
    --epochs 5 \
    --batch-size 8

# Full fine-tuning on A100 (maximum benchmark performance)
python -m gsr_cacl.train \
    --dataset finqa \
    --stage all \
    --encoder BAAI/bge-large-en-v1.5 \
    --finetune full \
    --epochs 5 \
    --batch-size 16

# Stage-by-stage training
python -m gsr_cacl.train --dataset finqa --stage identity --epochs 3
python -m gsr_cacl.train --dataset finqa --stage structural --epochs 3
python -m gsr_cacl.train --dataset finqa --stage joint --epochs 5
```

### 2. Run GSR Retrieval Benchmark

```bash
# On FinQA
python -m gsr_cacl.benchmark_gsr --mode gsr --dataset finqa --top-k 3

# On TAT-DQA (diverse tables)
python -m gsr_cacl.benchmark_gsr --mode gsr --dataset tatqa --top-k 3

# Hybrid GSR + BM25
python -m gsr_cacl.benchmark_gsr --mode hybridgsr --dataset finqa --top-k 3
```

### 3. Compare with Baselines

```bash
python -m gsr_cacl.evaluate_comparison --dataset finqa --methods gsr hybridgsr --save results.csv
```

---

## Architecture

```
Query Q
  ├─► Metadata extraction (company, year, sector)
  ├─► Table KG Construction
  │     markdown table ──parse──► Cell nodes
  │                          │
  │                          └──► Template Matching (15 IFRS/GAAP patterns)
  │                                │
  │                                └──► Accounting edges (ω ∈ {+1,−1,0})
  │                                      │
  │                                      └──► Fallback: positional graph
  │
  └─► Joint Scoring:
        s(Q,D) = α·sim_text(Q,D)
               + β·sim_entity(Q,G_D)
               + γ·ConstraintScore(G_D, Q)
             └─ ε-tolerance: exp(−|ω·v_u − v_v| / max(|v_v|, ε))
```

---

## Key Components

### `gsr_cacl/templates/` — IFRS/GAAP Template Library
15 accounting templates covering ~80-90% of financial tables:
- Income Statement (Revenue → COGS → Gross Profit → Net Income)
- Balance Sheet (Assets = Liabilities + Equity)
- Cash Flow, Revenue by Segment, EPS, EBITDA, Ratios, ...

### `gsr_cacl/kg/` — Knowledge Graph Construction
- `build_constraint_kg()`: Parse markdown → nodes + edges
- `build_kg_from_markdown()`: One-shot KG from table string
- Support for additive (ω=+1), subtractive (ω=−1), positional (ω=0) edges

### `gsr_cacl/encoders/` — GAT Encoder
- Edge-aware message passing with ω-weighted attention
- 2-layer GAT with sinusoidal positional encoding for row/col indices

### `gsr_cacl/scoring/` — Constraint-Aware Scoring
- `compute_constraint_score()`: ε-tolerance differentiable scoring
- `compute_entity_score()`: Company + year + sector matching
- `JointScorer`: Learnable α, β, γ weights

### `gsr_cacl/negative_sampler/` — CHAP Negative Generator
Three types, all satisfying Zero-Sum property:
- **CHAP-A**: Additive violation (change 1 cell → equation broken)
- **CHAP-S**: Scale violation (M → B, ratio broken)
- **CHAP-E**: Entity/year swap (same structure, wrong entity)

### `gsr_cacl/training/` — Joint Training
- `TripletLoss`: Margin-based contrastive loss
- `ConstraintViolationLoss`: Penalises scoring constraint-violating negatives high
- `CACLLoss`: L = L_triplet + λ · L_constraint

### `gsr_cacl/methods/` — RAG Methods
- `GSRRetrieval`: Full GSR pipeline with joint scoring
- `HybridGSR`: GSR + BM25 with RRF fusion

---

## Trainable Parameters Summary

| Component | Params | Frozen mode | LoRA mode | Full mode |
|-----------|--------|-------------|-----------|-----------|
| **TextEncoder** (bge-large) | 335M | Frozen | ~4.2M trainable | 335M trainable |
| **GATEncoder** | 1.22M | Trainable | Trainable | Trainable |
| **JointScorer** | 66.8K | Trainable | Trainable | Trainable |
| **Total trainable** | — | **1.29M** | **~5.5M** | **~336M** |

## Computational Cost

| Component | Complexity | Notes |
|---|---|---|
| KG Construction | O(n_cells) | ~57 cells avg (TAT-DQA), regex-based |
| GAT Encoding | O(V·E) | V≈57, E≈5-10, sparse attention |
| Constraint Scoring | O(E_c) | E_c≈5-10, constant time |
| Text Encoding (forward) | O(seq_len²·d) | Standard transformer, max_len=512 |
| **Total inference** | **~1.2–1.4× baseline** | Pre-indexed KG construction |
| CHAP Generation | O(n_cells) | Offline pre-generation, zero runtime cost |

---

## Expected Results

Based on T²-RAGBench reported baselines:

| Method | MRR@3 | Recall@3 |
|---|---|---|
| BM25 | 0.280 | 0.36 |
| Hybrid BM25 | 0.352 | 0.45 |
| ColBERTv2 | 0.310 | 0.40 |
| **GSR (ours)** | **≥ 0.40** | **≥ 0.50** |
| **HybridGSR (ours)** | **≥ 0.42** | **≥ 0.52** |

GSR targets: outperform HybridBM25 by capturing accounting identities that
lexical/dense methods miss.

---

## File Structure

```
ours/source/
├── pyproject.toml
├── README.md
├── Makefile
├── conf/
│   └── dataset/
│       ├── gsr_finqa.yaml
│       └── gsr_tatqa.yaml
└── src/
    └── gsr_cacl/
        ├── __init__.py
        ├── benchmark_gsr.py          ← Main benchmark entry point
        ├── evaluate.py               ← CLI entry
        ├── evaluate_comparison.py     ← Baseline comparison
        ├── train.py                   ← Joint training script
        ├── template_coverage_analysis.py
        │
        ├── core/                      ← Core data structures
        │   └── __init__.py            Document, RetrievalResult, DatasetSplit
        │
        ├── templates/                 ← IFRS/GAAP template library
        │   ├── __init__.py            (re-exports)
        │   ├── data_structures.py     AccountingConstraint, AccountingTemplate
        │   ├── library.py             15 templates + TEMPLATE_REGISTRY
        │   └── matching.py            match_template(), normalize_header()
        │
        ├── kg/                        ← Constraint KG construction
        │   ├── __init__.py            (re-exports)
        │   ├── data_structures.py     KGNode, KGEdge, ConstraintKG
        │   ├── parser.py              parse_markdown_rows(), parse_number()
        │   └── builder.py             build_constraint_kg(), build_kg_from_markdown()
        │
        ├── encoders/                  ← GAT graph encoder
        │   ├── __init__.py            (re-exports)
        │   ├── positional.py          SinusoidalPositionalEncoding
        │   ├── gat_layer.py           GATLayer (edge-aware attention)
        │   └── gat_encoder.py         GATEncoder (2-layer, 4-head)
        │
        ├── scoring/                   ← Constraint-aware scoring
        │   ├── __init__.py            (re-exports)
        │   ├── constraint_score.py    compute_constraint_score(), compute_entity_score()
        │   └── joint_scorer.py        JointScorer (learnable α, β, γ)
        │
        ├── negative_sampler/          ← CHAP negative generation
        │   ├── __init__.py            (re-exports)
        │   └── chap.py               CHAPNegativeSampler, apply_chap_{a,s,e}
        │
        ├── training/                  ← CACL training loop
        │   ├── __init__.py            (re-exports)
        │   ├── losses.py              TripletLoss, ConstraintViolationLoss, CACLLoss
        │   ├── data.py                RetrievalSample, RetrievalDataset
        │   └── trainer.py             train_gsr_cacl()
        │
        ├── methods/                   ← GSR RAG methods
        │   ├── __init__.py            (re-exports)
        │   └── gsr_retrieval.py       GSRRetrieval, HybridGSR
        │
        └── datasets/                  ← Dataset loading (HuggingFace)
            ├── __init__.py            (re-exports)
            ├── gsr_document.py        GSRDocument, extract_table()
            └── wrappers.py            load_t2ragbench_split(), build_gsr_corpus()
```

---

## Baselines for Comparison

| Method | Source | Priority |
|---|---|---|
| HELIOS (ACL 2025) | Multi-granular table-text retrieval | 🔴 Critical |
| THYME (EMNLP 2025) | Field-aware hybrid matching | 🔴 Critical |
| THoRR (2024) | Two-stage table retrieval | 🔴 Critical |
| HybridBM25 | T²-RAGBench best reported | ✅ Baseline |
| ColBERT-v2 | Late interaction baseline | ✅ Baseline |

---

## Extending to New Domains

The template library (`gsr_cacl/templates/`) is domain-specific.
To extend to new domains:

1. Add domain-specific accounting identities to `TEMPLATES`
2. Update `_HEADER_SYNONYMS` with domain vocabulary
3. Adjust `epsilon` tolerance for domain numerical precision

---

## References

- Strich et al. (EACL 2026): T²-RAGBench
- Halliday (1994): An Introduction to Functional Grammar
- Shannon (1948): A Mathematical Theory of Communication
- Singh & Gupta (NAACL 2023): Constraint-guided accounting KG
- ConFIT (Agents4Science 2025): Semantic-Preserving Perturbation (cf. CHAP differentiation)
