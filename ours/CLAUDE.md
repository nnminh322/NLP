# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GSR-CACL (Graph-Structured Retrieval with Constraint-Aware Contrastive Learning) — a financial document retrieval system for the T2-RAGBench benchmark (EACL 2026). Combines text encoding, knowledge graph construction from financial tables, and contrastive learning to retrieve relevant documents for financial QA.

## Development Setup

All development commands run from `source/` (not the repo root).

```bash
cd source
pip install -e ".[dev]"
pip install peft accelerate transformers faiss-cpu
# GPU variant: pip install -e ".[dev,gpu]"
```

Verify: `python -c "from gsr_cacl import GATEncoder, JointScorer; print('OK')"`

## Common Commands

All via Makefile in `source/`:

```bash
make lint              # ruff check + format
make test              # pytest src/gsr_cacl/tests/
make test-kg           # KG builder smoke test
make test-chap         # CHAP sampler smoke test
make test-templates    # Template library smoke test
make clean             # Remove outputs/, .egg-info, pycache
```

Training (3-stage curriculum):
```bash
make train-all-stages                          # All stages sequentially
make train-identity                            # Stage 1 only
make train-structural                          # Stage 2 only
make train-joint                               # Stage 3 only
python -m gsr_cacl.train --dataset finqa --stage all --preset t4 --gradient-checkpointing
```

Benchmarking:
```bash
make benchmark DATASET=finqa MODE=gsr          # Single benchmark
make benchmark-all                             # All datasets/modes
make compare                                   # Compare vs baselines, save CSV
python -m gsr_cacl.benchmark_gsr --mode gsr --dataset finqa --top-k 3
```

## Architecture

### Package: `source/src/gsr_cacl/`

**Core pipeline flow:** Query → TextEncoder → JointScorer(text_sim, entity_sim, constraint_score) → ranked documents

- **`core/`** — Dataclasses: `Document`, `RetrievalResult`, `DatasetSplit`
- **`datasets/`** — `load_t2ragbench_split()`, `build_gsr_corpus()`, `GSRDocument` (document + pre-computed KG metadata)
- **`kg/`** — Knowledge graph construction from markdown tables: `KGNode`, `KGEdge`, `ConstraintKG`, `build_constraint_kg()`
- **`templates/`** — 15 IFRS/GAAP accounting templates, header synonym matching. Extend templates in `library.py`, synonyms in `matching.py`
- **`encoders/`** — `TextEncoder` (BGE wrapper, supports full/lora/frozen), `EntityEncoder`+`SharedEncoder` (shared BGE backbone), `GATEncoder` (2-layer edge-aware GAT), `NumericEncoder` (v1=log-scale, v2=ScaleAware), `SinusoidalPositionalEncoding`
- **`scoring/`** — `JointScorer`: `s(Q,D) = α·s_text + β·s_entity + γ·CS(G_D)`. `compute_constraint_score()` v1 (fixed ε) / v2 (relative tolerance)
- **`negative_sampler/`** — `CHAPNegativeSampler`: CHAP-A (answer-overlap), CHAP-S (structure), CHAP-E (entity) hard negative strategies
- **`training/`** — `TripletLoss`, `CACLLoss`, `EntitySupConLoss`, `RetrievalDataset`, `train_gsr_cacl()` with 3-stage curriculum
- **`methods/`** — `GSRRetrieval`, `HybridGSR` (FAISS + BM25 + RRF fusion), `GSR_REGISTRY`

### Training Stages

| Stage | Loss | Modules Trained | Default Epochs |
|-------|------|-----------------|----------------|
| 1 — Identity | TripletLoss + EntitySupConLoss | SharedEncoder + JointScorer | 3 |
| 2 — Structural | MSE(CS, 1.0) | + GATEncoder | 3 |
| 3 — Joint CACL | CACLLoss + EntitySupConLoss | All | 5 |

### Configuration

Hydra configs in `source/conf/`. Default model: `BAAI/bge-large-en-v1.5` (d=1024), LoRA r=16 alpha=32. GAT: 2 layers, 4 heads, hidden 256.

Key CLI flags: `--contr1 v2` (ScaleAware numeric encoder), `--contr2 v2` (relative constraint tolerance) — recommended for best results.

Hardware presets: `--preset t4|a100|v100|cpu` control encoder mode, batch size, and VRAM usage.

### Dataset

HuggingFace `G4KMU/t2-ragbench` — three subsets: FinQA, ConvFinQA, TAT-DQA. Each row has `question`, `context` (text + markdown table), `context_id`, company metadata.

## Code Quality

- **Linter/formatter:** ruff (line-length 100, Python 3.10 target, rules: E/F/W/I/N/B/SIM/C4)
- **Type checker:** mypy (strict return types)
- **Tests:** pytest (testpaths: `tests`, pythonpath: `src`)
