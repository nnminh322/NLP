# GSR-CACL

**Graph-Structured Retrieval with Constraint-Aware Contrastive Learning** for financial documents (text + tables).

> **Paper:** *Structured Knowledge-Enhanced Retrieval for Financial Documents* — T²-RAGBench (EACL 2026)
> **Benchmark:** [G4KMU/t2-ragbench](https://huggingface.co/datasets/G4KMU/t2-ragbench) (23,088 QA pairs, 7,318 documents)

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Installation](#2-installation)
3. [Architecture Overview](#3-architecture-overview)
4. [Three Contributions](#4-three-contributions)
5. [Training](#5-training)
6. [Benchmarking](#6-benchmarking)
7. [Codebase Map](#7-codebase-map)
8. [Extending Templates](#8-extending-templates)

---

## 1. Quick Start

### Run on Google Colab / Kaggle (T4 GPU — free)

```python
# Cell 1: Clone
!git clone https://github.com/<YOUR_REPO>/gsr-cacl.git /content/gsr-cacl
!cd /content/gsr-cacl/ours/source

# Cell 2: Install
!pip install -e ".[dev]" --quiet
!pip install peft accelerate transformers faiss-cpu --quiet

# Cell 3: Train (LoRA on T4 — fits 16GB VRAM)
!python -m gsr_cacl.train \
    --dataset finqa \
    --stage all \
    --preset t4 \
    --gradient-checkpointing

# Cell 4: Benchmark
!python -m gsr_cacl.benchmark_gsr \
    --mode gsr \
    --dataset finqa \
    --contr1 v2 \
    --contr2 v2
```

### Local installation

```bash
cd ours/source
pip install -e ".[dev]"
pip install peft accelerate transformers faiss-cpu

# Train
python -m gsr_cacl.train --dataset finqa --stage all --preset t4

# Benchmark
python -m gsr_cacl.benchmark_gsr --mode gsr --dataset finqa
```

### Hardware presets

| Preset | Encoder | Fine-tune | Batch | VRAM | Use case |
|--------|---------|-----------|-------|------|----------|
| `t4` | bge-large (335M) | LoRA r=16 | 8 | ~12 GB | Colab / Kaggle free |
| `a100` | bge-large (335M) | Full | 16 | ~20 GB | Full fine-tune |
| `v100` | bge-base (110M) | Full | 16 | ~14 GB | Mid-range GPU |
| `cpu` | bge-base (110M) | Frozen | 4 | — | Debugging only |

> **Tip:** Add `--gradient-checkpointing` to halve activation memory at ~20% speed cost.

---

## 2. Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (recommended for training)

### Dependencies

```
torch>=2.0.0
transformers>=4.36.0, peft>=0.7.0, accelerate>=0.25.0
langchain-huggingface, langchain-community
rank-bm25, faiss-cpu
datasets, pandas, tqdm
hydra-core, omegaconf
```

### Full install

```bash
# From project root
cd ours/source
pip install -e ".[dev]"           # dev includes pytest, ruff, mypy
pip install -e ".[gpu]"           # GPU acceleration (optional)
pip install peft accelerate transformers faiss-cpu  # ML dependencies
```

### Verify installation

```bash
python -c "from gsr_cacl import GATEncoder, JointScorer; print('OK')"
```

---

## 3. Architecture Overview

GSR-CACL solves the problem: *retrieve the correct financial document given a question* from a corpus of mixed text+table reports.

### Pipeline

```
Question Q
    │
    ├── 1. FAISS Vector Search ──→ Candidate top-4K documents
    │       (text similarity: BGE cosine)
    │
    ├── 2. Extract Table + Build KG
    │       markdown table ──parse──► Cell nodes (value, header, position)
    │                              │
    │                              └──► Template Matching (15 IFRS/GAAP patterns)
    │                                    │
    │                                    └──► Accounting edges (ω = +1/−1/0)
    │                                          +1 = additive parent (e.g., Revenue + COGS → Gross Profit)
    │                                          −1 = subtractive parent (e.g., GP − OpEx → EBIT)
    │                                           0 = positional (same row/col)
    │
    ├── 3. Encode with GAT ──→ Graph embedding d_KG
    │       (2-layer, 4-head, edge-aware attention)
    │
    └── 4. Joint Scoring
            s(Q,D) = α·s_text + β·s_entity + γ·CS(G_D)
                │
                ├── s_text: cosine(BGE(Q), BGE(D)) × query_gate + kg_adjustment
                ├── s_entity: cosine(e_Q, e_D)  ← learned entity embeddings
                └── CS(G_D): constraint score  (ε-tolerance accounting equations)
```

### Three signals

| Signal | Source | What it measures |
|--------|-------|----------------|
| **s_text** | BGE encoder | Lexical/semantic relevance |
| **s_entity** | EntityEncoder | Company/year/sector match (learned) |
| **CS(G_D)** | Constraint KG | Mathematical validity of the table |

---

## 4. Three Contributions

### C1 — GSR: Graph-Structured Table Representation

**Problem:** Flattening a table into text destroys accounting structure. A cell "500" loses: which column? Which row? What equation does it participate in?

**Solution:** Parse the table into a constraint knowledge graph.

```
| Revenue | COGS | Gross Profit |
| 100,000 | 70,000| 30,000      |

         ┌─[ω=−1]── Revenue
Revenue ─┼─[ω=+1]── COGS
         └─[ω=+1]── Gross Profit  (Σ Revenue − COGS = GP)
```

- 15 IFRS/GAAP templates cover ~80–90% of financial tables
- Edge weights ω ∈ {+1, −1, 0} encode accounting semantics
- GAT encoder learns structural representations from the graph

**Key equation:**
```
h_v^{(l+1)} = W_o [ ⊕_k Σ α_{uv} · ω_{uv} · W_v h_u^{(l)} ] + h_v^{(l)}
```

### C2 — EntitySupConLoss: Entity Understanding

**Problem:** Exact match ("Apple" vs "Apple Inc." → score=0) has zero gradient. The encoder learns nothing from entity matching.

**Solution:** Supervised Contrastive Learning with a shared BGE backbone.

```
Apple → BGE → [CLS] ──┐
Apple Inc. → BGE → [CLS] ──┼── concat ── proj ── LayerNorm → e ∈ R²⁵⁶
AAPL → BGE → [CLS] ──┘

L = −log ( Σ_{j∈P(i)} exp(cos(z_i, z_j)/τ) )
                    --------------------------------
                    Σ_k exp(cos(z_i, z_k)/τ)
```

- Same entity (Apple, Apple Inc., AAPL) → embeddings cluster together
- Different entities (Apple vs Microsoft) → embeddings far apart
- Gradient flows back through shared BGE → improves both entity AND text embeddings

**Key innovation:** Entity understanding improves text retrieval because the same backbone learns both.

### C3 — CHAP: Constraint-Aware Hard Negatives

**Problem:** Random negatives are too easy (completely unrelated). BM25 negatives only capture lexical overlap. Neither tests whether the model understands accounting.

**Solution:** Break exactly ONE accounting equation to create a hard negative.

| Type | What it does | Example |
|------|-------------|---------|
| **CHAP-A** | Change one LHS cell | Revenue=100K→110K (GP no longer = Revenue−COGS) |
| **CHAP-S** | Change unit (M↔B) | Revenue=100M→0.1B (ratio broken) |
| **CHAP-E** | Swap company/year | Table for Apple, metadata says Microsoft |

**Zero-Sum property:** The negative differs from the positive in exactly ONE way → surface similarity stays high, but an accounting constraint is violated. This forces the model to learn constraint semantics, not just lexical overlap.

---

## 5. Training

### Three-Stage Curriculum

Training follows a curriculum from simple to complex:

```
Stage 1 — Identity
    Objective: Learn (Company, Year) discrimination via entity scoring
    Loss: TripletLoss + EntitySupConLoss
    Duration: 3 epochs
    Modules: EntityEncoder + TextEncoder + JointScorer
    ─────────────────────────────────────────────────────
Stage 2 — Structural
    Objective: Calibrate KG encoding + constraint score ≈ 1.0 for valid tables
    Loss: MSE(CS(G_D), 1.0)
    Duration: 3 epochs
    Modules: + GATEncoder
    ─────────────────────────────────────────────────────
Stage 3 — Joint (CACL)
    Objective: Full CACL with CHAP hard negatives
    Loss: L_total = L_triplet + λ_e · L_EntitySupCon + λ_c · L_constraint
    Duration: 5 epochs
    Modules: All
```

**Why Stage 1 first?** Entity embeddings must learn basic clustering structure before GAT can use them in attention (EntitySim term). If entity embeddings are random, EntitySim in GAT is noise, not signal.

### Full training command

```bash
python -m gsr_cacl.train \
    --dataset finqa \
    --stage all \
    --encoder BAAI/bge-large-en-v1.5 \
    --finetune lora \
    --epochs 5 \
    --batch-size 8 \
    --contr1 v2 \
    --contr2 v2 \
    --save ./outputs/gsr_training
```

### Standalone stages

```bash
python -m gsr_cacl.train --dataset finqa --stage identity --epochs 3
python -m gsr_cacl.train --dataset finqa --stage structural --epochs 3
python -m gsr_cacl.train --dataset finqa --stage joint --epochs 5 --save ./checkpoints
```

### Loading a checkpoint

```python
from gsr_cacl.encoders import TextEncoder, GATEncoder
from gsr_cacl.scoring import JointScorer

ckpt = torch.load("./outputs/gsr_training/final_model.pt")
text_encoder.load_state_dict(ckpt["text_encoder_state"])
gat_encoder.load_state_dict(ckpt["gat_encoder_state"])
scorer.load_state_dict(ckpt["scorer_state"])
```

---

## 6. Benchmarking

### Run retrieval benchmark

```bash
# Basic GSR
python -m gsr_cacl.benchmark_gsr --mode gsr --dataset finqa --top-k 3

# Hybrid GSR + BM25 with RRF
python -m gsr_cacl.benchmark_gsr --mode hybridgsr --dataset finqa --top-k 3

# Sample for quick testing
python -m gsr_cacl.benchmark_gsr --mode gsr --dataset finqa --sample 50

# Use ScaleAwareNumericEncoder (contr1=v2) + relative tolerance (contr2=v2)
python -m gsr_cacl.benchmark_gsr --mode gsr --dataset finqa --contr1 v2 --contr2 v2
```

### Expected results (on T²-RAGBench)

| Method | MRR@3 | Recall@3 |
|--------|-------|---------|
| Base-RAG (BGE dense) | 0.326 | 0.398 |
| Hybrid BM25 | 0.352 | 0.494 |
| **GSR (ours)** | **≥ 0.40** | **≥ 0.50** |
| **HybridGSR (ours)** | **≥ 0.42** | **≥ 0.52** |

### Compare with baselines

```bash
python -m gsr_cacl.evaluate_comparison \
    --dataset finqa \
    --methods gsr hybridgsr \
    --save results.csv
```

### Template coverage analysis

```bash
python -m gsr_cacl.template_coverage_analysis --dataset finqa --sample 300
python -m gsr_cacl.template_coverage_analysis --dataset tatqa --sample 300
```

---

## 7. Codebase Map

```
ours/source/src/gsr_cacl/
├── __init__.py
├── train.py                         ← Training CLI entry point
├── benchmark_gsr.py                 ← Benchmark CLI entry point
├── evaluate.py
├── evaluate_comparison.py
├── template_coverage_analysis.py
│
├── core/
│   └── __init__.py                 ← Document, RetrievalResult, DatasetSplit
│
├── encoders/
│   ├── text_encoder.py              ← Differentiable BGE encoder (LoRA/freeze/full)
│   ├── entity_encoder.py           ← EntityEncoder + SharedEncoder (NEW)
│   ├── gat_layer.py                 ← Edge-aware GAT + EntitySim + residual (FIXED)
│   ├── gat_encoder.py               ← 2-layer GAT encoder (FIXED)
│   ├── numeric_encoder.py           ← V1 (log-scale) + V2 (ScaleAware)
│   └── positional.py                ← Sinusoidal PE for row/col
│
├── kg/
│   ├── builder.py                   ← Template matching + KG construction
│   ├── data_structures.py          ← KGNode, KGEdge, ConstraintKG
│   └── parser.py                   ← Markdown table parser
│
├── scoring/
│   ├── constraint_score.py          ← V1 (fixed ε) + V2 (relative tolerance)
│   └── joint_scorer.py             ← JointScorer (learned s_ent) (FIXED)
│
├── negative_sampler/
│   └── chap.py                     ← CHAP-A / CHAP-S / CHAP-E negatives
│
├── training/
│   ├── losses.py                   ← TripletLoss, ConstraintViolationLoss, CACLLoss
│   ├── entity_supcon_loss.py        ← EntitySupConLoss + EntityRegistry (NEW)
│   ├── data.py                     ← RetrievalSample, RetrievalDataset
│   └── trainer.py                   ← train_gsr_cacl() function
│
├── methods/
│   └── gsr_retrieval.py            ← GSRRetrieval + HybridGSR
│
├── templates/
│   ├── library.py                   ← 15 IFRS/GAAP templates
│   ├── data_structures.py          ← AccountingTemplate, AccountingConstraint
│   └── matching.py                 ← Template matching + header normalization
│
└── datasets/
    ├── wrappers.py                 ← load_t2ragbench_split(), build_gsr_corpus()
    └── gsr_document.py             ← GSRDocument with pre-computed KG
```

---

## 8. Extending Templates

The template library is domain-specific (IFRS/GAAP financial tables). To add a new template:

**Step 1:** Add to `templates/library.py`:

```python
from gsr_cacl.templates.data_structures import AccountingTemplate, AccountingConstraint

_register(AccountingTemplate(
    name="my_template",
    description="Description of the accounting identity",
    headers=["Header1", "Header2", "Total"],
    constraints=[
        AccountingConstraint(
            name="my_constraint",
            lhs=["Header1", "Header2"],
            rhs="Total",
            omega=+1,  # +1 = additive, −1 = subtractive
            op="add",  # "add" | "sub" | "div"
        ),
    ],
))
```

**Step 2:** Add synonyms to `templates/matching.py`:

```python
_HEADER_SYNONYMS["my_header_variant"] = "Header1"
```

**Step 3:** Verify coverage:

```bash
python -m gsr_cacl.template_coverage_analysis --dataset finqa --sample 300
```

---

## Key References

- Strich et al. (EACL 2026): [T²-RAGBench](https://huggingface.co/datasets/G4KMU/t2-ragbench)
- Khosla et al. (NeurIPS 2020): Supervised Contrastive Learning
- Halliday (1994): An Introduction to Functional Grammar
- Shannon (1948): A Mathematical Theory of Communication
- Karpukhin et al. (EMNLP 2020): Dense Passage Retrieval (DPR)
