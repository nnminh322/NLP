# GSR-CACL Improvement: Entity-Aware JointScorer

---

## Current Architecture

The `JointScorer.forward()` computes:

```
s(Q, D) = α · s_text(Q, D)  +  β · s_entity(Q, D)  +  γ · s_constraint(D)
```

Three independent branches:

| Branch | Method | Limitation |
|---|---|---|
| `s_text` | BGE cosine + query gate + kg_adjustment | Limited: text signal only |
| `s_entity` | Exact hash match on company/year/sector | **No semantic matching** |
| `s_constraint` | MLP over [raw_score, violated_ratio, edges] | **No entity context** |

The `s_entity` branch is the weakest:

```python
# JointScorer.forward_entity() — lines 100–107
match = (query_meta == doc_meta).float()   # 0 or 1 per field
return match.mean(dim=-1)                   # 0.0 / 0.33 / 0.67 / 1.0
```

```
Problem:
  Query: company_name = "Apple"
  Doc:   company_name = "Apple Inc."
  → Exact match: FALSE → s_entity contribution = 0
```

The GAT also has no entity context:

```python
# GATEncoder._build_cell_embeddings() — lines 117–139
# Node features: header + numeric + row_pos + col_pos
# No company information anywhere
```

---

## Proposed Improvements

### P3a — Semantic Entity Matching

**What it changes:** `JointScorer.forward_entity()` — from exact match to BGE cosine.

**Before (v1):**
```python
# Exact string comparison
meta_q = [hash("apple"),     hash("2023"), hash("technology")]
meta_d = [hash("apple inc."), hash("2023"), hash("technology")]
match  = (meta_q == meta_d)     → [0, 1, 1] → 0.67
```

**After (v2):**
```python
# BGE cosine similarity
q_emb = BGE("apple")      → [0.12, -0.34, ...]
d_emb = BGE("apple inc.") → [0.11, -0.33, ...]
sim   = cosine(q_emb, d_emb) → 0.87
```

**Result:** "Apple" now matches "Apple Inc." with 0.87 similarity instead of 0. The text encoder is shared — no new model, no new parameters.

---

### P3b — Entity-Aware Constraint Weighting

**What it changes:** `compute_constraint_score()` — constraint violations are weighted by whether they are within-company or cross-company.

**Before (v1):**
```python
# All constraints scored identically
for edge in accounting_edges:
    residual = abs(edge.omega * src_val - tgt_val)
    score = exp(-residual / max(|tgt_val|, ε))
    # No check: are src and tgt from the same company?
```

**After (v2):**
```python
# Cross-company constraints are structurally invalid
for edge in accounting_edges:
    if cross_company(edge.src, edge.tgt, doc_entity_id):
        edge_score = 0.0   # impossible constraint
        violated += 1
    else:
        edge_score = exp(-residual / max(|tgt_val|, ε))
```

**Why it matters:** In a wrong document, cross-company constraints may accidentally appear valid numerically but are semantically impossible. Detecting them adds a strong disambiguation signal.

---

### P3c — Entity-Aware Node Features

**What it changes:** `GATEncoder._build_cell_embeddings()` — nodes carry company context.

**Before (v1):**
```python
# Each node has: header + numeric + row_pos + col_pos
# Nodes for "Apple" tables and "Microsoft" tables are indistinguishable
```

**After (v2):**
```python
# Each node has: header + numeric + row_pos + col_pos + company_embed
# company_embed = embedding of document's company_name
# Shape: [embed_dim + 2*(embed_dim//4) + entity_embed_dim] → input_proj
```

**Result:** GAT message passing now propagates company context through the graph. Attention weights are modulated by company similarity. Nodes from the same company subgraph aggregate differently.

---

## Full Architecture: Before → After

### Before

```
Document D
  │
  ├─► BGE(D) ──► text_emb
  │
  ├─► KG ──► GATEncoder
  │             └─ node: [header, numeric, row, col]  ← NO company context
  │
  └─► compute_constraint_score()
                └─ all constraints weighted equally  ← NO entity context

Query Q
  ├─► BGE(Q) ──► query_emb
  │
  └─► metadata: {company: "Apple", year: "2023"}

JointScorer:
  s_text      = cosine(BGE(Q), BGE(D)) + gate + kg_adjust
  s_entity    = exact_match(company_q, company_d)     ← PROBLEM: 0 for variants
  s_constraint = MLP([raw_cs, violated_ratio, edges])
  s           = α·s_text + β·s_entity + γ·s_constraint
```

### After (with P3a + P3b + P3c)

```
Document D
  │
  ├─► BGE(D) ──► text_emb
  │
  ├─► KG ──► GATEncoder
  │             └─ node: [header, numeric, row, col, company_embed]  ← NEW
  │
  └─► compute_constraint_score()
                └─ cross-company edges → score = 0.0           ← NEW

Query Q
  ├─► BGE(Q) ──► query_emb
  │
  └─► metadata: {company: "Apple", year: "2023"}

JointScorer:
  s_text      = cosine(BGE(Q), BGE(D)) + gate + kg_adjust
  s_entity    = cosine(BGE(company_q), BGE(company_d))         ← NEW: soft match
  s_constraint = MLP([raw_cs, violated_ratio, edges])
                + cross_company_penalty                           ← NEW: entity-weighted
  s           = α·s_text + β·s_entity + γ·s_constraint
```

**What changed:** 3 lines of code modified. The flow, the loss, the training pipeline — all unchanged.

---

## Summary Table

| | Before | After |
|---|---|---|
| `s_entity` | Exact string match (0 or 1) | BGE cosine similarity (0.0–1.0) |
| `s_constraint` | All constraints equal | Cross-company → 0.0 penalty |
| GAT nodes | [header, numeric, row, col] | [header, numeric, row, col, company] |
| Model changes | — | 3 components modified, no new models |
| Parameters added | — | None for P3a, ~320K max for P3c |
| Backward compatible | — | `v1` = original behavior |

---

## Ablation

```bash
# Baseline (original)
python -m gsr_cacl.train --dataset finqa --stage all \
    --entity-a v1 --entity-b v1 --entity-c v1

# P3a only: soft entity matching
python -m gsr_cacl.train --dataset finqa --stage all \
    --entity-a v2 --entity-b v1 --entity-c v1

# P3b only: entity-aware constraints
python -m gsr_cacl.train --dataset finqa --stage all \
    --entity-a v1 --entity-b v2 --entity-c v1

# P3c only: entity-aware node features
python -m gsr_cacl.train --dataset finqa --stage all \
    --entity-a v1 --entity-b v1 --entity-c v2

# All three
python -m gsr_cacl.train --dataset finqa --stage all \
    --entity-a v2 --entity-b v2 --entity-c v2
```

---

## Appendix A — P3a: BGE Soft Entity Matching

### Implementation

```python
# In JointScorer.forward_entity()
if self.entity_mode == "v2":
    # Encode entity fields with BGE
    q_embs = self.text_encoder([query_company_text])   # [1, embed_dim]
    d_embs = self.text_encoder([doc_company_text])     # [1, embed_dim]
    sim = torch.cosine_similarity(q_embs, d_embs, dim=-1)
    return sim  # continuous 0.0–1.0
else:
    # Original exact match
    match = (query_meta == doc_meta).float()
    return match.mean(dim=-1)
```

No new model. The `text_encoder` (BGE) is already in memory. Entity field embeddings are computed on-the-fly.

### Fields

| Field | Example Q | Example D | Note |
|---|---|---|---|
| company_name | "Apple" | "Apple Inc." | High variance — primary benefit |
| report_year | "2023" | "2023" | Typically exact match |
| company_sector | "technology" | "Technology" | Case-insensitive, low variance |

---

## Appendix B — P3b: Cross-Company Constraint Detection

### Implementation

```python
def compute_constraint_score_v2(kg, doc_company_id):
    # For each accounting edge (src → tgt):
    #   If node belongs to a different company than doc_company_id:
    #       edge_score = 0.0  (structurally invalid)
    #   Else:
    #       edge_score = exp(-residual / max(|tgt|, ε))
    pass
```

### When cross-company edges appear

In T²-RAGBench, each document belongs to one company. Cross-company edges would only appear if:
1. Tables from different companies are concatenated
2. The KG builder incorrectly merges tables from multiple documents

In practice, P3b's main contribution is **rejecting wrong documents where constraints are accidentally numerically valid but semantically impossible.**

---

## Appendix C — P3c: Company Embedding in GAT

### Implementation

```python
class GATEncoder(nn.Module):
    def __init__(self, ..., entity_embed_dim: int = 32):
        self.company_embed = nn.Embedding(num_embeddings=10000, embedding_dim=entity_embed_dim)

    def _build_cell_embeddings(self, kg, device):
        # ... header and numeric features as before ...

        # Company embedding for all nodes in this document
        company_id = self._company_name_to_id(kg.source_company)
        V = len(kg.nodes)
        comp_emb = self.company_embed(
            torch.tensor([company_id], device=device)
        ).expand(V, -1)  # [V, entity_embed_dim]

        # Concatenate: [cell_features ⊕ company_embed]
        all_features = torch.cat([cell_embed, row_pos, col_pos, comp_emb], dim=-1)
        h = self.input_proj(all_features)  # [V, hidden_dim]
```

**Parameters added:** `10,000 × 32 = 320,000` — negligible vs 335M BGE backbone.

### Alternative: BGE-based entity embedding

```python
# Instead of a separate lookup table, encode with BGE:
company_text = doc.meta_data.get("company_name", "")
company_emb = self.entity_proj(self.text_encoder([company_text]))  # [1, embed_dim]
# Projects BGE output to entity_embed_dim, trained jointly with GAT
```

This shares the text encoder backbone. During training, company embeddings update jointly with the rest of the model.
