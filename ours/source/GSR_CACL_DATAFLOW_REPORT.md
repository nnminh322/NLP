# GSR-CACL Dataflow Report
## Chi tiết toàn bộ flow từ raw data → Joint Scorer

---

## Tổng quan Kiến trúc

```
Query Q (text + company/year/sector)
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│                    SHARED BACKBONE f_θ                      │
│            (BGE / E5 / LLM2Vec + LoRA)                       │
│                   shared θ gradient                          │
└──────────┬──────────────────────────────────┬───────────────┘
           │                                  │
     ┌─────┴──────┐                    ┌──────┴──────┐
     │Text Signal │                    │Entity Signal │
     │   (blue)   │                    │   (orange)   │
     └────────────┘                    └──────────────┘

Document D (text + meta + Table T)
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│                    SHARED BACKBONE f_θ                      │
│                   (SAME INSTANCE)                           │
└──────────┬──────────────────────────────────┬───────────────┘
           │                                  │
     ┌─────┴──────┐                    ┌──────┴──────┐
     │Text Signal │                    │Entity Signal │
     └────────────┘                    └──────────────┘

Table T ⊂ D
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│              Structural Signal (teal)                       │
│  Template → KG → Edge-Aware GAT → d_KG + CS                │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    ┌──────┴──────┐
                    │Joint Scorer │
                    │  (purple)   │
                    │s(Q,D) score │
                    └─────────────┘
```

---

## 1. INPUT

Một sample `RetrievalSample` có:
```
{
  query: "Apple: What is the revenue in 2023?",   # company embedded in text
  positive_context: "In FY 2023, Apple reported revenue...",
  company_name: "Apple Inc.",      # hoặc "AAPL", "Apple"
  report_year: "2023",
  company_sector: "Technology",
}
```

**QUAN TRỌNG: Cả Query Q VÀ Document D đều có đủ 3 metadata fields.**

Từ `load_training_data()` trong `train.py`, tất cả 3 fields đều được populate:
```python
sample = RetrievalSample(
    query=f"{row.get('company_name', '')}: {row.get('question', '')}",
    company_name=str(row.get("company_name", "")),
    report_year=str(row.get("report_year", "")),
    company_sector=str(row.get("company_sector", "")),
)
```

Hai inputs chính:
- **Query Q**: text + (company, year, sector) metadata — dùng **original name** ("AAPL")
- **Document D**: text + (company, year, sector) metadata + Table T — dùng **canonical name** ("Apple Inc.")
    
    # LoRA applied here if finetune="lora"
    # The SAME self.backbone is used for BOTH text AND entity encoding
```

Điểm mấu chốt: **MỘT backbone duy nhất** (`self.backbone`), cùng weights θ, dùng cho cả text encoding và entity encoding. Gradient từ entity loss chảy ngược vào shared backbone → cải thiện cả text lẫn entity representations.

---

## 3. TEXT SIGNAL (Blue Lane)

**File**: `ours/source/src/gsr_cacl/encoders/entity_encoder.py` → method `text_encode()`

```
Text Flow:
──────────────────────────────────────────────────────────────────
Raw text (query)  ──tokenizer──▶  token_ids  ──backbone──▶ last_hidden_state
                    [B, seq_len]                    [B, seq_len, d]

last_hidden_state[:, 0, :]  ──[CLS] token──▶  [B, d]
        │
        ▼
LayerNorm([B, d])   ───nomalization──▶  [B, d]
        │
        ▼
L2 normalize  ───||x||=1──▶  [B, d]  ← q_text ∈ ℝ^d

──────────────────────────────────────────────────────────────────
Raw text (doc)    ──same pipeline──▶  d_text ∈ ℝ^d
```

Code trong `SharedEncoder.text_encode()`:
```python
def text_encode(self, texts: list[str], normalize=True):
    inputs = self.tokenizer(texts, ...)  # [B, seq_len]
    outputs = self.backbone(**inputs)     # [B, seq_len, d]
    embeds = outputs.last_hidden_state[:, 0, :]  # [CLS] → [B, d]
    embeds = self.text_norm(embeds)       # LayerNorm
    embeds = F.normalize(embeds, p=2, dim=-1)  # L2
    return embeds  # [B, d]
```

**Output**: `q_text ∈ ℝ^d`, `d_text ∈ ℝ^d` — L2-normalized text embeddings.

---

## 4. ENTITY SIGNAL (Orange Lane)

**File**: `ours/source/src/gsr_cacl/encoders/entity_encoder.py` → class `EntityEncoder`

### 4.1 Three Independent BGE Passes

```
company "Apple Inc."
    │
    ▼ tokenizer("Apple Inc.") → BGE backbone → last_hidden_state[:,0,:]
    ──→ [CLS] → e_company ∈ ℝ^d

year "2023"
    │
    ▼ tokenizer("2023") → BGE backbone → last_hidden_state[:,0,:]
    ──→ [CLS] → e_year ∈ ℝ^d

sector "Technology"
    │
    ▼ tokenizer("Technology") → BGE backbone → last_hidden_state[:,0,:]
    ──→ [CLS] → e_sector ∈ ℝ^d
```

### 4.2 Concatenation + MLP Projection

```python
# Code trong EntityEncoder.encode()
combined = torch.cat([e_company, e_year, e_sector], dim=-1)  # [3*d]
entity = self.proj(combined)   # Linear(3d, d) → ReLU → Drop → Linear(d, d_e)
entity = self.norm(entity)     # LayerNorm
entity = F.normalize(entity, p=2, dim=-1)  # L2 normalize
return entity  # [d_e], e.g., 256
```

Architecture:
```
e_company ⊕ e_year ⊕ e_sector  →  [3*d]
       │
       ▼
Linear(3d, d)  →  ReLU  →  Dropout  →  Linear(d, d_e)
       │
       ▼
LayerNorm(d_e)  →  L2 normalize  →  e_Q / e_D ∈ ℝ^{d_e}
```

**Output**: `e_Q ∈ ℝ^{d_e}`, `e_D ∈ ℝ^{d_e}` — L2-normalized entity embeddings.

### 4.3 Entity SupCon Loss

**File**: `ours/source/src/gsr_cacl/training/entity_supcon_loss.py`

```python
class EntitySupConLoss(nn.Module):
    def forward(self, embeddings, labels):
        # embeddings: [B, d_e] L2-normalized
        # labels: [B] canonical entity names
        
        similarity = torch.matmul(embeddings, embeddings.T) / temperature  # [B, B]
        # positive_mask[i,j] = 1 iff labels[i] == labels[j] and i != j
        
        # Loss = -log( Σ_{j∈P(i)} exp(cos/τ) / Σ_{k≠i} exp(cos/τ) )
        return per_sample_loss.mean()
```

- **Positive pairs**: "Apple" ↔ "Apple Inc.", "AAPL" ↔ "Apple Inc." (từ `EntityRegistry.CIK_MAPPING`)
- **Temperature τ = 0.07**: Làm phân bố similarity sharpness hơn, buộc embeddings phải rất close với tất cả positives và rất far từ negatives
- **Gradient**: ∂ℒ_e/∂θ chảy qua `EntityEncoder.proj` → `EntityEncoder.backbone` (= shared f_θ)

---

## 5. STRUCTURAL SIGNAL (Teal Lane)

**File**: `ours/source/src/gsr_cacl/encoders/gat_encoder.py` → class `GATEncoder`

### 5.1 Table T → Constraint KG

**File**: `ours/source/src/gsr_cacl/kg/builder.py`

```
Table T (from D)
   │
   ▼ Template Matching (IFRS/GAAP patterns)
   
Constraint KG:
  Nodes: cells {header, value, text, row_idx, col_idx}
  Edges: signed edges ω ∈ {+1, −1}
         +1 = accounting identity (e.g., Revenue − COGS = Gross Profit)
         −1 = constraint violation potential

Example signed KG:
  Revenue(100M) ──(+1)──► Gross Profit
  COGS(60M)     ──(+1)──► Gross Profit(40M)
  Revenue ──(−1)──► COGS  [violation if Revenue < COGS]
```

### 5.2 Node Features Construction (Eq.4)

```python
# GATEncoder._build_cell_embeddings()
cell_embed = BGE(cell_text)  # [V, d]  hoặc header_hash + numeric_proj fallback

# GATEncoder.forward()
row_pos = SinusoidalPE(row_idx)      # [V, d//4]
col_pos = SinusoidalPE(col_idx)      # [V, d//4]
entity_proj = entity_proj(e_D)        # [V, d]  (project e_D → d)

x_v = [cell_embed ⊕ row_pos ⊕ col_pos ⊕ entity_proj]  # [V, d + 2*d/4 + d] = [V, 2d]
x_v = input_proj(x_v)  # Linear(2d, hidden_dim) → LN → ReLU → Drop → [V, 256]
```

### 5.3 Edge-Aware GAT (2 layers)

**File**: `ours/source/src/gsr_cacl/encoders/gat_layer.py`

```
Layer 1:
  edge_index: adjacency matrix
  edge_weight: ω ∈ {+1, −1} (signed)

  Attention score (Eq.6):
    e_uv = <Q_u, K_v>/√d_k + Proj(ω_uv) + σ(scale) · EntitySim(e_u, e_v)
         = attention + edge_bias + entity_similarity

  Message passing:
    h_v^{(l+1)} = W_o [ ⊕_k Σ α_uv · ω_uv · W_v h_u^{(l)} ] + h_v^{(l)}
                 └──────────── residual ────────────┘

Layer 2: Same architecture
```

### 5.4 Graph-Level Representation (Eq.9)

```python
def encode_graph(self, kg, entity_embeddings=None):
    node_embeds = self.forward(kg, entity_embeddings=entity_embeddings)
    d_KG = node_embeds.mean(dim=0)  # mean pooling → [hidden_dim=256]
    return d_KG
```

### 5.5 Constraint Scoring

**File**: `ours/source/src/gsr_cacl/scoring/constraint_score.py`

```python
def compute_constraint_score(kg, version="v1"):
    for each edge (u→v) with weight ω:
        score_uv = exp(-|ω·v_u − v_v| / max(|v_v|, ε))
    CS = (1/|E_c|) Σ score_uv  ∈ [0, 1]
    return ConstraintScoringResult(
        constraint_score=CS,
        violated_count=count(ω·v_u < v_v),
        total_count=|E_c|
    )
```

**Output**: `d_KG ∈ ℝ^{256}` (graph embedding) + `CS` (constraint score ∈ [0,1])

---

## 6. JOINT SCORER

**File**: `ours/source/src/gsr_cacl/scoring/joint_scorer.py` → class `JointScorer`

### 6.1 Inputs

```python
def forward(
    query_text_embed: [B, text_embed_dim=1024],   # q_text
    doc_text_embed: [B, text_embed_dim=1024],      # d_text
    kg_embed: [B, kg_embed_dim=256],              # d_KG
    query_entity_embed: [B, entity_embed_dim=256], # e_Q
    doc_entity_embed: [B, entity_embed_dim=256],   # e_D
    constraint_features: [B, 3],  # [raw_cs, violated_ratio, edge_norm]
) → [B] scores
```

### 6.2 s_text — Text Similarity with KG Enrichment (Eq.16)

```python
def forward_text_sim(self, q, d, kg_embed):
    # Base cosine similarity
    sim = torch.cosine_similarity(q, d, dim=-1)  # cos(q_text, d_text) ∈ [-1,1]
    
    # Query-dependent gating
    gate_val = self.gate(q).squeeze(-1)          # MLP → Sigmoid → [0,1]
    gated_sim = sim * (0.5 + 0.5 * gate_val)     # [0.5, 1.5] × sim
    
    # KG structural enrichment
    combined = torch.cat([d, kg_embed], dim=-1)  # [d + 256]
    kg_adj = self.text_kg_proj(combined).squeeze(-1)  # MLP → Tanh → [-1,1]
    
    return gated_sim + 0.2 * kg_adj
```

**Formula**: `s_text = cos(q_text, d_text) × (0.5 + 0.5·σ(gate(q_text))) + 0.2·MLP([d_text ⊕ d_KG])`

### 6.3 s_entity — Entity Cosine Similarity (Eq.17)

```python
def forward_entity_sim(self, q_e, d_e):
    return torch.cosine_similarity(q_e, d_e, dim=-1)  # cos(e_Q, e_D)
```

**Formula**: `s_entity = cos(e_Q, e_D)` — gradient flows back through EntityEncoder → shared backbone.

### 6.4 s_struct — Constraint Score Refinement (Eq.10 + MLP)

```python
def forward_constraint(self, constraint_features):
    # constraint_features = [raw_cs, violated_ratio, edge_norm] ∈ ℝ^3
    return self.constraint_proj(constraint_features)  # MLP → Sigmoid → [0,1]
```

**Formula**: `s_struct = MLP_cs([raw_cs, violated_ratio, edge_norm])`

### 6.5 Final Score (Eq.15)

```python
@property
def alpha(self): return F.softplus(self.log_alpha)  # ≥ 0
@property
def beta(self):  return F.softplus(self.log_beta)   # ≥ 0
@property  
def gamma(self): return F.softplus(self.log_gamma)   # ≥ 0

def forward(self, ...):
    s_text = self.forward_text_sim(...)
    s_entity = self.forward_entity_sim(...)
    s_constraint = self.forward_constraint(...)
    
    return self.alpha * s_text + self.beta * s_entity + self.gamma * s_constraint
```

**Formula**: `s(Q, D) = α·s_text + β·s_entity + γ·s_struct`, α,β,γ ≥ 0 (softplus-constrained).

---

## 7. COMPLETE TRAINING FLOW (3-Stage Curriculum)

### Stage 1 — Identity Pretraining

```
Batch of B samples → encoder → [q_text, d_text, e_Q, e_D]

Triplet Loss:
  pos_score = scorer(q_text, d_text, kg_dummy, e_Q, e_D, cs_dummy)
  neg_score = scorer(q_text, d_text, kg_dummy, e_Q, e_D_neg)
  loss_triplet = TripletLoss(pos_score, neg_score)

EntitySupCon Loss:
  labels = entity_registry.build_entity_labels(companies)
  loss_supcon = EntitySupConLoss(e_Q, labels)

Total loss = loss_triplet + loss_supcon
Gradient → encoder.backbone (shared f_θ) + scorer
```

### Stage 2 — Structural Pretraining

```
For each sample:
  Table T → build_constraint_kg() → KG
  KG → GATEncoder.encode_graph() → d_KG
  KG → compute_constraint_score() → CS features

loss = MSE(scorer.forward_constraint(CS_feats), target=1.0)
     + variance_penalty on d_KG (encourage distinct representations)

Gradient → encoder + scorer + GATEncoder
```

### Stage 3 — Joint CACL

```
For each sample:
  q_text = encoder.text_encode([query])
  d_text = encoder.text_encode([positive_context])
  e_Q = encoder.entity_encode(companies, years, sectors)  # Q uses original names
  e_D = encoder.entity_encode(canonical_names, years, sectors)  # D uses canonical
  pos_KG = build_constraint_kg(positive_context)
  pos_d_KG = GATEncoder.encode_graph(pos_KG, e_Q)  # optional entity sim
  pos_CS = compute_constraint_score(pos_KG)
  pos_score = scorer(q_text, d_text, pos_d_KG, e_Q, e_D, pos_CS)

  CHAP generates hard negatives:
  neg_KG = perturb(pos_KG) → "Revenue + COGS = Gross Profit" broken
  neg_d_KG = GATEncoder.encode_graph(neg_KG)
  neg_score = scorer(q_text, neg_d_text, neg_d_KG, e_Q, e_D, neg_CS)
  is_violated = True/False

  loss_cacl = CACLLoss(pos_score, neg_score, is_violated)
  loss_supcon = EntitySupConLoss(e_Q, entity_labels)
  
  total = loss_cacl + λ_entity * loss_supcon
  Gradient → encoder + scorer + GATEncoder (all 3 components jointly)
```

---

## 8. Key Insights từ Codebase

### Shared Backbone (Mấu chốt)
```python
class SharedEncoder:
    self.backbone = AutoModel(...)  # MỘT instance
    self.entity_encoder = EntityEncoder(backbone=self.backbone)  # shares same backbone
    
    def text_encode(self, texts):
        return self.backbone(...)  # gradient flows through shared backbone
        
    def entity_encode(self, companies, years, sectors):
        return self.entity_encoder(...)  # gradient ALSO flows to shared backbone
```

Điều này có nghĩa: entity loss gradient và text loss gradient CÙNG update `self.backbone` weights. Entity understanding (từ SupCon) cải thiện text representations (cho retrieval).

### Entity Name Resolution (Fix Issue 5)
```python
# Query: use original name ("AAPL")
# Document: use canonical name ("Apple Inc.") from registry
q_company = sample.company_name
d_company = entity_registry.get_canonical_name(sample.company_name) or sample.company_name

q_entity_emb = encoder.entity_encode([q_company], ...)
d_entity_emb = encoder.entity_encode([d_company], ...)
# Now cos(e_Q, e_D) is maximized for same entities despite different name formats
```

### CHAP Negative Sampling
```python
chap_sampler = CHAPNegativeSampler(chap_a_prob=0.5, chap_s_prob=0.3, chap_e_prob=0.2)

# A: change amounts (e.g., Revenue 100M → 120M, identity breaks)
# S: swap rows/columns (structure perturbation)  
# E: remove/change entities (entity perturbation)

# Negative is VIOLATED → used in CACL constraint loss
# is_violated = True when CHAP breaks an accounting identity
```

---

## 9. Tensor Shapes Summary

| Signal | Shape | Description |
|--------|-------|-------------|
| q_text, d_text | [B, 1024] | L2-normalized text embeddings from BGE |
| e_Q, e_D | [B, 256] | L2-normalized entity embeddings |
| d_KG | [B, 256] | Graph-level embedding from GAT mean pooling |
| constraint_features | [B, 3] | [raw_cs, violated_ratio, edge_norm] |
| s_text | [B] | scalar per sample |
| s_entity | [B] | scalar per sample |
| s_struct | [B] | scalar per sample |
| s(Q,D) final | [B] | α·s_text + β·s_entity + γ·s_struct |

---

## 10. All Source Files

| File | Role |
|------|------|
| `encoders/entity_encoder.py` | SharedEncoder (text + entity encoding) |
| `encoders/gat_encoder.py` | GATEncoder (table → KG → graph embed) |
| `encoders/gat_layer.py` | GATLayer (edge-aware attention) |
| `scoring/joint_scorer.py` | JointScorer (fuse all signals) |
| `scoring/constraint_score.py` | Constraint scoring module |
| `kg/builder.py` | KG construction from tables |
| `training/entity_supcon_loss.py` | SupCon loss + EntityRegistry |
| `train.py` | 3-stage training loop |