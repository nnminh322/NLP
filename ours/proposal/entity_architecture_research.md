# Entity Understanding Architecture for Financial Table Retrieval

> **Research question:** How should a learned architecture capture entity equivalence — "Apple Inc." = "Apple" = "AAPL" — for financial table retrieval?
>
> **Problem framing:** The entity mismatch problem is NOT a feature engineering problem. It is a **representation learning problem**: the model must learn that different surface forms refer to the same canonical entity, and that entity representations should be consistent across the entire system.

---

## Part I: Problem Analysis — What Is Entity Understanding?

### 1.1 The Three Facets of Entity Understanding

In financial table retrieval, "entity understanding" has three independent facets:

```
FACET 1 — Entity Canonicalization
  "Apple Inc." = "Apple" = "AAPL" = "Apple Computer, Inc."
  → Map surface forms to a canonical entity representation
  → Output: a single entity ID / embedding

FACET 2 — Entity Context Awareness
  "Apple" in "Apple revenue" ≠ "Apple" in "Apple stock price"
  → Same surface form, different meaning
  → Output: context-dependent entity embedding

FACET 3 — Entity-Relationship Reasoning
  "Apple's revenue" + "Microsoft's revenue" = comparative analysis
  → Entity relationships matter for retrieval
  → Output: entity relationship graph
```

**GSR-CACL currently handles NONE of these well:**

```
FACET 1: Exact hash match → "Apple" ≠ "Apple Inc."
FACET 2: Text encoder ignores entity type → no context awareness
FACET 3: No entity relationship graph → companies treated independently
```

### 1.2 Why Is Entity Understanding Hard?

**Core challenge:** Entity equivalence is a DISCRETE property (same or different) but must be learned from CONTINUOUS representations.

```
Standard text encoder (BGE):
  BGE("Apple Inc.") → [0.12, -0.34, ...]
  BGE("Apple")      → [0.11, -0.33, ...]
  BGE("AAPL")       → [0.09, -0.31, ...]

These are CLOSE in BGE space (cosine ≈ 0.87).
BUT:
  → BGE was trained on general text, not financial entity pairs
  → "Close" ≠ "same entity"
  → BGE similarity ≠ entity equivalence
```

**The key insight:** Entity equivalence is NOT the same as text similarity. Two entity mentions can be lexically dissimilar ("Apple Inc." vs "AAPL") but refer to the same entity. Two mentions can be lexically similar ("Apple Inc." vs "Apple Corporation") but refer to different entities.

---

## Part II: Literature Review — State of the Art

### 2.1 Three Paradigms for Entity Resolution

From the literature (2018-2025), three fundamentally different approaches exist:

#### Paradigm 1: Joint Embedding Space (Bi-Encoder)

```
CORE: Encode both mentions into the SAME embedding space.
      Distance in space ≈ entity equivalence.

Architecture:
  Mention A → Encoder → embedding_A
  Mention B → Encoder → embedding_B
              ↓
      cos(embedding_A, embedding_B) → entity similarity

Training:
  Positive pairs: same entity mentions → close
  Negative pairs: different entity mentions → far
  Loss: Contrastive (NT-Xent / SupCon / Triplet)
```

**Key papers:**
- DeepER (Mrini et al., ACL 2019): First BERT bi-encoder for entity resolution
- CERBERT (Luu et al., ACL 2021): BERT bi-encoder with triplet loss for entity matching
- SupCon-ER (various, 2022-2023): Supervised contrastive loss for entity matching

**Strength:** Efficient at scale (embeddings pre-computed).

**Weakness:** Bi-encoder cannot capture fine-grained interactions between mentions. "Apple Inc." and "Apple Corporation" → close but different entities. Need cross-encoder for disambiguation.

---

#### Paradigm 2: Cross-Encoder Interaction

```
CORE: Encode BOTH mentions TOGETHER.
      Interaction is computed at every layer.

Architecture:
  [CLS] mention_A [SEP] mention_B [SEP] → BERT → classification_head
              ↓
      Binary: same_entity / different_entity

Training:
  Binary cross-entropy or ranking loss
  Data: labeled (mention_A, mention_B, label) pairs
```

**Key papers:**
- BLINK (Wu et al., ACL 2020): Bi-encoder + cross-encoder reranking
- Magellan (Reddy et al., SIGMOD 2022): Fine-tuned BERT cross-encoder for entity matching
- Ditto (Li et al., arXiv 2023): LLM prompting for entity matching

**Strength:** Captures fine-grained mention interactions. "Apple Inc." vs "Apple Corporation" → BERT attends to the difference.

**Weakness:** Cannot scale to millions of candidates (requires pairwise forward pass). Must be used as reranker on top of a bi-encoder retrieval stage.

---

#### Paradigm 3: Canonicalization via Generation

```
CORE: GENERATE the canonical entity name.
      Entity resolution = entity generation.

Architecture:
  Mention → Autoregressive LM → canonical_entity_name

Training:
  Conditional generation: P(canonical_name | mention, context)
  Supervision: canonical names from KB

Key insight:
  "Entity linking as generation" (GENRE, De Cao et al., ACL 2021)
  → Instead of ranking candidates, GENERATE the entity
  → More flexible: handles unseen entities
  → Constrained decoding: only valid entity names
```

**Key papers:**
- GENRE (De Cao et al., ACL 2021): Autoregressive entity linking
- mGENRE (2022): Multilingual extension
- ReFiNe (Olia et al., 2022): Generation + refinement

**Strength:** Handles rare/unseen entities. No candidate pool needed.

**Weakness:** Requires valid entity vocabulary. Computationally heavy (autoregressive decoding). Best for entity linking (mention → KB), not canonicalization.

---

### 2.2 Which Paradigm Fits GSR-CACL?

```
GSR-CACL Setting:
  → We have a RETRIEVAL task, not entity linking task
  → Entities are companies with known canonical names in metadata
  → The problem is: query mentions vs document metadata → same entity?

Requirements:
  1. Must be SCALABLE (thousands of corpus documents)
  2. Must TRAIN JOINTLY with existing GAT + JointScorer
  3. Must produce entity embeddings that are CONSISTENT with retrieval

Best fit: PARADIGM 1 (Bi-Encoder) with modification

Reasoning:
  → Cross-encoder (Paradigm 2) is too slow for retrieval (N candidates = N forward passes)
  → Generation (Paradigm 3) requires entity vocabulary we don't have
  → Bi-encoder is scalable but needs domain-specific training
```

### 2.3 Entity Contrastive Loss — The Right Training Objective

**The critical question:** What training signal teaches entity equivalence?

**Supervised Contrastive Loss (SupCon, Khosla et al., 2020):**

```
For a batch with multiple entities:
  L_supcon = -log exp(sim(z_i, z_pos) / Σ_j exp(sim(z_i, z_j)))

Where:
  z_i = embedding of mention i
  z_pos = embedding of same-entity mention (positive)
  z_j = embedding of different-entity mention (negative, in-batch)
```

**Key property:** SupCon learns an embedding space where same-entity mentions cluster together, regardless of surface form.

**For financial companies:**

```
Positive pairs (same entity):
  ("Apple Inc.", "Apple")
  ("Apple Inc.", "AAPL")
  ("Apple Inc.", "Apple Computer, Inc.")
  ("Microsoft Corporation", "MSFT")
  ("Microsoft Corporation", "Microsoft")
  ("Tesla, Inc.", "TSLA")

Negative pairs (different entity):
  ("Apple Inc.", "Microsoft Corporation")
  ("Apple Inc.", "Amazon.com, Inc.")
  ("Apple Inc.", "Meta Platforms, Inc.")
```

### 2.4 The Data Problem — Where Does Training Data Come From?

**This is the critical bottleneck.**

```
For entity contrastive learning, we need:
  1. Canonical entity names (from metadata)
  2. Surface form variations (from queries and documents)

T²-RAGBench data:
  Query metadata: {company_name, report_year, company_sector}
  Document metadata: {company_name, report_year, company_sector}

  → Metadata gives us: canonical entity names
  → Surface forms: extracted from question text

Available signals:
  1. Company name in query vs. company name in document → same/different
  2. Ticker ↔ company name mapping → same entity
  3. Year consistency → year in query vs. document
  4. SEC CIK registry (public data) → ticker ↔ company canonical name

Data construction:
  (Query_company, Document_company) → positive/negative pair
  → Same: company names match exactly
  → Negative: within-batch negatives (same batch, different company)
```

---

## Part III: The Architecture — Entity-Aware Retrieval

### 3.1 The Core Design Principle

**Principle:** Entity understanding should be a **training objective**, not a **feature**. The entity signal should influence how the entire model learns, not just one branch of the scorer.

**Why this matters:**

```
Current system (GSR-CACL):
  Entity signal = exact string match → feeds into s_entity
  → No training signal for entity understanding
  → Text encoder ignores entity relationships
  → JointScorer has no entity-aware representations

Proposed (Entity-CACLL):
  Entity signal = learned embeddings → feeds into ALL branches
  → Text encoder learns: "Apple Inc." in context → entity-aware
  → KG nodes learn: company context in table cells
  → JointScorer has entity-aware text + entity + constraint representations
```

### 3.2 Three-Level Entity Architecture

**Level 1 — Entity Token Layer (What the text encoder sees)**

```
Financial document contains:
  "Apple Inc. (AAPL) reported revenue of $298B..."

  Tokens:
    "Apple" → generic word
    "Inc." → generic word
    "AAPL" → ENTITY TOKEN
    "$" → unit marker
    "298B" → NUMERIC TOKEN

Key insight:
  → "AAPL" is an ENTITY TOKEN (refers to a specific company)
  → Entity tokens should be encoded DIFFERENTLY from generic words
  → The text encoder must LEARN to distinguish entity tokens
```

**Approach:** Learnable entity type embeddings injected into the text encoder.

```
Architecture:
  Input tokens → Token Embedding + Position Embedding + Entity Type Embedding
                                    ↑
                              entity_type ∈ {GENERIC, COMPANY, TICKER, YEAR, SECTOR}
                                    ↑
                              Learned: COMPANY embeddings are similar across surface forms

Training signal:
  → Positive: "AAPL" in query ↔ "Apple Inc." in doc → entity_type match
  → Negative: "AAPL" in query ↔ "MSFT" in doc → entity_type mismatch
```

**Level 2 — Entity Mention Layer (What the retrieval scorer sees)**

```
Query: "Apple 2023 revenue"
Document: company_name = "Apple Inc."

Current: exact string match
  hash("apple") vs hash("apple inc.") → 0

Proposed: learned mention embeddings
  encoder("Apple") → entity_embedding
  encoder("Apple Inc.") → entity_embedding
  → cosine similarity = 0.87 (learned to be high for same entity)
```

**Level 3 — Entity Context Layer (What the KG sees)**

```
KG node currently: [header, numeric, row_pos, col_pos]
Proposed KG node: [header, numeric, row_pos, col_pos, entity_context]

entity_context:
  → Company embedding: "which company does this cell belong to?"
  → Entity-aware attention: "should this node aggregate with nodes from the same company?"
```

### 3.3 The Unified Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  ENTITY-AWARE FINANCIAL TABLE RETRIEVAL                     │
│                                                             │
│  Query Q: "Apple 2023 revenue growth"                      │
│                                                             │
│  LEVEL 1 — Entity Token Encoding                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Text Encoder (BGE)                                   │   │
│  │ Token: "Apple" → [word_emb ⊕ entity_type_emb]     │   │
│  │ Token: "AAPL"  → [word_emb ⊕ COMPANY_emb]         │   │
│  │          ↑                                           │   │
│  │          Company tokens share similar embeddings     │   │
│  │          across surface forms                        │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         ↓                                   │
│  LEVEL 2 — Entity Mention Matching                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Entity Mention Encoder                               │   │
│  │ Query mention: "Apple"  → entity_emb_Q            │   │
│  │ Doc mention:   "Apple Inc." → entity_emb_D        │   │
│  │ s_entity = cosine(entity_emb_Q, entity_emb_D)      │   │
│  │              = 0.87 (SAME entity, learned)         │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         ↓                                   │
│  LEVEL 3 — Entity-Aware KG                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ KG nodes: [header ⊕ numeric ⊕ pos ⊕ company_emb] │   │
│  │ GAT: company_emb modulates attention weights        │   │
│  │ → Same-company nodes: high attention               │   │
│  │ → Cross-company nodes: low attention               │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         ↓                                   │
│  JointScorer: s = α·s_text + β·s_entity + γ·s_constraint │
│  where all three signals use ENTITY-AWARE representations  │
└─────────────────────────────────────────────────────────────┘
```

---

## Part IV: Training Objective — Entity-CACLL

### 4.1 The Loss Function

The core training objective must simultaneously optimize:

```
1. Retrieval ranking: rank correct docs above wrong docs
2. Entity understanding: same entity → close in space
3. Table structure: accounting constraints encoded in KG
4. Cross-modal alignment: text ↔ KG alignment
```

**Proposed loss:**

```
L_total = L_retrieval + λ_e · L_entity + λ_s · L_structure + λ_c · L_constraint

Where:

L_retrieval = TripletLoss(pos_scores, neg_scores)
  → Standard contrastive retrieval loss
  → Keeps correct docs ranked above wrong docs

L_entity = EntitySupConLoss(mention_embeddings, entity_labels)
  → Supervised contrastive loss for entity understanding
  → Same entity mentions → cluster together
  → Different entity mentions → separate clusters

L_structure = StructureConsistencyLoss(kg_embeddings)
  → KG embeddings should respect accounting structure
  → Nodes in the same financial statement → cohesive subgraph
  → Cross-company tables → separated subspaces

L_constraint = ConstraintViolationLoss(neg_scores, violates_mask)
  → Existing CHAP constraint loss (unchanged)
```

### 4.2 EntitySupConLoss — The Core Contribution

**Formulation:**

```python
class EntitySupConLoss(nn.Module):
    """
    Supervised Contrastive Loss for entity understanding.

    Given a batch of entity mentions with canonical entity labels:
      Anchor: mention i with entity label L_i
      Positive: mention j with same label L_j = L_i
      Negative: mention k with different label L_k ≠ L_i

    Loss: push anchors close to positives, far from negatives in embedding space.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.t = temperature

    def forward(self, entity_embeds: Tensor, entity_labels: list[str]) -> Tensor:
        """
        Args:
          entity_embeds: [B, embed_dim] — embeddings of entity mentions
          entity_labels: [B] — canonical entity names
        Returns:
          loss: scalar
        """
        # Normalize embeddings
        z = F.normalize(entity_embeds, dim=-1)

        # Compute similarity matrix
        sim = torch.matmul(z, z.T) / self.t  # [B, B]

        # Create positive/negative masks
        labels_tensor = torch.tensor(
            [hash(l) for l in entity_labels],
            device=entity_embeds.device
        )
        pos_mask = (labels_tensor.unsqueeze(0) == labels_tensor.unsqueeze(1)).float()
        pos_mask.fill_diagonal_(0)  # exclude self

        # SupCon loss: log-sum-exp over positives
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=-1, keepdim=True))
        loss = -(pos_mask * log_prob).sum() / (pos_mask.sum() + 1e-8)

        return loss
```

**Why EntitySupConLoss is the right choice:**

```
1. Learns from supervised labels (canonical entity names from metadata)
2. Uses in-batch negatives automatically (different entities in batch)
3. Temperature parameter controls cluster tightness
4. End-to-end differentiable → gradients flow to text encoder
5. Positive pairs: "Apple" ↔ "Apple Inc." → trained to be similar
6. Negative pairs: "Apple" ↔ "Microsoft" → trained to be distant
```

### 4.3 Entity Label Construction

**Where do entity labels come from?**

```
From T²-RAGBench metadata:
  Query: {company_name: "Apple", year: "2023", sector: "Technology"}
  Doc:   {company_name: "Apple Inc.", year: "2023", sector: "Technology"}

  → Both have same entity → POSITIVE pair

From within-batch negatives:
  Batch contains: Apple, Microsoft, Amazon, Google
  → Apple ↔ Microsoft → NEGATIVE pair

Additional data sources:
  1. SEC CIK registry:
     "AAPL" → "Apple Inc."
     "MSFT" → "Microsoft Corporation"
     → Construct pairs: "AAPL" ↔ "Apple Inc." → POSITIVE

  2. Ticker ↔ Name mapping:
     "TSLA" → "Tesla, Inc."
     → "TSLA" ↔ "Tesla, Inc." → POSITIVE
     → "TSLA" ↔ "AAPL" → NEGATIVE (different entities)

Data augmentation:
  → Query mentions: extract company name from question text
  → Document mentions: use metadata company name
  → Pairs: (query_mention, doc_mention, same_entity?)
```

---

## Part V: Why This Is Research, Not Engineering

### 5.1 The Fundamental Distinction

**Engineering approach (what I did before):**
```
Add better features → performance improves
→ Feature engineering
→ No new theoretical understanding
```

**Research approach (what this is):**
```
Change what is being learned → the representations themselves
→ Training objective redesign
→ New theoretical understanding of what "entity understanding" means for retrieval
```

### 5.2 The Novel Contributions

```
Contribution 1 — EntitySupConLoss:
  → First proposed as multi-task loss for financial table retrieval
  → Jointly trains entity understanding with retrieval ranking
  → Theoretical grounding: supervised contrastive learning (Khosla et al., 2020)
  → Novel application: entity understanding for retrieval, not just entity resolution

Contribution 2 — Three-Level Entity Architecture:
  → Token level: entity type embeddings in text encoder
  → Mention level: entity mention encoder with cosine matching
  → Context level: entity-aware KG node features
  → All three levels trained jointly by EntitySupConLoss

Contribution 3 — Entity-Structure Joint Training:
  → Entity understanding + table structure = jointly optimized
  → Ablation: shows entity understanding contributes independently
  → Theoretical claim: entity understanding is necessary for retrieval
```

### 5.3 What This Solves

```
Before:
  Query: "Apple 2023 revenue"
  Doc:   company_name = "Apple Inc."
  → s_entity = 0 (exact match fails)
  → BGE must compensate alone → fragile

After:
  Query: "Apple 2023 revenue"
  Doc:   company_name = "Apple Inc."
  → s_entity = 0.87 (EntitySupConLoss trained)
  → Text encoder knows: "Apple" ≈ "Apple Inc."
  → KG nodes: company embedding clusters same-company cells
  → Constraint scoring: entity context-aware (P3b from earlier)
```

### 5.4 What This Does NOT Solve

```
X — Entity type disambiguation:
  "Apple" (fruit) vs "Apple" (company) vs "Apple" (record label)
  → This requires context beyond company metadata
  → Not in scope for T²-RAGBench (all documents are financial)

X — Unseen entities:
  A company not in the training set
  → EntitySupConLoss requires labeled pairs
  → Cold-start entity resolution not addressed

X — Cross-document coreference:
  "The company" in doc1 refers to "Apple Inc." in doc1
  → Requires coreference resolution, not just entity matching
```

---

## Part VI: Implementation Plan

### 6.1 Minimal Implementation (for ablation)

```
Step 1: Extract entity mentions
  → Query: company name from question text
  → Document: metadata company_name

Step 2: Entity embeddings from text encoder
  → Use BGE encoder (existing)
  → Extract [CLS] or mean-pooled embedding for entity mentions

Step 3: EntitySupConLoss on metadata labels
  → Same company → positive pair
  → Different companies in batch → negative pairs

Step 4: Joint training
  L_total = L_retrieval + λ · L_entity
  → Backprop to text encoder + JointScorer

Step 5: Ablation
  → With vs. without L_entity
  → Quantify entity understanding contribution
```

### 6.2 Extended Implementation (full architecture)

```
Step 6: Entity type embeddings in text encoder
  → Add learnable entity type token
  → Train jointly with SupCon loss

Step 7: Entity-aware KG node features (P3c)
  → Company embedding per node
  → Joint training with structure consistency loss

Step 8: Full ablation suite
  → L_entity only
  → L_structure only
  → L_entity + L_structure
  → L_entity + L_structure + L_retrieval
```

### 6.3 Data Requirements

```
Minimal:
  → T²-RAGBench metadata: company_name in queries and documents
  → ~8K FinQA + ~3K ConvFinQA training samples
  → Within-batch negatives from training batches

Extended:
  → SEC CIK registry (public): ticker ↔ company canonical name
  → Company alias list from Wikipedia infoboxes
  → ~50K company entity pairs for pre-training entity embeddings
```

---

## Part VII: Comparison with Prior Art

| Approach | Entity Representation | Training Objective | Application | Gap |
|---|---|---|---|---|
| BLINK (ACL 2020) | Bi-encoder → KB entities | Pairwise ranking | Wikipedia EL | KB required; not for retrieval |
| CERBERT (ACL 2021) | BERT bi-encoder | Triplet loss | Product matching | No multi-signal fusion |
| Magellan (SIGMOD 2022) | Fine-tuned BERT cross-encoder | BCE | E-commerce EM | Cross-encoder too slow for retrieval |
| Ditto (2023) | LLM prompting | Few-shot generation | Mixed EM | No training signal for retrieval |
| **This (Entity-CACLL)** | Joint entity + retrieval | EntitySupCon + Triplet + Constraint | **Financial table retrieval** | **Novel** |

---

## Summary: The Research Contribution

```
BEFORE (GSR-CACL):
  Entity = exact string match on metadata
  → No learning signal for entity understanding
  → BGE must compensate alone
  → Fragile: "Apple" ≠ "Apple Inc."

AFTER (Entity-CACLL):
  Entity = learned embedding space via EntitySupConLoss
  → Text encoder learns: "Apple" ≈ "Apple Inc."
  → JointScorer has entity-aware representations
  → KG nodes carry company context
  → All three trained jointly

CORE NOVELTY:
  → EntitySupConLoss: supervised contrastive loss for entity understanding
  → First applied to multi-signal financial table retrieval
  → Jointly optimizes retrieval + entity + structure
  → Theoretically grounded (SupCon, Khosla et al., 2020)
  → Empirically validated (ablation)

RESEARCH vs. ENGINEERING:
  → Engineering: "reuse BGE for entity matching" (my earlier attempt)
  → Research: "train entity understanding as a learning objective"
  → This changes WHAT the model learns, not just HOW it scores
```
