# Critical Analysis: GSR-CACL Research Contributions

> **Purpose:** Evaluate the current contributions honestly, identify what is engineering vs. research, and propose three substantively different directions for strengthening the methodology.

---

## Part I: What We Actually Have

### 1.1 The GSR Contribution — What Is Novel, What Is Not

**The claim:**
> "We propose Graph-Structured Retrieval (GSR): represent financial tables as Constraint Knowledge Graphs, encode them with GAT, score with JointScorer."

**What is actually novel:**

```
Constraint Knowledge Graph
  → Nodes: table cells (as in any graph representation of a table)
  → Edges: accounting identities (Revenue − COGS = Gross Profit)

This IS novel for retrieval:
  - No prior work uses accounting constraint edges for document retrieval
  - Edges encode domain knowledge (IFRS/GAAP) into the retrieval graph
```

**What is NOT novel:**

```
GAT (Graph Attention Network) — Velickovic et al., ICLR 2018
JointScorer (weighted combination of signals) — standard practice
Template matching for table parsing — standard practice
```

**The critical gap:**

The accounting edges are **hard-coded from templates**. The GAT does not learn accounting structure — it aggregates node features through pre-defined edges. The architecture is:

```
Template → ω ∈ {+1, −1, 0} for each edge  ← DOMAIN KNOWLEDGE, HARDCODED
    ↓
GAT aggregates node features through these edges
    ↓
Does GAT learn "Revenue + COGS = Gross Profit"?
→ NO. The equation is already encoded in ω.
→ GAT learns: "how much does node A influence node B given their features"
```

**The honest framing:**

```
GSR = "we inject accounting domain knowledge via hard-coded template edges,
       then use a GAT to learn feature aggregation through those edges"
```

This is a reasonable architecture. But the research contribution is **domain knowledge injection**, not **learned accounting reasoning**.

---

### 1.2 The CACL Contribution — What Is Novel, What Is Not

**The claim:**
> "We propose Constraint-Aware Contrastive Learning (CACL): CHAP negative sampling + CACL loss + 3-stage curriculum."

**What is actually novel:**

```
CHAP (Contrastive Hard-negative via Accounting Perturbations):
  → Perturb cell values to BREAK accounting identities
  → Creates hard negatives that are lexically similar but numerically wrong
  → This is genuinely creative and justified

Zero-Sum property:
  → Exactly one component changed per perturbation
  → Guarantees constraint violation while maintaining surface similarity
```

**What is NOT novel:**

```
TripletLoss — standard since 2015 (FaceNet)
3-stage curriculum — common practice (pre-training → fine-tuning)
Margin-based contrastive learning — standard since BERT era
```

**The critical gap — Loss analysis:**

The training loop (Stage 3):

```python
Loss = TripletLoss(pos_scores, neg_scores)
     + λ × ConstraintViolationLoss(neg_scores, violates_mask)
```

**What this loss optimizes:**

```
→ "Rank correct docs above wrong docs" (TripletLoss)
→ "Constraint-violating docs should score low" (ConstraintViolationLoss)

→ What it does NOT optimize:
→ "Represent table structure better"
→ "Encode accounting relationships more accurately"
→ "Understand entity relationships"
```

**The honest framing:**

```
CACL = TripletLoss + CHAP negatives + 3-stage curriculum

CHAP negatives are novel and justified.
The loss is a standard contrastive retrieval loss.
```

The loss trains the model to **rank correctly**, not to **understand tables better**. These are different objectives.

---

### 1.3 The Real Research Gap

**Three independent sub-problems in financial table retrieval:**

```
Sub-problem A: Entity Understanding
  "Apple Inc." = "Apple" = "AAPL" = "Apple Computer, Inc."
  Current: hash matching → 0 signal
  What is needed: learned entity representations

Sub-problem B: Table Structure Representation
  Tables have hierarchical structure: line items → subtotals → totals
  Current: GAT aggregates features through hard-coded edges
  What is needed: model learns structure representation

Sub-problem C: Constraint-Aware Reasoning
  Accounting identities are constraints on the representation space
  Current: constraints used only for scoring (post-hoc)
  What is needed: constraints guide representation learning
```

**Each sub-problem requires a different type of learning signal.**

---

## Part II: What We Built — Engineering vs. Research

### 2.1 P0: Scale-Aware Numeric Encoding — ENGINEERING

**What it does:**
- Replace `[log|v|, sign, is_zero, bucket]` with `[magnitude_bin, mantissa_bin, unit]`

**Why it is engineering:**

```
No theoretical justification:
  → Why 24 magnitude bins? Why not 20 or 30?
  → Why 20 mantissa bins? Why not 10 or 40?
  → These are arbitrary hyperparameter choices

No learning objective change:
  → Same loss function
  → Same gradient flow
  → Only changes input features to the GAT

No research contribution:
  → "We tried more numeric features" = ablation, not method
```

**The right question to ask:**

```
Does the GAT need better numeric features, or does it need a DIFFERENT way
to learn numerical relationships?

If the GAT already aggregates numeric features through accounting edges,
why would better numeric features change the representation?

Answer: They wouldn't, unless the GAT is actually LEARNING accounting relationships,
not just aggregating through them.
```

---

### 2.2 P2: Adaptive ε — ENGINEERING

**What it does:**
- Change `exp(-residual / max(|v|, ε))` to `exp(-residual / (|v| × rel_tol))`

**Why it is engineering:**

```
rel_tol = 1e-3 is chosen by intuition, not theory.

Why 0.1% relative tolerance?
  → Financial materiality thresholds vary by company, by jurisdiction
  → 0.1% is reasonable but not justified

No theoretical framework:
  → Where does 1e-3 come from?
  → Is this optimal for all constraint types?
  → Should it be learned, not fixed?
```

**The right question to ask:**

```
Is the constraint satisfaction score differentiable and learnable,
or is it a fixed heuristic?

If it is learnable → should be trained via loss, not hand-tuned
If it is fixed → why use it in a learned system at all?
```

---

### 2.3 P3a/b/c: BGE Soft Entity Matching — ENGINEERING

**What it does:**
- Replace exact string match with BGE cosine similarity

**Why it is engineering:**

```
BGE is pre-trained for semantic similarity (sentence-level, general domain).
It is NOT trained for entity resolution in financial text.

Using BGE for entity matching:
  → Re-purposes a general semantic model for a specific task
  → No training signal specifically for entity understanding
  → No entity-specific loss

The BGE entity representation is trained on:
  → General text (Wikipedia, Web)
  → Not financial SEC filings
  → Not company name variations
```

**What is actually needed:**

```
An ENTITY CONTRASTIVE LOSS:
  Positive: "Apple Inc." ↔ "Apple" ↔ "AAPL" → close in embedding space
  Negative: "Apple Inc." ↔ "Microsoft Corp." → distant in embedding space

This is a DIFFERENT training objective from retrieval TripletLoss.
BGE cosine similarity without entity-specific training is weak signal.
```

---

### 2.4 Summary: Engineering vs. Research

| Component | Type | Problem |
|---|---|---|
| GSR: KG + GAT + JointScorer | Research | Edge weights hard-coded; GAT doesn't learn accounting |
| CACL: CHAP + TripletLoss | Research | Loss optimizes ranking, not representation |
| CACL: 3-stage curriculum | Engineering | Heuristic; no theory for why 3 stages |
| P0: Scale-aware encoding | Engineering | Arbitrary feature choices |
| P2: Adaptive ε | Engineering | Arbitrary tolerance value |
| P3a: BGE entity match | Engineering | Wrong training objective for entity |
| P3b: Entity-aware constraint | Engineering | Gating, not learning signal |
| P3c: Entity node features | Engineering | Isolated embeddings, no alignment loss |

---

## Part III: Three Research Directions

### Direction A — Strengthen the Learning Objective

**Core idea:** The current loss optimizes retrieval ranking, not table understanding. Add training objectives that explicitly improve representations.

**Theoretical foundation:**

```
Information Retrieval Learning (Järel et al., 2020):
  Contrastive loss ≠ retrieval loss
  → Contrastive: learn a representation space
  → Retrieval: rank correctly

Multi-task learning theory (Caruana, 1997):
  → Multiple objectives → better representations if tasks are related
  → Entity understanding + retrieval ranking = related tasks

Loss composition theory (Kendall et al., 2018):
  → Multi-task loss should be learned, not hand-weighted
```

**What this direction proposes:**

```
Loss = RetrievalTripletLoss
     + α · EntityContrastiveLoss      ← NEW: entity representations
     + β · StructuralConsistencyLoss  ← NEW: table structure
     + γ · ConstraintAwareLoss        ← NEW: constraint reasoning

Where:
  EntityContrastiveLoss:
    → Anchor: (Q_company, D_company) positive pairs
    → Positive: same canonical entity
    → Negative: different entity
    → Objective: same entity → close, different → far

  StructuralConsistencyLoss:
    → For each KG: consistency between node features and edge structure
    → Nodes connected by ω=+1 edge should have consistent features
    → Nodes connected by ω=-1 edge should have predictable feature difference
    → Objective: GAT representations should respect accounting structure

  ConstraintAwareLoss:
    → For positive docs: constraint satisfaction → 1.0
    → Gradient flows through constraint satisfaction into GAT
    → Model learns: representations that satisfy constraints are better
```

**Why this is research:**

```
1. EntityContrastiveLoss:
   → First proposed as multi-task loss for financial entity retrieval
   → Not in prior work on T²-RAGBench

2. StructuralConsistencyLoss:
   → Directly addresses the gap: GAT learns accounting structure
   → Based on graph representation learning theory
   → Novel application to financial table domain

3. Loss composition:
   → Learned loss weights (Kendall et al.) vs. hand-tuned λ
   → Adaptive weighting across tasks
```

**Novelty claim:**

> "We propose Multi-Task CACL (MT-CACL): extending the CACL loss with entity contrastive and structural consistency objectives, trained jointly with learned loss weights. This addresses the gap where the original CACL loss optimizes retrieval ranking but not table representation quality."

---

### Direction B — Strengthen the Architecture

**Core idea:** GAT aggregates through hard-coded edges. Make the architecture learn to represent table structure.

**Theoretical foundation:**

```
Relational Graph Convolutional Networks (R-GCN, Schlichtkrull et al., 2018):
  → Edge-type-specific message passing
  → Different W for different relation types
  → Learns relational structure from data

Heterogeneous Graph Transformer (HGT, Hu et al., 2020):
  → Microsoft research
  → Type-aware attention across heterogeneous node/edge types
  → State-of-the-art for knowledge graph reasoning

Inductive Logic Programming meets GNN (ILP-GNN, 2020-2023):
  → Logic rules as supervision for graph representations
  → Accounting identities as first-order logic rules
  → GNN learns to respect logical constraints
```

**What this direction proposes:**

```
Option B1: Heterogeneous GAT (HGT-style)
  → Different W for different constraint types:
    - w_additive: for +1 edges (Revenue + COGS = GP)
    - w_subtractive: for −1 edges (Revenue − COGS = GP)
    - w_ratio: for ratio constraints (NI / Revenue = Margin)
    - w_positional: for positional edges
  → GAT learns: "what does an additive relationship look like?"
  → NOT hard-coded: ω tells WHICH relationship, GAT learns HOW it behaves

Option B2: Neuro-Symbolic GAT
  → Accounting identities as logic constraints
  → Message passing is modulated by constraint satisfaction
  → If edge is satisfied: normal aggregation
  → If edge is violated: down-weight aggregation
  → Architecture itself enforces accounting consistency

Option B3: Cross-Modal GAT
  → Text tokens attend to table cells
  → Table cells attend to text tokens
  → Query-aware table representation
  → NOT pre-GAT: fusion happens inside the GAT
```

**Why this is research:**

```
Option B1:
  → First application of HGT to financial table representation
  → Learns accounting relationships from data, not hard-coded
  → Novel: domain-specific relational structure learning

Option B2:
  → Novel: neuro-symbolic integration for financial tables
  → Different from prior work: constraints guide representation, not just scoring
  → Based on ILP-GNN literature

Option B3:
  → Cross-modal attention is established (VL models, table QA)
  → Novel: query-aware table encoding for RETRIEVAL (not QA)
  → Different from TaBERT: fusion before final score, not for QA generation
```

**Novelty claim:**

> "We propose Relation-Aware GAT (RA-GAT): replacing homogeneous GAT message passing with heterogeneous message passing where each relation type (accounting additive, subtractive, ratio, positional) has a learned transformation matrix. Unlike hard-coded ω weights in GSR, RA-GAT learns the semantic behavior of each relation type from data."

---

### Direction C — Rebuild from Problem Framing

**Core idea:** The contributions as framed (GSR + CACL) are a retrieval system. The research framing should be about **table understanding for retrieval**, not a retrieval system with table features.

**What the problem actually is:**

```
T²-RAGBench = "Given a numerical question, retrieve the correct table"

Two sub-problems:
  1. TABLE UNDERSTANDING: What does this table represent?
     - What companies/entities?
     - What is the structure (income statement, balance sheet)?
     - What are the numerical relationships?

  2. SEMANTIC MATCHING: Does this table answer the question?
     - Is the question about this company?
     - Does the table contain the relevant information?
     - Are the numbers consistent?

GSR + CACL address BOTH, but:
  → GSR addresses TABLE UNDERSTANDING (via KG + GAT)
  → CACL addresses SEMANTIC MATCHING (via contrastive learning)
  → BUT: these are trained independently
```

**The research gap:**

```
Information Bottleneck for Multi-Signal Retrieval:
  → Each signal (text, entity, constraint) carries different information
  → Information-theoretic analysis: are signals redundant?
  → Optimal fusion: from theory, not heuristics

End-to-End Table Understanding for Retrieval:
  → Current: parse → KG → GAT → score → rank (pipeline)
  → Missing: joint optimization of understanding + matching
  → TaBERT-style: table + text → unified representation
  → Novel: for RETRIEVAL, not QA
```

**What this direction proposes:**

```
Problem reframing:
  "We study TABLE UNDERSTANDING for NUMERICAL RETRIEVAL.
   Not: 'how to retrieve with table features'
   But: 'what does it mean to understand a table for retrieval?'"

Framework:
  1. Define "table understanding" as:
     → Entity resolution: what entities does this table describe?
     → Structure recognition: what accounting template?
     → Constraint satisfaction: are numerical relationships valid?

  2. Show that:
     → GSR = explicit encoding of structure recognition
     → CACL = training objective for entity + constraint understanding
     → Together = complete table understanding pipeline

  3. Novel contribution:
     → Theoretical framework connecting table understanding and retrieval
     → GSR-CACL as an INSTANCE of this framework
     → Ablation as evidence for framework validity
```

**Novelty claim:**

> "We propose a Table Understanding for Retrieval (TUR) framework: a theoretical and empirical analysis of what it means for a retrieval system to 'understand' a financial table. We show that GSR encodes structural understanding, CACL trains semantic understanding, and their combination achieves state-of-the-art by jointly optimizing both. This reframes the contribution from 'a better retrieval system' to 'a theory of table understanding for retrieval.'"

---

## Part IV: Decision Matrix

| Direction | Novelty | Effort | Risk | Research Quality |
|---|---|---|---|---|
| **A: Strengthen Loss** | Medium-High | Medium | Low | High — theoretically grounded, clear ablation |
| **B: Strengthen Architecture** | High | High | Medium | Medium-High — needs extensive ablation |
| **C: Rebuild Framing** | High | Low | Low | High — reframing is itself a contribution |

### Recommendation

**Direction A is the most feasible and highest-impact short-term improvement:**

```
Reasoning:
  1. CHAP negatives are already the right training signal
  2. Just need to ADD entity + structural objectives
  3. Existing architecture (GAT + JointScorer) stays intact
  4. Minimal code changes: new loss terms, same training loop
  5. Clear ablation: original CACL loss vs. MT-CACL loss
  6. Strong theoretical foundation (multi-task learning)

Direction B is the most impactful long-term:
  1. HGT-style architecture is genuinely novel
  2. Directly addresses the GAT limitation
  3. But: high effort, needs extensive re-implementation

Direction C is the most important for paper quality:
  1. Reframing elevates the contribution
  2. Without it: "we built a better retrieval system"
  3. With it: "we study table understanding for retrieval"
  4. This is the difference between a systems paper and a research paper
```

**The minimum viable research contribution:**

```
If we only do Direction C (reframing):
  → Current GSR + CACL is presented as a THEORY of table understanding
  → Ablation shows each component contributes to understanding
  → This is publishable if the experiments support the framing

If we do Direction A on top:
  → Stronger ablation: with/without entity loss, with/without structural loss
  → More compelling evidence for the theory
  → Higher quality publication venue
```

---

## Part V: What Must Be Done Regardless of Direction

### 5.1 Ablation Requirements

Any reviewer will ask: "What does each component actually contribute?"

```
Mandatory ablation:
  1. GSR only (no CACL training) vs. CACL only (GSR architecture)
     → Quantify contribution of architecture vs. learning

  2. Each negative type separately (CHAP-A vs CHAP-S vs CHAP-E)
     → CHAP-E (entity swap) should help most for entity understanding

  3. 3-stage curriculum ablation (1+2+3 vs. 1+3 vs. 3 only)
     → Is Stage 2 (structural) actually necessary?

  4. Constraint signal ablation (with vs. without constraint scoring)
     → Does accounting constraint signal matter?
     → Or is text similarity sufficient?
```

### 5.2 Theoretical Justification Requirements

For any improvement to be research (not engineering):

```
→ MUST answer: Why this approach, not another?
→ MUST cite: Theoretical framework or prior empirical evidence
→ MUST validate: Ablation or analysis showing the theory matches practice
→ MUST acknowledge: Limitations and when the approach fails
```

### 5.3 The Fundamental Question

Before any implementation, answer:

```
What is the ONE thing that current GSR + CACL cannot do,
that the improvement enables?

If the answer is:
  → "It can't distinguish Apple Inc. from Apple"
  → Then: Direction A (entity loss) is right

  → "The GAT doesn't learn accounting structure"
  → Then: Direction B (HGT architecture) is right

  → "The paper frames this as retrieval, not understanding"
  → Then: Direction C (reframing) is right

If the answer is: "it doesn't perform well enough"
→ Then: This is engineering, not research.
```

---

## Summary: Research vs. Engineering

| | Engineering | Research |
|---|---|---|
| **Problem** | "Doesn't work well enough" | "What does it mean to understand a table?" |
| **Solution** | Better features / parameters | New theoretical framework |
| **Justification** | Empirical (ablation) | Theoretical + Empirical |
| **Contribution** | Improved numbers | New understanding |
| **GSR + CACL** | Competitive benchmark numbers | Theory of table understanding for retrieval |
| **P0/P2/P3** | Engineering (better features) | Direction A/B/C (theoretically grounded) |
