# GSR-CACL: Architecture Diagrams

> **Notation convention.** All symbols match the paper text:
> $Q$ = query, $D$ = document, $\mathbf{q}$ = query embedding,
> $\mathbf{d}_\text{text}$ = doc text embedding, $\mathbf{d}_\text{KG}$ = graph embedding,
> $G_D = (\mathcal{V}, \mathcal{E}, \omega)$ = constraint KG,
> $\oplus$ = concatenation, $\otimes$ = element-wise product.

---

## Figure 1 — GSR-CACL Framework

> **Main figure.** Two contributions: C1 (Graph-Structured Retrieval) operates at inference; C2 (Constraint-Aware Contrastive Learning) operates at training. Both share the Joint Scorer $\phi_\theta$.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontFamily': 'Helvetica, Arial, sans-serif', 'fontSize': '13px', 'lineColor': '#64748b'}}}%%
flowchart LR
    classDef io     fill:#f9fafb,stroke:#6b7280,color:#1f2937,stroke-width:1.5px,rx:3
    classDef enc    fill:#eff6ff,stroke:#2563eb,color:#1e3a5f,stroke-width:1.5px,rx:3
    classDef kg     fill:#f0fdf4,stroke:#16a34a,color:#14532d,stroke-width:1.5px,rx:3
    classDef score  fill:#faf5ff,stroke:#7c3aed,color:#4c1d95,stroke-width:1.5px,rx:3
    classDef train  fill:#fef2f2,stroke:#dc2626,color:#7f1d1d,stroke-width:1.5px,stroke-dasharray:5 5,rx:3

    Q(["Q, meta_Q"]):::io
    D(["D = (text, table, meta_D)"]):::io

    Q  -->  EQ["f_θ  Text Encoder"]:::enc
    D  -->  ED["f_θ  Text Encoder"]:::enc
    D  -->  KG["KG\nConstruction\n(§2)"]:::kg

    KG -->|"G_D"| GAT["GAT ×2"]:::kg
    GAT -->|"d_KG ∈ ℝʰ"| JS

    EQ -->|"q ∈ ℝᵈ"| JS["Scorer φ_θ\ns = α·s_text + β·s_ent + γ·CS"]:::score
    ED -->|"d_text ∈ ℝᵈ"| JS

    JS --> TOP(["Top-K"]):::io

    subgraph C2 [" C2 · CACL  — training only "]
        direction LR
        CHAP["CHAP"]:::train
        LCACL["ℒ_CACL"]:::train
        CHAP -->|"C⁻"| LCACL
    end

    D -.-> CHAP
    LCACL -.->|"∇_θ"| JS
```

---

## Figure 2 — Constraint KG Construction

> **Novel.** A financial table is parsed, matched against a library $\mathcal{T}$ of 15 IFRS/GAAP templates, and converted into a constraint knowledge graph $G_D = (\mathcal{V}, \mathcal{E}, \omega)$. Edges carry accounting semantics: $\omega = +1$ (additive), $\omega = -1$ (subtractive), $\omega = 0$ (positional fallback).

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontFamily': 'Helvetica, Arial, sans-serif', 'fontSize': '13px', 'lineColor': '#64748b'}}}%%
flowchart TD
    classDef io    fill:#f9fafb,stroke:#6b7280,color:#1f2937,stroke-width:1.5px,rx:3
    classDef proc  fill:#eff6ff,stroke:#2563eb,color:#1e3a5f,stroke-width:1.5px,rx:3
    classDef kg    fill:#f0fdf4,stroke:#16a34a,color:#14532d,stroke-width:1.5px,rx:3
    classDef cond  fill:#fffbeb,stroke:#d97706,color:#78350f,stroke-width:1.5px,rx:3

    TABLE(["Financial Table\n(markdown)"]):::io

    TABLE --> PARSE["Parse rows\nheaders + cells"]:::proc
    PARSE --> MATCH["Template Match\nconf = |H_table ∩ H_tmpl| / max(|H|)"]:::proc

    MATCH --> CHECK{"conf ≥ 0.5 ?"}:::cond

    CHECK -->|"Yes"| ACCT["Accounting edges\nω ∈ {+1, −1}\nfrom template constraints"]:::kg
    CHECK -->|"No"| POS["Positional edges\nω = 0\nsame-row / same-col"]:::kg

    subgraph GD [" G_D = (V, E, ω) "]
        direction LR
        V["V : one node per cell\n(value, row, col, header)"]:::kg
        E["E : ω-labeled edges"]:::kg
    end

    ACCT --> GD
    POS  --> GD
    PARSE --> V
```

---

## Figure 3 — GAT Encoder

> Node features are constructed by concatenating a cell embedding with sinusoidal positional encodings, then projected and refined through $L = 2$ edge-aware GAT layers. The graph-level representation $\mathbf{d}_\text{KG}$ is obtained by mean pooling.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontFamily': 'Helvetica, Arial, sans-serif', 'fontSize': '13px', 'lineColor': '#64748b'}}}%%
flowchart LR
    classDef io    fill:#f9fafb,stroke:#6b7280,color:#1f2937,stroke-width:1.5px,rx:3
    classDef feat  fill:#eff6ff,stroke:#2563eb,color:#1e3a5f,stroke-width:1.5px,rx:3
    classDef gat   fill:#f0fdf4,stroke:#16a34a,color:#14532d,stroke-width:1.5px,rx:3
    classDef out   fill:#faf5ff,stroke:#7c3aed,color:#4c1d95,stroke-width:1.5px,rx:3

    GD(["G_D"]):::io

    subgraph FEAT [" Node Features "]
        direction TB
        CELL["x_cell ∈ ℝ⁷⁶⁸"]:::feat
        ROW["PE_row ∈ ℝ¹⁹²"]:::feat
        COL["PE_col ∈ ℝ¹⁹²"]:::feat
        CONCAT["⊕   →  ℝ¹¹⁵²"]:::feat
        CELL --> CONCAT
        ROW  --> CONCAT
        COL  --> CONCAT
    end

    GD --> FEAT

    CONCAT -->|"W_proj"| H0["h⁰ ∈ ℝ²⁵⁶\nLayerNorm · ReLU"]:::gat

    subgraph GATL [" GAT ×L  (L=2, H=4 heads) "]
        direction LR
        L1["GATLayer¹"]:::gat
        L2["GATLayer²"]:::gat
        L1 --> L2
    end

    H0 --> L1
    L2 -->|"mean pool"| DKG(["d_KG ∈ ℝ²⁵⁶"]):::out
```

---

## Figure 3b — Edge-Aware Attention (detail of one GATLayer)

> Each head computes attention biased by the constraint weight $\omega$, ensuring the network respects accounting identities.
>
> $$e_{uv}^{(h)} = \frac{\langle W_q \mathbf{h}_u,\; W_k \mathbf{h}_v \rangle}{\sqrt{d_h}} + \text{Proj}(\omega_{uv})$$
> $$\mathbf{h}_v' = W_o \Big[\,\big\|_{h=1}^{H}\; \sum_{u \in \mathcal{N}(v)} \alpha_{uv}^{(h)} \cdot \omega_{uv} \cdot W_v^{(h)} \mathbf{h}_u \Big]$$

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontFamily': 'Helvetica, Arial, sans-serif', 'fontSize': '13px', 'lineColor': '#64748b'}}}%%
flowchart LR
    classDef io    fill:#f9fafb,stroke:#6b7280,color:#1f2937,stroke-width:1.5px,rx:3
    classDef attn  fill:#faf5ff,stroke:#7c3aed,color:#4c1d95,stroke-width:1.5px,rx:3
    classDef msg   fill:#eff6ff,stroke:#2563eb,color:#1e3a5f,stroke-width:1.5px,rx:3

    HU(["h_u"]):::io
    HV(["h_v"]):::io
    W(["ω_uv"]):::io

    HU -->|"W_q"| DOT["⟨Q, K⟩ / √d_h"]:::attn
    HV -->|"W_k"| DOT
    W  --> BIAS["Proj(ω)"]:::attn
    DOT --> ADD["⊕"]:::attn
    BIAS --> ADD
    ADD -->|"softmax"| ALPHA["α_uv"]:::attn

    HU -->|"W_v"| VAL["V·h_u"]:::msg
    ALPHA --> MUL["⊗  α · ω · Vh_u"]:::msg
    W --> MUL
    VAL --> MUL

    MUL -->|"Σ_N(v)"| AGG["W_o · concat heads"]:::msg
    AGG --> OUT(["h_v'"]):::io
```

---

## Figure 4 — Joint Scorer $\phi_\theta$

> Three complementary signals — textual, entity, and structural — are combined via learned positive weights $(\alpha, \beta, \gamma)$ constrained through softplus.
>
> $$s(Q, D) = \alpha \cdot s_\text{text} + \beta \cdot s_\text{ent} + \gamma \cdot \text{CS}(G_D)$$

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontFamily': 'Helvetica, Arial, sans-serif', 'fontSize': '13px', 'lineColor': '#64748b'}}}%%
flowchart TD
    classDef io    fill:#f9fafb,stroke:#6b7280,color:#1f2937,stroke-width:1.5px,rx:3
    classDef sig   fill:#eff6ff,stroke:#2563eb,color:#1e3a5f,stroke-width:1.5px,rx:3
    classDef out   fill:#faf5ff,stroke:#7c3aed,color:#4c1d95,stroke-width:2px,rx:3

    Q(["q ∈ ℝᵈ"]):::io
    DT(["d_text ∈ ℝᵈ"]):::io
    DK(["d_KG ∈ ℝʰ"]):::io
    META(["meta_Q , meta_D"]):::io

    Q  --> STEXT["s_text = cos(q, d_text)"]:::sig
    DT --> STEXT

    META --> SENT["s_ent = ⅓ Σ 𝟙[m_Q = m_D]"]:::sig

    DK --> SCS["CS(G_D) = Σ exp(−|ω·v_u − v_v| / max(|v_v|, ε))"]:::sig

    STEXT -->|"× α"| COMB["s(Q, D) = α · s_text + β · s_ent + γ · CS"]:::out
    SENT  -->|"× β"| COMB
    SCS   -->|"× γ"| COMB
```

---

## Figure 5 — CHAP Negative Sampler

> CHAP creates hard negatives $C^-$ from the positive $C^+$ by violating exactly one accounting identity. Three perturbation types target different invariants, producing samples that are textually similar but structurally broken.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontFamily': 'Helvetica, Arial, sans-serif', 'fontSize': '13px', 'lineColor': '#64748b'}}}%%
flowchart LR
    classDef io    fill:#f9fafb,stroke:#6b7280,color:#1f2937,stroke-width:1.5px,rx:3
    classDef typeA fill:#eff6ff,stroke:#2563eb,color:#1e3a5f,stroke-width:1.5px,rx:3
    classDef typeS fill:#f0fdf4,stroke:#16a34a,color:#14532d,stroke-width:1.5px,rx:3
    classDef typeE fill:#faf5ff,stroke:#7c3aed,color:#4c1d95,stroke-width:1.5px,rx:3
    classDef neg   fill:#fef2f2,stroke:#dc2626,color:#7f1d1d,stroke-width:2px,rx:3

    POS(["C⁺"]):::io

    POS -->|"p=0.5"| A["A · Additive\nperturb one cell\nΣ identity breaks"]:::typeA
    POS -->|"p=0.3"| S["S · Scale\nchange magnitude\n×10³ or ×10⁻³"]:::typeS
    POS -->|"p=0.2"| E["E · Entity\nswap company / year"]:::typeE

    A --> NEG(["C⁻"]):::neg
    S --> NEG
    E --> NEG
```

---

## Figure 6 — CACL Training Objective

> The full loss $\mathcal{L}_\text{CACL}$ combines a margin-based triplet term (push $s^+ > s^- + m$) with a constraint violation penalty (suppress scores of broken documents).
>
> $$\mathcal{L}_\text{CACL} = \mathcal{L}_\text{triplet} + \lambda \cdot \mathcal{L}_\text{constraint}$$

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontFamily': 'Helvetica, Arial, sans-serif', 'fontSize': '13px', 'lineColor': '#64748b'}}}%%
flowchart LR
    classDef io    fill:#f9fafb,stroke:#6b7280,color:#1f2937,stroke-width:1.5px,rx:3
    classDef model fill:#eff6ff,stroke:#2563eb,color:#1e3a5f,stroke-width:1.5px,rx:3
    classDef loss  fill:#fef2f2,stroke:#dc2626,color:#7f1d1d,stroke-width:1.5px,rx:3
    classDef total fill:#fef2f2,stroke:#dc2626,color:#7f1d1d,stroke-width:2.5px,rx:3

    T(["(Q, C⁺, C⁻)"]):::io

    T --> SC["φ_θ\ns⁺ = s(Q, C⁺)\ns⁻ = s(Q, C⁻)"]:::model

    SC --> LT["ℒ_trip\nmax(0, m − s⁺ + s⁻)"]:::loss
    SC --> LC["ℒ_con\n−log σ(−s⁻) · 𝟙[violated]"]:::loss

    LT -->|"1"| LCACL["ℒ_CACL"]:::total
    LC -->|"λ"| LCACL

    LCACL -.->|"∇_θ"| SC
```

---

## Figure 7 — Three-Stage Curriculum Training

> Weights are transferred sequentially. Each stage provides a progressively stronger initialisation for the next.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontFamily': 'Helvetica, Arial, sans-serif', 'fontSize': '13px', 'lineColor': '#64748b'}}}%%
flowchart LR
    classDef stage fill:#fffbeb,stroke:#d97706,color:#78350f,stroke-width:1.5px,rx:3
    classDef out   fill:#faf5ff,stroke:#7c3aed,color:#4c1d95,stroke-width:2px,rx:3

    S1["Stage 1\nIdentity\nlearn entity discrimination"]:::stage
    S2["Stage 2\nStructural\ncalibrate CS ≈ 1"]:::stage
    S3["Stage 3\nJoint CACL\nfull contrastive + CHAP"]:::stage
    OUT(["θ*"]):::out

    S1 -->|"θ₁"| S2 -->|"θ₂"| S3 -->|"θ₃"| OUT
```
