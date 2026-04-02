# Structured Knowledge-Enhanced Retrieval for Financial Documents

---

## 1. Đặt vấn đề

### 1.1. Bài toán

Retrieval-Augmented Generation (RAG) trên tài liệu tài chính là bài toán truy xuất đoạn context (chứa cả văn bản lẫn bảng số liệu) phù hợp nhất với một câu hỏi tài chính, phục vụ cho bước sinh câu trả lời phía sau. Đây là bài toán quan trọng vì hầu hết thông tin tài chính nằm trong báo cáo hàng năm, báo cáo quý — nơi dữ liệu được trình bày dưới dạng hỗn hợp text + table.

Trong thiết lập **unknown-context** (câu hỏi không chỉ rõ context nào), hệ thống cần tìm đúng context từ hàng trăm đến hàng nghìn documents trong corpus. Đây là nơi các retriever hiện tại gặp khó.

### 1.2. Scope

Chúng tôi tập trung vào bài toán **retrieval** (truy xuất), không phải generation (sinh câu trả lời). Cụ thể:

- **Đầu vào:** câu hỏi tài chính $Q$ + corpus gồm $N$ documents $\{D_1, ..., D_N\}$, mỗi document chứa text narrative + markdown table + metadata (tên công ty, năm báo cáo, ngành).
- **Đầu ra:** top-$K$ documents liên quan nhất.
- **Metric:** MRR@3, Recall@1/3/5, NDCG@3 — chuẩn hoá theo T²-RAGBench (EACL 2026).

### 1.3. Khoảng cách hiện tại

Các hệ thống retrieval tốt nhất hiện tại (Hybrid BM25, dense retrievers) đạt khoảng 35% MRR@3 trên benchmark tài chính — kém Oracle Context (biết trước context đúng) tới **30 điểm phần trăm**. Điều này cho thấy standard retrieval chưa khai thác được bản chất đặc thù của tài liệu tài chính.

---

## 2. Phân tích bản chất vấn đề

### 2.1. Bốn hiện tượng cốt lõi

Tài liệu tài chính không phải văn bản tự nhiên thông thường. Chúng là hệ thống mã hoá tuân theo luật ngôn ngữ đặc thù + ràng buộc toán học bất biến. Bốn hiện tượng sau khiến standard retrieval thất bại:

**Hiện tượng 1: Lexical Overlap Illusion (Ảo giác trùng từ vựng).** Các document từ cùng một công ty có mức tương đồng từ vựng rất cao (gấp nhiều lần so với document từ công ty khác). Hệ quả: trong phần lớn trường hợp, context *sai* lại có điểm similarity cao hơn context *đúng* — vì chúng dùng chung thuật ngữ, cùng công ty, nhưng khác năm hoặc khác bảng.

**Hiện tượng 2: Mathematical Inconsistency (Mất nhất quán toán học).** Tài liệu tài chính bị ràng buộc bởi các phương trình kế toán ẩn:

$$\text{Revenue} - \text{COGS} = \text{Gross Profit}$$
$$\text{Assets} = \text{Liabilities} + \text{Equity}$$

Khi bảng bị "flatten" thành văn bản phẳng, quan hệ cha-con giữa các ô số bị phá hủy. Retriever không thể phân biệt được context nào thỏa mãn ràng buộc toán học liên quan đến câu hỏi.

**Hiện tượng 3: Semantic Shock (Bất ngờ ngữ nghĩa).** Trong gần nửa số trường hợp, query match tốt với phần text narrative nhưng câu trả lời thực sự nằm trong table. Dense retriever dựa trên embedding văn bản sẽ bị "hút" về text narrative — bỏ lỡ thông tin bảng.

**Hiện tượng 4: Numerical Density Paradox (Nghịch lý mật độ số).** Hầu hết queries có tất cả con số cần thiết nằm trong context đúng. Nhưng cùng con số đó cũng xuất hiện trong hàng chục context khác — số liệu vừa là manh mối vừa là bẫy cho retriever.

### 2.2. Bốn định luật ngôn ngữ tài chính

**Law of Formulaic Encoding** (Halliday, 1994): Ngữ nghĩa không nằm ở từ ngữ, mà ở **vị trí trong cấu trúc**. Mode (mixed text + tabular) bị phá huỷ khi flatten.

**Law of Lexical Compression** (Shannon, 1948): Entropy/token trong specialized register thấp hơn văn bản thông thường — từ chuyên ngành có xác suất cố định (Zipf's Law deviation).

**Law of Numerical Anchoring:** Con số là điểm cố định — thay đổi số neo → ngữ nghĩa bị huỷ bỏ.

**Law of Multi-layered Semantics:** Một context chứa trung bình nhiều câu hỏi khác nhau → biểu diễn phải đa diện, không thể dùng một vector đơn.

### 2.3. Ba giả thuyết nghiên cứu

**H1 (Consistency):** Retrieval vượt trội khi bảo toàn tính tự hợp toán học — tức là khi hệ thống kiểm tra được constraint satisfaction của context.

**H2 (Structural Alignment):** Kết quả retrieval tỉ lệ thuận với mức độ căn chỉnh giữa ba không gian: Textual, Structural, và Numerical.

**H3 (Hard Negative Validity):** Khả năng phân biệt chỉ thực sự được kiểm chứng khi model đối mặt với mẫu nhiễu có cấu trúc (constraint-directed negatives), không phải negative ngẫu nhiên.

### 2.4. Luận giải từ lý thuyết thông tin

Thông tin trong một bảng tài chính $C$ có thể phân tách:

$$H(T(C)) = H_T(T) + H_S(T) + H_N(T)$$

Trong đó $H_T$ là entropy text, $H_S$ là entropy cấu trúc (vị trí ô, quan hệ cha-con), $H_N$ là entropy số liệu. Khi flatten bảng thành văn bản:

$$I_{loss} \geq I_S + I_{discriminative}(H_N)$$

Mất mát thông tin cấu trúc + thông tin phân biệt số liệu là **không tránh khỏi** khi dùng biểu diễn vector đơn. → Multi-space representation là **cần thiết** (necessary condition).

---

## 3. Ý tưởng thiết kế

### 3.1. Insight chính

**Tài liệu tài chính có cấu trúc graph ngầm — các ô số được nối bởi các đẳng thức kế toán (accounting identities).** Knowledge Graph là ngôn ngữ tự nhiên để biểu diễn cấu trúc này. Nếu ta khai thác được cấu trúc ẩn đó, retriever sẽ phân biệt được context đúng và context "gần giống nhưng sai".

### 3.2. Tổng quan hai contribution

Chúng tôi đề xuất hai thành phần bổ trợ lẫn nhau:

**Contribution 1 — GSR (Graph-Structured Retrieval):** Thay vì coi bảng tài chính như văn bản phẳng, GSR xây dựng một Knowledge Graph cho mỗi bảng, trong đó các ô số (cells) là node và các ràng buộc kế toán (Revenue − COGS = Gross Profit) là edge. Một GAT encoder mã hoá graph này, và hệ thống chấm điểm kết hợp: (1) tương đồng text thông thường, (2) khớp metadata đúng công ty/năm, (3) mức tuân thủ ràng buộc kế toán của bảng. Nói đơn giản: GSR "hiểu" rằng context tốt là context mà các con số trong đó thoả mãn phương trình kế toán liên quan đến câu hỏi.

**Contribution 2 — CACL (Constraint-Aware Contrastive Learning):** Để huấn luyện GSR phân biệt tinh tế, chúng tôi tạo ra các mẫu nhiễu đặc biệt gọi là CHAP (Contrastive Hard-negative via Accounting Perturbations). Thay vì lấy negative ngẫu nhiên, CHAP thay đổi đúng một ô số trong bảng gốc — khiến phương trình kế toán bị vi phạm — tạo ra negative "rất giống đúng nhưng chắc chắn sai". Model được huấn luyện với loss kết hợp: vừa triplet loss thông thường, vừa penalty thêm khi gán điểm cao cho document vi phạm constraint.

**Hai contribution củng cố nhau:** GSR cung cấp graph structure để CACL học từ, và CACL cung cấp training signal chất lượng cao để GSR cải thiện constraint awareness.

### 3.3. Khác biệt với công trình liên quan

| Method | Biểu diễn bảng | Ràng buộc kế toán | Mục đích | Khác biệt với GSR |
|--------|----------------|-------------------|----------|-------------------|
| HELIOS (ACL 2025) | Bipartite subgraph (table ↔ text) | Không | Multi-hop QA | GSR dùng **accounting edges**, không phải table-text edges |
| THYME (EMNLP 2025) | Field-aware matching | Không | Field retrieval | GSR hiểu **quan hệ giữa fields** (qua constraints) |
| THoRR (2024) | Header concatenation | Không | Two-stage retrieval | GSR xây dựng **full KG**, không chỉ headers |
| ConFIT (2025) | SPP perturbation | Không (dùng lexicon) | Extraction | CHAP dùng **accounting identity violations**, không phải lexicon |

**Positioning:** Tất cả công trình trước dùng cấu trúc bảng cho QA hoặc field matching. GSR là phương pháp đầu tiên mô hình hoá **accounting identities như tín hiệu retrieval** — coi constraint satisfaction là bằng chứng của relevance.

---

## 4. Chi tiết kiến trúc

### 4.1. GSR Pipeline tổng thể

```
Query Q
  │
  ├─► Metadata extraction: (company, year) từ metadata sẵn có
  │
  ├─► Table KG Construction:
  │     table_md ──parse──► Cell nodes + Header mapping
  │                           │
  │                           └──► Template Matching (IFRS/GAAP templates)
  │                                 │
  │                                 └──► Constraint edges (ω ∈ {+1, −1})
  │                                       │
  │                                       └──► Partial KG
  │                                            (fallback: positional graph nếu
  │                                             template không match)
  │
  └─► GAT Encoder: node embeddings + edge-aware aggregation
          │
          ▼
  Joint Scoring:
    s(Q, D) = α · sim_text(Q, D)
             + β · sim_entity(Q, G_D)
             + γ · ConstraintScore(G_D)
          │
          ▼
  Top-K Retrieval
```

### 4.2. Component 1: Template-Based KG Construction

**Bước 1 — Parse markdown table:** Tách bảng markdown thành header row + data rows. Mỗi ô trở thành một KGNode với thông tin: giá trị số, header cột, vị trí (row, col).

**Bước 2 — Template matching:** So khớp danh sách headers với thư viện 15 IFRS/GAAP accounting templates. Mỗi template định nghĩa các đẳng thức kế toán cần thoả mãn:

| Template | Ràng buộc chính |
|----------|----------------|
| Income Statement | Revenue − COGS = Gross Profit; Gross Profit − OpEx = Operating Income |
| Balance Sheet (Assets) | Current Assets + Non-Current Assets = Total Assets |
| Balance Sheet (L+E) | Total Liabilities + Equity = Total |
| Cash Flow | Operating CF + Investing CF + Financing CF = Net CF |
| Revenue by Segment | Segment₁ + Segment₂ + ... = Total Revenue |
| Quarterly Breakdown | Q1 + Q2 + Q3 + Q4 = Annual |
| Debt Schedule | Long-Term Debt + Short-Term Debt = Total Debt |
| ... (15 templates tổng cộng) | ... |

**Bước 3 — Build edges:** Với template matched, tạo constraint edges giữa các ô liên quan. Mỗi edge mang trọng số $\omega \in \{+1, -1\}$ biểu thị quan hệ cộng (+1) hay trừ (−1).

**Bước 4 — Fallback:** Nếu confidence matching < 0.7, dùng positional graph (edge nối các ô cùng hàng/cùng cột) thay vì constraint edges.

**Coverage estimate:** ~80–90% bảng trong FinQA, ~70% trong TAT-DQA match với ít nhất một template. Phần còn lại dùng fallback.

### 4.3. Component 2: GAT Encoder

Node embedding khởi tạo:
$$h_v^{(0)} = \text{BGE}(cell\_text) \oplus \text{PosEnc}(row) \oplus \text{PosEnc}(col)$$

Edge-aware message passing:
$$h_v^{(l+1)} = \text{LeakyReLU}\left(\sum_{u \in \mathcal{N}(v)} \alpha_{vu} \cdot \omega_{vu} \cdot W^{(l)} \cdot h_u^{(l)}\right)$$

Trong đó:
- $\omega_{vu} \in \{+1, -1, 0\}$: trọng số edge (accounting hoặc positional)
- $\alpha_{vu}$: attention coefficient (softmax trên neighbors)
- $W^{(l)}$: ma trận trọng số học được

Kiến trúc: 2 GAT layers, 4 heads, hidden dim 256. Output: embedding cho mỗi node → pool thành document-level embedding.

### 4.4. Component 3: Constraint-Aware Scoring

**Constraint Score** (ε-tolerance, differentiable):
$$\text{CS}(G) = \frac{1}{|\mathcal{E}_c|} \sum_{(u,v) \in \mathcal{E}_c} \exp\left(-\frac{|\omega_{uv} \cdot v_u - v_v|}{\max(|v_v|, \epsilon)}\right)$$

Trực giác: nếu Revenue − COGS đúng bằng Gross Profit, score → 1.0. Nếu lệch, score giảm theo hàm exp.

**Entity Matching Score:**
$$s_{entity}(Q, D) = \text{match}(company_Q, company_D) + \text{match}(year_Q, year_D) + \text{match}(sector_Q, sector_D)$$

**Joint Score:**
$$s(Q, D) = \alpha \cdot \text{sim}_{text}(Q, D) + \beta \cdot s_{entity}(Q, D) + \gamma \cdot \text{CS}(G_D)$$

Với $\alpha, \beta, \gamma$ là trọng số học được (softplus-constrained positive).

### 4.5. CHAP Negative Generation

Ba loại perturbation, đều tuân theo **Zero-Sum property** (thay đổi đúng 1 thành phần, giữ nguyên cấu trúc):

| Type | Cách tạo | Constraint bị vi phạm | Ví dụ |
|------|---------|----------------------|-------|
| **CHAP-A** (Additive) | Thay đổi 1 ô con, giữ nguyên ô tổng | $A + B \neq Total$ | Revenue=100M → 120M (Gross Profit equation vỡ) |
| **CHAP-S** (Scale) | Đổi đơn vị (M → B), parent không đổi | Scale mismatch | Revenue 500M → 0.5B (tỷ lệ vỡ) |
| **CHAP-E** (Entity) | Đổi company/year, giữ nguyên cấu trúc | Entity/temporal mismatch | Apple/2023 → Apple/2022 |

Mỗi negative "rất giống" positive nhưng **chắc chắn vi phạm** ít nhất một constraint → hard negative chất lượng cao.

### 4.6. CACL Training Objective

$$\mathcal{L} = \mathcal{L}_{triplet} + \lambda \cdot \mathcal{L}_{constraint}$$

**Triplet Loss:**
$$\mathcal{L}_{triplet} = \frac{1}{N} \sum_{i=1}^{N} \max\left(0, m - s(Q_i, C_i^+) + s(Q_i, C_i^-)\right)$$

**Constraint Violation Loss:**
$$\mathcal{L}_{constraint} = -\frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{violates}(C_i^-, G_{C_i^-})] \cdot \log \sigma(s(Q_i, C_i^-))$$

Trực giác: model không chỉ bị phạt khi negative có score cao — mà bị phạt **nặng hơn** khi negative vi phạm constraint mà vẫn được gán điểm cao. Điều này buộc model phải học constraint semantics.

### 4.7. Three-Stage Training

| Stage | Mục tiêu | Epochs |
|-------|---------|--------|
| Stage 1: Identity Pretraining | Học phân biệt (company, year) pairs | 3 |
| Stage 2: Structural Pretraining | Học KG encoding + constraint scoring | 3 |
| Stage 3: Joint Finetuning (CACL) | Full objective với CHAP negatives | 5 |

Training loop:
```
for (Q, C⁺, C⁻_CHAP) in batch:
    G⁺ = build_constraint_kg(C⁺)
    G⁻ = build_constraint_kg(C⁻_CHAP)
    s⁺ = scoring(Q, C⁺, G⁺)
    s⁻ = scoring(Q, C⁻, G⁻)
    loss = triplet_loss(s⁺, s⁻) + λ · constraint_loss(s⁻, G⁻)
    loss.backward()
```

---

## 5. Thực nghiệm

### 5.1. Setup

- **Benchmark:** T²-RAGBench (EACL 2026) — 3 subsets: FinQA, ConvFinQA, TAT-DQA
- **Metrics:** MRR@3, Recall@1/3/5, NDCG@3
- **Embedding backbone:** BGE (intfloat/multilingual-e5-large-instruct)
- **Device:** CUDA (GPU khuyến nghị)
- **Pre-indexing:** KG được xây offline cho toàn bộ corpus → inference chỉ cần lookup + encode query

### 5.2. Baselines

| Baseline | Mô tả |
|----------|--------|
| BM25 | Lexical baseline |
| Hybrid BM25 | T²-RAGBench best reported |
| ColBERT-v2 | Late interaction baseline |
| E5-Mistral-7B | Dense retrieval SOTA |
| HELIOS (ACL 2025) | Multi-granular table-text retrieval |
| THYME (EMNLP 2025) | Field-aware hybrid matching |
| THoRR (2024) | Two-stage table retrieval |
| ConFIT (2025) | Semantic-Preserving Perturbation (so sánh CHAP) |

### 5.3. Kịch bản thử nghiệm

**Exp 1 — Main Results:** So sánh GSR, HybridGSR (GSR + BM25 + RRF) với tất cả baselines trên 3 datasets. Kỳ vọng: GSR vượt Hybrid BM25 ≥ 5 điểm MRR@3.

**Exp 2 — Ablation Studies:** Tách từng thành phần để đo contribution riêng.

| Ablation | Mô tả | Kiểm chứng |
|----------|--------|-----------|
| GSR − KG | Bỏ Constraint KG | Đo contribution của KG |
| GSR − Constraint Score | KG có nhưng không dùng constraint score | Đo constraint scoring |
| CACL − CHAP | Dùng random negatives thay vì CHAP | Kiểm chứng H3 |
| CACL − $\mathcal{L}_{constraint}$ | Chỉ triplet loss | Đo constraint regularization |
| GSR + CACL | Full system | Main result |

**Exp 3 — CHAP Negative Evaluation:** Đo accuracy trên 3 loại negatives (Standard random, BM25 hard, CHAP). Kỳ vọng: GSR + CACL đạt accuracy cao tương đương trên cả 3 types → model đã thực sự học constraint semantics, không chỉ overfit vào một dạng negative.

**Exp 4 — Cross-Domain Generalization:** Train trên FinQA + ConvFinQA, test trên TAT-DQA (diverse sectors, cấu trúc bảng khác). Đánh giá khả năng generalise của template-based approach.

**Exp 5 — Template Coverage Survey:** Chạy template matching trên toàn bộ corpus để kiểm chứng claim coverage 80–90%.

**Exp 6 — Error Analysis:** Phân tích chi tiết các failure cases, phân loại theo nguyên nhân: template miss, entity confusion, numerical ambiguity.

### 5.4. Chi phí tính toán

| Component | Complexity | Per-document | Ghi chú |
|-----------|-----------|-------------|---------|
| KG Construction | $O(n_{cells})$ | ~57 cells (avg) | Template matching = regex → nhanh |
| GAT Encoding | $O(V \cdot E)$ | $V \approx 57$, $E \approx 10$ | Sparse attention, 2 layers |
| Entity Matching | $O(1)$ | Constant | Metadata lookup |
| Constraint Scoring | $O(E_c)$ | $E_c \approx 5$–$10$ | Phép tính số học đơn giản |
| **Tổng inference** | | **~1.2–1.4× baseline** | |
| CHAP Generation | $O(n_{cells})$ | Pre-generate offline | Zero runtime cost |

**Pre-indexing:** KG xây offline cho mọi document → inference chỉ cần lookup và encode query.

---

## 6. Cấu trúc Paper dự kiến

```
1. Introduction                            (1 page)
2. Background & Related Work               (1.5 pages)
3. Problem Analysis                        (1 page)
     — 4 hiện tượng + 4 định luật + 3 giả thuyết
4. GSR: Graph-Structured Retrieval         (1.5 pages)
     — KG Construction, GAT Encoder, Constraint Scoring
5. CACL: Constraint-Aware Contrastive      (1.5 pages)
     — CHAP Generation, Training Objective
6. Experiments                             (2 pages)
     — Main results, Ablation, Cross-domain, Error analysis
7. Conclusion                              (0.5 page)
```

---

## Tham khảo chính

- **Benchmark:** Strich et al. (EACL 2026) — T²-RAGBench
- **Graph-based Retrieval:** HELIOS (ACL 2025), THYME (EMNLP 2025), THoRR (2024)
- **Constraint KG:** Singh & Gupta (NAACL 2023)
- **Hard Negative Mining:** ConFIT (2025), DPR (Karpukhin et al., 2020)
- **Language Theory:** Halliday (1994), Shannon (1948), Cover & Thomas (2006)
- **Financial QA:** FinQA (Chen et al., 2021), ConvFinQA (Chen et al., 2022), TAT-DQA (Zhu et al., 2022)
