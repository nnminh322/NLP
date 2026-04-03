# Structured Knowledge-Enhanced Retrieval for Financial Documents

---

# Truy xuất tài liệu tài chính tăng cường cấu trúc tri thức

## Tóm tắt

Truy xuất ngữ cảnh chính xác từ các báo cáo tài chính hỗn hợp văn bản và bảng biểu là một bước then chốt trong các hệ thống RAG cho lĩnh vực tài chính. Các phương pháp truy xuất hiện tại gặp khó khăn do bỏ qua các đặc trưng ngôn ngữ và ràng buộc toán học vốn có của tài liệu tài chính. Bài báo này phân tích bốn hiện tượng nền tảng gây ra thất bại của các retriever thông thường: tính thể loại đặc thù, ảo tưởng trùng lặp từ vựng, mất nhất quán toán học và bất ngờ ngữ nghĩa. Từ đó, chúng tôi đề xuất GSR (Graph-Structured Retrieval) – một phương pháp biểu diễn bảng tài chính dưới dạng đồ thị tri thức với các cạnh ràng buộc kế toán, kết hợp với cơ chế chấm điểm nhận biết ràng buộc. Để huấn luyện, chúng tôi phát triển CACL (Constraint-Aware Contrastive Learning) và kỹ thuật sinh mẫu âm CHAP dựa trên phá vỡ có chủ đích các đẳng thức kế toán. Nghiên cứu này thực nghiệm trên bộ benmark cho miền dữ liệu tài chính tiêu chuẩn T²-RAGBench.

## 1. Giới thiệu

Truy xuất ngữ cảnh phù hợp từ kho tài liệu tài chính (báo cáo thường niên, báo cáo quý) là bài toán nền tảng cho các hệ thống sinh tăng cường truy xuất (RAG) trong lĩnh vực này. Đầu vào là một câu hỏi tài chính \(Q\) và một kho gồm \(N\) tài liệu \(\{D_1, ..., D_N\}\), mỗi tài liệu chứa văn bản tự sự, bảng dạng markdown và siêu dữ liệu (tên công ty, năm báo cáo, ngành). Đầu ra là danh sách top-\(K\) tài liệu liên quan nhất, đánh giá qua các chỉ số MRR@3, Recall@1/3/5, NDCG@3 theo chuẩn của T²-RAGBench [1].

Thách thức chính nằm ở tính chất hỗn hợp của thông tin tài chính: câu trả lời thường đòi hỏi kết hợp cả văn bản lẫn bảng số liệu. Các hệ thống truy xuất tốt nhất hiện nay, bao gồm hybrid BM25 và dense retriever, chỉ đạt khoảng 35% MRR@3 trên các benchmark tài chính – một khoảng cách đáng kể so với mức lý tưởng. Nguyên nhân sâu xa là các phương pháp này chưa khai thác được bản chất đặc thù của tài liệu tài chính.

Bài báo này đóng góp:

1.  **Phân tích lý thuyết** bốn hiện tượng cốt lõi khiến truy xuất thông thường thất bại trên tài liệu tài chính, cùng ba giả thuyết nghiên cứu có thể kiểm chứng.
2.  **GSR (Graph-Structured Retrieval)** – một kiến trúc truy xuất biểu diễn mỗi bảng dưới dạng đồ thị tri thức với các nút là ô số và các cạnh là ràng buộc kế toán (ví dụ: Doanh thu – Giá vốn hàng bán = Lợi nhuận gộp). Điểm số liên quan kết hợp độ tương đồng văn bản, khớp siêu dữ liệu và mức độ thỏa mãn ràng buộc.
3.  **CACL (Constraint-Aware Contrastive Learning)** cùng kỹ thuật sinh mẫu âm CHAP, trong đó các mẫu âm được tạo ra bằng cách phá vỡ có chủ đích một đẳng thức kế toán trong bảng gốc, tạo ra các mẫu nhiễu “rất giống nhưng chắc chắn sai” – giúp mô hình học khả năng phân biệt tinh tế.
4.  **Đánh giá thực nghiệm** trên ba tập con của T²-RAGBench, bao gồm phân tích ablation, đánh giá khả năng tổng quát hóa chéo lĩnh vực và phân tích lỗi chi tiết.

Phần còn lại của bài báo được tổ chức như sau: Mục 2 điểm qua các công trình liên quan. Mục 3 phân tích sâu bản chất của tài liệu tài chính và đề xuất ba giả thuyết nghiên cứu. Mục 4 trình bày chi tiết phương pháp GSR và CACL. Mục 5 mô tả thiết lập thực nghiệm. Mục 6 báo cáo và thảo luận kết quả. Mục 7 kết luận.


### 1.2. Scope

Chúng tôi tập trung vào bài toán **only retrieval** (truy xuất). Cụ thể:

- **Đầu vào:** câu hỏi tài chính $Q$ + corpus gồm $N$ documents $\{D_1, ..., D_N\}$, mỗi document chứa text narrative + markdown table + metadata (tên công ty, năm báo cáo, ngành).
- **Đầu ra:** top-$K$ documents liên quan nhất.
- **Metric:** MRR@3, Recall@1/3/5, NDCG@3 — chuẩn hoá theo T²-RAGBench (EACL 2026).

### 1.3. Khoảng cách hiện tại

Các hệ thống retrieval tốt nhất hiện tại (Hybrid BM25, dense retrievers) đạt khoảng 35% MRR@3 trên benchmark tài chính — kém Oracle Context (biết trước context đúng) tới **30 điểm phần trăm**. Điều này cho thấy standard retrieval chưa khai thác được bản chất đặc thù của tài liệu tài chính.

---
## 2. Công trình liên quan

**Truy xuất bảng trong tài liệu tài chính.** HELIOS [2] sử dụng đồ thị hai phía giữa bảng và văn bản cho câu hỏi đa bước, nhưng không mô hình hóa các ràng buộc kế toán giữa các ô số. THYME [3] đề xuất ghép nối theo trường (field-aware matching) nhưng chỉ dừng ở mức header. THoRR [4] là phương pháp truy xuất hai giai đoạn dựa trên nối header, thiếu biểu diễn toàn cục các mối quan hệ số học. ConFIT [5] sử dụng nhiễu loạn bảo toàn ngữ nghĩa dựa trên từ điển chuyên ngành, khác với cách tiếp cận dựa trên vi phạm đẳng thức kế toán của chúng tôi.

**Học đối lập và mẫu âm khó.** DPR [6] đặt nền móng cho việc sử dụng mẫu âm ngẫu nhiên và mẫu âm từ BM25. Các công trình gần đây như ConFIT [5] chỉ ra lợi ích của mẫu âm được tạo có chủ đích. Tuy nhiên, chưa có phương pháp nào tận dụng các ràng buộc toán học vốn có của bảng tài chính để tạo mẫu âm khó một cách có hệ thống.

**Lý thuyết ngôn ngữ cho văn bản chuyên ngành.** Halliday [7] chỉ ra rằng ngữ nghĩa trong các văn bản có tính công thức cao không nằm ở từ mà ở vị trí cấu trúc. Shannon [8] và các nghiên cứu về entropy từ vựng cho thấy từ chuyên ngành có phân phối xác suất khác biệt so với văn bản thông thường. Chúng tôi kế thừa các quan điểm này để giải thích hiện tượng thất bại của các embedding tổng quát.

## 3. Phân tích bản chất của tài liệu tài chính

Chúng tôi cho rằng tài liệu tài chính không phải là văn bản tự nhiên thông thường mà là một hệ thống mã hóa tuân theo các luật ngôn ngữ và ràng buộc toán học bất biến. Bốn hiện tượng sau đây giải thích tại sao các phương pháp truy xuất thành công trên văn bản đa dạng lại sụp đổ trên miền tài chính.

### 3.1. Bốn hiện tượng đặc thù

**Hiện tượng 1: Tính thể loại đặc thù (Genre-specificity).** Báo cáo tài chính có ba đặc trưng nền tảng dựa trên phân tích thể loại [7, 9]:
- *Tính lặp lại cao*: Các cấu trúc câu gần như cố định (ví dụ: “Lợi nhuận ròng tăng X% lên Y triệu”). Điều này làm bão hòa embedding tổng quát, khiến chúng mất khả năng phân biệt giữa các ngữ cảnh khác nhau.
- *Nén từ vựng*: Một từ chuyên ngành mang nhiều nghĩa kỹ thuật tùy ngữ cảnh (ví dụ: “thu nhập” có thể là thu nhập ròng, thu nhập hoạt động, thu nhập trước thuế). Embedding tổng quát không phân biệt được các sắc thái này.
- *Cấu trúc đa phương thức cố định*: Mỗi báo cáo tuân theo trình tự [văn bản] → [bảng] → [chú thích] → [văn bản]. Làm phẳng cấu trúc này sẽ phá hủy các mối quan hệ vốn có.

> *"Do đó có thể nói rằng Báo cáo tài chính không phải là văn bản tự nhiên mà là một dạng mã hóa (encoding) của các sự kiện kinh tế theo một ngữ pháp cố định — một thứ 'ngôn ngữ hình thức hóa'."*

**Hiện tượng 2: Ảo tưởng trùng lặp từ vựng (Lexical Overlap Illusion).** Trong tài liệu tài chính, độ trùng lặp từ vựng cao không đồng nghĩa với tương đồng ngữ nghĩa cao. Cùng một từ như “doanh thu” trong đoạn văn bản có thể chỉ doanh thu toàn công ty, nhưng trong một bảng cụ thể có thể chỉ doanh thu theo phân khúc. Ngữ nghĩa không nằm trong từ mà nằm trong cấu trúc. Do đó, truy xuất thành công đòi hỏi tách biệt tín hiệu từ vựng (các từ xuất hiện) và tín hiệu cấu trúc (vị trí của từ trong tài liệu).

**Hiện tượng 3: Mất nhất quán toán học (Mathematical Inconsistency).** Các báo cáo tài chính bị ràng buộc bởi các phương trình kế toán ẩn:
$$\text{Doanh thu} - \text{Giá vốn hàng bán} = \text{Lợi nhuận gộp}$$
$$\text{Tài sản} = \text{Nợ phải trả} + \text{Vốn chủ sở hữu}$$

Khi một bảng bị “làm phẳng” thành văn bản, quan hệ cha – con giữa các ô số bị phá hủy. Một retriever không thể phân biệt được ngữ cảnh nào thực sự thỏa mãn các ràng buộc toán học liên quan đến câu hỏi.

**Hiện tượng 4: Nghịch lý mật độ số (Numerical Density Paradox).** Hầu hết các câu hỏi tài chính đều có các con số cần thiết nằm trong ngữ cảnh đúng. Tuy nhiên, cùng một con số đó cũng xuất hiện trong hàng chục ngữ cảnh khác. Số liệu vừa là manh mối mạnh vừa là cái bẫy cho retriever.

### 3.2. Các định luật ngôn ngữ học trong tài liệu tài chính

Từ các nghiên cứu ngôn ngữ học ứng dụng [7,8,10] và khảo sát của chúng tôi, có bốn định luật ngôn ngữ chi phối cấu trúc thông tin của báo cáo tài chính:

- **Định luật mã hóa công thức (Law of Formulaic Encoding)** – *Halliday (1994)*: Ngữ nghĩa không nằm ở từ ngữ, mà ở vị trí trong cấu trúc. Chế độ hỗn hợp (văn bản + bảng) bị phá hủy khi làm phẳng.
- **Định luật nén từ vựng (Law of Lexical Compression)** – *Shannon (1948)*: Entropy trên mỗi token trong văn bản chuyên ngành thấp hơn văn bản thông thường – từ chuyên ngành có xác suất xuất hiện cố định (độ lệch so với định luật Zipf).
- **Định luật neo số (Law of Numerical Anchoring):** Con số là điểm cố định – thay đổi số neo sẽ hủy bỏ ngữ nghĩa.
- **Định luật ngữ nghĩa đa lớp (Law of Multi-layered Semantics):** Một ngữ cảnh chứa trung bình nhiều câu hỏi khác nhau → biểu diễn phải đa diện, không thể dùng một vector đơn.

### 3.3. Ba giả thuyết nghiên cứu

Từ các phân tích trên, chúng tôi đề xuất ba giả thuyết có thể kiểm chứng:

- **H1 (Tính nhất quán):** Truy xuất vượt trội khi bảo toàn tính tự hợp toán học – tức là khi hệ thống kiểm tra được mức độ thỏa mãn ràng buộc của ngữ cảnh.
- **H2 (Căn chỉnh cấu trúc):** Kết quả truy xuất tỷ lệ thuận với mức độ căn chỉnh giữa ba không gian: văn bản, cấu trúc và số liệu.
- **H3 (Giá trị của mẫu âm khó):** Khả năng phân biệt chỉ thực sự được kiểm chứng khi mô hình đối mặt với các mẫu nhiễu có cấu trúc (dựa trên ràng buộc), không phải mẫu âm ngẫu nhiên.

### 3.4. Luận giải từ lý thuyết thông tin

Thông tin trong một bảng tài chính \(C\) có thể phân tách thành ba thành phần:
$$H(T(C)) = H_T(T) + H_S(T) + H_N(T)$$

trong đó \(H_T\) là entropy văn bản, \(H_S\) là entropy cấu trúc (vị trí ô, quan hệ cha-con), \(H_N\) là entropy số liệu. Khi làm phẳng bảng thành văn bản, lượng thông tin mất đi là:
$$I_{loss} \geq I_S + I_{discriminative}(H_N)$$

Mất mát thông tin cấu trúc và thông tin phân biệt số liệu là không thể tránh khỏi khi dùng biểu diễn vector đơn. Do đó, biểu diễn đa không gian là điều kiện cần.
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

## Tham khảo

[1] Strich et al. T²-RAGBench: A Benchmark for Table-Text Retrieval in Financial Documents. EACL, 2026.

[2] HELIOS: Multi-hop QA over Financial Tables and Text. ACL, 2025.

[3] THYME: Field-aware Hybrid Matching for Table Retrieval. EMNLP, 2025.

[4] THoRR: Two-stage Table Retrieval with Header Concatenation. 2024.

[5] ConFIT: Semantic-Preserving Perturbation for Hard Negative Mining. 2025.

[6] Karpukhin et al. Dense Passage Retrieval for Open-Domain Question Answering. EMNLP, 2020.

[7] Halliday, M.A.K. An Introduction to Functional Grammar. 1994.

[8] Shannon, C.E. A Mathematical Theory of Communication. Bell System Technical Journal, 1948.

[9] Swales, J.M. Genre Analysis: English in Academic and Research Settings. 1990.

[10] Cover, T.M. & Thomas, J.A. Elements of Information Theory. 2006.

[11] Chen et al. FinQA: A Dataset for Numerical Reasoning over Financial Reports. ACL, 2021.

[12] Chen et al. ConvFinQA: Exploring Conversational Numerical Reasoning over Financial Reports. ACL, 2022.

[13] Zhu et al. TAT-DQA: Question Answering over Financial Tables with Text and Tables. EMNLP, 2022.

---