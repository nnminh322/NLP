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

## 4. Proposed Method: GSR-CACL

Mục này trình bày phương pháp đề xuất theo hướng top-down: bắt đầu từ định nghĩa bài toán hình thức (§4.1), tổng quan kiến trúc (§4.2), phân tách từng module với cơ sở toán học (§4.3–§4.6), và kết thúc bằng hàm mục tiêu huấn luyện (§4.7–§4.8). Hình tổng quan kiến trúc xem tại *Figure 1* trong `beauty_architecture.md`.

### 4.1. Problem Formulation

Cho kho tài liệu $\mathcal{C} = \{C_1, C_2, \ldots, C_N\}$, trong đó mỗi tài liệu $C_i = (t_i, T_i, m_i)$ bao gồm:

- $t_i$: đoạn văn bản tự sự (narrative text),
- $T_i$: bảng dạng markdown (markdown table),
- $m_i = (\text{company}_i, \text{year}_i, \text{sector}_i)$: siêu dữ liệu (metadata).

Cho câu hỏi $Q$ cùng siêu dữ liệu $m_Q$, mục tiêu là xây dựng hàm truy xuất:

$$f_\theta: Q \times \mathcal{C} \mapsto \left[C_{k}^{*}\right]_{k=1}^{K}$$

trả về $K$ tài liệu liên quan nhất, tối ưu hóa theo MRR@$k$ và Recall@$k$.

**Ký hiệu xuyên suốt bài báo:**

| Ký hiệu | Ý nghĩa |
|----------|---------|
| $\mathbf{q} \in \mathbb{R}^d$ | Embedding của câu hỏi $Q$ |
| $\mathbf{d}_{\text{text}} \in \mathbb{R}^d$ | Embedding văn bản của tài liệu $C$ |
| $G_D = (\mathcal{V}, \mathcal{E}, \omega)$ | Đồ thị tri thức ràng buộc (Constraint KG) của bảng $T$ |
| $\mathbf{d}_{\text{KG}} \in \mathbb{R}^h$ | Biểu diễn đồ thị (graph embedding) sau GAT |
| $\phi_\theta$ | Hàm chấm điểm kết hợp (Joint Scorer) |
| $s(Q, C)$ | Điểm liên quan tổng hợp |
| $\alpha, \beta, \gamma$ | Trọng số học được (softplus-constrained) |
| $\oplus$ | Phép nối (concatenation) |
| $\otimes$ | Phép nhân từng phần tử (element-wise product) |
| $\omega_{uv} \in \{+1, -1, 0\}$ | Trọng số cạnh kế toán |

### 4.2. Architecture Overview

GSR-CACL bao gồm hai đóng góp bổ trợ lẫn nhau (xem *Figure 1*):

**Contribution 1 — GSR (Graph-Structured Retrieval) — Inference.** Thay vì coi bảng tài chính như văn bản phẳng, GSR biểu diễn mỗi bảng dưới dạng đồ thị tri thức $G_D = (\mathcal{V}, \mathcal{E}, \omega)$ với các nút là ô số và các cạnh là ràng buộc kế toán. Hàm chấm điểm $\phi_\theta$ kết hợp ba tín hiệu: tương đồng văn bản, khớp thực thể, và mức tuân thủ ràng buộc:

$$s(Q, C) = \alpha \cdot s_\text{text}(Q, C) + \beta \cdot s_\text{ent}(Q, C) + \gamma \cdot \text{CS}(G_D) \tag{1}$$

**Contribution 2 — CACL (Constraint-Aware Contrastive Learning) — Training.** CACL huấn luyện $\phi_\theta$ bằng cách tạo mẫu âm khó (hard negatives) qua kỹ thuật CHAP — phá vỡ có chủ đích đúng một đẳng thức kế toán. Hàm mất mát kết hợp:

$$\mathcal{L}_\text{CACL} = \mathcal{L}_\text{triplet} + \lambda \cdot \mathcal{L}_\text{constraint} \tag{2}$$

**Mối quan hệ:** GSR cung cấp cấu trúc đồ thị để CACL học từ; CACL cung cấp tín hiệu huấn luyện chất lượng cao để GSR cải thiện khả năng nhận biết ràng buộc.

### 4.3. Template-Based KG Construction

> *Tham chiếu: Figure 2 — Constraint KG Construction*

Mỗi bảng markdown $T$ được chuyển đổi thành đồ thị tri thức $G_D = (\mathcal{V}, \mathcal{E}, \omega)$ qua bốn bước:

**Bước 1 — Parse.** Tách bảng thành tập header $\mathcal{H}$ và ma trận ô $\{c_{ij}\}$. Mỗi ô trở thành nút $v \in \mathcal{V}$ với thuộc tính $(value, row, col, header)$.

**Bước 2 — Template Matching.** So khớp $\mathcal{H}$ với thư viện $\mathcal{T}$ gồm 15 template IFRS/GAAP. Độ tin cậy:

$$\text{conf}(\mathcal{H}, \tau) = \frac{|\{h \in \mathcal{H} \mid \text{normalize}(h) \in \mathcal{H}_\tau\}|}{\max(|\mathcal{H}|, |\mathcal{H}_\tau|)}, \quad \tau \in \mathcal{T} \tag{3}$$

| Template | Ràng buộc đại diện |
|----------|-------------------|
| Income Statement | Revenue $-$ COGS $=$ Gross Profit; GP $-$ OpEx $=$ EBIT |
| Balance Sheet (Assets) | Current Assets $+$ Non-Current Assets $=$ Total Assets |
| Balance Sheet (L+E) | Total Liabilities $+$ Equity $=$ Total Assets |
| Cash Flow Statement | OCF $+$ ICF $+$ FCF $=$ Net Cash Flow |
| Revenue by Segment | $\sum_i$ Segment$_i$ $=$ Total Revenue |
| Quarterly Breakdown | Q1 $+$ Q2 $+$ Q3 $+$ Q4 $=$ Annual |
| ... (15 templates) | ... |

**Bước 3 — Edge Construction.** Nếu $\text{conf} \geq \tau_\text{min}$ (mặc định 0.5), tạo cạnh ràng buộc kế toán (accounting edges) với trọng số ngữ nghĩa:

$$\omega_{uv} = \begin{cases} +1 & \text{nếu } v_u \text{ cộng vào } v_v \text{ (additive)} \\ -1 & \text{nếu } v_u \text{ trừ từ } v_v \text{ (subtractive)} \end{cases} \tag{4}$$

**Bước 4 — Fallback.** Nếu $\text{conf} < \tau_\text{min}$, tạo cạnh vị trí (positional edges) $\omega = 0$ cho các ô cùng hàng/cùng cột.

**Coverage estimate:** ~80–90% bảng trong FinQA/ConvFinQA và ~70% trong TAT-DQA khớp ít nhất một template (sẽ được kiểm chứng trong §6.5).

### 4.4. Edge-Aware GAT Encoder

> *Tham chiếu: Figure 3 & 3b — GAT Encoder*

**Node Feature Construction.** Mỗi nút $v$ tại vị trí $(r, c)$ được biểu diễn bằng phép nối:

$$\mathbf{x}_v = f_\theta(\text{cell\_text}_v) \oplus \text{PE}_\text{row}(r) \oplus \text{PE}_\text{col}(c) \in \mathbb{R}^{d + 2p} \tag{5}$$

trong đó $f_\theta$ là text encoder (§4.2), $\text{PE}$ là mã hóa vị trí dạng sinusoidal với $p = d/4$. Phép chiếu đầu vào:

$$\mathbf{h}_v^{(0)} = \text{ReLU}\big(\text{LayerNorm}(W_\text{proj}\, \mathbf{x}_v + b_\text{proj})\big) \in \mathbb{R}^{h_\text{dim}} \tag{6}$$

**Edge-Aware Multi-Head Attention.** Tại mỗi layer $l$, mỗi head $k \in \{1,\ldots,H\}$ tính attention có thiên hướng theo trọng số ràng buộc $\omega$:

$$e_{uv}^{(k)} = \frac{\langle W_q^{(k)} \mathbf{h}_u^{(l)},\; W_k^{(k)} \mathbf{h}_v^{(l)} \rangle}{\sqrt{d_k}} + \text{Proj}(\omega_{uv}) \tag{7}$$

$$\alpha_{uv}^{(k)} = \frac{\exp(e_{uv}^{(k)})}{\sum_{w \in \mathcal{N}(v)} \exp(e_{wv}^{(k)})} \tag{8}$$

**Message Passing & Aggregation:**

$$\mathbf{h}_v^{(l+1)} = W_o \bigg[\Big\|_{k=1}^{H} \sum_{u \in \mathcal{N}(v)} \alpha_{uv}^{(k)} \cdot \omega_{uv} \cdot W_v^{(k)} \mathbf{h}_u^{(l)} \bigg] + \mathbf{h}_v^{(l)} \tag{9}$$

với residual connection. Kiến trúc: $L = 2$ layers, $H = 4$ heads, $h_\text{dim} = 256$.

**Graph-Level Representation.** Biểu diễn cấp tài liệu qua mean pooling:

$$\mathbf{d}_\text{KG} = \frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} \mathbf{h}_v^{(L)} \in \mathbb{R}^{h_\text{dim}} \tag{10}$$

### 4.5. Joint Scorer $\phi_\theta$

> *Tham chiếu: Figure 4 — Joint Scorer*

Hàm chấm điểm kết hợp ba tín hiệu bổ trợ (Eq. 1), mỗi tín hiệu nắm bắt một khía cạnh khác nhau của sự liên quan:

**Text Similarity** — Tương đồng ngữ nghĩa giữa câu hỏi và tài liệu:

$$s_\text{text}(Q, C) = \cos(\mathbf{q}, \mathbf{d}_\text{text}) = \frac{\mathbf{q}^\top \mathbf{d}_\text{text}}{\|\mathbf{q}\| \cdot \|\mathbf{d}_\text{text}\|} \in [-1, 1] \tag{11}$$

**Entity Score** — Mức khớp siêu dữ liệu (công ty, năm, ngành):

$$s_\text{ent}(Q, C) = \frac{1}{3} \sum_{k \in \{\text{co}, \text{yr}, \text{se}\}} \mathbb{1}[m_Q^k = m_C^k] \in [0, 1] \tag{12}$$

**Constraint Score** — Mức tuân thủ đẳng thức kế toán ($\varepsilon$-tolerance, differentiable):

$$\text{CS}(G_D) = \frac{1}{|\mathcal{E}_c|} \sum_{(u,v,\omega) \in \mathcal{E}_c} \exp\!\left(-\frac{|\omega \cdot v_u - v_v|}{\max(|v_v|, \varepsilon)}\right) \in (0, 1] \tag{13}$$

**Trọng số học được:** $\alpha = \text{softplus}(\log\hat{\alpha})$, $\beta = \text{softplus}(\log\hat{\beta})$, $\gamma = \text{softplus}(\log\hat{\gamma})$ — luôn dương. Giá trị khởi tạo: $\alpha_0 = 0.5$, $\beta_0 = 0.3$, $\gamma_0 = 0.2$.

### 4.6. CHAP: Constraint-Aware Hard Negative Generation

> *Tham chiếu: Figure 5 — CHAP Negative Sampler*

Thay vì mẫu âm ngẫu nhiên (DPR [6]) hoặc BM25 hard negatives, CHAP tạo mẫu âm $C^-$ từ tài liệu dương $C^+$ bằng cách phá vỡ có chủ đích **đúng một** đẳng thức kế toán. Ba kiểu nhiễu loạn:

| Kiểu | Phép biến đổi | Ràng buộc bị vi phạm | Xác suất |
|------|---------------|----------------------|----------|
| **A** (Additive) | Thay đổi 1 ô con, giữ nguyên ô tổng | $\sum_i v_{child_i} \neq v_{parent}$ | $p = 0.5$ |
| **S** (Scale) | Biến đổi bậc đại lượng ($\times 10^3$ hoặc $\times 10^{-3}$) | Tỷ lệ giữa các ô bị phá vỡ | $p = 0.3$ |
| **E** (Entity) | Hoán đổi company/year trong metadata | Sai khớp thực thể/thời gian | $p = 0.2$ |

**Tính chất Zero-Sum:** Mỗi $C^-$ chỉ khác $C^+$ đúng một thành phần → mẫu âm có độ tương đồng bề mặt rất cao nhưng **chắc chắn** vi phạm ít nhất một ràng buộc. Điều này buộc mô hình phải học phân biệt dựa trên cấu trúc, không phải từ vựng.

### 4.7. Training Objective: $\mathcal{L}_\text{CACL}$

> *Tham chiếu: Figure 6 — CACL Training Objective*

Hàm mất mát kết hợp hai thành phần (Eq. 2):

**Triplet Loss** — Đẩy điểm tài liệu dương cao hơn tài liệu âm một khoảng margin $m$:

$$\mathcal{L}_\text{triplet} = \frac{1}{N} \sum_{i=1}^{N} \max\!\big(0,\; m - s(Q_i, C_i^+) + s(Q_i, C_i^-)\big) \tag{14}$$

**Constraint Violation Loss** — Phạt nặng khi mô hình gán điểm cao cho tài liệu vi phạm ràng buộc:

$$\mathcal{L}_\text{constraint} = -\frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{violated}(C_i^-, G_{C_i^-})] \cdot \log\sigma(-s(Q_i, C_i^-)) \tag{15}$$

**Trực giác:** $\mathcal{L}_\text{triplet}$ dạy mô hình phân biệt đúng/sai; $\mathcal{L}_\text{constraint}$ dạy mô hình *tại sao* sai — vì vi phạm ràng buộc kế toán. Siêu tham số $\lambda$ điều chỉnh cân bằng hai tín hiệu.

### 4.8. Three-Stage Curriculum Training

> *Tham chiếu: Figure 7 — Three-Stage Curriculum*

Huấn luyện theo chương trình (curriculum) ba giai đoạn, trọng số được chuyển tuần tự $\theta_1 \to \theta_2 \to \theta_3 = \theta^*$:

| Giai đoạn | Module được huấn luyện | Mục tiêu | Loss |
|-----------|----------------------|---------|------|
| **Stage 1: Identity** | $f_\theta$ + $\phi_\theta$ (text + entity) | Học phân biệt $(company, year)$ | $\mathcal{L}_\text{triplet}$ (chỉ $s_\text{text} + s_\text{ent}$) |
| **Stage 2: Structural** | $f_\theta$ + GAT + $\phi_\theta$ | Hiệu chuẩn $\text{CS} \approx 1$ cho tài liệu hợp lệ | $\mathcal{L}_\text{triplet}$ (full $s$) |
| **Stage 3: Joint CACL** | $f_\theta$ + GAT + $\phi_\theta$ + CHAP | Tối ưu toàn diện với mẫu âm khó | $\mathcal{L}_\text{CACL}$ (Eq. 2) |

Mỗi giai đoạn cung cấp khởi tạo tốt hơn cho giai đoạn tiếp theo, tránh collapse khi huấn luyện toàn bộ từ đầu.

---

## 5. Experimental Setup

### 5.1. Datasets

Chúng tôi đánh giá trên T²-RAGBench [1] — benchmark chuẩn cho truy xuất tài liệu tài chính hỗn hợp văn bản-bảng, bao gồm ba tập con:

| Subset | Documents | QA Pairs | Avg. Token/Doc | Domain |
|--------|-----------|----------|----------------|--------|
| FinQA [11] | 2,789 | 8,281 | 950.4 | S&P 500 financial reports |
| ConvFinQA [12] | 1,806 | 3,458 | 890.9 | S&P 500 (conversational) |
| TAT-DQA [13] | 2,723 | 11,349 | 915.3 | Diverse financial reports |
| **Tổng** | **7,318** | **23,088** | **924.2** | |

Mỗi mẫu là bộ ba $(Q, A, C)$ với câu hỏi context-independent đã được xác nhận bởi chuyên gia (91.3% context-independent, Cohen's $\kappa = 0.87$). Câu hỏi yêu cầu truy xuất ngữ cảnh đúng trước khi thực hiện suy luận số học.

### 5.2. Baselines

Các baselines được phân nhóm theo chiến lược truy xuất để tạo bối cảnh đối chiếu công bằng:

**Nhóm 1 — Giới hạn trên/dưới (Bounds):**

| Method | Mô tả |
|--------|-------|
| Pretrained-Only | Không dùng retriever, LLM trả lời từ pre-training knowledge |
| Oracle Context | Ngữ cảnh đúng được cung cấp trực tiếp (giới hạn trên) |

**Nhóm 2 — Basic RAG Methods:**

| Method | Mô tả |
|--------|-------|
| Base-RAG [6] | Dense retrieval chuẩn (embed query → cosine similarity) |
| Hybrid BM25 | Kết hợp sparse (BM25) + dense retrieval — **best reported** trên T²-RAGBench |
| Reranker | Cross-encoder reranking sau initial retrieval |

**Nhóm 3 — Advanced RAG Methods:**

| Method | Mô tả |
|--------|-------|
| HyDE | Sinh câu trả lời giả → dùng làm query mới |
| Summarization | Tóm tắt ngữ cảnh trước khi retrieval |
| SumContext | Retrieval từ bản tóm tắt, nhưng trả về ngữ cảnh gốc đầy đủ |

**Nhóm 4 — Table-Aware Methods (SOTA cho bảng):**

| Method | Mô tả |
|--------|-------|
| THoRR [4] | Two-stage retrieval dựa trên header concatenation |
| ConFIT [5] | Semantic-Preserving Perturbation — so sánh trực tiếp với CHAP |

**Nhóm 5 — Our Methods:**

| Method | Mô tả |
|--------|-------|
| GSR | Graph-Structured Retrieval (Contribution 1) |
| GSR + CACL | Full system với CHAP training (Contribution 1 + 2) |
| HybridGSR | GSR + BM25 + Reciprocal Rank Fusion |

### 5.3. Evaluation Metrics

| Metric | Ý nghĩa | Lý do lựa chọn |
|--------|---------|-----------------|
| **MRR@3** | Mean Reciprocal Rank tại $k=3$ | Metric chính của T²-RAGBench; đánh giá thứ hạng tài liệu đúng trong top-3 |
| **Recall@3** | Tỷ lệ câu hỏi có tài liệu đúng trong top-3 | Đánh giá coverage; giới hạn $k=3$ vì avg. doc length = 924 tokens |
| **NM (Number Match)** | Câu trả lời số đúng (tolerance $\varepsilon = 10^{-2}$, scale-invariant) | End-to-end metric: retrieval đúng → reasoning đúng |
| **Recall@1, Recall@5** | Supplementary recall tại $k=1$ và $k=5$ | Phân tích chi tiết chất lượng retrieval |

Chúng tôi tập trung vào **MRR@3 và R@3** cho retrieval evaluation (vì bài toán chính là retrieval), và báo cáo NM cho end-to-end reference.

### 5.4. Implementation Details

| Tham số | Giá trị |
|---------|---------|
| Text encoder | BAAI/bge-large-en-v1.5 ($d = 1024$) |
| Fine-tuning strategy | LoRA ($r = 16$, $\alpha = 32$, target: $W_q, W_v$) |
| GAT layers / heads / hidden | $L = 2$ / $H = 4$ / $h_\text{dim} = 256$ |
| Positional encoding dim | $p = 192$ (sinusoidal) |
| Node feature input dim | $d + 2p = 1024 + 384 = 1408$ |
| Template library size | $|\mathcal{T}| = 15$ templates IFRS/GAAP |
| Template matching threshold | $\tau_\text{min} = 0.5$ |
| Training stages (epochs) | Stage 1: 3 / Stage 2: 3 / Stage 3: 5 |
| Batch size | 8 (T4 GPU) hoặc 16 (A100) |
| Learning rate | $2 \times 10^{-5}$ (AdamW, cosine schedule) |
| Margin $m$ | 0.3 |
| Constraint weight $\lambda$ | 0.5 |
| CHAP ratios (A/S/E) | 0.5 / 0.3 / 0.2 |
| Optimizer | AdamW ($\beta_1 = 0.9$, $\beta_2 = 0.999$, weight decay $= 0.01$) |
| Hardware | NVIDIA T4 16GB (Kaggle) hoặc A100 40GB |

**Offline Pre-indexing:** KG construction và GAT encoding được thực hiện offline cho toàn bộ corpus. Tại inference, chỉ cần encode query và tra cứu.

| Component | Complexity | Per-document avg. |
|-----------|-----------|-------------------|
| KG Construction | $O(|\mathcal{V}|)$ | ~57 cells |
| GAT Encoding | $O(|\mathcal{V}| \cdot |\mathcal{E}|)$ | $|\mathcal{V}| \approx 57$, $|\mathcal{E}| \approx 10$ |
| Constraint Scoring | $O(|\mathcal{E}_c|)$ | ~5–10 edges |
| CHAP Generation | $O(|\mathcal{V}|)$ | Pre-generated offline |
| **Tổng inference overhead** | | **~1.2–1.4× so với Base-RAG** |

---

## 6. Expected Results & Analysis

> **Lưu ý:** Đây là proposal — các bảng kết quả được thiết kế sẵn với baseline numbers từ T²-RAGBench [1] (Table 3). Kết quả của phương pháp đề xuất (GSR, GSR+CACL, HybridGSR) sẽ được điền sau khi chạy thực nghiệm.

### 6.1. Main Results

**Exp 1 — So sánh toàn diện với baselines.** Bảng dưới đây báo cáo hiệu năng retrieval (MRR@3, R@3) và end-to-end (NM) trên ba tập con. Generator sử dụng Llama 3.3-70B.

| Method | FinQA | | | ConvFinQA | | | TAT-DQA | | | W. Avg | | |
|--------|-------|---|---|-----------|---|---|---------|---|---|--------|---|---|
| | NM | MRR@3 | R@3 | NM | MRR@3 | R@3 | NM | MRR@3 | R@3 | NM | MRR@3 | R@3 |
| *Bounds* | | | | | | | | | | | | |
| Pretrained-Only | 7.9 | — | — | 2.8 | — | — | 3.7 | — | — | 5.1 | — | — |
| Oracle Context | 76.2 | 100 | 100 | 75.8 | 100 | 100 | 69.2 | 100 | 100 | 72.7 | 100 | 100 |
| *Basic RAG* | | | | | | | | | | | | |
| Base-RAG | 39.5 | 38.7 | 49.7 | 47.4 | 42.2 | 53.8 | 29.6 | 25.2 | 28.4 | 35.8 | 32.6 | 39.8 |
| Hybrid BM25 | 41.7 | 40.0 | 53.0 | 50.3 | 43.5 | 57.2 | 37.4 | 29.2 | 44.4 | 40.9 | 35.2 | 49.4 |
| Reranker | 32.4 | 29.0 | 36.2 | 37.3 | 32.3 | 40.5 | 27.0 | 22.8 | 28.4 | 30.5 | 26.4 | 33.0 |
| *Advanced RAG* | | | | | | | | | | | | |
| HyDE | 38.4 | 35.4 | 45.7 | 44.8 | 39.8 | 50.9 | 26.7 | 20.8 | 26.7 | 33.6 | 28.9 | 37.1 |
| Summarization | 27.3 | 47.3 | 59.5 | 35.2 | 52.1 | 63.8 | 14.6 | 24.7 | 31.5 | 22.2 | 36.9 | 46.4 |
| SumContext | 47.2 | 47.3 | 59.4 | 55.5 | 52.1 | 63.8 | 29.1 | 24.8 | 31.4 | 39.5 | 37.0 | 46.3 |
| *Table-Aware* | | | | | | | | | | | | |
| THoRR | — | — | — | — | — | — | — | — | — | — | — | — |
| ConFIT | — | — | — | — | — | — | — | — | — | — | — | — |
| *Ours* | | | | | | | | | | | | |
| GSR | — | — | — | — | — | — | — | — | — | — | — | — |
| GSR + CACL | — | — | — | — | — | — | — | — | — | — | — | — |
| HybridGSR | — | — | — | — | — | — | — | — | — | — | — | — |

**Kỳ vọng:** GSR + CACL vượt Hybrid BM25 ≥ 5 điểm MRR@3 (W. Avg), đặc biệt trên TAT-DQA nơi cấu trúc bảng phức tạp hơn. HybridGSR (kết hợp BM25 sparse signal) kỳ vọng đạt kết quả cao nhất.

**Phân tích dự kiến:**
- *Tại sao GSR vượt trội?* Nhờ Constraint Score phân biệt được tài liệu có cấu trúc toán học hợp lệ — tín hiệu mà dense/sparse retriever bỏ qua.
- *Tại sao TAT-DQA khó nhất?* Cấu trúc bảng đa dạng hơn, template coverage thấp hơn (~70% vs ~85%).
- *So sánh với SumContext:* SumContext đạt MRR@3 cao nhờ denoising, nhưng GSR+CACL kỳ vọng vượt nhờ tín hiệu cấu trúc mà summarization không bảo toàn.

### 6.2. Ablation Study

**Exp 2 — Đóng góp của từng module.** Loại bỏ lần lượt từng thành phần đề xuất tại §4 để đo mức sụt giảm hiệu năng, chứng minh không module nào bị "thừa".

| Variant | Thành phần bị loại | Kiểm chứng | MRR@3 (W. Avg) | $\Delta$ MRR@3 |
|---------|-------------------|-----------|-----------------|----------------|
| GSR + CACL (full) | — | — | — | — |
| $-$ Constraint KG | Bỏ toàn bộ KG, chỉ dùng text + entity | H1 | — | — |
| $-$ Constraint Score | Giữ KG cho GAT nhưng bỏ $\gamma \cdot \text{CS}$ khỏi $\phi_\theta$ | H1 | — | — |
| $-$ GAT Encoder | Bỏ GAT, chỉ dùng text + entity + CS truyền thống | H2 | — | — |
| $-$ Entity Score | Bỏ $\beta \cdot s_\text{ent}$ khỏi $\phi_\theta$ | — | — | — |
| $-$ CHAP → Random Neg. | Thay CHAP bằng random negatives | H3 | — | — |
| $-$ CHAP → BM25 Hard Neg. | Thay CHAP bằng BM25 hard negatives | H3 | — | — |
| $-$ $\mathcal{L}_\text{constraint}$ | Chỉ dùng $\mathcal{L}_\text{triplet}$ | — | — | — |
| $-$ Curriculum (direct Stage 3) | Bỏ Stage 1 + 2, train trực tiếp Stage 3 | — | — | — |

**Kỳ vọng:** Mỗi dòng ablation đều cho thấy $\Delta$ MRR@3 < 0, với:
- $-$ Constraint KG: sụt giảm lớn nhất → chứng minh H1.
- $-$ CHAP → Random: sụt giảm đáng kể → chứng minh H3.
- $-$ Curriculum: sụt giảm vừa phải → curriculum giúp nhưng không critical.

### 6.3. CHAP Negative Type Analysis

**Exp 3 — Đánh giá khả năng phân biệt trên từng loại mẫu âm.** Đo MRR@3 khi inference trên 3 loại negatives khác nhau:

| Negative Source | Base-RAG | Hybrid BM25 | GSR | GSR + CACL |
|----------------|----------|-------------|-----|------------|
| Random negatives | — | — | — | — |
| BM25 hard negatives | — | — | — | — |
| CHAP negatives | — | — | — | — |

**Kỳ vọng:** GSR + CACL đạt hiệu năng cao và đồng đều trên cả 3 loại → mô hình đã học *constraint semantics* thực sự, không chỉ overfit vào dạng negative cụ thể.

### 6.4. Cross-Domain Generalization

**Exp 4 — Đánh giá khả năng tổng quát hóa chéo.** Train trên FinQA + ConvFinQA (cùng nguồn FinTabNet), test trên TAT-DQA (nguồn khác, cấu trúc bảng đa dạng hơn).

| Train Set | Test Set | MRR@3 | R@3 | $\Delta$ vs. in-domain |
|-----------|----------|-------|-----|----------------------|
| FinQA + ConvFinQA | TAT-DQA | — | — | — |
| TAT-DQA | FinQA | — | — | — |
| All (in-domain) | All (in-domain) | — | — | baseline |

**Kỳ vọng:** Nhờ template-based approach dựa trên chuẩn IFRS/GAAP, mô hình tổng quát hóa tốt giữa các nguồn dữ liệu tài chính — template coverage là yếu tố quyết định.

### 6.5. Template Coverage Analysis

**Exp 5 — Kiểm chứng claim coverage.** Chạy template matching trên toàn bộ corpus để kiểm chứng ước lượng §4.3.

| Subset | Total Tables | Matched ($\geq \tau_\text{min}$) | Coverage (%) | Avg. Accounting Edges |
|--------|-------------|--------------------------------|-------------|----------------------|
| FinQA | — | — | — | — |
| ConvFinQA | — | — | — | — |
| TAT-DQA | — | — | — | — |

### 6.6. Error Analysis

**Exp 6 — Phân tích lỗi định tính.** Lấy mẫu ngẫu nhiên các trường hợp GSR + CACL truy xuất sai, phân loại theo nguyên nhân:

| Loại lỗi | Mô tả | Tỷ lệ (%) |
|----------|-------|-----------|
| Template Miss | Bảng không khớp template nào → fallback positional | — |
| Entity Confusion | Cùng công ty, khác năm (hoặc ngược lại) | — |
| Numerical Ambiguity | Cùng con số xuất hiện ở nhiều ngữ cảnh | — |
| Complex Table | Bảng lồng nhau / multi-level headers | — |
| Text-Only Answer | Câu trả lời nằm trong narrative, không dùng bảng | — |

**Mục tiêu:** Chỉ ra giới hạn cụ thể của phương pháp, gợi ý hướng cải tiến (mở rộng template library, multi-level table parsing).

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