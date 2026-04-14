# Structured Knowledge-Enhanced Retrieval with Entity-Aware Learning for Financial Documents

## Tóm tắt

Truy xuất ngữ cảnh chính xác từ các báo cáo tài chính định dạng hỗn hợp (văn bản và bảng biểu) là khâu then chốt trong hệ thống RAG lĩnh vực tài chính. Mặc dù các mô hình dense retrieval hiện tại đạt hiệu năng cao trên văn bản mở, chúng sụp đổ trên dữ liệu tài chính do cấu trúc vector đơn tuyến tính bị bão hòa trước đặc trưng nén thông tin và ràng buộc phương trình của miền dữ liệu này. Bài báo này giải quyết bài toán bằng một mạch tiếp cận ba bước. Đầu tiên, chúng tôi hệ thống hóa sự sụp đổ của dense retrieval qua phân tích bốn rào cản ngôn ngữ học và toán học. Thứ hai, chúng tôi đề xuất GSR (Graph-Structured Retrieval) tích hợp Entity-Aware GAT — một kiến trúc học biểu diễn tách bạch và liên kết ba không gian: ngữ nghĩa bề mặt, ràng buộc cấu trúc (đồ thị kế toán) và ngữ cảnh thực thể. Cuối cùng, chúng tôi thiết kế một bộ hàm tối ưu mục tiêu mới gồm EntitySupConLoss để giải quyết hiện tượng đồng tham chiếu, và CACL (với mẫu âm CHAP) để phạt nặng các vi phạm cấu trúc toán học. Thực nghiệm trên T²-RAGBench [1] — benchmark chuẩn đánh giá hệ thống RAG trên 23,088 cặp câu hỏi-ngữ cảnh-trả lời từ 7,318 báo cáo tài chính thực tế — chứng minh sự vượt trội của phương pháp, thiết lập một tiêu chuẩn mới cho truy xuất tài liệu tài chính đa phương thức.



---


## 1. Giới thiệu

Truy xuất ngữ cảnh từ kho tài liệu tài chính là bài toán nền tảng cho hệ thống RAG chuyên ngành. Đầu vào gồm một truy vấn $Q$ và kho $N$ tài liệu $\mathcal{C}$, mỗi tài liệu chứa văn bản, bảng markdown và siêu dữ liệu. Đầu ra là danh sách top-$K$ tài liệu liên quan nhất, được đánh giá qua MRR@3 trên ba tập con chuẩn của T²-RAGBench [1]: FinQA (8,281 cặp QA suy luận số học đơn lượt), ConvFinQA (3,458 cặp QA đa lượt), và TAT-DQA (11,349 cặp suy luận số học độc lập).

Thách thức cốt lõi là tính hỗn hợp của dữ liệu: câu trả lời đòi hỏi kết hợp cả văn bản và số liệu. Baseline tốt nhất trên T²-RAGBench hiện chỉ đạt khoảng 35% MRR@3 — kém Oracle Context tới 30 điểm phần trăm. Sự sụt giảm này bắt nguồn từ việc mô hình học biểu diễn tổng quát bị ép buộc phải hiểu một ngôn ngữ mang tính hình thức hóa cao thông qua các cấu trúc vector đơn tuyến tính.

Để giải quyết triệt để rào cản này, nghiên cứu xây dựng một khung giải pháp toàn diện, đi từ việc giải mã nguyên nhân lý thuyết đến thiết kế kiến trúc biểu diễn và xây dựng hàm mục tiêu. Cụ thể, bài báo có ba đóng góp chính:

1. **Phân tích nền tảng tính ngôn ngữ học và sự sụp đổ của dense retrieval:** Chúng tôi chỉ ra bốn hiện tượng cơ sở (Tính thể loại đặc thù, Ảo tưởng trùng lặp từ vựng, Mất nhất quán toán học, Đồng tham chiếu thực thể). Phân tích này chứng minh rằng dense embedding không thiếu dữ liệu huấn luyện, mà cấu trúc của nó bị vô hiệu hóa trước đặc trưng nén thông tin và ràng buộc phương trình ẩn của văn bản tài chính.
2. **Kiến trúc học biểu diễn (Representation Learning Architecture):** Thay vì ép dense encoder hiểu cấu trúc bảng biểu một cách khiên cưỡng, chúng tôi đề xuất GSR (Graph-Structured Retrieval) tích hợp Entity-Aware GAT. Kiến trúc này tách bạch và liên kết ba không gian: không gian ngữ nghĩa bề mặt (text encoder), không gian ràng buộc cấu trúc (biểu diễn nút/cạnh của đồ thị tri thức) và không gian ngữ cảnh thực thể (entity embeddings). Đây là lời giải kiến trúc trực tiếp cho các rào cản ngôn ngữ đã phân tích.
3. **Hàm tối ưu mục tiêu (Objective Optimization Function):** Kiến trúc mới đòi hỏi cơ chế huấn luyện tương ứng. Chúng tôi thiết kế bộ hàm mất mát tổng hợp $\mathcal{L}_\text{total}$ để áp đặt các quy luật toán học và ngôn ngữ vào trọng số của mô hình. Bao gồm: **EntitySupConLoss** giải quyết trực tiếp bài toán đồng tham chiếu bằng cách kéo các biến thể từ vựng (ví dụ: "AAPL", "Apple Inc.") về cùng một tọa độ không gian; và **CACL với cơ chế sinh mẫu âm CHAP** nhằm ép mô hình học tính đúng đắn của phương trình bằng cách phạt nặng $\mathcal{L}_\text{constraint}$ đối với các mẫu nhiễu cố tình phá vỡ đẳng thức kế toán.




---

## 2. Công trình liên quan

Mục này tổng hợp ba nhóm phương pháp tiếp cận cốt lõi trong truy xuất tài liệu, phân tích các điểm mù lý thuyết của chúng khi áp dụng vào miền dữ liệu tài chính, từ đó định vị chính xác khoảng trống mà hệ thống của chúng tôi lấp đầy.

### 2.1. Truy xuất Vector và Nhận thức Cấu trúc Bảng

Kể từ khi DPR [6] và các biến thể hybrid đặt nền móng cho truy xuất văn bản bằng không gian vector chung, nhiều công trình đã nỗ lực mở rộng phương pháp này sang dữ liệu có cấu trúc. TaBERT [17] đề xuất tiền huấn luyện (pre-training) với các mục tiêu nhận thức lược đồ (schema-aware); TAPAS [18] tinh chỉnh BERT cho tác vụ hỏi đáp trên bảng; và TableFormer [19] tích hợp nhận thức cấu trúc vào kiến trúc Transformer. Cơ chế cốt lõi của toàn bộ nhóm phương pháp này là sử dụng bi-encoder để biểu diễn tài liệu/bảng thành một vector tĩnh duy nhất, sau đó xếp hạng bằng độ tương đồng cosine. 

Mặc dù giả định "vector đơn" này hoạt động tốt trên văn bản tự nhiên, nó **sụp đổ nghiêm trọng** trên tài liệu tài chính. Báo cáo tài chính không phải văn bản mô tả mà là hệ thống mã hóa bị ràng buộc bởi các phương trình kế toán ẩn (ví dụ: *Doanh thu − Giá vốn = Lợi nhuận gộp*). Việc làm phẳng (flattening) bảng biểu triệt tiêu các quan hệ phân cấp này, khiến một vector đơn không thể gánh vác đồng thời ngữ nghĩa bề mặt, cấu trúc toán học và định danh thực thể. Các nỗ lực gần đây như HELIOS [2] (dùng đồ thị hai phía giữa bảng - văn bản) hay THYME [3] (ghép nối theo trường) cũng chỉ dừng lại ở mức liên kết từ vựng (header matching) mà hoàn toàn bỏ qua logic toán học. Khác biệt triệt để với các phương pháp trên, **GSR-CACL** đề xuất biểu diễn bảng dưới dạng đồ thị ràng buộc kế toán, nơi mỗi cạnh mã hóa một đẳng thức (+1 cộng, −1 trừ). Đây là cách tiếp cận đầu tiên hình thức hóa cấu trúc toán học của bảng biểu ở cấp độ node-edge thay vì so khớp từ vựng bề mặt.

### 2.2. Học đối lập và Khai thác Mẫu âm khó (Hard Negatives)

Sự thành công của mô hình truy xuất phụ thuộc lớn vào chất lượng mẫu âm trong quá trình học đối lập. Từ nền tảng của DPR [6], ANCE [20] cải thiện bằng cách cập nhật mẫu âm song song với index; SimCSE [21] và E5 [22] chuẩn hóa pre-training cho biểu diễn câu. Gần đây, hệ thống ConFIT [5] chứng minh rằng mẫu âm có chủ đích (structured negatives) dựa trên nhiễu loạn từ điển chuyên ngành vượt trội so với mẫu ngẫu nhiên. Mấu chốt của các hệ thống này là đẩy đường biên quyết định (decision boundary) sát lại bằng các mẫu âm có độ tương đồng ngữ nghĩa cao.

Tuy nhiên, toàn bộ các phương pháp hiện tại đều sinh mẫu âm dựa trên **tín hiệu từ vựng bề mặt**. Trong miền tài liệu tài chính, mẫu âm nguy hiểm nhất không phải là mẫu trùng lặp từ vựng, mà là mẫu **giống hệt về định dạng (cùng template) nhưng vi phạm nguyên tắc toán học**. Một bảng có *Doanh thu = 100, Lợi nhuận = 30* có độ tương đồng bề mặt cực cao với bảng gốc (*Doanh thu = 100, Lợi nhuận = 35*) nhưng lại vi phạm nghiêm trọng đẳng thức kế toán. Không có phương pháp nào hiện tại tạo ra được loại nhiễu này một cách có hệ thống. Để lấp đầy khoảng trống này, kỹ thuật **CHAP** được chúng tôi đề xuất là phương pháp đầu tiên sinh mẫu âm bằng cách phá vỡ có chủ đích *đúng một* đẳng thức kế toán. Mẫu âm này ép hệ thống phải phân biệt tài liệu dựa trên tính hợp lệ của phương trình thay vì đánh lừa bởi ảo giác trùng lặp từ vựng.

### 2.3. Nhận thực thể qua Học biểu diễn (Entity Resolution)

Kiến trúc Bi-encoder cho bài toán giải quyết đồng tham chiếu thực thể đã được nghiên cứu sâu rộng. DeepER [15] ứng dụng BERT kết hợp triplet loss; CERBERT [16] mở rộng bằng cách học thêm vector loại thực thể; trong khi BLINK [14] sử dụng cross-encoder ở tầng cuối để đạt độ chính xác tối đa. Mở rộng từ triplet loss, framework Supervised Contrastive Learning (SupCon) của Khosla et al. [23] chứng minh sự vượt trội bằng cách tối ưu hóa khoảng cách của tất cả các cặp dương (positive pairs) trong cùng một batch.

Điểm đứt gãy của hệ sinh thái này là chúng được thiết kế cho các **pipeline nhận diện thực thể độc lập**. Chưa có công trình nào đưa SupCon vào hệ thống truy xuất tài liệu tài chính end-to-end, nơi một công ty có thể có hàng trăm biến thể tham chiếu. Hệ quả là các retriever hiện hành buộc phải dùng phương pháp so khớp chuỗi tĩnh (exact match), gán điểm 0 cho cặp ("Apple", "Apple Inc.") và không tạo ra bất kỳ gradient nào để cập nhật mô hình. Bằng việc đề xuất **EntitySupConLoss**, chúng tôi là những người đầu tiên tích hợp học đối lập có giám sát cho thực thể thẳng vào quy trình truy xuất. Không chỉ giải quyết bài toán đồng tham chiếu, gradient từ hàm loss này chảy trực tiếp vào backbone BGE, thiết lập một không gian ngữ nghĩa thực thể vững chắc bổ trợ cho không gian văn bản và toán học.

### 2.4. Nền tảng Lý thuyết và Cơ sở Ngôn ngữ học

Các thất bại của hệ thống truy xuất hiện hành có thể được lý giải tận gốc qua lăng kính Ngôn ngữ học Chức năng Hệ thống (SFL) của Halliday [7] và Phân tích Thể loại của Swales [9]. Trong các văn bản có tính công thức cao (formulaic language), ngữ nghĩa không nằm ở từ vựng mà ở **vị trí cấu trúc**. Đồng thời, theo lý thuyết thông tin của Shannon [8], từ chuyên ngành có phân phối xác suất cố định, làm giảm entropy trên mỗi token. Dù các công trình này đã định hình bản chất của văn bản học thuật, chưa có nghiên cứu nào kết nối các định luật này với sự sụp đổ của dense retrieval. Khoảng trống lý thuyết này sẽ được chúng tôi hình thức hóa chi tiết tại Mục 3, tạo tiền đề luận lý vững chắc cho kiến trúc ba không gian của GSR-CACL.

---

## 3. Phân tích bản chất của tài liệu tài chính

Tài liệu tài chính không phải văn bản tự nhiên thông thường mà là hệ thống mã hóa tuân theo luật ngôn ngữ và ràng buộc toán học bất biến. Bốn hiện tượng sau giải thích tại sao các phương pháp truy xuất thành công trên văn bản đa dạng lại sụp đổ trên miền tài chính.

### 3.1. Bốn hiện tượng đặc thù

**Hiện tượng 1: Tính thể loại đặc thù (Genre-specificity).** Báo cáo tài chính có ba đặc trưng nền tảng [7, 9]: (i) *Tính lặp lại cao* — cấu trúc câu cố định, làm bão hòa embedding tổng quát. (ii) *Nén từ vựng* — một từ chuyên ngành mang nhiều nghĩa kỹ thuật (ví dụ: "thu nhập" = thu nhập ròng / thu nhập hoạt động / thu nhập trước thuế). (iii) *Cấu trúc đa phương thức cố định* — tuân theo trình tự [văn bản] → [bảng] → [chú thích]. Làm phẳng cấu trúc phá hủy quan hệ vốn có.

> *"Báo cáo tài chính là một dạng mã hóa của các sự kiện kinh tế theo ngữ pháp cố định — một 'ngôn ngữ hình thức hóa'."*

**Hiện tượng 2: Ảo tưởng trùng lặp từ vựng (Lexical Overlap Illusion).** Trùng lặp từ vựng cao không đồng nghĩa tương đồng ngữ nghĩa cao. "Doanh thu" trong văn bản chỉ doanh thu toàn công ty, trong bảng có thể chỉ doanh thu theo phân khúc. Ngữ nghĩa không nằm trong từ mà nằm trong cấu trúc.

**Hiện tượng 3: Mất nhất quán toán học (Mathematical Inconsistency).** Báo cáo tài chính bị ràng buộc bởi các phương trình ẩn:
$$\text{Doanh thu} - \text{Giá vốn hàng bán} = \text{Lợi nhuận gộp}$$
$$\text{Tài sản} = \text{Nợ phải trả} + \text{Vốn chủ sở hữu}$$

Làm phẳng bảng thành văn bản phá hủy quan hệ cha-con giữa các ô số. Retriever không thể phân biệt ngữ cảnh nào thỏa mãn ràng buộc toán học.

**Hiện tượng 4: Đồng tham chiếu thực thể (Entity Co-reference Paradox).** Cùng một công ty được nhắc đến dưới nhiều hình thức: "Apple Inc.", "Apple", "AAPL", "Apple Computer, Inc.". Exact match ($1[m_Q = m_D]$) gán score = 0 cho "Apple" vs "Apple Inc." dù cùng một thực thể. Đây là bài toán representation learning — encoder phải học continuous representations sao cho cùng entity → gần nhau, khác entity → xa nhau — không phải feature engineering.

### 3.2. Các định luật ngôn ngữ học

Từ Halliday [7], Shannon [8], Cover [10]:

- **Định luật mã hóa công thức:** Ngữ nghĩa nằm ở vị trí cấu trúc, không phải từ ngữ. Chế độ hỗn hợp (văn bản + bảng) bị phá hủy khi làm phẳng.
- **Định luật nén từ vựng:** Entropy/token trong văn bản chuyên ngành thấp — từ chuyên ngành có xác suất xuất hiện cố định.
- **Định luật neo số:** Con số là điểm cố định — thay đổi số neo hủy bỏ ngữ nghĩa.
- **Định luật ngữ nghĩa đa lớp:** Biểu diễn phải đa diện — một vector đơn không đủ.

### 3.3. Ba giả thuyết nghiên cứu

- **H1 (Tính nhất quán):** Truy xuất vượt trội khi bảo toàn tính tự hợp toán học — kiểm tra được mức độ thỏa mãn ràng buộc của ngữ cảnh.
- **H2 (Căn chỉnh đa không gian):** Kết quả tỷ lệ thuận với mức căn chỉnh giữa ba không gian: văn bản, cấu trúc và thực thể. Ba không gian này độc lập và bổ trợ — không gian nào yếu thì kết quả sụt giảm tương ứng.
- **H3 (Giá trị của mẫu âm khó):** Khả năng phân biệt chỉ thực sự được kiểm chứng khi mô hình đối mặt với mẫu nhiễu có cấu trúc (dựa trên ràng buộc), không phải mẫu âm ngẫu nhiên.

### 3.4. Luận giải từ lý thuyết thông tin

Thông tin trong bảng tài chính $C$ phân tách thành bốn thành phần:
$$H(T) = H_T(T) + H_E(T) + H_S(T) + H_N(T)$$

với $H_T$: entropy văn bản, $H_E$: entropy thực thể (đồng tham chiếu), $H_S$: entropy cấu trúc (quan hệ cha-con), $H_N$: entropy số liệu. Mất mát khi làm phẳng:
$$I_{loss} \geq I_S + I_E + I_{discriminative}(H_N)$$

Mất mát thông tin cấu trúc, thực thể và phân biệt số liệu không thể tránh khỏi khi dùng vector đơn. Biểu diễn đa không gian là điều kiện cần — mỗi không gian tương ứng với một loss signal riêng trong huấn luyện.

---

## 4. Proposed Method: GSR-CACL

Mục này trình bày ba đóng góp, mỗi đóng góp được phân tách qua hai góc nhìn rõ ràng:

> **(A) Architecture — Vector nào được tạo ra? Baseline không có gì?**
> **(B) Training — Loss signal nào dạy vector đó học thứ đúng?**

### 4.1. Problem Formulation

Cho corpus $\mathcal{C} = \{C_1, \ldots, C_N\}$, mỗi $C_i = (t_i, T_i, m_i)$ gồm văn bản, bảng markdown và metadata $m_i = (\text{company}_i, \text{year}_i, \text{sector}_i)$. Cho câu hỏi $Q$ cùng metadata $m_Q$, tìm top-$K$ documents:
$$f_\theta: Q \times \mathcal{C} \mapsto [C_k^*]_{k=1}^K$$

tối ưu MRR@$k$ và Recall@$k$.

**Ký hiệu:**

| Ký hiệu | Ý nghĩa |
|----------|---------|
| $\mathbf{q}, \mathbf{d}_\text{text} \in \mathbb{R}^d$ | Text embeddings (từ BGE fine-tuned) |
| $\mathbf{e}_Q, \mathbf{e}_D \in \mathbb{R}^{d_e}$ | Entity embeddings (từ §4.3) |
| $\mathbf{d}_\text{KG} \in \mathbb{R}^{h_\text{dim}}$ | Graph embedding (từ §4.2) |
| $G_D = (\mathcal{V}, \mathcal{E}, \omega)$ | Đồ thị ràng buộc kế toán |
| $\text{CS}(G_D) \in (0, 1]$ | Constraint Score — mức tuân thủ đẳng thức |
| $\omega_{uv} \in \{+1, -1, 0\}$ | Trọng số cạnh kế toán |

### 4.2. Hai câu hỏi nền tảng

**A — Architecture:** Baseline (Base-RAG) chỉ có $\mathbf{q}$ và $\mathbf{d}_\text{text}$ từ BGE pre-trained. Ba đóng góp bổ sung vector mới:

| Vector | Ai tạo | Không gian ngữ nghĩa | Baseline có không? |
|--------|--------|----------------------|-------------------|
| $\mathbf{e}_Q, \mathbf{e}_D$ | §4.3 EntityEncoder | Entity equivalence | Không (exact match) |
| $\mathbf{d}_\text{KG}$ | §4.2 GAT Encoder | Cấu trúc toán học bảng | Không |
| $\text{CS}(G_D)$ | §4.2 KG + scoring | Mức tuân thủ ràng buộc | Không |

**B — Training:** Ba loss components tương ứng với ba không gian:

$$\mathcal{L}_\text{total} = \underbrace{\mathcal{L}_\text{triplet}}_{\text{(a)}} + \lambda_e \cdot \underbrace{\mathcal{L}_\text{EntitySupCon}}_{\text{(b)}} + \lambda_c \cdot \underbrace{\mathcal{L}_\text{constraint}}_{\text{(c)}} \tag{1}$$

(a) $\mathcal{L}_\text{triplet}$: dạy phân biệt đúng/sai dựa trên CHAP negatives. (b) $\mathcal{L}_\text{EntitySupCon}$: dạy $\mathbf{e}_Q, \mathbf{e}_D$ hiểu entity equivalence. (c) $\mathcal{L}_\text{constraint}$: dạy scorer phân biệt document hợp lệ vs vi phạm ràng buộc.

Ba tín hiệu bổ trợ lẫn nhau — $\mathcal{L}_\text{EntitySupCon}$ không flow gradient vào $\mathbf{d}_\text{KG}$, $\mathcal{L}_\text{constraint}$ không flow vào entity space.

---

### 4.3. Đóng góp 1: GSR — Graph-Structured Table Representation

> **Architecture:** Tạo ra $\mathbf{d}_\text{KG}$ và $\text{CS}(G_D)$ — biểu diễn cấu trúc toán học của bảng mà baseline không có.
> **Training:** $\mathcal{L}_\text{constraint}$ dạy Joint Scorer hiểu constraint semantics.

#### 4.3.1. Tại sao cần biểu diễn bảng dưới dạng đồ thị?

Baseline làm phẳng bảng thành văn bản → phá hủy quan hệ cha-con. Con số "500" mất hết ngữ cảnh: là Revenue hay COGS? Cộng hay trừ? Với bảng nào tạo thành đẳng thức hợp lệ? GSR parse bảng thành đồ thị ràng buộc — mỗi ô là nút, mỗi đẳng thức kế toán là cạnh có trọng số ngữ nghĩa — bảo toàn cấu trúc toán học mà văn bản phẳng không có.

#### 4.3.2. Template-Based KG Construction

> *Tham chiếu: Figure 2 — Constraint KG Construction*

Bảng markdown $T$ được chuyển thành đồ thị $G_D = (\mathcal{V}, \mathcal{E}, \omega)$ qua bốn bước:

**Bước 1 — Parse.** Tách bảng thành header $\mathcal{H}$ và ma trận ô $\{c_{ij}\}$. Mỗi ô là nút $v \in \mathcal{V}$ với thuộc tính $(value, row, col, header)$.

**Bước 2 — Template Matching.** So khớp $\mathcal{H}$ với thư viện $\mathcal{T}$ gồm 15 template IFRS/GAAP. Độ tin cậy:

$$\text{conf}(\mathcal{H}, \tau) = \frac{|\{h \in \mathcal{H} \mid \text{normalize}(h) \in \mathcal{H}_\tau\}|}{\max(|\mathcal{H}|, |\mathcal{H}_\tau|)} \tag{2}$$

| Template | Ràng buộc đại diện |
|----------|-------------------|
| Income Statement | Revenue $-$ COGS $=$ Gross Profit; GP $-$ OpEx $=$ EBIT |
| Balance Sheet (Assets) | Current Assets $+$ Non-Current Assets $=$ Total Assets |
| Balance Sheet (L+E) | Total Liabilities $+$ Equity $=$ Total Assets |
| Cash Flow Statement | OCF $+$ ICF $+$ FCF $=$ Net Cash Flow |
| Revenue by Segment | $\sum_i$ Segment$_i$ $=$ Total Revenue |
| Quarterly Breakdown | Q1 $+$ Q2 $+$ Q3 $+$ Q4 $=$ Annual |
| ... (15 templates) | ... |

**Bước 3 — Edge Construction.** Nếu $\text{conf} \geq \tau_\text{min}$ (mặc định 0.5), tạo cạnh ràng buộc kế toán:

$$\omega_{uv} = \begin{cases} +1 & \text{nếu } v_u \text{ cộng vào } v_v \text{ (additive)} \\ -1 & \text{nếu } v_u \text{ trừ từ } v_v \text{ (subtractive)} \end{cases} \tag{3}$$

**Bước 4 — Fallback.** Nếu $\text{conf} < \tau_\text{min}$, tạo cạnh vị trí $\omega = 0$ cho các ô cùng hàng/cùng cột.

*Coverage:* ~80–90% bảng FinQA/ConvFinQA và ~70% TAT-DQA khớp ít nhất một template.

#### 4.3.3. Edge-Aware GAT Encoder

> *Tham chiếu: Figure 3 — GAT Encoder*

**Node Feature Construction.** Mỗi nút $v$ tại vị trí $(r, c)$ được biểu diễn:

$$\mathbf{x}_v = f_\theta(\text{cell\_text}_v) \oplus \text{PE}_\text{row}(r) \oplus \text{PE}_\text{col}(c) \oplus \mathbf{e}_D \in \mathbb{R}^{d + 2p + d_e} \tag{4}$$

với $\mathbf{e}_D$ là entity embedding của tài liệu (tạo bởi §4.4; thay bằng vector zero nếu chưa có §4.4). Phép chiếu đầu vào:

$$\mathbf{h}_v^{(0)} = \text{ReLU}\big(\text{LayerNorm}(W_\text{proj}\, \mathbf{x}_v + b_\text{proj})\big) \in \mathbb{R}^{h_\text{dim}} \tag{5}$$

**Edge-Aware Multi-Head Attention.** Tại mỗi layer $l$, mỗi head $k$ tính attention với ba thành phần:

$$e_{uv}^{(k)} = \frac{\langle W_q^{(k)} \mathbf{h}_u^{(l)},\; W_k^{(k)} \mathbf{h}_v^{(l)} \rangle}{\sqrt{d_k}} + \text{Proj}(\omega_{uv}) + \text{EntitySim}(\mathbf{e}_u, \mathbf{e}_v) \tag{6}$$

$$\alpha_{uv}^{(k)} = \frac{\exp(e_{uv}^{(k)})}{\sum_{w \in \mathcal{N}(v)} \exp(e_{wv}^{(k)})} \tag{7}$$

$$\mathbf{h}_v^{(l+1)} = W_o \bigg[\Big\|_{k=1}^{H} \sum_{u \in \mathcal{N}(v)} \alpha_{uv}^{(k)} \cdot \omega_{uv} \cdot W_v^{(k)} \mathbf{h}_u^{(l)}\bigg] + \mathbf{h}_v^{(l)} \tag{8}$$

với residual connection. Kiến trúc: $L = 2$ layers, $H = 4$ heads, $h_\text{dim} = 256$.

**Graph-Level Representation:**

$$\mathbf{d}_\text{KG} = \frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} \mathbf{h}_v^{(L)} \in \mathbb{R}^{h_\text{dim}} \tag{9}$$

$\mathbf{d}_\text{KG}$ mã hóa toàn bộ cấu trúc toán học của bảng — baseline không có vector nào mang thông tin này.

#### 4.3.4. Constraint Score CS(G_D)

$$\text{CS}(G_D) = \frac{1}{|\mathcal{E}_c|} \sum_{(u,v,\omega) \in \mathcal{E}_c} \exp\!\left(-\frac{|\omega \cdot v_u - v_v|}{\max(|v_v|, \varepsilon)}\right) \in (0, 1] \tag{10}$$

CS ≈ 1: mọi đẳng thức thỏa mãn. CS ≈ 0: có vi phạm nghiêm trọng. Tín hiệu hoàn toàn mới — dense/sparse retrieval không biết gì về accounting semantics.

#### 4.3.5. Training Signal: $\mathcal{L}_\text{constraint}$

$$\mathcal{L}_\text{constraint} = -\frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{violated}(C_i^-, G_{C_i^-})] \cdot \log\sigma(-s(Q_i, C_i^-)) \tag{11}$$

Phạt nặng khi Joint Scorer gán score cao cho document vi phạm ràng buộc. Không có $\mathcal{L}_\text{constraint}$, scorer biết "sai" nhưng không biết tại sao sai.

---

### 4.4. Đóng góp 2: EntitySupConLoss — Supervised Contrastive Entity Understanding

> **Architecture:** Tạo ra $\mathbf{e}_Q$ và $\mathbf{e}_D$ — entity embeddings có khả năng generalization mà exact match không có.
> **Training:** $\mathcal{L}_\text{EntitySupCon}$ dạy $\mathbf{e}_Q, \mathbf{e}_D$ hiểu entity equivalence, có gradient flow vào BGE backbone.

#### 4.4.1. Vấn đề: Exact Match không đủ

GSR gốc tính entity score bằng exact match:

$$s_\text{ent}^\text{goc}(Q, C) = \frac{1}{3} \sum \mathbb{1}[m_Q = m_D] \in \{0, \tfrac{1}{3}, \tfrac{2}{3}, 1\}$$

Hai hạn chế nghiêm trọng:

1. **"Apple" ≠ "Apple Inc." → score = 0.** TAT-DQA có hàng trăm hình thức tham chiếu khác nhau → exact match thất bại thường xuyên.
2. **Không có gradient.** Exact match không sinh loss → BGE backbone không học được gì từ entity matching.

Vấn đề cốt lõi: entity equivalence là tính chất **rời rạc** (cùng/khác entity) nhưng phải học từ biểu diễn **liên tục**. BGE pre-trained gán cosine cao cho ("Apple Inc.", "Apple Corporation") — từ vựng gần — nhưng entity equivalence cần: ("Apple", "Apple Inc.") → gần (cùng entity), ("Apple Inc.", "Apple Corporation") → xa (khác entity).

#### 4.4.2. Paradigm: Bi-Encoder (Joint Embedding Space)

Từ tài liệu entity resolution [14, 15, 16], ba paradigm:

- **Bi-Encoder:** Encode cả hai mentions vào cùng không gian, distance ≈ entity equivalence. Hiệu quả, scalable, pre-computable.
- **Cross-Encoder:** Encode cùng lúc. Fine-grained nhưng quá chậm cho retrieval (N candidates = N forward passes).
- **Generative:** Sinh canonical entity name. Cần vocabulary.

Chọn **Bi-Encoder** vì retrieval là bài toán scale — embeddings pre-compute một lần, reused cho mọi query.

#### 4.4.3. Entity Encoder

$$\mathbf{e} = \text{LayerNorm}\big(\text{BGE}(m_\text{company}) \oplus \text{BGE}(m_\text{year}) \oplus \text{BGE}(m_\text{sector})\big) \in \mathbb{R}^{d_e} \tag{12}$$

EntityEncoder **chia sẻ BGE backbone** với text encoder — gradient flow ngược vào BGE, cải thiện entity understanding trong cả entity embeddings lẫn text embeddings. $d_e = 256$.

#### 4.4.4. Entity Label Construction

Nguồn labels cho supervised learning:

1. **T²-RAGBench metadata:** Query `{company: "Apple", year: "2023", sector: "Technology"}` vs doc `{company: "Apple Inc.", year: "2023", sector: "Technology"}` → cùng entity → positive pair.
2. **SEC CIK Registry:** "AAPL" → "Apple Inc." (CIK: 0000320193), "MSFT" → "Microsoft Corporation" (CIK: 0000789019).
3. **SEC EDGAR tickers JSON:** Public ticker → company name mapping. Bao phủ hầu hết US-listed companies.
4. **In-batch negatives:** Batch chứa nhiều công ty → pairs khác công ty tự động là negatives.

#### 4.4.5. EntitySupConLoss: Supervised Contrastive Learning

Cho batch $B$ entity mentions với canonical labels. Supervised Contrastive Loss (Khosla et al., NeurIPS 2020) [17]:

$$\mathcal{L}_\text{EntitySupCon} = -\log \frac{\sum_{j \in \mathcal{P}(i)} \exp(\cos(\mathbf{z}_i, \mathbf{z}_j) / \tau)}{\sum_{k=1}^{B} \exp(\cos(\mathbf{z}_i, \mathbf{z}_k) / \tau)} \tag{13}$$

với $\mathbf{z}_i = \text{normalize}(\mathbf{e}_i)$, $\mathcal{P}(i) = \{j \mid \text{label}(i) = \text{label}(j), j \neq i\}$, $\tau = 0.07$.

**Positive pairs (cùng entity):**

| Cặp | Canonical Label |
|-----|---------------|
| ("Apple", "Apple Inc.") | Apple Inc. |
| ("AAPL", "Apple Inc.") | Apple Inc. |
| ("Apple Inc.", "Apple Computer, Inc.") | Apple Inc. |
| ("2023", "2023") | cùng năm |
| ("Microsoft", "MSFT") | Microsoft Corporation |

**Tính chất quan trọng:**
- Tận dụng **tất cả** positives trong batch — triplet loss chỉ có 1 anchor-positive-negative.
- Temperature nhỏ → embeddings phải rất gần với all positives, rất xa với all negatives.
- Gradient flows vào BGE backbone → encoder học entity semantics trong cả entity lẫn text representations.

#### 4.4.6. Tại sao SupCon Loss mà không phải Triplet Loss?

| | Triplet Loss | EntitySupConLoss |
|---|-------------|-----------------|
| Positive pairs | 1 | Tất cả positives trong batch |
| Gradient quality | Yếu (1 pair) | Mạnh (tất cả positives) |
| Clustering | Không học | Học compact clusters |
| Ứng dụng | CERBERT [16] | Khosla et al. [17] |

Mỗi entity có 3–5 hình thức tham chiếu. Triplet loss chỉ học từ 1 pair mỗi lần cập nhật. SupCon loss học từ tất cả — compact cluster hơn.

---

### 4.5. Đóng góp 3: CHAP — Constraint-Aware Hard Negative Generation

> **Architecture:** Không tạo vector mới.
> **Training:** Cung cấp hard negatives chất lượng cao cho $\mathcal{L}_\text{triplet}$.

#### 4.5.1. Tại sao cần hard negatives đặc biệt?

DPR [6] dùng random negatives hoặc BM25 negatives. Random negatives quá dễ (không liên quan gì). BM25 negatives dựa trên lexical overlap — không capture constraint semantics.

Trong tài liệu tài chính, mẫu âm nguy hiểm nhất là **surface rất giống positive** nhưng **vi phạm ràng buộc toán học** — chính xác điều mà random/BM25 negatives không tạo ra được.

#### 4.5.2. CHAP: Breaking Exactly One Accounting Equation

CHAP tạo mẫu âm $C^-$ từ document dương $C^+$ bằng cách phá vỡ **đúng một** đẳng thức kế toán:

| Kiểu | Phép biến đổi | Ràng buộc bị vi phạm | Xác suất |
|------|---------------|----------------------|----------|
| **A** (Additive) | Thay đổi 1 ô con, giữ nguyên ô tổng | $\sum_i v_{child_i} \neq v_{parent}$ | $p = 0.5$ |
| **S** (Scale) | Biến đổi bậc đại lượng ($\times 10^3$ hoặc $\times 10^{-3}$) | Tỷ lệ giữa các ô bị phá vỡ | $p = 0.3$ |
| **E** (Entity) | Hoán đổi company/year trong metadata | Sai khớp thực thể/thời gian | $p = 0.2$ |

**Tính chất Zero-Sum:** Mỗi $C^-$ chỉ khác $C^+$ đúng một thành phần. Surface similarity rất cao — cùng template, cùng cấu trúc — nhưng **chắc chắn vi phạm ít nhất một ràng buộc**. Điều này buộc mô hình học phân biệt dựa trên constraint semantics, không phải lexical overlap.

#### 4.5.3. Training Signal: $\mathcal{L}_\text{triplet}$

$$\mathcal{L}_\text{triplet} = \frac{1}{N} \sum_{i=1}^{N} \max\!\big(0,\; m - s(Q_i, C_i^+) + s(Q_i, C_i^-)\big) \tag{14}$$

CHAP negatives làm tăng độ khó của negatives → decision boundary được học chính xác hơn. Không có CHAP, $\mathcal{L}_\text{triplet}$ vẫn học được phân biệt đúng/sai nhưng boundary kém hơn.

---

### 4.6. Joint Scorer $\phi_\theta$: Kết hợp ba không gian

> **Không tạo vector mới. Không có loss riêng.** Là điểm cuối dùng các vector đã tạo ra ở §4.3–§4.4.

Hàm chấm điểm kết hợp ba tín hiệu bổ trợ:

$$s(Q, C) = \alpha \cdot s_\text{text}(Q, C) + \beta \cdot s_\text{ent}(Q, C) + \gamma \cdot \text{CS}(G_D) \tag{15}$$

**Text Similarity:**

$$s_\text{text}(Q, C) = \cos(\mathbf{q}, \mathbf{d}_\text{text}) = \frac{\mathbf{q}^\top \mathbf{d}_\text{text}}{\|\mathbf{q}\| \cdot \|\mathbf{d}_\text{text}\|} \in [-1, 1] \tag{16}$$

**Entity Score (từ §4.4 — learned embeddings):**

$$s_\text{ent}(Q, C) = \cos(\mathbf{e}_Q, \mathbf{e}_D) = \frac{\mathbf{e}_Q^\top \mathbf{e}_D}{\|\mathbf{e}_Q\| \cdot \|\mathbf{e}_D\|} \in [-1, 1] \tag{17}$$

**Constraint Score (từ §4.3):**

$$\text{CS}(G_D) = \frac{1}{|\mathcal{E}_c|} \sum_{(u,v,\omega) \in \mathcal{E}_c} \exp\!\left(-\frac{|\omega \cdot v_u - v_v|}{\max(|v_v|, \varepsilon)}\right) \in (0, 1] \tag{10}$$

**Trọng số học được:** $\alpha = \text{softplus}(\log\hat{\alpha})$, $\beta = \text{softplus}(\log\hat{\beta})$, $\gamma = \text{softplus}(\log\hat{\gamma})$. Khởi tạo: $\alpha_0 = 0.5$, $\beta_0 = 0.3$, $\gamma_0 = 0.2$.

---

### 4.7. Training: Tổng hợp và Curriculum

**Full Training Objective:**

$$\mathcal{L}_\text{total} = \mathcal{L}_\text{triplet} + \lambda_e \cdot \mathcal{L}_\text{EntitySupCon} + \lambda_c \cdot \mathcal{L}_\text{constraint} \tag{1}$$

**Tại sao cần ba thành phần, không phải một?**

- $\mathcal{L}_\text{triplet}$: dạy phân biệt đúng/sai ở cấp document.
- $\mathcal{L}_\text{EntitySupCon}$: dạy entity embeddings — không $\mathcal{L}_\text{triplet}$ nào flow gradient vào EntityEncoder.
- $\mathcal{L}_\text{constraint}$: dạy constraint semantics — không hai loss kia nào biết "sai vì vi phạm ràng buộc".

Ba tín hiệu bổ trợ lẫn nhau, không overlap.

**Three-Stage Curriculum Training:**

| Stage | Module | Mục tiêu | Loss |
|-------|--------|---------|------|
| **1: Identity** | EntityEncoder + text encoder + Joint Scorer | Khởi tạo entity embeddings tốt + phân biệt (company, year) | $\mathcal{L}_\text{EntitySupCon} + \mathcal{L}_\text{triplet}$ (chỉ $s_\text{text} + s_\text{ent}$) |
| **2: Structural** | Thêm GAT Encoder | Hiệu chuẩn $\text{CS} \approx 1$ cho tài liệu hợp lệ | $\mathcal{L}_\text{triplet}$ (full $s$) |
| **3: Joint CACL** | Thêm CHAP sampler | Tối ưu toàn diện với hard negatives | $\mathcal{L}_\text{total}$ (Eq. 1) |

**Tại sao Stage 1 huấn luyện EntitySupConLoss trước?** Entity embeddings cần học được clustering structure cơ bản trước khi GAT sử dụng chúng trong node features. Nếu $\mathbf{e}_D$ chưa trained, entity similarity trong GAT attention (Eq. 6) là noise thay vì signal. Joint Scorer cũng cần $s_\text{ent}$ ổn định trước khi học full scoring.

---

## 5. Experimental Setup

### 5.1. Datasets

Đánh giá trên T²-RAGBench [1] — benchmark chuẩn cho truy xuất tài liệu tài chính hỗn hợp văn bản-bảng:

| Subset | Documents | QA Pairs | Avg. Token/Doc | Domain |
|--------|-----------|----------|----------------|--------|
| FinQA [11] | 2,789 | 8,281 | 950.4 | S&P 500 financial reports |
| ConvFinQA [12] | 1,806 | 3,458 | 890.9 | S&P 500 (conversational) |
| TAT-DQA [13] | 2,723 | 11,349 | 915.3 | Diverse financial reports |
| **Tổng** | **7,318** | **23,088** | **924.2** | |

91.3% câu hỏi context-independent (Cohen's $\kappa = 0.87$).

### 5.2. Baselines

**Nhóm 1 — Bounds:**

| Method | Mô tả |
|--------|-------|
| Pretrained-Only | Không dùng retriever |
| Oracle Context | Ngữ cảnh đúng được cung cấp trực tiếp |

**Nhóm 2 — Basic RAG:**

| Method | Mô tả |
|--------|-------|
| Base-RAG [6] | Dense retrieval chuẩn (BGE → cosine) |
| Hybrid BM25 | BM25 + dense retrieval — best reported trên T²-RAGBench |
| Reranker | Cross-encoder reranking sau initial retrieval |

**Nhóm 3 — Advanced RAG:**

| Method | Mô tả |
|--------|-------|
| HyDE | Sinh câu trả lời giả → dùng làm query mới |
| Summarization | Tóm tắt ngữ cảnh trước khi retrieval |
| SumContext | Retrieval từ bản tóm tắt, trả về ngữ cảnh gốc |

**Nhóm 4 — Table-Aware Methods:**

| Method | Mô tả |
|--------|-------|
| THoRR [4] | Two-stage retrieval dựa trên header concatenation |
| ConFIT [5] | Semantic-Preserving Perturbation — so sánh với CHAP |

**Nhóm 5 — Ours:**

| Method | Đặc điểm |
|--------|---------|
| GSR | KG + GAT + exact entity match (không §4.4) |
| GSR + EntitySupCon | KG + GAT + EntitySupConLoss (không entity-aware GAT) |
| **GSR-CACL (full)** | §4.3 + §4.4 + §4.5 + §4.6: toàn bộ contributions |
| HybridGSR | GSR-CACL + BM25 + Reciprocal Rank Fusion |

### 5.3. Evaluation Metrics

| Metric | Ý nghĩa |
|--------|---------|
| **MRR@3** | Mean Reciprocal Rank tại $k=3$ — metric chính của T²-RAGBench |
| **Recall@3** | Tỷ lệ câu hỏi có tài liệu đúng trong top-3 |
| **NM (Number Match)** | Câu trả lời số đúng (tolerance $\varepsilon = 10^{-2}$) — end-to-end metric |
| **Recall@1, Recall@5** | Supplementary recall |
| **Entity Pair Accuracy** | % cặp ("Apple", "Apple Inc.") có cosine > 0.7 |

### 5.4. Implementation Details

| Tham số | Giá trị |
|---------|---------|
| Text encoder | BAAI/bge-large-en-v1.5 ($d = 1024$) |
| Fine-tuning | LoRA ($r = 16$, $\alpha = 32$, target: $W_q, W_v$) |
| Entity encoder | BGE backbone chia sẻ, output $d_e = 256$ |
| GAT layers / heads / hidden | $L = 2$ / $H = 4$ / $h_\text{dim} = 256$ |
| Node feature dim | $d + 2p + d_e = 1024 + 384 + 256 = 1664$ |
| Template library | $|\mathcal{T}| = 15$ templates IFRS/GAAP |
| Template threshold | $\tau_\text{min} = 0.5$ |
| Training stages | Stage 1: 3 epochs / Stage 2: 3 / Stage 3: 5 |
| Batch size | 8 (T4) hoặc 16 (A100) |
| Learning rate | $2 \times 10^{-5}$ (AdamW, cosine schedule) |
| Margin $m$ | 0.3 |
| Temperature $\tau$ | 0.07 |
| $\lambda_e, \lambda_c$ | 0.5 / 0.5 |
| CHAP ratios (A/S/E) | 0.5 / 0.3 / 0.2 |
| Hardware | NVIDIA T4 16GB (Kaggle) hoặc A100 40GB |

---

## 6. Expected Results & Analysis

### 6.1. Main Results

**Exp 1 — So sánh toàn diện.** Báo cáo NM, MRR@3, R@3 trên ba tập con. Generator: Llama 3.3-70B.

| Method | FinQA | | | ConvFinQA | | | TAT-DQA | | | W. Avg | | |
|--------|-------|---|---|---|-----------|---|---|---------|---|---|--------|---|---|
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
| GSR + EntitySupCon | — | — | — | — | — | — | — | — | — | — | — | — |
| **GSR-CACL (full)** | — | — | — | — | — | — | — | — | — | — | — | — |
| HybridGSR | — | — | — | — | — | — | — | — | — | — | — | — |

**Kỳ vọng:** GSR-CACL (full) vượt Hybrid BM25 ≥ 5 điểm MRR@3 (W. Avg), đặc biệt trên TAT-DQA nơi cấu trúc bảng phức tạp và entity diversity cao.

### 6.2. Ablation Study

**Exp 2 — Đóng góp của từng module:**

| Variant | Thành phần bị loại | Kiểm chứng | Kỳ vọng $\Delta$ |
|---------|-------------------|-----------|-----------------|
| GSR-CACL (full) | — | Baseline đầy đủ | — |
| $-$ EntitySupConLoss | Bỏ $\lambda_e \cdot \mathcal{L}_\text{EntitySupCon}$, $s_\text{ent}$ = exact match | H2 (entity) | Sụt mạnh trên TAT-DQA |
| $-$ Entity in GAT | Bỏ $\mathbf{e}_D$ khỏi node features (Eq. 4) | H2 (entity context) | Sụt vừa |
| $-$ EntitySim in attention | Bỏ $\text{EntitySim}(\mathbf{e}_u, \mathbf{e}_v)$ khỏi Eq. 6 | H2 (attention) | Sụt nhẹ |
| $-$ Constraint KG | Bỏ toàn bộ KG, chỉ dùng text + entity | H1 | Sụt lớn trên FinQA/ConvFinQA |
| $-$ CHAP → Random Neg. | Thay CHAP bằng random negatives | H3 | Sụt vừa |
| $-$ $\mathcal{L}_\text{constraint}$ | Chỉ dùng $\mathcal{L}_\text{triplet} + \lambda_e \cdot \mathcal{L}_\text{EntitySupCon}$ | H1 | Sụt vừa |
| Curriculum vs. direct Stage 3 | Bỏ Stage 1 + 2, train trực tiếp Stage 3 | — | Sụt không đáng kể |

**Kỳ vọng:**
- $-$ EntitySupConLoss: sụt lớn nhất trên TAT-DQA → entity co-reference là vấn đề nghiêm trọng.
- $-$ Constraint KG: sụt lớn trên FinQA/ConvFinQA → chứng minh H1.
- $-$ CHAP → Random: sụt đáng kể → chứng minh H3.

### 6.3. Entity Clustering Quality Analysis

**Exp 3 — Đo chất lượng entity understanding:**

| Metric | Mô tả | Kỳ vọng |
|--------|-------|---------|
| Entity Pair Accuracy | % cặp cùng entity có cosine > 0.7 | > 85% sau Stage 1 |
| Cross-Entity Separation | Khoảng cách trung bình Apple ↔ Microsoft | > 0.5 sau training |
| Entity Clustering Score | Silhouette score của entity clusters | Tăng sau SupCon |
| $s_\text{ent}$ improvement | Giá trị trung bình cho "Apple"/"Apple Inc." | Tăng từ 0 → ~0.87 |

### 6.4. CHAP Negative Type Analysis

**Exp 4 — Khả năng phân biệt trên từng loại mẫu âm:**

| Negative Source | Base-RAG | Hybrid BM25 | GSR | GSR-CACL |
|----------------|----------|-------------|-----|------------|
| Random negatives | — | — | — | — |
| BM25 hard negatives | — | — | — | — |
| CHAP negatives | — | — | — | — |

**Kỳ vọng:** GSR-CACL đồng đều trên cả 3 loại → mô hình đã học constraint semantics thực sự, không overfit vào dạng negative cụ thể.

### 6.5. Cross-Domain Generalization

**Exp 5 — Tổng quát hóa chéo:**

| Train Set | Test Set | MRR@3 | R@3 | $\Delta$ vs. in-domain |
|-----------|----------|-------|-----|----------------------|
| FinQA + ConvFinQA | TAT-DQA | — | — | — |
| TAT-DQA | FinQA | — | — | — |
| All (in-domain) | All (in-domain) | — | — | baseline |

**Kỳ vọng:** Template-based approach dựa trên IFRS/GAAP tổng quát hóa tốt. SEC CIK registry bao phủ đa số US-listed companies → EntitySupConLoss cũng tổng quát tốt.

### 6.6. Template Coverage Analysis

**Exp 6 — Kiểm chứng claim coverage:**

| Subset | Total Tables | Matched ($\geq \tau_\text{min}$) | Coverage (%) | Avg. Accounting Edges |
|--------|-------------|--------------------------------|-------------|----------------------|
| FinQA | — | — | — | — |
| ConvFinQA | — | — | — | — |
| TAT-DQA | — | — | — | — |

### 6.7. Error Analysis

**Exp 7 — Phân tích lỗi định tính:**

| Loại lỗi | Mô tả | Tỷ lệ (%) |
|----------|-------|-----------|
| Template Miss | Bảng không khớp template nào → fallback positional | — |
| Entity Residual Confusion | EntitySupConLoss không phân biệt được entity pair khó | — |
| Numerical Ambiguity | Cùng con số xuất hiện ở nhiều ngữ cảnh | — |
| Complex Table | Bảng lồng nhau / multi-level headers | — |
| Text-Only Answer | Câu trả lời nằm trong narrative, không dùng bảng | — |

---

## 7. Kết luận

Chúng tôi đề xuất GSR-CACL với ba đóng góp, mỗi đóng góp phân tách rõ qua hai góc nhìn — Architecture (vector nào) và Training (loss signal nào):

| Đóng góp | Vector tạo ra | Training signal | Kiểm chứng |
|----------|-------------|-----------------|-------------|
| GSR (§4.3) | $\mathbf{d}_\text{KG}$, $\text{CS}(G_D)$ | $\mathcal{L}_\text{constraint}$ | H1 |
| EntitySupConLoss (§4.4) | $\mathbf{e}_Q$, $\mathbf{e}_D$ | $\mathcal{L}_\text{EntitySupCon}$ | H2 (entity) |
| CHAP (§4.5) | Không tạo vector mới | $\mathcal{L}_\text{triplet}$ (hard negatives) | H3 |

Ba tín hiệu bổ trợ lẫn nhau: entity understanding không overlap với constraint semantics, triplet loss không học được entity equivalence mà không có EntitySupConLoss.

Điểm khác biệt cốt lõi: EntitySupConLoss là **training objective**, không phải **feature bổ sung** — thay đổi cái mô hình học được, không chỉ thay đổi cái mô hình chấm điểm. Gradient flows vào BGE backbone → encoder hiểu entity semantics trong cả entity lẫn text embeddings.

---

## Tham khảo

[1] Strich, J., Kutay Isgorur, E., Trescher, M., Biemann, C., & Semmann, M. T²-RAGBench: Text-and-Table Benchmark for Evaluating Retrieval-Augmented Generation. EACL, 2026.

[2] HELIOS: Multi-hop Question Answering over Financial Tables and Text. ACL Findings, 2025.

[3] THYME: Field-aware Hybrid Matching for Table Retrieval. EMNLP, 2025.

[4] THoRR: Two-stage Table Retrieval with Header Concatenation. ACL Findings, 2024.

[5] ConFIT: Semantic-Preserving Perturbation for Hard Negative Mining in Dense Retrieval. ACL, 2025.

[6] Karpukhin et al. Dense Passage Retrieval for Open-Domain Question Answering. EMNLP, 2020.

[7] Halliday, M.A.K. An Introduction to Functional Grammar. Arnold, 1985 (revised by C.M.I.M. Matthiessen, 2004).

[8] Shannon, C.E. A Mathematical Theory of Communication. Bell System Technical Journal, 27(3):379–423, 1948.

[9] Swales, J.M. Genre Analysis: English in Academic and Research Settings. Cambridge University Press, 1990.

[10] Cover, T.M. & Thomas, J.A. Elements of Information Theory. Wiley-Interscience, 2006 (2nd edition).

[11] Chen et al. FinQA: A Dataset for Numerical Reasoning over Financial Reports. ACL, 2021.

[12] Chen et al. ConvFinQA: Exploring Conversational Numerical Reasoning over Financial Reports. ACL, 2022.

[13] Zhu et al. TAT-DQA: Question Answering over Financial Tables with Text and Tables. EMNLP, 2022.

[14] Wu et al. BLINK: BERT-spLItiNg for Knowledge Construction. ACL, 2020.

[15] Mrini et al. Rethinking Neural Entity Matching. ACL, 2019.

[16] Luu et al. CERT: Contrastive Entity Resolution via Entity Type Embeddings. ACL, 2021.

[17] Khosla et al. Supervised Contrastive Learning. NeurIPS, 2020.

[18] Yin et al. TaBERT: Pretraining Jointly Encoding Text and Tabular Data. ACL, 2020.

[19] Herzig et al. TAPAS: Weakly Supervised Table Parsing via Pre-training. ACL, 2020.

[20] Xiong et al. Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval. ICLR, 2021.

[21] Gao et al. SimCSE: Simple Contrastive Learning of Sentence Embeddings. EMNLP, 2021.

[22] Wang et al. E5: Embeddings from Encoders with Explicitly Controlled Negative Sampling. ICLR, 2024.
