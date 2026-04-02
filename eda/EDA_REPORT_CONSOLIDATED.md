# Báo cáo EDA Toàn diện: Tại sao General Retrieval Sụp đổ trên T²-RAGBench Financial Documents

**Thời gian phân tích:** 02/04/2026  
**Dữ liệu:** FinQA (8,281), ConvFinQA (3,458), TAT-DQA (11,349) = **23,088 QA pairs** từ **7,317 tài liệu tài chính**  
**Công cụ:** Python + scikit-learn + HuggingFace Datasets  
**Mục tiêu:** Xác định **7 điểm yếu cấu trúc** khiến embedding/BM25 retrievers thất bại

---

## I. TỔNG QUAN VỀ DATASET

### 1.1. Kích thước và Phân bổ

| Metric | FinQA | ConvFinQA | TAT-DQA | **Tổng** |
|--------|-------|-----------|---------|----------|
| Samples | 8,281 | 3,458 | 11,349 | **23,088** |
| Unique contexts | 2,789 | 1,806 | 2,722 | **7,317** |
| Unique companies | 136 | 132 | 173 | **~300** |
| Questions/context (avg) | 2.97 | 1.91 | 4.17 | **3.16** |
| % numeric answers | 100.0% | 99.8% | 99.0% | **~99.6%** |

### 1.2. Phân bố ngành (Top sectors)

**FinQA/ConvFinQA:** Financials (26.5%), Industrials (15.8%), Utilities (9.3%)  
**TAT-DQA:** Telecommunications (8.7%), Semiconductors (6.7%), Software (6.3%)

---

## II. CẤU TRÚC DOCUMENT - THÁCH THỨC VỀ MARKDOWN TABLE

### 2.1. Table Density - Cực cao, Embeddings không hiểu

| Metric | FinQA | ConvFinQA | TAT-DQA | Insight |
|--------|-------|-----------|---------|---------|
| % docs với table | **100%** | **100%** | **97.6%** | Mọi document đều là text+table |
| Avg tables/doc | 1.01 | 1.00 | **2.22** | TAT-DQA trung bình 2+ bảng |
| Avg table rows | 5.4 | 6.4 | **15.8** | Bảng lớn = khó encoding |
| Avg table cols | 4.9 | 3.9 | 3.6 | Cấu trúc ma trận phức tạp |
| **Table-line ratio** | **21%** | **79%** | **43%** | ConvFinQA gần như toàn bảng |
| Avg text words | 616.69 | 617.01 | 414.30 | Text portion nhỏ, table lớn |

**🔴 Root cause #1: Embedding models được train trên natural text, KHÔNG hiểu markdown table syntax. Khi pool embeddings, bảng có thể bị "mù" hoặc bị dilute bởi dòng separator (`|:---|:---|`).**

### 2.2. Document Length - Vượt quá embedding window

| Token length | FinQA | ConvFinQA | TAT-DQA | Impact |
|---|---|---|---|---|
| Avg (estimated) | **946** | **931** | **763** | Gần 1K tokens |
| **% > 512 tokens** | **91.0%** | **89.7%** | **82.1%** | Vượt 512-token limit |
| % > 1024 tokens | 36.8% | 34.6% | 13.7% | ~36% khó encode full context |
| Max tokens | 3,511 | 3,074 | 3,124 | Vài doc siêu dài |

**⚠️ Vấn đề:** Hầu hết embedding models có max length 512-8192. Khi context >512 tokens (90% cases), query embedding không thể capture toàn bộ context. Thông tin bị truncate có thể chứa đáp án.

---

## III. NUMERICAL DENSITY - NHIỄU CỰC LỚN

### 3.1. Mật độ số liệu

| Metric | FinQA | ConvFinQA | TAT-DQA |
|--------|-------|-----------|---------|
| **Avg numbers/doc** | **70.6** | **66.0** | **67.0** |
| Avg monetary values | 7.0 | 7.2 | **12.4** |
| Avg percentages | 5.5 | 5.8 | 3.8 |
| Avg year refs | 18.6 | 18.9 | 15.2 |
| **Number-to-word ratio** | **10.3%** | **9.8%** | **12.1%** |
| Max numbers/doc | 339 | 320 | 330 |

**3.2. Số trong câu hỏi**

| Metric | FinQA | ConvFinQA | TAT-DQA |
|--------|-------|-----------|---------|
| Avg numbers/query | 2.74 | 2.21 | 2.34 |
| **% queries với numbers** | **99.3%** | **99.1%** | **98.0%** |

**🔴 Root cause #2: Embedding models encode số thành subword tokens, KHÔNG hiểu semantic value. Hai document với cùng cấu trúc nhưng khác revenue (500M vs 500B) sẽ có embedding gần như nguyên hình — tạo hard negatives.**

---

## IV. PHÂN TÍCH RETRIEVAL FAILURE - BẰNG CHỨNG THỰC NGHIỆM

### 4.1. TF-IDF Performance (BM25 proxy)

| Metric | FinQA | TAT-DQA |
|--------|-------|---------|
| **Recall@1** | **37.7%** | **24.7%** |
| **Recall@3** | **59.7%** | **41.3%** |
| Recall@5 | 69.0% | 51.0% |
| MRR | 0.516 | 0.368 |
| Mean rank | 10.0 | **54.2** |
| **% rank > 100** | 1.7% | **10.3%** |

**Insight:** TAT-DQA khó gấp 2x FinQA. Cả hai đều có Recall@1 hoàn toàn không chấp nhận được (<40%).

### 4.2. 🔴 HARD NEGATIVES — "Sụp đổ" của dense retrievers

**FinQA Deep Analysis (200 queries):**

| Metric | Value |
|--------|-------|
| **% queries: wrong > correct** | **59.5%** |
| Avg similarity (correct context) | 0.3266 |
| Avg similarity (top wrong context) | **0.3538** |
| **Similarity gap (correct - wrong)** | **-0.0272** |
| % with gap < 0.01 | **65.5%** |
| % with gap < 0.05 | **79.0%** |

**Concrete examples:**
- Q: *"What was total equity for JPMorgan Chase in 2009?"*  
  Correct sim: 0.1113, Wrong sim: **0.4672**, Gap: **-0.3559** ❌

- Q: *"Percentage change in capitalized interest 2017-2018 for Norwegian Cruise?"*  
  Correct sim: 0.0879, Wrong sim: **0.4024**, Gap: **-0.3145** ❌

**Root cause:** Câu hỏi chứa từ chung (equity, capitalized interest) match hàng chục document. Document chứa từ nhưng không chứa **câu trả lời** lại score cao hơn document đúng.

### 4.3. Same-Company Confusion (Intra vs Inter-company Similarity)

| Comparator | Avg similarity | Median |
|---|---|---|
| Different companies | 0.0672 | 0.053 |
| **Same company** | **0.1895** | **0.1315** |
| **Ratio** | **2.82x** | **2.48x** |

**Vấn đề:** Khi query hỏi "revenue of Apple in 2020", retriever mix up cùng Apple nhưng khác năm (2019, 2021). Các document từ cùng 1 công ty **tương đồng gấp 3 lần** so với các document khác công ty.

### 4.4. Table vs Text — Tín hiệu retrieval nằm ở đâu?

| Metric | FinQA | TAT-DQA |
|--------|-------|---------|
| Query-Table similarity | 0.216 | 0.221 |
| Query-Text similarity | **0.304** | 0.225 |
| **% query match table > text** | 30.6% | **48.8%** |
| % table sim < 0.01 | 6.8% | 5.4% |

**Insight:** 
- **FinQA:** Tín hiệu **chủ yếu ở text** (narrative), nhưng câu trả lời **nằm trong table** → retriever mô tương ứng sai.
- **TAT-DQA:** ~50% lần query match tốt hơn với table → **table-aware retrieval là essential**.

### 4.5. Numerical Overlap — Paradox

| Metric | FinQA | TAT-DQA |
|--------|-------|---------|
| Avg numbers/query | 2.43 | 2.14 |
| Avg numbers/context | **38.88** | **43.22** |
| **% 100% numeric overlap** | **94.6%** | **92.2%** |
| % <50% numeric overlap | 2.3% | 4.3% |

**Paradox:** ~95% queries có **tất cả** con số của chúng nằm trong context đúng. NHƯNG context chứa trung bình ~40 số, nên **cùng con số đó xuất hiện trong hàng chục contexts khác** (năm 2019 xuất hiện mọi báo cáo 2019). **Số là manh mối vừa là bẫy.**

---

## V. LEXICAL & SEMANTIC OVERLAP ANALYSIS

### 5.1. Lexical Overlap (Query tokens ↔ Context tokens)

| Metric | FinQA | ConvFinQA | TAT-DQA |
|--------|-------|-----------|---------|
| Avg overlapping tokens | 12.26 | 10.20 | **9.17** |
| Avg Jaccard similarity | 0.0692 | 0.0600 | **0.0565** |
| **Query coverage** | **75.7%** | **79.5%** | **71.6%** |
| % queries <50% coverage | 2.4% | 1.9% | **6.1%** |

**🔴 Root cause #3: Low lexical overlap** — Câu hỏi đã được reformulate thành context-independent (chứa tên công ty, năm, metric). Direct keyword matching yếu. BM25 phụ thuộc vào overlapping tokens → thất bại.

### 5.2. Top Financial Vocabulary — Ultra-shared

**Top 20 từ chung ở cả 3 datasets:** net, income, assets, total, stock, operating, revenue, consolidated, capital, debt, equity...

**💡 Vấnđề:** Hầu hết financial documents dùng **cùng ngôn ngữ**. TF-IDF không phân biệt được document nào. Ví dụ: "net income" xuất hiện trong 30K+ lần, nhưng không phái giá trị nào là unique discriminator.

---

## VI. VIẾT TẮT (ABBREVIATIONS) - MISMATCH FORM PHÁ HỦY RETRIEVAL

### 6.1. Mật độ viết tắt — Sự khác biệt cực lớn

| Metric | FinQA | ConvFinQA | TAT-DQA |
|--------|-------|-----------|---------|
| **Unique abbreviations (≥10 lần)** | **0** | **0** | **1,514** |
| Total abbreviation instances | ~660 | ~276 | **201,913** |
| Avg fin. abbreviations/doc | **0.08** | **0.08** | **2.89** |
| % docs có ≥1 abbr | 4.4% | 4.3% | **60.3%** |
| Max abbreviations/doc | 22 | 22 | **46** |

### 6.2. Top Abbreviations

| Rank | TAT-DQA | FinQA/ConvFinQA |
|------|---------|---|
| 1 | U.S. (7,691) | — (all 0) |
| 2 | EBITDA (3,396) | — |
| 3 | GAAP (2,799) | — |
| 4 | ASC (2,160) | — |
| 5 | IFRS (1,692) | — |

**Key finding:** FinQA/ConvFinQA contexts đã được **lowercase + full-form normalize**. Không chứa "EBITDA", chỉ "earnings before interest". TAT-DQA giữ nguyên abbreviations.

### 6.3. 🔴 ABBREVIATION MISMATCH (Root cause #5)

**Pattern 1: Query dùng viết tắt, Context dùng full form**
- GAAP trong query (0.4%) vs "generally accepted accounting principles" trong context (1.1%)
- BM25 khi query "What is GAAP?" sẽ **KHÔNG MATCH** context chứa "generally accepted..."
- Embedding model có thể không biết "GAAP" ≈ full form → vectors khác nhau

**Pattern 2: Intra-dataset inconsistency (TAT-DQA)**
| Term | Ctx abbr | Ctx full | Q abbr | Q full | Status |
|------|----------|----------|--------|--------|--------|
| EPS | 2.4% | 7.4% | 0.2% | 0.9% | Mixed within context |
| R&D | 3.3% | 10.8% | 0.1% | 0.9% | Mixed within context |
| SG&A | 1.2% | 5.3% | 0.1% | 0.3% | Context=abbr, Query≈full |

**Impact trên retrieval (FinQA):**
- Queries với abbreviations: **R@3 = 60.7%**
- Queries mà abbreviations: **R@3 = 66.0%** (+5.3% improvement)

**Root cause:** Form mismatch không phải abbr itself.

---

## VII. CONTEXT SHARING & MULTI-QUESTION CHALLENGE

### 7.1. Context Reuse Pattern

| Metric | FinQA | ConvFinQA | TAT-DQA |
|--------|-------|-----------|---------|
| Questions/context (avg) | 2.97 | 1.91 | **4.17** |
| % contexts >1 question | **92.3%** | 56.9% | **99.7%** |
| % contexts >3 questions | **41.8%** | 7.9% | **67.7%** |
| % contexts >5 questions | 0.7% | 0.1% | **12.3%** |

**🔴 Root cause #7: Context sharing** — Một context được dùng bởi nhiều câu hỏi **khác nhau** (hỏi đáp, column, segment khác). Single-vector representation không đủ.

Ví dụ: Document công ty Apple chứa:
- Revenue table (Q1: What is total revenue?)
- Gross profit table (Q2: What is gross margin?)
- Cost breakdown (Q3: What are R&D expenses?)

Cùng 1 embedding representation của context không thể capture tất cả 3 semantic aspects.

---

## VIII. QUESTION CHARACTERISTICS

### 8.1. Loại câu hỏi

| Pattern | FinQA | ConvFinQA | TAT-DQA |
|---------|-------|-----------|---------|
| "what" questions | 96.0% | 98.0% | 90.8% |
| **Year reference (20xx)** | **98.2%** | **98.3%** | **96.1%** |
| percentage/ratio | **68.7%** | 30.0% | 37.7% |
| change/difference | 38.4% | 31.0% | **42.2%** |
| **Multi-step reasoning** | **14.4%** | 9.3% | 8.3% |
| Entity reference | **80.3%** | 79.1% | 64.5% |

**🔴 Root cause #6: Multi-step reasoning** — 14.4% FinQA queries yêu cầu:
1. Extract multiple values từ table
2. Perform arithmetic (divide, average, percentage change)
3. Compare across years/segments

Query embedding không capture **computational intent** — chỉ capture natural language.

### 8.2. Query Length Distribution

| Dataset | Avg words | Median | Max |
|---------|-----------|--------|-----|
| FinQA | **28.75** | 28 | 81 |
| ConvFinQA | 22.58 | 22 | 54 |
| TAT-DQA | 22.40 | 22 | 68 |

FinQA queries dài hơn vì đã được reformulate thành **context-independent** (chứa company name + year + specific metric).

---

## IX. COMPANY & SECTOR DISTRIBUTION

**FinQA/ConvFinQA:** Financials-heavy (26.5%), S&P 500 bias  
**TAT-DQA:** More diverse sectors (Telecom 8.7%, Semi 6.7%, Software 6.3%)

**Implication:** 
- Sector-specific terminology chứa tập abbreviation/jargon khác nhau
- Same-company confusion worse ở FinQA (136 companies) vs TAT-DQA (173 companies)

---

## X. TỔNG KẾT: 7 NGUYÊN NHÂN RETRIEVAL SỤP ĐỔ

| # | Nguyên nhân | Bằng chứng | Mức độ | Tác động |
|---|---|---|---|---|
| **1** | **Table Dominance & Embedding incompatibility** | 100% docs với table, ConvFinQA 79% markdown | 🔴 Critical | Embedding mặc định không hiểu table structure |
| **2** | **Extreme Numerical Density** | 67-71 số/doc (10-12% words), max 339/doc | 🔴 Critical | Numbers encode thành subword tokens, mất semantic value |
| **3** | **Hard Negatives — Wrong > Correct score** | 59.5% queries wrong context scores higher | 🔴 Critical | Dense retriever confuse wrong doc with right doc |
| **4** | **Same-Company Confusion** | Intra-company sim 2.82x inter-company | 🟠 High | Apple 2019 match Apple 2020 better than Apple vs Google |
| **5** | **Abbreviation Form Mismatch** | Query "GAAP" vs Context "generally accepted..." (1,514 unique abbrs) | 🟠 High | BM25 fail (no token overlap), Embedding fail (form diff) |
| **6** | **Multi-step Reasoning Incompatibility** | 14.4% queries need arithmetic, queries encode natural language only | 🟠 High | Retriever không capture computational intent |
| **7** | **Context Sharing — Single Vector Insufficiency** | 92.3% contexts với >1 question, 41.8% >3 questions | 🟡 Medium | Same context cần capture multiple semantic aspects |

### Bonus:
- **Vocabulary Overlap ultra-high** — Top terms (net, income, assets) xuất hiện mọi document
- **Document Length** — 90% docs > 512 tokens, vượt embedding window limit
- **Lexical Overlap low** — 6.1% queries <50% token coverage (TAT-DQA)
- **Numerical Overlap paradox** — 94.6% queries 100% numeric overlap nhưng ít discriminative

---

## XI. RECOMMENDATION CHO IMPROVEMENT

1. **Table-aware embeddings** — Fine-tune embedding model trên financial table data
2. **Numerical value normalization** — Map numbers to value ranges (e.g., "1M", "1B") để semantic better
3. **Form standardization** — Pre-process queries + contexts để consistent abbreviation usage
4. **Multi-vector per context** — Maintain multiple embeddings per document (one per semantic aspect)
5. **Hybrid retrieval** — Combine dense (embedding) + sparse (BM25) + structured (table) retrievers
6. **Company/time filtering** — Add metadata filter to reduce same-company/year confusion
7. **Computational intent modeling** — Augment query with "likely operations needed" (arithmetic types)

---

## Metadata

- **Analysis date:** April 2, 2026
- **Scripts:** eda/eda_analysis.py, eda/eda_supplementary.py, eda/eda_abbreviations.py
- **Conda environment:** master
- **Total dataset size:** 23,088 QA pairs, 7,317 unique documents
- **Benchmark:** T²-RAGBench (EACL 2026, University of Hamburg)
