# GSR-CACL Entity Signal EDA Report
## T²-RAGBench: FinQA, ConvFinQA, TAT-DQA (23,088 samples)

**Script**: `eda/gsr_entity_signal_eda.py`  
**Output**: `eda/entity_signal_eda_output.txt`

---

## Tổng quan Datasets

| Dataset | N | Metadata richness | Unique characteristics |
|---------|---|-------------------|------------------------|
| **FinQA** | 8,281 | ★★★★★ (21 cols) | company_symbol, cik, industry, headquarters, founded |
| **ConvFinQA** | 3,458 | ★★★★★ (21 cols) | company_symbol, cik, industry, headquarters, founded |
| **TAT-DQA** | 11,349 | ★★★ (11 cols) | NO ticker/cik/industry, only company_name, year, sector |

---

## 1. METADATA FORM (Explicit vs Implicit)

### Question: Metadata tồn tại ở dạng nào?

### FINDING: **Tất cả 3 datasets đều có EXPLICIT metadata columns**

```
FinQA columns:
  company_name, company_symbol, report_year, company_sector,
  company_industry, company_headquarters, company_cik, company_founded

ConvFinQA columns:
  company_name, company_symbol, report_year, company_sector,
  company_industry, company_headquarters, company_cik, company_founded

TAT-DQA columns:
  company_name, report_year, company_sector
  ⚠️ NO company_symbol, NO cik, NO industry
```

### Metadata embedded in Query text (IMPLICIT)

| Dataset | Query embeds company_name in text |
|---------|----------------------------------|
| FinQA | 7,459/8,281 = **90.1%** ✅ |
| ConvFinQA | 3,069/3,458 = **88.8%** ✅ |
| TAT-DQA | 2,799/11,349 = **24.7%** ⚠️ |

**→ CRITICAL**: TAT-DQA chỉ có **24.7%** queries chứa company_name trong text.  
**75.3% queries TAT-DQA KHÔNG embed company name** → Entity phải lấy từ metadata column.

### Metadata trong Document (Context)

| Dataset | Doc has company (meta) | Doc has year (meta) | Doc has sector (meta) |
|---------|----------------------|---------------------|---------------------|
| FinQA | 100% | 100% | 100% |
| ConvFinQA | 100% | 100% | 100% |
| TAT-DQA | 100% | 100% | **85.2%** ⚠️ |

---

## 2. ENTITY CO-REFERENCE ANALYSIS

### Question: Có bao nhiêu Query/Document có entity đồng tham chiếu?

### DOCUMENT Entity Signal (TỪ METADATA — 100% COVERED)

| Signal | FinQA | ConvFinQA | TAT-DQA |
|--------|-------|-----------|---------|
| company_name | 100% | 100% | 100% |
| report_year | 100% | 100% | 100% |
| company_sector | 100% | 100% | 85.2% |
| company_symbol | 100% | 100% | **0%** ❌ |

### QUERY Entity Signal (TỪ TEXT — Variable coverage)

| Entity type | FinQA | ConvFinQA | TAT-DQA |
|-------------|-------|-----------|---------|
| Same entity in text | 91.2% | 90.2% | **100%** ✅ |
| Ticker symbol (AAPL) | 19.1% | 16.8% | 15.2% |
| Year reference | 98.2% | 98.3% | 96.1% |
| Quarter reference | 2.5% | 1.4% | 2.0% |
| Full company name | 73.0% | 71.5% | 56.2% |

### KEY FINDING: Entity Name Form Mismatch (Alias Problem)

```
FinQA Query: "What is the net change in Entergy's net revenue..."
  → Query text: "Entergy" (company name embedded)
  → Doc metadata: "Entergy" + "ETR" (symbol)
  → e_Q = encode("Entergy"), e_D = encode("Entergy Corporation") ← NEEDS EntitySupConLoss!

ConvFinQA Query: "What was the net cash from operating activities for Jack Henry..."
  → Query text: "Jack Henry & Associates"
  → Doc metadata: "Jack Henry & Associates" + "JKHY"
  → Match ✓

TAT-DQA Query: "What was the amount of Income before income taxes for carpenter-technology-corp..."
  → Query text: "carpenter-technology-corp" (slug format)
  → Doc metadata: "carpenter-technology-corp" (NO symbol)
  → ⚠️ 75.3% queries DON'T embed company at all
```

### Implication for GSR-CACL:

- ✅ **EntitySignal VALID cho cả Q và D**: Cả 2 đều có metadata
- ✅ **EntitySupConLoss ESSENTIAL**: Resolve "AAPL" ↔ "Apple Inc." ↔ "Apple"
- ⚠️ **TAT-DQA challenge**: Query không embed company → e_Q chỉ từ metadata
- ⚠️ **TAT-DQA**: Không có ticker/cik → EntityRegistry không resolve được

---

## 3. TABLE PRESENCE

### Question: Có bao nhiêu Query/Document có table? Ở dạng nào?

### DOCUMENT Table Presence

| Dataset | Doc has table (in context) | Avg tables/doc | Max tables/doc |
|---------|---------------------------|-----------------|----------------|
| FinQA | 100% | 0.02 | 1 |
| ConvFinQA | 100% | **1.00** | 1 |
| TAT-DQA | 97.7% | **2.17** | 15 |

### QUERY Table Presence

| Dataset | Query has table | Query has table-related language |
|---------|-----------------|----------------------------------|
| FinQA | **0%** | 58.1% |
| ConvFinQA | **0%** | 44.0% |
| TAT-DQA | **0%** | 41.4% |

### Table Format: MARKDOWN PIPE (| | |)

```
Example from TAT-DQA:
| ($ in millions) | 2019          | 2018       | 2017       |
|-----------------|---------------|------------|------------|
```

### KEY FINDING: Query KHÔNG BAO GIỜ có table

**0% queries chứa markdown table trong text**

Queries **MÔ TẢ** table content, không **CHỨA** table:
- "What was the total operating expenses for American Airlines Group in 2018?"
- "In which year did the non-current assets in the Asia Pacific region exceed..."

### Implication for GSR-CACL:

- ✅ **StructuralSignal chỉ cần cho DOCUMENT**, không cần cho Query
- ✅ **GATEncoder chỉ encode document tables** → d_KG ∈ ℝ^{256}
- ✅ **Query chỉ cần TextSignal + EntitySignal** (không cần StructuralSignal)
- ⚠️ **TAT-DQA**: Nhiều bảng nhất (avg 2.17) → KHÓ hơn, cần multi-table reasoning

---

## 4. ACCOUNTING EQUATION / IDENTITY PRESENCE

### Question: Có bao nhiêu Query/Document có phương trình kế toán?

### DOCUMENT Accounting Identity Presence

| Dataset | Doc has accounting identity | Top patterns found |
|---------|---------------------------|-------------------|
| FinQA | 50.1% | cash flow (24.1%), net income (14.3%), operating income (7.6%) |
| ConvFinQA | 52.4% | cash flow (24.8%), net income (14.5%), operating income (8.6%) |
| TAT-DQA | 54.8% | cash flow (20.6%), net income (15.9%), operating income (11.0%) |

### QUERY Accounting Identity Presence

| Dataset | Query has accounting identity | Top patterns |
|---------|------------------------------|---------------|
| FinQA | 10.5% | cash flow (2.3%), operating income (1.9%), net income (1.8%) |
| ConvFinQA | 12.6% | totaling (2.7%), operating income (2.0%), summation (1.9%) |
| TAT-DQA | 15.3% | net income (2.5%), averaging (2.5%), cash flow (1.8%) |

### Computation Operation Keywords in Queries

| Operation | FinQA | ConvFinQA | TAT-DQA |
|-----------|-------|-----------|---------|
| total | **38.2%** | **31.2%** | **21.4%** |
| change | 20.6% | 17.8% | 25.1% |
| average | 11.1% | 5.3% | 14.1% |
| difference | 3.5% | 6.2% | 5.3% |
| sum | 0.2% | 1.9% | 1.2% |
| add | 0.1% | 0.1% | 0.1% |
| subtract | 0.0% | 0.1% | 0.0% |
| multiply | 0.0% | 0.2% | 0.0% |

### KEY FINDING: Implicit Computation (No Explicit Equations)

- **Queries KHÔNG chứa explicit equation** (e.g., "Revenue - COGS = Gross Profit")
- Queries chỉ chứa **COMPUTATION KEYWORDS**: "total", "average", "change", "difference"
- ~30-40% queries yêu cầu tính toán ("total", "average", "change")
- ~15% queries chứa accounting terminology

### Implication for GSR-CACL:

- ✅ **ConstraintScore VALID**: Documents có 50-55% chứa accounting identities
- ✅ **CHAP Negatives VALID**: Tạo hard negatives bằng cách break accounting identities
- ✅ **Query chỉ cần computation keywords**: Model học implicit computation từ keywords
- ⚠️ **ConstraintScore chỉ áp dụng cho documents có tables** (50-55% docs)

---

## 5. QUESTION TYPE DISTRIBUTION

| Question Type | FinQA | ConvFinQA | TAT-DQA |
|---------------|-------|-----------|---------|
| total/sum | **39.0%** | **33.5%** | 22.6% |
| change/difference | 36.7% | 29.8% | **41.3%** |
| percentage/ratio | 16.2% | 9.0% | 7.1% |
| average/mean | 11.1% | 5.3% | 14.1% |
| year_ref | 98.2% | 98.3% | 96.1% |

---

## 6. CROSS-DATASET SUMMARY TABLE

| Metric | FinQA (n=8281) | ConvFinQA (n=3458) | TAT-DQA (n=11349) |
|--------|-----------------|---------------------|-------------------|
| **Doc: company (meta)** | 100% | 100% | 100% |
| **Doc: year (meta)** | 100% | 100% | 100% |
| **Doc: sector (meta)** | 100% | 100% | 85% |
| **Doc: symbol (meta)** | 100% | 100% | **0%** ⚠️ |
| **Doc: table** | 100% | 100% | 98% |
| **Doc: accounting eq** | 50% | 52% | 55% |
| **Query: same entity in text** | 91% | 90% | **100%** |
| **Query: embeds company name** | 90% | 89% | **25%** ⚠️ |
| **Query: ticker** | 19% | 17% | 15% |
| **Query: year** | 98% | 98% | 96% |
| **Query: table** | 0% | 0% | 0% |
| **Query: accounting eq** | 11% | 13% | 15% |
| **Query: total/sum** | 39% | 34% | 23% |
| **Query: change/diff** | 37% | 30% | 41% |
| **Avg tables/doc** | 0.02 | **1.00** | **2.17** |

---

## 7. ARCHITECTURE IMPLICATIONS FOR GSR-CACL

### ✅ VALID DESIGN DECISIONS (Confirmed by EDA)

```
┌──────────────────────────────────────────────────────────────┐
│                    GSR-CACL ARCHITECTURE                     │
│                                                              │
│  Query Q:                                                    │
│    TextSignal ✓  (query text — always available)             │
│    EntitySignal ✓ (metadata — always available)              │
│    StructuralSignal ✗ (query never has tables)               │
│                                                              │
│  Document D:                                                 │
│    TextSignal ✓  (document text — always available)          │
│    EntitySignal ✓ (metadata — always available)              │
│    StructuralSignal ✓ (tables in 98-100% docs)               │
│                                                              │
│  JointScorer:                                                │
│    s_text = cos(q_text, d_text) × gate(q) + 0.2·MLP(d_KG)    │
│    s_entity = cos(e_Q, e_D)  ← NEEDS EntitySupConLoss        │
│    s_struct = MLP_cs(constraint_features)                    │
└──────────────────────────────────────────────────────────────┘
```

### ⚠️ CHALLENGES IDENTIFIED

| Challenge | Dataset | Impact | Mitigation |
|-----------|---------|--------|------------|
| **Ticker resolution** | FinQA, ConvFinQA | "AAPL" ≠ "Apple Inc." | EntitySupConLoss + CIK_MAPPING |
| **No ticker/cik** | TAT-DQA | EntityRegistry không resolve được | Fallback sang substring matching |
| **Query không embed company** | TAT-DQA (75%) | e_Q chỉ từ metadata | Dùng `company_name` column |
| **Multi-table documents** | TAT-DQA | Average 2.17 tables/doc | Cần multi-table KG encoding |
| **Alias mismatch** | All | "carpenter-technology-corp" vs "Carpenter Technology Corp" | Normalize + CIK fallback |
| **No explicit equations** | All queries | Chỉ implicit keywords | CHAP tạo hard negatives |

### ✅ EntitySupConLoss IS ESSENTIAL

```python
# TAT-DQA: "carpenter-technology-corp" (query) vs "carpenter-technology-corp" (doc)
# → BOTH have metadata company_name BUT text forms may differ

# FinQA: "Entergy" (query) vs "Entergy" (doc) + "ETR" (symbol)
# → Need EntitySupConLoss to learn "ETR" → "Entergy"

# ConvFinQA: "Jack Henry & Associates" (query) vs "Jack Henry & Associates" (doc)
# → Match nhưng ticker resolution vẫn cần
```

---

## 8. RECOMMENDATIONS

### Priority 1: Entity Resolution (EntitySupConLoss)

```python
# MUST DO:
1. Xây dựng EntityRegistry với CIK_MAPPING cho FinQA/ConvFinQA
2. Fallback heuristic cho TAT-DQA (substring matching)
3. Normalize company names: lowercase, remove special chars
4. EntitySupConLoss với τ=0.07 để cluster entity embeddings
```

### Priority 2: Multi-table Handling (TAT-DQA)

```python
# TAT-DQA có 2.17 tables/doc (max 15):
1. GATEncoder xử lý từng table riêng → vector riêng
2. Merge multi-table representations (concat/attention)
3. Hoặc flatten tất cả tables thành 1 KG lớn
```

### Priority 3: ConstraintScore Coverage

```python
# 50-55% docs có accounting identities
# 45-50% docs KHÔNG có → cần fallback strategy
Fallback: s_struct = 0.5 (neutral) cho docs không có constraints
```

### Priority 4: Query-Document Entity Alignment

```python
# 75% TAT-DQA queries KHÔNG embed company name
# → e_Q = encode(company_name_from_metadata) LUÔN
# → KHÔNG dùng entity_encode(query_text) cho TAT-DQA
```

---

## 9. FILES GENERATED

| File | Description |
|------|-------------|
| `eda/gsr_entity_signal_eda.py` | Main EDA script |
| `eda/entity_signal_eda_output.txt` | Raw console output |
| `eda/ENTITY_SIGNAL_EDA_REPORT.md` | This report |

---

## 10. DATASET SPECIFIC NOTES

### FinQA (n=8,281)
- Rich metadata: symbol, cik, industry, headquarters
- 90.1% queries embed company name
- Tables: Low density (avg 0.02 tables/doc)
- **Strength**: Easy entity matching (most queries embed company)
- **Challenge**: Low table density → ConstraintScore ít áp dụng

### ConvFinQA (n=3,458)
- Rich metadata: symbol, cik, industry, headquarters
- 88.8% queries embed company name
- Tables: Consistent (exactly 1 table/doc)
- **Strength**: 1-to-1 query-table matching
- **Challenge**: Conversational context (multi-turn)

### TAT-DQA (n=11,349)
- Sparse metadata: NO symbol, NO cik, NO industry
- **Only 24.7% queries embed company name** ⚠️
- Tables: High density (avg 2.17 tables/doc, max 15)
- **Strength**: Most challenging → best test for GSR-CACL
- **Challenge**: Multi-table, no ticker, implicit entity references
