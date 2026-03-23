## T²-RAGBench

T²-RAGBench is a realistic and rigorous benchmark for evaluating Retrieval-Augmented Generation (RAG) systems on financial documents combining text and tables **(over 12k Downloaded on Huggingface)**.
It contains 23,088 question-context-answer triples from 7,318 real-world financial reports, focusing on numerical reasoning and retrieval robustness.


----
### Benchmark Subsets

The benchmark comprises four subsets derived from financial datasets:

| Subset | Domain | # Documents | # QA Pairs | Avg. Tokens/Doc | Avg. Tokens/Question |
|--------|--------|-------------|-----------|-----------------|---------------------|
| FinQA | Finance | 2,789 | 8,281 | 950.4 | 39.2 |
| ConvFinQA | Finance | 1,806 | 3,458 | 890.9 | 30.9 |
| TAT-DQA | Finance | 2,723 | 11,349 | 915.3 | 31.7 |

---
### Getting Started

#### 1. Setup Environment
Ensure you have Python 3.12+ installed.

```bash
pip install -r requirements.txt
```

#### 2. Pre-fetch Data (Optional)
You can pre-cache datasets and models using the provided script:
```bash
python download_data.py -m 2
```

#### 3. Run Retrieval Baseline
Run the standalone benchmarking script with the dataset and retrieval mode:
```bash
# Available datasets: finqa, convfinqa, tatqa
# Available modes: base, hybrid, hyde, summarization
python src/g4k/evaluation/benchmark_retrieval.py --dataset finqa --mode base
```

**Mode Descriptions:**
- `base`: Standard vector retrieval (FAISS).
- `hybrid`: Combined Vector + BM25 (RankBM25).
- `hyde`: Hypothetical Document Embeddings (Needs LLM API).
- `summarization`: Retrieval on summarized context (Needs LLM API).

*Note: For `hyde` and `summarization` modes, ensure you have an LLM server (vLLM or OpenAI) running and accessible via the `base_url` in your `.env` or config.*

---

You can find more details about the benchmark in our [Paper](https://arxiv.org/abs/2506.12071), [Website](https://t2ragbench.demo.hcds.uni-hamburg.de/), and on the dataset on [Huggingface](https://huggingface.co/datasets/G4KMU/t2-ragbench).

For more details on the benchmark, please refer to our paper or write us an email at t2ragbench@gmail.com.