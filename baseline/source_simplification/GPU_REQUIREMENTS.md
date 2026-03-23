# GPU Requirements and Setup for Retrieval Benchmarking

To reproduce the retrieval metrics from the $T^2$-RAGBench paper, the following environment is recommended:

## Hardware Requirements

### Dense Retrieval (Embedding Models)
- **Model**: `intfloat/multilingual-e5-large-instruct` (1.1 GB)
- **Minimum VRAM**: 4GB (8GB recommended for batch processing)
- **Supported Backends**: CUDA (NVIDIA), MPS (Mac Apple Silicon), CPU (Slow)

### HyDE (Query Expansion)
- **Model**: Requires a Large Language Model (e.g., Llama 3.3-70B)
- **Minimum VRAM**: 
  - For local 70B model: 80GB+ (A100/H100) or Mac Studio with 64GB+ RAM.
  - **Alternative**: Use an API (OpenAI/Gemini/vLLM hosted) to avoid local GPU requirements.

## Setup Instructions

1. **Environment**: Use Python 3.12+ (as specified in `pyproject.toml`).
2. **Dependencies**:
   ```bash
   pip install .
   ```
3. **Execution**:
   Run the newly created benchmark script:
   ```bash
   python src/g4k/evaluation/benchmark_retrieval.py
   ```

## Retrieval Phase Metrics
The script will output metrics for all **3 Modes (Subcategories)** mentioned in the paper:
1.  **Baseline Mode**: `No Context` (Zero-shot) and `Known Context` (Oracle).
2.  **Basic RAG Mode**: `Base-RAG`, `Hybrid BM25`, and `Reranker`.
3.  **Advanced RAG Mode**: `HyDE`, `Summarization`, and `SumContext`.

The script calculates:
- **MRR@3** (Mean Reciprocal Rank)
- **Recall@1, 3, 5** (Recall at different cutoff points)
