"""GSR-CACL: Graph-Structured Retrieval + Constraint-Aware Contrastive Learning.

Paper: Structured Knowledge-Enhanced Retrieval for Financial Documents
Venue: EMNLP / SIGIR
Benchmark: T²-RAGBench (EACL 2026)

Contributions:
  C1: GSR — Graph-Structured Retrieval
       Knowledge Graph built from IFRS/GAAP accounting identities,
       GAT encoder with edge-aware message passing,
       constraint-aware scoring with ε-tolerance.

  C2: CACL — Constraint-Aware Contrastive Learning
       CHAP (Contrastive Hard-negative viA Accounting Perturbations),
       three negative types: additive, scale, entity swap,
       Zero-Sum property guarantees constraint-violating negatives.
"""

__version__ = "1.0.0"
__author__ = "natmin322"
