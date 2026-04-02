#!/usr/bin/env python3
"""
run_gsr.py — Main entry point to run GSR / HybridGSR on T²-RAGBench.

Mirrors the pattern of g4k's benchmark_retrieval.py but for GSR methods.

Usage:
    python -m gsr_cacl.run_gsr --mode gsr --dataset finqa --embedding intfloat/multilingual-e5-large-instruct
    python -m gsr_cacl.run_gsr --mode hybridgsr --dataset tatqa
    python -m gsr_cacl.run_gsr --mode gsr --dataset all

All results are saved to outputs/gsr_benchmark/.
"""

import sys
import os

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from gsr_cacl.benchmark_gsr import run_gsr_benchmark, main as cli_main

if __name__ == "__main__":
    cli_main()
