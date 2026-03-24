#!/usr/bin/env python3
"""Standalone script to benchmark retrieval phase of T2-RAGBench."""

import os
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from g4k.internal.abstractions import BatchInferenceRunner, SamplingParams

from g4k.evaluation.config import Config
from g4k.evaluation.evaluation import evaluation
from g4k.evaluation.factory_helper import (
    get_dataset_class,
    get_rag_method_class,
    get_response_format,
)
from g4k.utils import flatten_dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set path to hydra config
config_path = str((Path(__file__).parents[3] / "conf").resolve())

def run_benchmark(cfg: Config, mode: str, dataset: str) -> None:
    """Run a specific retrieval benchmark."""
    logger.info(f"Summary: Evaluating {mode} on {dataset}")
    
    # Update config for subset
    subset_cfg_path = Path(config_path) / "dataset" / f"{dataset}.yaml"
    if not subset_cfg_path.exists():
        logger.error(f"Dataset config not found: {subset_cfg_path}")
        return
    cfg.dataset = OmegaConf.load(str(subset_cfg_path))
    
    # Load method config
    method_cfg_path = Path(config_path) / "rag" / f"{mode}.yaml"
    if not method_cfg_path.exists():
        logger.error(f"RAG method config not found: {method_cfg_path}")
        return
    method_cfg = OmegaConf.load(str(method_cfg_path))
    
    # Setup runner (needed for advanced methods)
    sampling_params = SamplingParams(
        temperature=cfg.model.temperature, max_tokens=cfg.model.max_tokens
    )
    
    # Use method-specific runner if needed (for internal LLM calls in HyDE/Summarization)
    runner_model = method_cfg.get("model", cfg.model.model_name)
    runner_port = method_cfg.get("vllm_port", cfg.vllm_port)
    runner_url = f"http://localhost:{runner_port}/v1/" if runner_port else cfg.base_url
    
    runner = BatchInferenceRunner(sampling_params, runner_model, base_url=runner_url)
    
    # Setup embeddings
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading embedding model: {cfg.vector_db.embedding_function} on {device}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=cfg.vector_db.embedding_function,
        model_kwargs={'device': device}
    )
    
    # Load dataset
    qa_dataset = load_dataset(
        cfg.dataset.name, cfg.dataset.config_name, split=cfg.dataset.split
    ).to_pandas()
    dataset_cls = get_dataset_class(cfg.dataset.runner_name)
    dataset_obj = dataset_cls(
        qa_dataset,
        cfg.dataset.retrieval_query,
        cfg.dataset.meta_data_keys,
        cfg.dataset.document_percentage,
    )
    
    # Initialize RAG method
    rag_method_cls = get_rag_method_class(method_cfg.method)
    rag_method_instance = rag_method_cls(
        context_collection=dataset_obj.get_context_collection(),
        collection_name=cfg.vector_db.collection_name,
        embedding_function=embeddings,
        top_k=cfg.vector_db.top_k,
        retrieval_only=True,
        runner=runner,
        additional_prompt=method_cfg.get("prompt", ""),
        template_name=method_cfg.get("template_name", ""),
    )
    
    # Run Retrieval
    responses = dataset_obj.run(
        rag_method_instance,
        runner,
        sys_prompt={},
        template_name=cfg.dataset.template_name,
        # response_format=get_response_format(cfg), # IR only, don't need format
    )
    
    # Save results for evaluation
    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    output_dir = Path(f"outputs/benchmark_retrieval/{cfg.dataset.name}/{mode}/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from g4k.file_manager import FileManager
    json_dump = [response.to_dict() for response in responses.response_data]
    FileManager(str(output_dir / "inference_log.json")).dump_json(json_dump)
    
    # Run evaluation to get metrics
    cfg.rag.method = method_cfg.method
    cfg.rag.retrieval_only = True
    evaluation(cfg, results_path=output_dir)
    
    logger.info(f"Finished benchmark for {mode} on {dataset}")

def main() -> None:
    """Main entry point with custom argument parsing."""
    load_dotenv(".env")
    
    # Custom parsing to extract --mode and --dataset
    parser = argparse.ArgumentParser(description="T2-RAGBench Retrieval Benchmark")
    parser.add_argument("--mode", type=str, required=True, help="Retrieval mode (e.g. base, hybrid, hyde, summarization)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset (e.g. finqa, convfinqa, tatqa)")
    
    args = parser.parse_args()
    
    # Manual Hydra initialization
    with initialize(version_base=None, config_path="../../../conf"):
        cfg = compose(config_name="defaults")
    
    run_benchmark(cfg, args.mode, args.dataset)

if __name__ == "__main__":
    main()
