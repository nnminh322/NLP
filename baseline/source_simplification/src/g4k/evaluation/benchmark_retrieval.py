#!/usr/bin/env python3
"""Standalone script to benchmark retrieval phase of T2-RAGBench."""

import argparse
import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from encourage.llm import BatchInferenceRunner
from vllm import SamplingParams

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
    
    # Load dataset
    qa_dataset = load_dataset(cfg.dataset.name, split=cfg.dataset.split).to_pandas()
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
        embedding_function=cfg.vector_db.embedding_function,
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
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from g4k.file_manager import FileManager
    json_dump = [response.to_dict() for response in responses.response_data]
    FileManager(str(output_dir / "inference_log.json")).dump_json(json_dump)
    
    # Run evaluation to get metrics (MRR, Recall)
    cfg.rag.method = method_cfg.method
    cfg.rag.retrieval_only = True
    evaluation(cfg)
    
    logger.info(f"Finished benchmark for {mode} on {dataset}")

@hydra.main(version_base=None, config_path=config_path, config_name="defaults")
def main(cfg: Config) -> None:
    """Main entry point with custom argument parsing."""
    load_dotenv(".env")
    
    # Custom parsing to extract --mode and --dataset
    parser = argparse.ArgumentParser(description="T2-RAGBench Retrieval Benchmark")
    parser.add_argument("--mode", type=str, required=True, help="Retrieval mode (e.g. base, hybrid, hyde, summarization)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset (e.g. finqa, convfinqa, tatqa)")
    
    # We need to filter out Hydra arguments from sys.argv so they don't break our parser
    # Or just ignore unknown args
    args, unknown = parser.parse_known_args()
    
    run_benchmark(cfg, args.mode, args.dataset)

if __name__ == "__main__":
    main()
