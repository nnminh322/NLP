"""Module for evaluation of QA datasets."""

import logging
from pathlib import Path

import hydra
import mlflow
import mlflow.data.pandas_dataset
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from g4k.internal.abstractions import BatchInferenceRunner, ResponseWrapper, SamplingParams

from g4k.evaluation.config import Config
from g4k.evaluation.evaluation import main as evaluation
from g4k.evaluation.factory_helper import (
    get_dataset_class,
    get_rag_method_class,
    get_response_format,
)
from g4k.file_manager import FileManager
from g4k.utils import flatten_dict

config_path = str((Path(__file__).parents[3] / "conf").resolve())

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=config_path, config_name="defaults")
def main(cfg: Config) -> None:
    """Main function for evaluation of QA datasets."""
    # Load dataset from Huggingface
    load_dotenv(".env")

    sampling_params = SamplingParams(
        temperature=cfg.model.temperature, max_tokens=cfg.model.max_tokens
    )
    runner = BatchInferenceRunner(sampling_params, cfg.model.model_name, base_url=cfg.base_url)
    # sys_prompt = FileManager(cfg.dataset.sys_prompt_path).load_yaml()

    ## Run the Inference
    mlflow.set_tracking_uri(cfg.mlflow.uri)
    mlflow.set_experiment(experiment_name=cfg.dataset.name)

    with mlflow.start_run():
        qa_dataset = load_dataset(
            cfg.dataset.name, cfg.dataset.config_name, split=cfg.dataset.split
        ).to_pandas()
        dataset = get_dataset_class(cfg.dataset.runner_name)
        dataset_obj = dataset(
            qa_dataset,
            cfg.dataset.retrieval_query,
            cfg.dataset.meta_data_keys,
            cfg.dataset.document_percentage,
        )

        mlflow.log_params(flatten_dict(cfg))
        mlflow.log_params({"dataset_size": len(dataset_obj.get_data_frame())})
        mlflow.log_input(
            mlflow.data.pandas_dataset.from_pandas(
                dataset_obj.get_data_frame(), name=cfg.dataset.name
            ),
            context="inference",
        )

        intern_runner = runner
        if cfg.rag.method in ["Hyde", "Summarization", "SummarizationContextRAG"]:
            intern_runner = BatchInferenceRunner(
                sampling_params,
                cfg.rag.model,
                base_url=f"http://localhost:{cfg.rag.vllm_port}/v1/",
            )
            logger.info(
                f"""
                Using the following model for {cfg.rag.method}: {cfg.rag.model}
                on port {cfg.rag.vllm_port}
                """
            )

        with mlflow.start_span(name="root"):
            rag_method = get_rag_method_class(cfg.rag.method)
            rag_method_instance = rag_method(
                context_collection=dataset_obj.get_context_collection(),
                collection_name=cfg.vector_db.collection_name,
                embedding_function=cfg.vector_db.embedding_function,
                top_k=cfg.vector_db.top_k,
                retrieval_only=cfg.rag.retrieval_only,
                runner=intern_runner,
                additional_prompt=cfg.rag.prompt or "",
                template_name=cfg.rag.template_name,
            )
            responses: ResponseWrapper = dataset_obj.run(
                rag_method_instance,
                runner,
                sys_prompt={},
                template_name=cfg.dataset.template_name,
                response_format=get_response_format(cfg),
            )

        json_dump = [response.to_dict() for response in responses.response_data]
        FileManager(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + "/inference_log.json"
        ).dump_json(json_dump)
        json_dump = [flatten_dict(response.to_dict()) for response in responses.response_data]

        active_run = mlflow.active_run()
        run_name = active_run.info.run_name if active_run else "responses"

        try:
            mlflow.log_table(data=pd.DataFrame(json_dump), artifact_file=f"{run_name}.json")
        except Exception as e:
            print(f"Failed to log table to MLflow: {e}")

        # Evaluate the retrieval
        evaluation(cfg)


if __name__ == "__main__":
    main()
