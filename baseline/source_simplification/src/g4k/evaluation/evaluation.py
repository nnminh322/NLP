"""Module for evaluation of retrieval and QA results with MLflow tracking."""

import logging
from pathlib import Path

import hydra
import mlflow
from encourage.llm import BatchInferenceRunner, Response
from encourage.metrics import Metric, MetricOutput
from vllm import SamplingParams

from g4k.evaluation.config import Config
from g4k.evaluation.factory_helper import load_metrics
from g4k.file_manager import FileManager
from g4k.utils import flatten_dict

logger = logging.getLogger(__name__)
config_path = str((Path(__file__).parents[3] / "conf").resolve())


@hydra.main(version_base=None, config_path=config_path, config_name="defaults")
def main(cfg: Config) -> None:
    """Main function for evaluation of QA results with MLflow tracking."""
    # Set MLflow tracking configuration
    mlflow.set_tracking_uri(cfg.mlflow.uri)
    mlflow.set_experiment(experiment_name=cfg.dataset.name)

    # Run the evaluation with MLflow tracking
    if mlflow.active_run().info.run_id if mlflow.active_run() else False:  # type: ignore
        evaluation(cfg)
    else:
        with mlflow.start_run():
            evaluation(cfg)


def evaluation(cfg: Config) -> None:
    """Evaluate the QA results with MLflow tracking."""
    flat_config = flatten_dict(cfg)
    mlflow.log_params(flat_config)

    with mlflow.start_span(name="loading_results"):
        results_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        if not results_path.exists() or not results_path.is_dir():
            raise ValueError(f"Results folder not found: {results_path}")

        # Convert to ResponseDataCollection format
        responses_json = FileManager(list(results_path.glob("inference_log.json"))[0]).load_json()
        responses = [Response.from_dict(item) for item in responses_json]

        logger.info(f"Loaded {len(responses)} responses!")

    sampling_params = SamplingParams(
        temperature=cfg.model.temperature, max_tokens=cfg.model.max_tokens
    )
    runner = BatchInferenceRunner(sampling_params, cfg.model.model_name, base_url=cfg.base_url)

    # Load metrics
    metrics: list[Metric] = load_metrics(cfg.metrics, runner)
    metrics_log = []
    for metric in metrics:
        result: MetricOutput = metric(responses)
        metrics_log.append({metric.name: result.to_dict()})

        mlflow.log_metric(metric.name, result.score)

    FileManager(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + "/metrics_log.json"
    ).dump_json(metrics_log, pydantic_encoder=True)


if __name__ == "__main__":
    main()
