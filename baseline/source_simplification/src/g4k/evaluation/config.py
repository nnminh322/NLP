"""Configuration dataclasses for the hydra modules."""

from dataclasses import dataclass, field
from typing import Union


@dataclass
class Model:
    """Model configuration."""

    model_name: str
    model_name_short: str
    gpu_memory_utilization: float
    temperature: float
    max_tokens: int
    max_model_len: int
    top_p: float = 0.95
    tensor_parallel_size: int = 1
    quantization: str | None = None
    tool_call_parser: str | None = None


@dataclass
class Dataset:
    """Dataset configuration."""

    name: str
    split: str
    language: str
    context_key: str
    answer_key: str
    sys_prompt_path: str
    config_name: str | None = None
    tool_name: str = ""
    runner_name: str = ""
    meta_data_keys: list[str] = field(default_factory=list)
    retrieval_query: str = ""
    template_name: str = ""
    document_percentage: float = 1.0
    response_format: dict = field(default_factory=dict)


@dataclass
class MLFlowConfig:
    """MLFlow configuration."""

    experiment_id: str
    uri: str


@dataclass
class VectorDB:
    """VectorDB configuration."""

    collection_name: str
    top_k: int
    embedding_function: str = "default"


@dataclass
class RAGConfig:
    """RAG configuration."""

    method: str = ""
    prompt: str | None = None
    retrieval_only: bool = False
    template_name: str = ""
    model: str = ""
    vllm_port: int = 18123


@dataclass
class Config:
    """Configuration dataclass for the hydra modules."""

    model: Model
    dataset: Dataset
    mlflow: MLFlowConfig
    vector_db: VectorDB
    rag: RAGConfig
    metrics: list[Union[str, dict[str, dict[str, str]]]]
    vllm_port: int
    base_url: str
