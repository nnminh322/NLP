"""DatasetCollectionInterface class definition."""

from abc import ABC, abstractmethod

import pandas as pd
from g4k.internal.abstractions import BatchInferenceRunner, ResponseWrapper, Document, MetaData, RAGMethodInterface
from pydantic import BaseModel


class DatasetCollectionInterface(ABC):
    """Interface for dataset collections.

    This class defines the methods that must be implemented by any dataset collection
    class. It includes methods for initializing the dataset collection and running
    inference on the dataset.
    """

    @abstractmethod
    def __init__(
        self,
        df: pd.DataFrame,
        retrieval_query: str = "",
        meta_data_keys: list[str] = [],
        document_percentage: float = 1.0,
    ) -> None:
        pass

    @abstractmethod
    def get_context_collection(self) -> list[Document]:
        """Get the context collection."""
        pass

    @abstractmethod
    def get_data_frame(self) -> pd.DataFrame:
        """Get the DataFrame."""
        pass

    @abstractmethod
    def create_context_ids(self) -> None:
        """Create context ID for each sample and adds them to the Pydantic model."""
        pass

    @abstractmethod
    def create_prompt_meta_data(self) -> list[MetaData]:
        """Create metadata for the prompts."""
        pass

    @abstractmethod
    def prepare_contexts_for_db(self) -> list[Document]:
        """Transform the data class format to list of documents."""
        pass

    @abstractmethod
    def run(
        self,
        rag_method_instance: RAGMethodInterface,
        runner: BatchInferenceRunner,
        sys_prompt: dict,
        template_name: str,
        response_format: type[BaseModel] | str | None = None,
    ) -> ResponseWrapper:
        """Run the dataset."""
        pass
