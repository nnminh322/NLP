"""TatDQA dataset model."""

import json
import uuid
from typing import Optional, Any

import pandas as pd
from g4k.internal.abstractions import BatchInferenceRunner, ResponseWrapper, Document, MetaData, RAGMethodInterface
from pydantic import BaseModel

from g4k.datasets.base.dataset_interface import DatasetCollectionInterface


class TatQADatasetSample(BaseModel):
    """TatQA dataset model."""

    id: str
    question: str
    answer: str
    program_answer: str
    answer_type: str
    derivation: str
    page_numbers: Optional[Any] = None
    context_id: Optional[str] = None
    context: Optional[str] = None
    company_name: Optional[str] = None
    company_sector: Optional[str] = None
    report_year: Optional[Any] = None


class TatQADatasetCollection(DatasetCollectionInterface):
    """Collection of TatQA dataset samples.

    Source: https://nextplusplus.github.io/TAT-DQA/
    """

    def __init__(
        self,
        df: pd.DataFrame,
        retrieval_query: Optional[str] = "",
        meta_data_keys: list[str] = [],
        document_percentage: float = 1.0,
        corpus_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Initialize the dataset collection."""
        self.meta_data_keys = meta_data_keys
        self.retrieval_query = retrieval_query
        self.document_percentage = document_percentage

        # Create samples from DataFrame records
        self.samples = [TatQADatasetSample(**record) for record in df.to_dict(orient="records")]

        # Create context IDs for each sample
        self.create_context_ids()

        # Build corpus
        if corpus_df is not None:
            self.corpus_samples = [
                TatQADatasetSample(**record) for record in corpus_df.to_dict(orient="records")
            ]
            self.create_corpus_context_ids()
            self.context_collection = self.prepare_contexts_for_db(use_corpus=True)
        else:
            self.context_collection = self.prepare_contexts_for_db(use_corpus=False)

        # Create metadata and prepare user prompts using samples
        self.prompt_meta_data = self.create_prompt_meta_data()
        self.user_prompts = [sample.question for sample in self.samples]

    def get_context_collection(self) -> list[str]:
        """Get the context collection."""
        return self.context_collection

    def get_data_frame(self) -> pd.DataFrame:
        """Get the DataFrame."""
        return pd.DataFrame([sample.dict() for sample in self.samples])

    def _get_deterministic_id(self, content: str) -> str:
        """Generate a deterministic UUID based on content string."""
        import hashlib
        m = hashlib.md5()
        m.update(content.encode("utf-8"))
        return str(uuid.UUID(m.hexdigest()))

    def create_context_ids(self) -> None:
        """Create context ID for each sample using deterministic hashing."""
        for sample in self.samples:
            if sample.context:
                sample.context_id = self._get_deterministic_id(sample.context)

    def create_corpus_context_ids(self) -> None:
        """Create context ID for each corpus sample using deterministic hashing."""
        for sample in self.corpus_samples:
            if sample.context:
                sample.context_id = self._get_deterministic_id(sample.context)

    def create_prompt_meta_data(self) -> list[MetaData]:
        """Create metadata from samples."""
        meta_datas = []
        for _, sample in enumerate(self.samples):
            meta_data = MetaData(
                {
                    "reference_answer": sample.program_answer,
                    "id": sample.id,
                    "reference_document": Document(
                        id=uuid.UUID(sample.context_id),
                        content=sample.context,
                    ),
                }
            )
            meta_datas.append(meta_data)
        return meta_datas

    def prepare_contexts_for_db(self, use_corpus: bool = False) -> list[Document]:
        """Prepare contexts for the vector database."""
        source_samples = self.corpus_samples if use_corpus else self.samples
        
        # Deduplicate contexts by ID
        unique_contexts = {}
        for sample in source_samples:
            if not sample.context:
                continue
            ctx_id = sample.context_id or self._get_deterministic_id(sample.context)
            if ctx_id not in unique_contexts:
                unique_contexts[ctx_id] = Document(
                    id=uuid.UUID(ctx_id) if isinstance(ctx_id, str) else ctx_id,
                    content=sample.context,
                    meta_data=MetaData(
                        {key: getattr(sample, key, None) for key in self.meta_data_keys}
                    ),
                )
        
        collection = list(unique_contexts.values())
        
        # Sampling if needed
        if self.document_percentage < 1.0:
            import random
            random.seed(42)
            n = int(len(collection) * self.document_percentage)
            collection = random.sample(collection, n)
            
        return collection

    def run(
        self,
        rag_method_instance: RAGMethodInterface,
        runner: BatchInferenceRunner,
        sys_prompt: dict,
        template_name: str,
        response_format: type[BaseModel] | str | None = None,
    ) -> ResponseWrapper:
        """Run the dataset."""
        retrieval_query = self._generate_retrieval_queries()
        responses = rag_method_instance.run(
            runner,
            sys_prompt.get("round1", ""),
            self.user_prompts,
            self.prompt_meta_data,
            retrieval_queries=retrieval_query,
            response_format=response_format,
        )

        if rag_method_instance.retrieval_only:
            return responses

        return self.post_response_processing(responses)

    def post_response_processing(
        self,
        responses: ResponseWrapper,
    ) -> ResponseWrapper:
        """Post-process the response."""
        for response in responses.response_data:
            try:
                json_response = json.loads(response.response)
                response.response = json_response["computed_formula"]
                response.meta_data["reasoning_steps"] = json_response["reasoning_steps"]
                response.meta_data["final_formula"] = json_response["final_formula"]
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                print(f"Response: {response.response}")
                response.response = ""
        return responses

    def _generate_retrieval_queries(self) -> list[str]:
        """Generate queries from dataset."""
        queries = [sample.question for sample in self.samples]
        if self.retrieval_query:
            queries = [f"Instruct: {self.retrieval_query}\nQuery: {q}" for q in queries]
        return queries
