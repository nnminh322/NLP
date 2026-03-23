"""TatDQA dataset model."""

import json
import uuid
from typing import Optional

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
    page_numbers: Optional[int] = None
    context_id: Optional[str] = None
    context: Optional[str] = None
    company_name: Optional[str] = None
    company_sector: Optional[str] = None
    report_year: Optional[str] = None


class TatQADatasetCollection(DatasetCollectionInterface):
    """Collection of TatQA dataset samples.

    Source: https://nextplusplus.github.io/TAT-DQA/
    """

    def __init__(
        self,
        df: pd.DataFrame,
        retrieval_query: Optional[str] = "",
        meta_data_keys: list[str] = [],
    ) -> None:
        """Initialize the dataset collection."""
        self.meta_data_keys = meta_data_keys
        self.retrieval_query = retrieval_query

        # Create samples from DataFrame records
        self.samples = [TatQADatasetSample(**record) for record in df.to_dict(orient="records")]

        # Create context IDs for each sample
        self.create_context_ids()

        # Create metadata and prepare user prompts using samples
        self.prompt_meta_data = self.create_prompt_meta_data()
        self.user_prompts = [sample.question for sample in self.samples]
        self.context_collection = self.prepare_contexts_for_db()

    def get_context_collection(self) -> list[str]:
        """Get the context collection."""
        return self.context_collection

    def get_data_frame(self) -> pd.DataFrame:
        """Get the DataFrame."""
        return pd.DataFrame([sample.dict() for sample in self.samples])

    def create_context_ids(self) -> None:
        """Create context ID for each sample and adds them to the Pydantic model."""
        context_dict = {}
        for sample in self.samples:
            if sample.context not in context_dict:
                context_dict[sample.context] = str(uuid.uuid4())
            sample.context_id = context_dict[sample.context]

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

    def prepare_contexts_for_db(self) -> list[Document]:
        """Prepare contexts from samples for the vector database."""
        context_collection = []
        for sample in self.samples:
            if sample.context:
                context = Document(
                    id=sample.context_id,
                    content=sample.context,
                    meta_data=MetaData(
                        {key: getattr(sample, key, None) for key in self.meta_data_keys}
                    ),
                )
                context_collection.append(context)
        return context_collection

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
