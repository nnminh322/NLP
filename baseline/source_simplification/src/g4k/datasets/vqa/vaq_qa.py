"""VAQ dataset model for Visual Question Answering."""

import json
import uuid
from typing import Any, List, Optional

import pandas as pd
from encourage.llm import BatchInferenceRunner, ResponseWrapper
from encourage.prompts.context import Document
from encourage.prompts.meta_data import MetaData
from encourage.rag import RAGMethodInterface
from pydantic import BaseModel

from g4k.datasets.base.dataset_interface import DatasetCollectionInterface


class VAQDatasetSample(BaseModel):
    """VAQ dataset model for Visual Question Answering."""

    id: str
    question: Optional[str] = ""
    answer: Any
    answer_type: str
    category: str
    file_path: str
    company_symbol: Optional[str] = None
    report_year: Optional[Any] = None
    page_number: Optional[Any] = None
    vqa_split: Optional[str] = None
    Symbol: Optional[str] = None
    company_name: Optional[str] = None
    company_sector: Optional[str] = None
    company_industry: Optional[str] = None
    company_headquarters: Optional[str] = None
    company_date_added: Optional[str] = None
    company_cik: Optional[float] = None
    company_founded: Optional[str] = None
    filepath: Optional[str] = None
    tables: Optional[Any] = None
    annotations: Optional[Any] = None
    html_source: Optional[Any] = None
    split: Optional[str] = None
    table_id: Optional[int] = None
    context: Optional[Any] = None
    context_id: Optional[str] = None


class VAQDatasetCollection(DatasetCollectionInterface):
    """Collection of VAQ dataset samples for Visual Question Answering."""

    def __init__(
        self,
        df: pd.DataFrame,
        retrieval_query: str = "",
        meta_data_keys: List[str] = [],
    ) -> None:
        """Initialize the dataset collection.

        Args:
            df: DataFrame containing the dataset samples
            retrieval_query: Optional instruction for retrieval queries
            meta_data_keys: List of keys to include in metadata

        """
        self.meta_data_keys = meta_data_keys
        self.retrieval_query = retrieval_query

        # Create samples from DataFrame records
        self.df = df
        df = df.reset_index()
        df["id"] = df["index"].apply(lambda x: f"va_qa_{x}")
        self.samples = [
            VAQDatasetSample(**{str(k): v for k, v in record.items()})
            for record in df.to_dict(orient="records")
        ]

        # Create context IDs for each sample
        self.create_context_ids()

        # Create metadata and prepare user prompts using samples
        self.prompt_meta_data = self.create_prompt_meta_data()
        self.user_prompts = [sample.question for sample in self.samples]
        self.context_collection = self.prepare_contexts_for_db()

    def get_context_collection(self) -> List[Document]:
        """Get the context collection."""
        return self.context_collection

    def get_data_frame(self) -> pd.DataFrame:
        """Get the DataFrame."""
        return self.df.drop(columns=["tables", "annotations", "html_source"])

    def create_context_ids(self) -> None:
        """Create context ID for each sample and update both samples and qa_dataset."""
        # Create mapping of unique contexts to UUIDs
        context_dict = {}
        for sample in self.samples:
            if sample.context not in context_dict:
                context_dict[sample.context] = str(uuid.uuid4())
            sample.context_id = context_dict[sample.context]

    def create_prompt_meta_data(self) -> List[MetaData]:
        """Create metadata from samples."""
        meta_datas = []
        for sample in self.samples:
            meta_data_dict = {
                "reference_answer": sample.answer,
                "id": sample.id,
                "reference_document": Document(
                    id=sample.context_id or uuid.uuid4(),
                    content=sample.context or "",
                    meta_data=MetaData(
                        {
                            "file_path": sample.file_path,
                            "page_number": sample.page_number,
                        }
                    ),
                ),
            }
            # Add additional metadata if available
            if sample.company_name:
                meta_data_dict["company_name"] = sample.company_name
            if sample.report_year:
                meta_data_dict["report_year"] = sample.report_year
            if sample.table_id:
                meta_data_dict["table_id"] = sample.table_id

            meta_data = MetaData(meta_data_dict)
            meta_datas.append(meta_data)
        return meta_datas

    def prepare_contexts_for_db(self) -> List[Document]:
        """Prepare contexts from samples for the vector database."""
        context_collection = []
        for sample in self.samples:
            if sample.context:
                context = Document(
                    id=sample.context_id,
                    content=sample.context or "",
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
        template_name: str = "",
        response_format: type[BaseModel] | str | None = None,
    ) -> ResponseWrapper:
        """Run the dataset with a single round of queries.

        Args:
            rag_method_instance: RAG method to use
            runner: Batch inference runner
            sys_prompt: System prompts to use
            template_name: Template name to use
            response_format: Response format to use

        Returns:
            List of response wrappers

        """
        retrieval_query = self._generate_retrieval_queries()
        responses = rag_method_instance.run(
            runner,
            sys_prompt["round1"],
            self.user_prompts,
            self.prompt_meta_data,
            retrieval_queries=retrieval_query,
            response_format=response_format,
        )
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
        """Generate queries from samples."""
        queries = [f"{sample.company_name} : {sample.question}" for sample in self.samples]
        if self.retrieval_query:
            queries = [f"Instruct: {self.retrieval_query}\nQuery: {q}" for q in queries]
        return queries
