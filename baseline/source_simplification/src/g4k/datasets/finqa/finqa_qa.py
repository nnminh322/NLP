"""FinQA dataset model."""

import json
import uuid
from typing import Optional, Any

import pandas as pd
from g4k.internal.abstractions import BatchInferenceRunner, ResponseWrapper, Document, MetaData, RAGMethodInterface, PromptCollection, ResponseData
from pydantic import BaseModel

from g4k.datasets.base.dataset_interface import DatasetCollectionInterface
from g4k.datasets.finqa.postprocessing import extract_answer


class FinQADatasetSample(BaseModel):
    """FinQA dataset model."""

    id: str
    question: Optional[str] = None
    answer: Optional[str] = None
    program_solution: Optional[str] = None
    pre_text: Optional[str] = None
    post_text: Optional[str] = None
    table: Optional[str] = None
    context: Optional[str] = None
    report_year: Optional[Any] = None
    page_number: Optional[Any] = None
    company_symbol: Optional[str] = None
    company_name: Optional[str] = None
    company_sector: Optional[str] = None
    company_industry: Optional[str] = None
    company_headquarters: Optional[str] = None
    company_date_added: Optional[Any] = None
    company_cik: Optional[Any] = None
    company_founded: Optional[Any] = None
    question_de: Optional[str] = None
    context_de: Optional[str] = None
    program_answer: Optional[str] = None
    context_id: Optional[str] = None


class FinQADatasetCollection(DatasetCollectionInterface):
    """Collection of FinQA dataset samples."""

    def __init__(
        self,
        df: pd.DataFrame,
        retrieval_query: str = "",
        meta_data_keys: list[str] = [],
        document_percentage: float = 1.0,
        corpus_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Initialize the dataset collection."""
        self.meta_data_keys = meta_data_keys
        self.retrieval_query = retrieval_query
        self.document_percentage = document_percentage

        # Create samples from DataFrame records
        self.samples = [
            FinQADatasetSample(**{str(k): v for k, v in record.items()})
            for record in df.to_dict(orient="records")
        ]

        # Create context IDs for each sample
        self.create_context_ids()

        # Build corpus
        if corpus_df is not None:
            self.corpus_samples = [
                FinQADatasetSample(**{str(k): v for k, v in record.items()})
                for record in corpus_df.to_dict(orient="records")
            ]
            self.create_corpus_context_ids()
            self.context_collection = self.prepare_contexts_for_db(use_corpus=True)
        else:
            self.context_collection = self.prepare_contexts_for_db(use_corpus=False)

        # Create metadata and prepare user prompts using samples
        self.prompt_meta_data = self.create_prompt_meta_data()
        self.user_prompts = [sample.question for sample in self.samples]

    def get_context_collection(self) -> list[Document]:
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
                    "reference_answer": sample.program_answer or "",
                    "id": sample.id,
                    "reference_document": Document(
                        id=uuid.UUID(sample.context_id) if sample.context_id else uuid.uuid4(),
                        content=sample.context or "",
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
        template_name: str = "",
        response_format: type[BaseModel] | str | None = None,
    ) -> ResponseWrapper:
        """Run the dataset."""
        retrieval_query = self._generate_retrieval_queries()
        responses_r1: ResponseWrapper = rag_method_instance.run(
            runner,
            sys_prompt.get("round1", ""),
            self.user_prompts,
            self.prompt_meta_data,
            retrieval_queries=retrieval_query,
            response_format=response_format,
        )

        if rag_method_instance.retrieval_only:
            return responses_r1

        user_prompts = [response.response for response in responses_r1]
        prompt_collection = PromptCollection.create_prompts(
            sys_prompt["round2"],
            user_prompts,
            self.context_collection,
            self.prompt_meta_data,
            template_name=template_name,
        )
        responses_r2: ResponseWrapper = runner.run(prompt_collection)
        return self._add_meta_data(responses_r1, responses_r2)

    def _generate_retrieval_queries(self) -> list[str]:
        """Generate queries from samples."""
        queries = [f"{sample.company_name} : {sample.question}" for sample in self.samples]
        if self.retrieval_query:
            queries = [f"Instruct: {self.retrieval_query}\nQuery: {q}" for q in queries]
        return queries

    def _add_meta_data(
        self, responses: ResponseWrapper, responses_2: ResponseWrapper
    ) -> ResponseWrapper:
        """Add metadata to responses."""
        for response, response2, sample in zip(responses, responses_2, self.samples):
            response.meta_data["original_response"] = response.response
            response.meta_data["round2_sys_prompt"] = response2.sys_prompt
            response.meta_data["program_solution"] = sample.program_solution or ""
            extracted_response = response2.response.strip("\n")
            response.meta_data["extracted_response"] = extracted_response
            table = sample.table if sample.table else ""
            response.response = extract_answer(extracted_response, table)
        return responses


class FinQADatasetFineTuned(FinQADatasetCollection):
    """FinQA dataset model for fine-tuning."""

    def run(
        self,
        rag_method_instance: RAGMethodInterface,
        runner: BatchInferenceRunner,
        sys_prompt: dict,
        template_name: str = "",
        response_format: type[BaseModel] | str | None = None,
    ) -> ResponseWrapper:
        """Run the dataset."""
        retrieval_query = self._generate_retrieval_queries()
        return rag_method_instance.run(
            runner,
            sys_prompt["round1"],
            retrieval_queries=retrieval_query,
            response_format=response_format,
        )


class FinQADatasetOneShot(FinQADatasetCollection):
    """FinQA dataset model for fine-tuning."""

    def run(
        self,
        rag_method_instance: RAGMethodInterface,
        runner: BatchInferenceRunner,
        sys_prompt: dict,
        template_name: str = "",
        response_format: type[BaseModel] | str | None = None,
    ) -> ResponseWrapper:
        """Run the dataset."""
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
