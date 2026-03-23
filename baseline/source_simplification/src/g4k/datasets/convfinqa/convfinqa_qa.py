"""ConvFinQA dataset model."""

import json

from encourage.llm import BatchInferenceRunner, ResponseWrapper
from encourage.rag import RAGMethodInterface
from pydantic import BaseModel

from g4k.datasets.finqa.finqa_qa import FinQADatasetCollection


class ResponseModel(BaseModel):
    """Response model for ConvFinQA dataset."""

    answer: str
    reasoning: list[str]


class ConvFinQADataset(FinQADatasetCollection):
    """ConvFinQA dataset."""

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
