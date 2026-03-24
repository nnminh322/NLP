"""Internal abstractions to replace the private 'encourage' dependency."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel

@dataclass
class SamplingParams:
    """Mock for vllm.SamplingParams to avoid heavy dependency."""
    temperature: float = 0.0
    max_tokens: int = 512
    # Add other common params if needed
    top_p: float = 1.0
    stop: List[str] = field(default_factory=list)

@dataclass
class Document:
    """Internal Document class."""
    page_content: str = ""
    meta_data: Union[MetaData, Dict[str, Any]] = field(default_factory=dict)
    id: Any = None
    content: str = ""

    def __post_init__(self):
        if self.content and not self.page_content:
            self.page_content = self.content
        if self.page_content and not self.content:
            self.content = self.page_content

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_content": self.page_content,
            "meta_data": self.meta_data if isinstance(self.meta_data, dict) else self.meta_data.data,
            "id": self.id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        return cls(
            page_content=data.get("page_content", ""),
            meta_data=data.get("meta_data", {}),
            id=data.get("id")
        )
inline: 

@dataclass
class MetaData:
    """Internal MetaData class for prompt engineering."""
    data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Response:
    """Internal Response class for LLM outputs."""
    content: str
    raw: Any = None
    meta_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ResponseData:
    """Internal ResponseData class for RAG execution results."""
    query: str
    retrieved_docs: List[Document]
    generated_response: Optional[str] = None
    meta_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "retrieved_docs": [doc.to_dict() for doc in self.retrieved_docs],
            "generated_response": self.generated_response,
            "meta_data": self.meta_data
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResponseData":
        return cls(
            query=data["query"],
            retrieved_docs=[Document.from_dict(doc) for doc in data.get("retrieved_docs", [])],
            generated_response=data.get("generated_response"),
            meta_data=data.get("meta_data", {})
        )

@dataclass
class ResponseWrapper:
    """Wrapper for multiple responses."""
    response_data: List[ResponseData]

    def __iter__(self):
        return iter(self.response_data)

    def __len__(self):
        return len(self.response_data)

@dataclass
class Prompt:
    """Internal Prompt class."""
    system_prompt: str
    user_prompt: str
    meta_data: MetaData = field(default_factory=MetaData)

class PromptCollection:
    """Collection of prompts for batch processing."""
    def __init__(self, prompts: List[Prompt]):
        self.prompts = prompts

    def __iter__(self):
        return iter(self.prompts)

    def __len__(self):
        return len(self.prompts)

    @staticmethod
    def create_prompts(
        sys_prompt: str,
        user_prompts: List[str],
        contexts: List[Document],
        meta_data: List[MetaData],
        template_name: str = ""
    ) -> "PromptCollection":
        prompts = []
        for u_prompt, context, meta in zip(user_prompts, contexts, meta_data):
            # Simple template merging (placeholder for more complex logic if needed)
            full_user_prompt = f"Context: {context.page_content}\n\nQuestion: {u_prompt}"
            if template_name:
                full_user_prompt = f"Template: {template_name}\n{full_user_prompt}"
            
            prompts.append(Prompt(
                system_prompt=sys_prompt,
                user_prompt=full_user_prompt,
                meta_data=meta
            ))
        return PromptCollection(prompts)

class G4KRunner:
    """Basic LLM runner using OpenAI/vLLM API."""
    def __init__(self, sampling_params: Any, model: str, base_url: str):
        self.sampling_params = sampling_params
        self.model = model
        self.base_url = base_url
        
        # Initialize OpenAI client if needed, or just store for now
        # vLLM is OpenAI compatible
        import openai
        self.client = openai.OpenAI(api_key="empty", base_url=base_url)

    def generate(self, prompt: str) -> str:
        """Simple synchronous generation."""
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=self.sampling_params.max_tokens,
            temperature=self.sampling_params.temperature
        )
        return response.choices[0].text

    def batch_generate(self, prompts: List[str]) -> List[str]:
        """Simple batch generation (could be optimized)."""
        return [self.generate(p) for p in prompts]

    def run(self, prompt_collection: PromptCollection) -> ResponseWrapper:
        """Run a collection of prompts and return responses."""
        responses = []
        for prompt in prompt_collection:
            content = self.generate(f"{prompt.system_prompt}\n\n{prompt.user_prompt}")
            responses.append(ResponseData(
                query=prompt.user_prompt,
                retrieved_docs=[], # Internal runner calls may not have docs
                generated_response=content,
                meta_data=prompt.meta_data.data
            ))
        return ResponseWrapper(responses)

# Alias for backward compatibility during refactor
BatchInferenceRunner = G4KRunner

class RAGMethodInterface(ABC):
    """Interface for RAG methods."""
    @abstractmethod
    def retrieve(self, query: str) -> List[Document]:
        """Retrieve documents for a query."""
        pass

    @abstractmethod
    def run(
        self,
        runner: G4KRunner,
        sys_prompt: str,
        user_prompts: List[str] = [],
        prompt_meta_data: List[MetaData] = [],
        retrieval_queries: List[str] = [],
        response_format: Any = None,
    ) -> ResponseWrapper:
        """Run retrieval and optionally generation for a batch of queries."""
        pass
