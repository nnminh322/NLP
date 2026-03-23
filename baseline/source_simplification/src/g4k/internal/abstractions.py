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
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_content": self.page_content,
            "metadata": self.metadata,
            "id": self.id
        }

@dataclass
class MetaData:
    """Internal MetaData class for prompt engineering."""
    data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Response:
    """Internal Response class for LLM outputs."""
    content: str
    raw: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ResponseData:
    """Internal ResponseData class for RAG execution results."""
    query: str
    retrieved_docs: List[Document]
    generated_response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "retrieved_docs": [doc.to_dict() for doc in self.retrieved_docs],
            "generated_response": self.generated_response,
            "metadata": self.metadata
        }

@dataclass
class ResponseWrapper:
    """Wrapper for multiple responses."""
    response_data: List[ResponseData]

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

# Alias for backward compatibility during refactor
BatchInferenceRunner = G4KRunner

class RAGMethodInterface(ABC):
    """Interface for RAG methods."""
    @abstractmethod
    def retrieve(self, query: str) -> List[Document]:
        """Retrieve documents for a query."""
        pass
