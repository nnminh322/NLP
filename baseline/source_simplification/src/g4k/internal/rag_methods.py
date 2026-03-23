"""Internal RAG method implementations."""

from typing import List, Optional
import numpy as np
from g4k.internal.abstractions import Document, RAGMethodInterface, G4KRunner

class BaseRAG(RAGMethodInterface):
    """Standard vector-based retrieval."""
    def __init__(
        self, 
        context_collection: List[Document],
        embedding_function: Any,
        top_k: int = 5,
        **kwargs
    ):
        self.context_collection = context_collection
        self.embedding_function = embedding_function
        self.top_k = top_k
        
        # Initialize LangChain FAISS index
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document as LCDocument
        
        lc_docs = [LCDocument(page_content=doc.page_content, metadata=doc.metadata) for doc in context_collection]
        self.vector_store = FAISS.from_documents(lc_docs, embedding_function)

    def retrieve(self, query: str) -> List[Document]:
        docs = self.vector_store.similarity_search(query, k=self.top_k)
        return [Document(page_content=d.page_content, metadata=d.metadata) for d in docs]

class HybridBM25(RAGMethodInterface):
    """Hybrid retrieval (BM25 + Vector) with RRF fusion."""
    def __init__(
        self, 
        context_collection: List[Document],
        embedding_function: Any,
        top_k: int = 5,
        **kwargs
    ):
        self.context_collection = context_collection
        self.embedding_function = embedding_function
        self.top_k = top_k
        
        # Vector Store
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document as LCDocument
        lc_docs = [LCDocument(page_content=doc.page_content, metadata=doc.metadata) for doc in context_collection]
        self.vector_store = FAISS.from_documents(lc_docs, embedding_function)
        
        # BM25
        from rank_bm25 import BM25Okapi
        tokenized_corpus = [doc.page_content.split(" ") for doc in context_collection]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str) -> List[Document]:
        # Vector retrieval
        vector_docs = self.vector_store.similarity_search(query, k=self.top_k * 2)
        
        # BM25 retrieval
        tokenized_query = query.split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_n_indices = np.argsort(bm25_scores)[::-1][:self.top_k * 2]
        bm25_docs = [self.context_collection[i] for i in top_n_indices]
        
        # Simple RRF or just combine
        # For simplicity in baseline, we'll return top_k from both and deduplicate
        combined = []
        seen = set()
        
        # Interleave (simplified RRF behavior)
        for v_doc, b_doc in zip(vector_docs, bm25_docs):
            if v_doc.page_content not in seen:
                combined.append(Document(page_content=v_doc.page_content, metadata=v_doc.metadata))
                seen.add(v_doc.page_content)
            if b_doc.page_content not in seen:
                combined.append(b_doc)
                seen.add(b_doc.page_content)
            if len(combined) >= self.top_k:
                break
                
        return combined[:self.top_k]

class HydeRAG(BaseRAG):
    """HyDE retrieval: generate hypothetical doc then retrieve."""
    def __init__(self, runner: G4KRunner, prompt: str, **kwargs):
        super().__init__(**kwargs)
        self.runner = runner
        self.hyde_prompt = prompt

    def retrieve(self, query: str) -> List[Document]:
        hypothetical_doc = self.runner.generate(f"{self.hyde_prompt}\n\nQuestion: {query}\n\nDocument:")
        return super().retrieve(hypothetical_doc)

class SummarizationRAG(BaseRAG):
    """Summarization RAG: summarize query or docs (simplified)."""
    # In this baseline, summarization usually means summarizing the retrieved context.
    # But if it's "Retrieval Only", maybe it's summarizing the query?
    # Based on the config, it seems to be generating a summary of the query/intent.
    def __init__(self, runner: G4KRunner, prompt: str, **kwargs):
        super().__init__(**kwargs)
        self.runner = runner
        self.summarization_prompt = prompt

    def retrieve(self, query: str) -> List[Document]:
        summary_query = self.runner.generate(f"{self.summarization_prompt}\n\nInput: {query}\n\nSummary:")
        return super().retrieve(summary_query)

# Registry for the factory
RAG_REGISTRY = {
    "Base": BaseRAG,
    "HybridBM25": HybridBM25,
    "Hyde": HydeRAG,
    "Summarization": SummarizationRAG
}
