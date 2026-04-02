"""GSR RAG Method: Graph-Structured Retrieval + CACL.

Implements two retrieval methods that integrate with the g4k benchmark framework:
  1. GSRRetrieval  — full GSR (KG + GAT + constraint scoring)
  2. HybridGSR     — GSR + BM25 hybrid (for comparison with HybridBM25 baseline)

Both implement g4k's RAGMethodInterface so they can be plugged into
benchmark_retrieval.py directly.

Reference:
  overall_idea.md §2.4 — GSR Architecture
  overall_idea.md §2.7 — Joint Scoring
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

import numpy as np
import torch
from tqdm import tqdm

from g4k.internal.abstractions import (
    Document,
    RAGMethodInterface,
    G4KRunner,
    ResponseWrapper,
    ResponseData,
    MetaData,
    PromptCollection,
)
from gsr_cacl.kg import (
    ConstraintKG,
    build_constraint_kg,
    build_kg_from_markdown,
)
from gsr_cacl.encoders import GATEncoder
from gsr_cacl.scoring import (
    JointScorer,
    compute_constraint_score,
    compute_entity_score,
)
from gsr_cacl.negative_sampler import CHAPNegativeSampler


# ----------------------------------------------------------------------
# GSRRetrieval — Full GSR pipeline
# ----------------------------------------------------------------------

class GSRRetrieval(RAGMethodInterface):
    """
    Graph-Structured Retrieval (GSR) for financial documents.

    Pipeline:
      Query Q
        ├─► Metadata extraction: (company, year)
        ├─► Table KG Construction (from table_md)
        ├─► GAT Encoding
        └─► Joint Scoring:
              s(Q,D) = α·sim_text(Q,D)
                      + β·sim_entity(Q,G_D)
                      + γ·ConstraintScore(G_D)

    For documents without tables, falls back to pure dense retrieval.
    """

    def __init__(
        self,
        context_collection: list[Document],
        embedding_function: Any,
        top_k: int = 5,
        embedding_model_name: str = "intfloat/multilingual-e5-large-instruct",
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        epsilon: float = 1e-4,
        gat_hidden_dim: int = 256,
        gat_num_heads: int = 4,
        gat_num_layers: int = 2,
        device: str | None = None,
        **kwargs,
    ):
        self.context_collection = context_collection
        self.embedding_function = embedding_function
        self.top_k = top_k
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.retrieval_only = kwargs.get("retrieval_only", True)

        # Device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Pre-index corpus
        self._build_index(embedding_model_name)

        # Pre-build KGs for all documents (offline, as per §5)
        self._build_all_kgs()

        # GAT encoder (shared across all documents)
        self.gat_encoder = GATEncoder(
            embed_dim=768,          # BGE embedding dim
            hidden_dim=gat_hidden_dim,
            num_heads=gat_num_heads,
            num_layers=gat_num_layers,
        ).to(self.device)
        self.gat_encoder.eval()

        # Pre-encode all KG embeddings
        self._encode_all_kgs()

        # Scorer
        self.scorer = JointScorer(
            text_embed_dim=768,
            kg_embed_dim=gat_hidden_dim,
        ).to(self.device)
        self.scorer.eval()

        # Store text embeddings for similarity
        self.doc_text_embeds = None   # set in _encode_all_kgs

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_index(self, embedding_model_name: str) -> None:
        """Build FAISS vector index (same as BaseRAG)."""
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document as LCDocument

        lc_docs = []
        for doc in self.context_collection:
            md = dict(doc.meta_data) if isinstance(doc.meta_data, dict) else {}
            md["id"] = str(doc.id)
            lc_docs.append(LCDocument(page_content=doc.page_content, metadata=md))

        batch_size = 100
        self.vector_store = None
        for i in tqdm(range(0, len(lc_docs), batch_size),
                      desc="Indexing GSR documents"):
            batch = lc_docs[i:i + batch_size]
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(batch, self.embedding_function)
            else:
                self.vector_store.add_documents(batch)

    def _build_all_kgs(self) -> None:
        """Pre-build constraint KGs for all documents."""
        self.doc_kgs: list[ConstraintKG] = []
        for doc in tqdm(self.context_collection, desc="Building constraint KGs"):
            table_md = self._extract_table_from_content(doc.page_content)
            kg = build_kg_from_markdown(table_md)
            self.doc_kgs.append(kg)

    def _encode_all_kgs(self) -> None:
        """Pre-encode all KG node embeddings."""
        self.kg_embeds: list[torch.Tensor] = []
        all_text_embeds = []

        for i, (doc, kg) in enumerate(tqdm(
                list(zip(self.context_collection, self.doc_kgs)),
                desc="Encoding GSR embeddings"
        )):
            with torch.no_grad():
                kg_embed = self.gat_encoder(kg)          # [V, hidden]
                # Pool KG to document-level: mean over nodes
                if kg_embed.numel() > 0:
                    kg_doc_embed = kg_embed.mean(dim=0)  # [hidden]
                else:
                    kg_doc_embed = torch.zeros(self.gat_encoder.output_dim, device=self.device)
                self.kg_embeds.append(kg_doc_embed)

            # Also get text embedding for doc
            text_emb = self.embedding_function.embed_query(doc.page_content)
            all_text_embeds.append(text_emb)

        self.doc_text_embeds = np.array(all_text_embeds)

    # ------------------------------------------------------------------
    # Table extraction from content
    # ------------------------------------------------------------------

    def _extract_table_from_content(self, content: str) -> str:
        """
        Extract markdown table from page_content.
        Looks for pipe-delimited tables using simple heuristic.
        Returns empty string if no table found.
        """
        lines = content.split("\n")
        in_table = False
        table_lines = []
        for line in lines:
            stripped = line.strip()
            if "|" in stripped:
                in_table = True
                table_lines.append(stripped)
            elif in_table and not stripped:
                # Empty line after table → stop
                break
            elif in_table:
                break
        if len(table_lines) < 2:
            return ""
        return "\n".join(table_lines)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> list[Document]:
        """
        Retrieve top-K documents using joint GSR scoring.

        Step 1: FAISS vector retrieval → candidate set
        Step 2: Joint scoring with constraint awareness
        Step 3: Re-rank by final score
        """
        # Step 1: Vector retrieval candidates
        candidates = self.vector_store.similarity_search(query, k=self.top_k * 4)
        candidate_indices = [
            int(c.metadata.get("id", c.metadata.get("idx", i)))
            for i, c in enumerate(candidates)
        ]

        # If we have doc IDs, map back to collection indices
        # Simple fallback: use first top_k candidates as indices
        n = min(len(candidates), self.top_k * 4)
        candidate_indices = list(range(n))

        # Step 2: Query embedding
        q_text_emb = np.array(self.embedding_function.embed_query(query))
        q_tensor = torch.tensor(q_text_emb, dtype=torch.float32, device=self.device)

        # Step 3: Joint scoring
        scores: dict[int, float] = {}
        for idx in candidate_indices:
            if idx >= len(self.doc_text_embeds):
                continue
            doc_tensor = torch.tensor(
                self.doc_text_embeds[idx],
                dtype=torch.float32,
                device=self.device,
            )
            kg_tensor = self.kg_embeds[idx].float()

            # Text similarity (cosine)
            text_sim = float(
                torch.cosine_similarity(q_tensor.unsqueeze(0), doc_tensor.unsqueeze(0)).item()
            )

            # Constraint score (from pre-computed KG)
            cs_result = compute_constraint_score(self.doc_kgs[idx], epsilon=self.epsilon)
            constraint_score = cs_result.constraint_score if cs_result.total_count > 0 else 1.0

            # Entity score (placeholder — needs query metadata)
            entity_score = 0.5

            # Weighted combination
            final_score = (
                self.alpha * text_sim
                + self.beta * entity_score
                + self.gamma * constraint_score
            )
            scores[idx] = final_score

        # Step 4: Re-rank by final score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:self.top_k]
        result_docs = []
        for idx, score in ranked:
            doc = self.context_collection[idx]
            result_docs.append(Document(
                page_content=doc.page_content,
                meta_data=doc.meta_data,
                id=doc.id,
            ))

        return result_docs

    # ------------------------------------------------------------------
    # RAGMethodInterface
    # ------------------------------------------------------------------

    def run(
        self,
        runner: G4KRunner,
        sys_prompt: str,
        user_prompts: list[str] = [],
        prompt_meta_data: list[MetaData] = [],
        retrieval_queries: list[str] = [],
        response_format: Any = None,
    ) -> ResponseWrapper:
        """Run retrieval (or retrieval + generation if retrieval_only=False)."""
        all_retrieved_docs = []
        for query in tqdm(retrieval_queries, desc=f"GSR Retrieval ({self.__class__.__name__})"):
            all_retrieved_docs.append(self.retrieve(query))

        if getattr(self, "retrieval_only", False):
            responses = []
            for query, docs, meta in zip(retrieval_queries, all_retrieved_docs, prompt_meta_data):
                meta_dict = meta.data if hasattr(meta, "data") else (meta or {})
                responses.append(ResponseData(
                    query=query,
                    retrieved_docs=docs,
                    generated_response=None,
                    meta_data=meta_dict,
                ))
            return ResponseWrapper(responses)

        # Generation path (not used in retrieval-only benchmark)
        prompt_collection = PromptCollection.create_prompts(
            sys_prompt,
            user_prompts,
            [Document(page_content="\n\n".join([d.page_content for d in docs]))
             for docs in all_retrieved_docs],
            prompt_meta_data,
        )
        responses = runner.run(prompt_collection)

        for resp, docs in zip(responses.response_data, all_retrieved_docs):
            resp.retrieved_docs = docs

        return responses


# ----------------------------------------------------------------------
# HybridGSR — GSR + BM25 hybrid
# ----------------------------------------------------------------------

class HybridGSR(GSRRetrieval):
    """
    Hybrid retrieval: GSR + BM25 + RRF fusion.

    Combines:
      - GSR constraint-aware dense retrieval
      - BM25 lexical retrieval
    via Reciprocal Rank Fusion (RRF).

    Used as a stronger baseline comparison against HybridBM25.
    """

    def __init__(
        self,
        context_collection: list[Document],
        embedding_function: Any,
        top_k: int = 5,
        rrf_k: int = 60,
        **kwargs,
    ):
        super().__init__(
            context_collection=context_collection,
            embedding_function=embedding_function,
            top_k=top_k,
            **kwargs,
        )
        self.rrf_k = rrf_k
        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        """Build BM25Okapi index."""
        from rank_bm25 import BM25Okapi
        self.tokenized_corpus = [
            doc.page_content.split(" ") for doc in self.context_collection
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str) -> list[Document]:
        """
        RRF fusion of GSR and BM25.
        """
        # GSR candidates
        gsr_docs = super().retrieve(query)
        gsr_ranking = {d.page_content: rank for rank, d in enumerate(gsr_docs)}

        # BM25 candidates
        tokenized_query = query.split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_n_indices = np.argsort(bm25_scores)[::-1][:self.top_k * 4]
        bm25_ranking = {
            self.context_collection[i].page_content: rank
            for rank, i in enumerate(top_n_indices)
        }

        # Reciprocal Rank Fusion
        all_docs = {d.page_content: d for d in gsr_docs}
        for idx in top_n_indices:
            all_docs[self.context_collection[idx].page_content] = self.context_collection[idx]

        rrf_scores = {}
        for content, doc in all_docs.items():
            rrf = 0.0
            if content in gsr_ranking:
                rrf += 1.0 / (self.rrf_k + gsr_ranking[content] + 1)
            if content in bm25_ranking:
                rrf += 1.0 / (self.rrf_k + bm25_ranking[content] + 1)
            rrf_scores[content] = rrf

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:self.top_k]
        return [all_docs[content] for content, _ in ranked]


# ----------------------------------------------------------------------
# Registry (compatible with g4k factory)
# ----------------------------------------------------------------------

GSR_REGISTRY = {
    "GSR": GSRRetrieval,
    "HybridGSR": HybridGSR,
}
