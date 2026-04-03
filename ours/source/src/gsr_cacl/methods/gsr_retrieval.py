"""GSR Retrieval Methods: GSRRetrieval + HybridGSR.

Implements the full GSR pipeline (§4 of overall_idea.md):
  Query Q → FAISS candidates → Joint Scoring (text + entity + constraint) → Top-K

And the hybrid variant: GSR + BM25 with Reciprocal Rank Fusion (RRF).

No dependency on g4k — uses gsr_cacl.core.Document directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from gsr_cacl.core import Document
from gsr_cacl.kg.builder import build_kg_from_markdown
from gsr_cacl.kg.data_structures import ConstraintKG
from gsr_cacl.encoders.gat_encoder import GATEncoder
from gsr_cacl.scoring.constraint_score import compute_constraint_score
from gsr_cacl.scoring.joint_scorer import JointScorer


class GSRRetrieval:
    """
    Graph-Structured Retrieval (GSR) for financial documents.

    Pipeline per overall_idea.md §4.1:
      1. Build FAISS index over document embeddings  (text signal)
      2. Build Constraint KG for every document       (structure signal)
      3. At query time:
           a. FAISS → candidate set (4×top_k)
           b. Joint Score = α·sim_text + β·sim_entity + γ·CS(G_D)
           c. Return top-K by joint score
    """

    def __init__(
        self,
        corpus: list[Document],
        embedding_function: Any,
        top_k: int = 5,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        epsilon: float = 1e-4,
        gat_hidden_dim: int = 256,
        gat_num_heads: int = 4,
        gat_num_layers: int = 2,
        device: str | None = None,
    ):
        self.corpus = corpus
        self.embedding_function = embedding_function
        self.top_k = top_k
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Step 1: FAISS text index
        self._build_faiss_index()

        # Step 2: Constraint KGs for all docs
        self._build_all_kgs()

        # GAT encoder (§4.3)
        self.gat_encoder = GATEncoder(
            embed_dim=768,
            hidden_dim=gat_hidden_dim,
            num_heads=gat_num_heads,
            num_layers=gat_num_layers,
        ).to(self.device)
        self.gat_encoder.eval()

        # Pre-encode KGs
        self._encode_all_kgs()

        # Joint scorer (§4.4)
        self.scorer = JointScorer(
            text_embed_dim=768,
            kg_embed_dim=gat_hidden_dim,
        ).to(self.device)
        self.scorer.eval()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _build_faiss_index(self) -> None:
        """Build FAISS vector store from corpus documents."""
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document as LCDocument

        lc_docs = []
        for doc in self.corpus:
            md = dict(doc.meta_data) if isinstance(doc.meta_data, dict) else {}
            md["id"] = str(doc.id)
            lc_docs.append(LCDocument(page_content=doc.page_content, metadata=md))

        batch_size = 100
        self.vector_store = None
        for i in tqdm(range(0, len(lc_docs), batch_size), desc="Indexing documents (FAISS)"):
            batch = lc_docs[i : i + batch_size]
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(batch, self.embedding_function)
            else:
                self.vector_store.add_documents(batch)

        # Corpus id → index mapping for KG/embed lookup
        self._id_to_idx: dict[str, int] = {
            str(doc.id): i for i, doc in enumerate(self.corpus)
        }

        # Store text embeddings for scoring
        all_text_embeds = []
        for doc in tqdm(self.corpus, desc="Computing text embeddings"):
            emb = self.embedding_function.embed_query(doc.page_content)
            all_text_embeds.append(emb)
        self.doc_text_embeds = np.array(all_text_embeds)

    def _build_all_kgs(self) -> None:
        """Build a Constraint KG for every corpus document (§4.2)."""
        self.doc_kgs: list[ConstraintKG] = []
        for doc in tqdm(self.corpus, desc="Building constraint KGs"):
            table_md = self._extract_table(doc.page_content)
            kg = build_kg_from_markdown(table_md)
            self.doc_kgs.append(kg)

    def _encode_all_kgs(self) -> None:
        """Encode all KGs with GAT and cache graph-level embeddings (§4.3)."""
        self.kg_embeds: list[torch.Tensor] = []
        for kg in tqdm(self.doc_kgs, desc="Encoding KGs with GAT"):
            with torch.no_grad():
                node_embeds = self.gat_encoder(kg)
                if node_embeds.numel() > 0:
                    doc_embed = node_embeds.mean(dim=0)
                else:
                    doc_embed = torch.zeros(
                        self.gat_encoder.output_dim, device=self.device
                    )
                self.kg_embeds.append(doc_embed)

    # ------------------------------------------------------------------
    # Retrieval (§4.1 + §4.4)
    # ------------------------------------------------------------------

    def retrieve(self, query: str, query_meta: dict[str, Any] | None = None) -> list[Document]:
        """
        Retrieve top-K documents for a query using joint scoring.

        Joint Score = α·sim_text(Q,D) + β·sim_entity(Q,D) + γ·CS(G_D)
        """
        # FAISS first-stage retrieval → candidate LangChain Documents
        candidates = self.vector_store.similarity_search(query, k=self.top_k * 4)

        q_text_emb = np.array(self.embedding_function.embed_query(query))
        q_tensor = torch.tensor(q_text_emb, dtype=torch.float32, device=self.device)

        scores: list[tuple[int, float]] = []
        for cand in candidates:
            # Map FAISS candidate back to corpus index
            cand_id = cand.metadata.get("id", "")
            corpus_idx = self._id_to_idx.get(cand_id)
            if corpus_idx is None:
                continue

            # Text similarity
            doc_tensor = torch.tensor(
                self.doc_text_embeds[corpus_idx], dtype=torch.float32, device=self.device
            )
            text_sim = float(
                torch.cosine_similarity(
                    q_tensor.unsqueeze(0), doc_tensor.unsqueeze(0)
                ).item()
            )

            # Constraint score (§4.4)
            cs_result = compute_constraint_score(
                self.doc_kgs[corpus_idx], epsilon=self.epsilon
            )
            constraint_score = (
                cs_result.constraint_score if cs_result.total_count > 0 else 1.0
            )

            # Entity matching score (§4.4)
            entity_score = self._compute_entity_score(query_meta, corpus_idx)

            # Joint score
            final_score = (
                self.alpha * text_sim
                + self.beta * entity_score
                + self.gamma * constraint_score
            )
            scores.append((corpus_idx, final_score))

        ranked = sorted(scores, key=lambda x: x[1], reverse=True)[: self.top_k]
        return [self.corpus[idx] for idx, _ in ranked]

    def retrieve_batch(
        self,
        queries: list[str],
        queries_meta: list[dict[str, Any]] | None = None,
    ) -> list[list[Document]]:
        """Retrieve for a batch of queries."""
        if queries_meta is None:
            queries_meta = [None] * len(queries)
        results = []
        for q, meta in tqdm(
            zip(queries, queries_meta), total=len(queries), desc="GSR Retrieval"
        ):
            results.append(self.retrieve(q, query_meta=meta))
        return results

    # ------------------------------------------------------------------
    # Entity matching (§4.4)
    # ------------------------------------------------------------------

    def _compute_entity_score(
        self, query_meta: dict[str, Any] | None, doc_idx: int
    ) -> float:
        """
        Entity matching: company + year + sector.
        s_entity = match(company_Q, company_D) + match(year_Q, year_D) + match(sector_Q, sector_D)
        Each match contributes 1/3 when matched.
        """
        if query_meta is None:
            return 0.5  # neutral if no metadata

        doc_meta = self.corpus[doc_idx].meta_data
        score = 0.0
        for key in ("company_name", "report_year", "company_sector"):
            q_val = str(query_meta.get(key, "")).strip().lower()
            d_val = str(doc_meta.get(key, "")).strip().lower()
            if q_val and d_val and q_val == d_val:
                score += 1.0 / 3.0
        return score

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_table(content: str) -> str:
        """Extract the first markdown table from page content."""
        lines = content.split("\n")
        in_table = False
        table_lines = []
        for line in lines:
            stripped = line.strip()
            if "|" in stripped:
                in_table = True
                table_lines.append(stripped)
            elif in_table:
                break
        return "\n".join(table_lines) if len(table_lines) >= 2 else ""


class HybridGSR(GSRRetrieval):
    """
    Hybrid retrieval: GSR + BM25 with Reciprocal Rank Fusion (RRF).

    RRF score = 1/(k + rank_GSR) + 1/(k + rank_BM25)
    """

    def __init__(
        self,
        corpus: list[Document],
        embedding_function: Any,
        top_k: int = 5,
        rrf_k: int = 60,
        **kwargs,
    ):
        super().__init__(
            corpus=corpus,
            embedding_function=embedding_function,
            top_k=top_k,
            **kwargs,
        )
        self.rrf_k = rrf_k
        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        from rank_bm25 import BM25Okapi

        self.tokenized_corpus = [
            doc.page_content.split(" ") for doc in self.corpus
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str, query_meta: dict[str, Any] | None = None) -> list[Document]:
        # GSR ranking
        gsr_docs = super().retrieve(query, query_meta=query_meta)
        gsr_ranking = {d.id: rank for rank, d in enumerate(gsr_docs)}

        # BM25 ranking
        tokenized_query = query.split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_n_indices = np.argsort(bm25_scores)[::-1][: self.top_k * 4]
        bm25_ranking = {
            self.corpus[i].id: rank for rank, i in enumerate(top_n_indices)
        }

        # Merge candidates
        all_docs: dict[str, Document] = {d.id: d for d in gsr_docs}
        for idx in top_n_indices:
            all_docs[self.corpus[idx].id] = self.corpus[idx]

        # RRF fusion
        rrf_scores: dict[str, float] = {}
        for doc_id in all_docs:
            rrf = 0.0
            if doc_id in gsr_ranking:
                rrf += 1.0 / (self.rrf_k + gsr_ranking[doc_id] + 1)
            if doc_id in bm25_ranking:
                rrf += 1.0 / (self.rrf_k + bm25_ranking[doc_id] + 1)
            rrf_scores[doc_id] = rrf

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[
            : self.top_k
        ]
        return [all_docs[doc_id] for doc_id, _ in ranked]


GSR_REGISTRY: dict[str, type] = {
    "gsr": GSRRetrieval,
    "hybridgsr": HybridGSR,
}
