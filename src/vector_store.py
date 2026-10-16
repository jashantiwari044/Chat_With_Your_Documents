import re
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi


def simple_tokenize(text: str) -> List[str]:
    """Tokenizes text for BM25 keyword matching."""
    return re.findall(r"\w+", text.lower())


class VectorStoreManager:
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "document_knowledge_base",
        embedding_model_name: str = "all-mpnet-base-v2"
    ):
        """
        Manages ChromaDB vector store + BM25 keyword index for Hybrid Search.
        Uses 768-dim 'all-mpnet-base-v2' embeddings for semantic accuracy.
        """
        self.client = chromadb.PersistentClient(path=persist_dir)

        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

        # In-memory BM25 structures
        self.bm25: Optional[BM25Okapi] = None
        self.bm25_docs: List[str] = []
        self.bm25_metas: List[Dict[str, Any]] = []
        self.bm25_ids: List[str] = []

        # If collection already has data on disk, initialize the BM25 index
        if self.collection.count() > 0:
            self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        """Loads all documents from ChromaDB and builds the BM25 index."""
        data = self.collection.get(include=["documents", "metadatas"])
        if not data or not data["documents"]:
            return

        self.bm25_ids = data["ids"]
        self.bm25_docs = data["documents"]
        self.bm25_metas = data["metadatas"]

        tokenized_corpus = [simple_tokenize(doc) for doc in self.bm25_docs]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def index_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 50) -> None:
        """Upserts document chunks into ChromaDB and initializes BM25."""
        if not chunks:
            print("⚠️ No chunks provided to index.")
            return

        total = len(chunks)
        print(f"🧠 Embedding and storing {total} chunks into ChromaDB...")

        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            self.collection.upsert(
                ids=[item["id"] for item in batch],
                documents=[item["text"] for item in batch],
                metadatas=[item["metadata"] for item in batch]
            )
            print(f"   └── Indexed {min(i + batch_size, total)} / {total} chunks")

        # Rebuild BM25 index with fresh data
        print("⚡ Building BM25 keyword index for Hybrid Search...")
        self._build_bm25_index()
        print("✅ Indexing complete (Dense + BM25 Hybrid ready)!")

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Hybrid Retrieval: Combines Dense Vector Search + BM25 Keyword Search
        using Reciprocal Rank Fusion (RRF).
        """
        if self.collection.count() == 0:
            return []

        # Ensure BM25 is built
        if self.bm25 is None:
            self._build_bm25_index()

        k_candidates = min(max(top_k * 2, 20), self.collection.count())

        # 1. Dense Semantic Search (ChromaDB)
        dense_results = self.collection.query(
            query_texts=[query],
            n_results=k_candidates,
            include=["documents", "metadatas", "distances"]
        )

        dense_ids = dense_results["ids"][0]
        dense_docs = dense_results["documents"][0]
        dense_metas = dense_results["metadatas"][0]
        dense_dists = dense_results["distances"][0]

        # 2. BM25 Keyword Search
        query_tokens = simple_tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens) if self.bm25 else []

        # Sort BM25 by score descending
        bm25_ranked_indices = sorted(
            range(len(bm25_scores)),
            key=lambda idx: bm25_scores[idx],
            reverse=True
        )[:k_candidates]

        # 3. Reciprocal Rank Fusion (RRF)
        # Score = 1 / (60 + dense_rank) + 1 / (60 + bm25_rank)
        rrf_constant = 60
        doc_store: Dict[str, Dict[str, Any]] = {}
        rrf_scores: Dict[str, float] = {}

        # Add Dense ranks
        for rank, (doc_id, doc, meta, dist) in enumerate(
            zip(dense_ids, dense_docs, dense_metas, dense_dists), start=1
        ):
            doc_store[doc_id] = {
                "text": doc,
                "metadata": meta,
                "distance": round(float(dist), 4)
            }
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_constant + rank))

        # Add BM25 ranks
        for rank, idx in enumerate(bm25_ranked_indices, start=1):
            doc_id = self.bm25_ids[idx]
            if doc_id not in doc_store:
                doc_store[doc_id] = {
                    "text": self.bm25_docs[idx],
                    "metadata": self.bm25_metas[idx],
                    "distance": 0.5  # placeholder if not in dense top-k
                }
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_constant + rank))

        # Sort all candidates by RRF score descending
        sorted_doc_ids = sorted(
            rrf_scores.keys(),
            key=lambda did: rrf_scores[did],
            reverse=True
        )[:top_k]

        final_results = []
        for did in sorted_doc_ids:
            item = doc_store[did]
            item["rrf_score"] = round(rrf_scores[did], 6)
            final_results.append(item)

        return final_results

        # Sort all candidates by RRF score descending
        sorted_doc_ids = sorted(
            rrf_scores.keys(),
            key=lambda did: rrf_scores[did],
            reverse=True
        )[:top_k]

        final_results = []
        for did in sorted_doc_ids:
            item = doc_store[did]
            item["rrf_score"] = round(rrf_scores[did], 6)
            final_results.append(item)

        return final_results

