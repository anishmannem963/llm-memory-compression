"""
Memory Store — ChromaDB-backed persistent vector database
=========================================================
Stores MemoryEntry objects as embeddings + metadata.
Supports semantic search, filtering by tier, and persistence across sessions.
"""

import json
import time
from typing import List, Optional, Dict, Any
from pathlib import Path

from src.memory.memory_types import MemoryEntry, MemoryTier, MemoryQueryResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MemoryStore:
    """
    Persistent vector store for memory entries using ChromaDB.
    Each memory tier uses a separate ChromaDB collection.
    """

    TIER_COLLECTIONS = {
        MemoryTier.SHORT_TERM: "short_term",
        MemoryTier.WORKING:    "working_memory",
        MemoryTier.LONG_TERM:  "long_term",
    }

    def __init__(
        self,
        persist_dir: str = "./data/embeddings/chroma",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_prefix: str = "memory",
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_prefix = collection_prefix
        self._collections: Dict[MemoryTier, Any] = {}

        self._init_chromadb()
        self._init_embedding_function(embedding_model_name)

    def _init_chromadb(self):
        try:
            import chromadb
            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
            logger.info(f"ChromaDB initialized at {self.persist_dir}")
        except ImportError:
            raise ImportError(
                "ChromaDB not installed. Run: pip install chromadb"
            )

    def _init_embedding_function(self, model_name: str):
        try:
            from chromadb.utils import embedding_functions
            self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name
            )
            logger.info(f"Embedding function set: {model_name}")
        except Exception as e:
            logger.warning(f"Could not load embedding function: {e}. Using default.")
            self.embed_fn = None

    def _get_collection(self, tier: MemoryTier):
        if tier not in self._collections:
            name = f"{self.collection_prefix}_{self.TIER_COLLECTIONS[tier]}"
            kwargs = {"name": name, "metadata": {"hnsw:space": "cosine"}}
            if self.embed_fn:
                kwargs["embedding_function"] = self.embed_fn
            self._collections[tier] = self.client.get_or_create_collection(**kwargs)
        return self._collections[tier]

    # ------------------------------------------------------------------ #
    #  Write operations                                                    #
    # ------------------------------------------------------------------ #

    def add(self, entry: MemoryEntry) -> str:
        """Add a single MemoryEntry to its tier collection."""
        collection = self._get_collection(entry.tier)

        metadata = {
            "importance_score": entry.importance_score,
            "recency_score": entry.recency_score,
            "final_score": entry.final_score,
            "created_at": entry.created_at,
            "last_accessed": entry.last_accessed,
            "access_count": entry.access_count,
            "tier": entry.tier.value,
            "tags": json.dumps(entry.tags),
            "has_compression": entry.compressed_content is not None,
            "compression_ratio": entry.compression_ratio or 1.0,
        }

        collection.add(
            ids=[entry.entry_id],
            documents=[entry.active_content],
            metadatas=[metadata],
        )
        logger.debug(f"Added entry {entry.entry_id} to {entry.tier.value}")
        return entry.entry_id

    def add_batch(self, entries: List[MemoryEntry]) -> List[str]:
        """Add multiple entries efficiently."""
        ids = []
        for entry in entries:
            ids.append(self.add(entry))
        logger.info(f"Added {len(ids)} entries to memory store")
        return ids

    def update(self, entry: MemoryEntry):
        """Update an existing entry (e.g. after compression)."""
        collection = self._get_collection(entry.tier)
        collection.update(
            ids=[entry.entry_id],
            documents=[entry.active_content],
            metadatas=[{
                "importance_score": entry.importance_score,
                "final_score": entry.final_score,
                "last_accessed": entry.last_accessed,
                "access_count": entry.access_count,
                "has_compression": entry.compressed_content is not None,
                "compression_ratio": entry.compression_ratio or 1.0,
            }],
        )

    # ------------------------------------------------------------------ #
    #  Query operations                                                    #
    # ------------------------------------------------------------------ #

    def query(
        self,
        query_text: str,
        tier: MemoryTier = MemoryTier.LONG_TERM,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[MemoryQueryResult]:
        """
        Semantic search over a memory tier.
        Returns ranked list of MemoryQueryResult.
        """
        collection = self._get_collection(tier)

        try:
            results = collection.query(
                query_texts=[query_text],
                n_results=min(top_k, collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.warning(f"Query failed on {tier.value}: {e}")
            return []

        if not results["ids"][0]:
            return []

        query_results = []
        for rank, (doc_id, doc, meta, dist) in enumerate(zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score [0, 1]
            similarity = 1.0 - (dist / 2.0)

            if similarity < min_score:
                continue

            entry = MemoryEntry(
                entry_id=doc_id,
                tier=MemoryTier(meta.get("tier", tier.value)),
                original_content=doc,
                importance_score=meta.get("importance_score", 0.0),
                final_score=meta.get("final_score", 0.0),
                created_at=meta.get("created_at", 0.0),
                last_accessed=meta.get("last_accessed", 0.0),
                access_count=meta.get("access_count", 0),
                tags=json.loads(meta.get("tags", "[]")),
            )

            query_results.append(MemoryQueryResult(
                entry=entry,
                similarity_score=similarity,
                rank=rank + 1,
            ))

        return query_results

    def query_all_tiers(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> List[MemoryQueryResult]:
        """Search across all three memory tiers and merge results."""
        all_results = []
        for tier in MemoryTier:
            results = self.query(query_text, tier=tier, top_k=top_k)
            all_results.extend(results)

        # Re-rank by similarity
        all_results.sort(key=lambda r: r.similarity_score, reverse=True)
        for i, r in enumerate(all_results[:top_k]):
            r.rank = i + 1
        return all_results[:top_k]

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    def count(self, tier: Optional[MemoryTier] = None) -> int:
        if tier:
            return self._get_collection(tier).count()
        return sum(self._get_collection(t).count() for t in MemoryTier)

    def delete(self, entry_id: str, tier: MemoryTier):
        self._get_collection(tier).delete(ids=[entry_id])

    def clear_tier(self, tier: MemoryTier):
        collection = self._get_collection(tier)
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)
        logger.info(f"Cleared {tier.value} ({len(all_ids)} entries)")

    def stats(self) -> Dict[str, int]:
        return {tier.value: self._get_collection(tier).count() for tier in MemoryTier}
