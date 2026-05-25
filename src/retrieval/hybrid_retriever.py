"""
Hybrid Retriever — Phase 3 Core
=================================
Combines two retrieval strategies:

  1. Dense (semantic)  — FAISS vector search
     Finds memories that MEAN the same thing as the query,
     even if they use different words.
     e.g. query "cloud deployment" matches "AWS ECS setup"

  2. Sparse (keyword)  — BM25 ranking
     Finds memories that contain the EXACT words in the query.
     Strong for proper nouns, IDs, technical terms.
     e.g. query "Kubernetes EKS" matches entries with those exact words

  3. Hybrid fusion     — Reciprocal Rank Fusion (RRF)
     Merges the two ranked lists into one superior list.
     A result that ranks well in BOTH lists scores highest.
     This consistently outperforms either method alone.

Why hybrid beats pure semantic:
  - Semantic search misses exact-match terms (names, IDs, version numbers)
  - BM25 misses paraphrases and synonyms
  - Fusion gets the best of both worlds

Research backing: Hybrid retrieval outperforms dense-only by 8-15%
on standard QA benchmarks (MS-MARCO, Natural Questions).
"""

import math
import time
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass

from src.memory.memory_types import MemoryEntry, MemoryTier, MemoryQueryResult
from src.memory.memory_store import MemoryStore
from src.utils.logger import get_logger
from src.utils.token_counter import count_tokens

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Extended result with both semantic and keyword scores."""
    entry: MemoryEntry
    semantic_score: float   # FAISS cosine similarity
    keyword_score: float    # BM25 score
    fusion_score: float     # RRF combined score
    rank: int
    retrieval_latency_ms: float = 0.0

    def __repr__(self):
        return (
            f"RetrievalResult(rank={self.rank}, "
            f"fusion={self.fusion_score:.3f}, "
            f"sem={self.semantic_score:.3f}, "
            f"kw={self.keyword_score:.3f})"
        )


class BM25Index:
    """
    BM25 (Best Match 25) — the gold standard sparse retrieval algorithm.
    Used by Elasticsearch, Solr, and most search engines internally.

    BM25 formula:
      score(D, Q) = sum over terms t in Q of:
        IDF(t) * (tf(t,D) * (k1+1)) / (tf(t,D) + k1*(1 - b + b*|D|/avgdl))

    Where:
      tf(t,D)   = term frequency of t in document D
      IDF(t)    = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
      |D|       = length of document D in words
      avgdl     = average document length
      k1, b     = tuning parameters (default 1.5, 0.75)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: List[str] = []
        self._doc_ids: List[str] = []
        self._tf: List[Dict[str, int]] = []
        self._df: Dict[str, int] = defaultdict(int)
        self._avgdl: float = 0.0
        self._idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + lowercase tokenizer."""
        import re
        return re.findall(r'\b[a-z0-9]+\b', text.lower())

    def add_documents(self, doc_ids: List[str], documents: List[str]):
        """Build BM25 index from a list of documents."""
        self._docs = documents
        self._doc_ids = doc_ids
        self._tf = []
        self._df = defaultdict(int)

        # Compute term frequencies
        for doc in documents:
            tokens = self._tokenize(doc)
            tf: Dict[str, int] = defaultdict(int)
            for token in tokens:
                tf[token] += 1
            self._tf.append(dict(tf))
            for token in set(tokens):
                self._df[token] += 1

        # Average document length
        lengths = [len(self._tokenize(d)) for d in documents]
        self._avgdl = sum(lengths) / max(len(lengths), 1)

        # Precompute IDF
        N = len(documents)
        for term, df in self._df.items():
            self._idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)

        logger.debug(f"BM25 index built: {len(documents)} documents, {len(self._idf)} unique terms")

    def add_document(self, doc_id: str, document: str):
        """Incrementally add a single document (slower, use add_documents for bulk)."""
        tokens = self._tokenize(document)
        tf: Dict[str, int] = defaultdict(int)
        for token in tokens:
            tf[token] += 1

        self._docs.append(document)
        self._doc_ids.append(doc_id)
        self._tf.append(dict(tf))

        for token in set(tokens):
            self._df[token] += 1

        # Recompute avgdl and IDF (simplified: full recompute)
        lengths = [len(self._tokenize(d)) for d in self._docs]
        self._avgdl = sum(lengths) / max(len(lengths), 1)

        N = len(self._docs)
        for term, df in self._df.items():
            self._idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search the index for the top-k documents matching the query.
        Returns list of (doc_id, bm25_score) sorted by score descending.
        """
        if not self._docs:
            return []

        query_tokens = self._tokenize(query)
        scores: Dict[int, float] = defaultdict(float)

        for token in query_tokens:
            if token not in self._idf:
                continue
            idf = self._idf[token]
            for i, tf_dict in enumerate(self._tf):
                tf = tf_dict.get(token, 0)
                if tf == 0:
                    continue
                doc_len = sum(tf_dict.values())
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self._avgdl, 1))
                scores[i] += idf * (numerator / denominator)

        # Sort by score, return top-k
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self._doc_ids[i], score) for i, score in sorted_scores]

    @property
    def size(self) -> int:
        return len(self._docs)


class HybridRetriever:
    """
    Hybrid retrieval combining ChromaDB semantic search + BM25 keyword search.
    Uses Reciprocal Rank Fusion to merge results.

    RRF formula:
      rrf_score(d) = sum over rankers r of: 1 / (k + rank_r(d))
      where k=60 is a constant that reduces the impact of high rankings

    RRF is robust: it doesn't require score normalization and
    works well even when the two rankers have very different score distributions.
    """

    def __init__(
        self,
        store: MemoryStore,
        rrf_k: int = 60,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4,
        top_k: int = 5,
    ):
        self.store = store
        self.rrf_k = rrf_k
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.top_k = top_k

        # Separate BM25 index per memory tier
        self._bm25: Dict[str, BM25Index] = {
            tier.value: BM25Index() for tier in MemoryTier
        }
        self._doc_cache: Dict[str, MemoryEntry] = {}  # entry_id → entry

        logger.info(
            f"HybridRetriever initialized: "
            f"semantic_weight={semantic_weight}, keyword_weight={keyword_weight}, "
            f"rrf_k={rrf_k}"
        )

    # ------------------------------------------------------------------ #
    #  Index management                                                    #
    # ------------------------------------------------------------------ #

    def index_entry(self, entry: MemoryEntry):
        """Add a single entry to the BM25 index."""
        self._bm25[entry.tier.value].add_document(
            entry.entry_id, entry.active_content
        )
        self._doc_cache[entry.entry_id] = entry

    def index_entries(self, entries: List[MemoryEntry]):
        """Bulk-index a list of entries."""
        tier_groups: Dict[str, Tuple[List[str], List[str]]] = defaultdict(lambda: ([], []))
        for entry in entries:
            tier_groups[entry.tier.value][0].append(entry.entry_id)
            tier_groups[entry.tier.value][1].append(entry.active_content)
            self._doc_cache[entry.entry_id] = entry

        for tier_name, (ids, docs) in tier_groups.items():
            if ids:
                self._bm25[tier_name].add_documents(ids, docs)

        logger.info(f"Indexed {len(entries)} entries across {len(tier_groups)} tiers")

    # ------------------------------------------------------------------ #
    #  Retrieval                                                           #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        tiers: Optional[List[MemoryTier]] = None,
    ) -> List[RetrievalResult]:
        """
        Main retrieval method. Returns hybrid-ranked results.

        Args:
            query:   The search query
            top_k:   Number of results (default: self.top_k)
            tiers:   Which memory tiers to search (default: all)
        """
        t0 = time.time()
        top_k = top_k or self.top_k
        tiers = tiers or list(MemoryTier)
        fetch_k = top_k * 3  # fetch more than needed before fusion

        # 1. Semantic retrieval (ChromaDB)
        semantic_results: Dict[str, float] = {}
        for tier in tiers:
            results = self.store.query(query, tier=tier, top_k=fetch_k)
            for r in results:
                eid = r.entry.entry_id
                # Keep best score if entry appears in multiple tiers
                if eid not in semantic_results or r.similarity_score > semantic_results[eid]:
                    semantic_results[eid] = r.similarity_score
                    self._doc_cache[eid] = r.entry

        # 2. Keyword retrieval (BM25)
        keyword_results: Dict[str, float] = {}
        for tier in tiers:
            bm25_hits = self._bm25[tier.value].search(query, top_k=fetch_k)
            for eid, score in bm25_hits:
                if eid not in keyword_results or score > keyword_results[eid]:
                    keyword_results[eid] = score

        # 3. Reciprocal Rank Fusion
        fused = self._rrf_fuse(semantic_results, keyword_results, top_k)

        latency_ms = (time.time() - t0) * 1000

        results = []
        for rank, (eid, fusion_score) in enumerate(fused, 1):
            entry = self._doc_cache.get(eid)
            if entry is None:
                continue
            results.append(RetrievalResult(
                entry=entry,
                semantic_score=semantic_results.get(eid, 0.0),
                keyword_score=keyword_results.get(eid, 0.0),
                fusion_score=fusion_score,
                rank=rank,
                retrieval_latency_ms=latency_ms / max(len(fused), 1),
            ))

        logger.debug(
            f"Hybrid retrieval: {len(results)} results in {latency_ms:.1f}ms "
            f"(sem={len(semantic_results)}, kw={len(keyword_results)})"
        )
        return results

    def _rrf_fuse(
        self,
        semantic: Dict[str, float],
        keyword: Dict[str, float],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """
        Reciprocal Rank Fusion.
        Assigns RRF score to each doc based on its rank in each list.
        """
        # Build rank lists
        sem_ranked = sorted(semantic.items(), key=lambda x: x[1], reverse=True)
        kw_ranked  = sorted(keyword.items(),  key=lambda x: x[1], reverse=True)

        sem_rank = {eid: rank + 1 for rank, (eid, _) in enumerate(sem_ranked)}
        kw_rank  = {eid: rank + 1 for rank, (eid, _) in enumerate(kw_ranked)}

        # All candidate IDs
        all_ids = set(semantic.keys()) | set(keyword.keys())

        rrf_scores: Dict[str, float] = {}
        for eid in all_ids:
            score = 0.0
            if eid in sem_rank:
                score += self.semantic_weight / (self.rrf_k + sem_rank[eid])
            if eid in kw_rank:
                score += self.keyword_weight / (self.rrf_k + kw_rank[eid])
            rrf_scores[eid] = score

        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    # ------------------------------------------------------------------ #
    #  Stats                                                               #
    # ------------------------------------------------------------------ #

    def stats(self) -> Dict:
        return {
            "indexed_docs": {tier: self._bm25[tier].size for tier in self._bm25},
            "cached_entries": len(self._doc_cache),
            "config": {
                "rrf_k": self.rrf_k,
                "semantic_weight": self.semantic_weight,
                "keyword_weight": self.keyword_weight,
            },
        }
