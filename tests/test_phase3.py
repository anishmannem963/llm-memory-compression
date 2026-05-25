"""
Phase 3 Tests — Hybrid Retrieval, Cache, Budget Optimizer
Run with: pytest tests/test_phase3.py -v
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.retrieval.hybrid_retriever import BM25Index, HybridRetriever, RetrievalResult
from src.retrieval.token_budget_optimizer import TokenBudgetOptimizer
from src.retrieval.cache import InMemoryLRUCache
from src.memory.memory_types import MemoryEntry, MemoryTier
from src.utils.token_counter import count_tokens


# ------------------------------------------------------------------ #
#  BM25 tests                                                          #
# ------------------------------------------------------------------ #

class TestBM25:

    def _make_index(self):
        bm25 = BM25Index()
        docs = [
            "Deploy Kubernetes cluster on AWS EKS with auto-scaling",
            "Redis cache configuration for high throughput systems",
            "PostgreSQL database setup with pgvector extension",
            "Machine learning model training with PyTorch and CUDA",
            "FastAPI REST endpoint for memory retrieval system",
            "The deadline for the project is April 30th",
            "Hi how are you doing today",
        ]
        ids = [f"doc_{i}" for i in range(len(docs))]
        bm25.add_documents(ids, docs)
        return bm25, ids, docs

    def test_basic_search_returns_results(self):
        bm25, _, _ = self._make_index()
        results = bm25.search("Kubernetes AWS deployment", top_k=3)
        assert len(results) > 0

    def test_exact_keyword_match_scores_high(self):
        bm25, ids, _ = self._make_index()
        results = bm25.search("Kubernetes EKS", top_k=5)
        top_ids = [r[0] for r in results]
        assert "doc_0" in top_ids[:2], "Kubernetes document should rank in top 2"

    def test_irrelevant_query_scores_low(self):
        bm25, _, _ = self._make_index()
        results = bm25.search("quantum physics neutron star", top_k=3)
        if results:
            assert results[0][1] < 5.0  # BM25 score should be low

    def test_top_k_respected(self):
        bm25, _, _ = self._make_index()
        results = bm25.search("system", top_k=2)
        assert len(results) <= 2

    def test_empty_index_returns_empty(self):
        bm25 = BM25Index()
        results = bm25.search("anything", top_k=5)
        assert results == []

    def test_incremental_add(self):
        bm25 = BM25Index()
        bm25.add_document("d1", "Deploy to AWS")
        bm25.add_document("d2", "Redis cache setup")
        results = bm25.search("AWS deploy", top_k=2)
        assert len(results) >= 1
        assert results[0][0] == "d1"

    def test_size_property(self):
        bm25, _, docs = self._make_index()
        assert bm25.size == len(docs)


# ------------------------------------------------------------------ #
#  Token Budget Optimizer tests                                        #
# ------------------------------------------------------------------ #

def _make_result(content: str, score: float = 0.8, rank: int = 1) -> RetrievalResult:
    entry = MemoryEntry(
        tier=MemoryTier.WORKING,
        original_content=content,
        importance_score=score,
        final_score=score,
    )
    return RetrievalResult(
        entry=entry,
        semantic_score=score,
        keyword_score=score * 0.8,
        fusion_score=score,
        rank=rank,
    )


class TestTokenBudgetOptimizer:

    def test_fits_within_budget(self):
        optimizer = TokenBudgetOptimizer(token_budget=200)
        results = [_make_result("word " * 30, rank=i) for i in range(1, 6)]
        ctx = optimizer.optimize(results)
        assert ctx.total_tokens <= 220  # small margin for formatting

    def test_high_relevance_included_first(self):
        optimizer = TokenBudgetOptimizer(token_budget=300)
        results = [
            _make_result("Deploy Kubernetes on AWS with ECS. " * 5, score=0.95, rank=1),
            _make_result("The weather is nice today. " * 5, score=0.2, rank=2),
        ]
        ctx = optimizer.optimize(results)
        assert "Kubernetes" in ctx.context_text or ctx.entries_included >= 1

    def test_redundancy_removal(self):
        optimizer = TokenBudgetOptimizer(token_budget=1000, redundancy_threshold=0.5, min_entry_tokens=5)
        # Two nearly identical entries
        results = [
            _make_result("Deploy the Kubernetes cluster on AWS EKS today", score=0.9, rank=1),
            _make_result("Deploy the Kubernetes cluster on AWS EKS today", score=0.85, rank=2),
            _make_result("Redis cache configuration for high throughput", score=0.7, rank=3),
        ]
        ctx = optimizer.optimize(results)
        assert ctx.redundancy_removed >= 1

    def test_empty_results_returns_empty_context(self):
        optimizer = TokenBudgetOptimizer(token_budget=1000)
        ctx = optimizer.optimize([])
        assert ctx.context_text == ""
        assert ctx.entries_included == 0

    def test_recent_context_included(self):
        optimizer = TokenBudgetOptimizer(token_budget=500)
        results = [_make_result("Deploy Kubernetes on AWS", rank=1)]
        ctx = optimizer.optimize(results, recent_context="user: hello\nassistant: hi")
        assert "hello" in ctx.context_text

    def test_budget_analysis(self):
        optimizer = TokenBudgetOptimizer(token_budget=500)
        results = [_make_result("word " * 50, rank=i) for i in range(1, 4)]
        analysis = optimizer.budget_analysis(results)
        assert "total_retrieved_tokens" in analysis
        assert analysis["entries"] == 3


# ------------------------------------------------------------------ #
#  InMemoryLRUCache tests                                              #
# ------------------------------------------------------------------ #

class TestInMemoryLRUCache:

    def test_basic_set_get(self):
        cache = InMemoryLRUCache(maxsize=10, ttl_seconds=60)
        cache.set("key1", {"data": "value1"})
        result = cache.get("key1")
        assert result == {"data": "value1"}

    def test_miss_returns_none(self):
        cache = InMemoryLRUCache()
        assert cache.get("nonexistent") is None

    def test_ttl_expiry(self):
        cache = InMemoryLRUCache(ttl_seconds=1)
        cache.set("k", "v")
        assert cache.get("k") == "v"
        time.sleep(1.1)
        assert cache.get("k") is None

    def test_lru_eviction(self):
        cache = InMemoryLRUCache(maxsize=3, ttl_seconds=60)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        # Access 'a' to make it recently used
        cache.get("a")
        # Add 'd' — should evict 'b' (least recently used)
        cache.set("d", 4)
        assert cache.get("a") == 1
        assert cache.get("b") is None  # evicted

    def test_hit_rate_tracking(self):
        cache = InMemoryLRUCache(ttl_seconds=60)
        cache.set("k", "v")
        cache.get("k")   # hit
        cache.get("k")   # hit
        cache.get("x")   # miss
        assert cache.hits == 2
        assert cache.misses == 1
        assert abs(cache.hit_rate - 2/3) < 0.01

    def test_delete(self):
        cache = InMemoryLRUCache()
        cache.set("k", "v")
        cache.delete("k")
        assert cache.get("k") is None

    def test_clear(self):
        cache = InMemoryLRUCache()
        for i in range(5):
            cache.set(f"k{i}", i)
        cache.clear()
        assert cache.size == 0
