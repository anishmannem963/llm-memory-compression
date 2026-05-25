"""
Memory Manager — Phase 3 (final upgrade)
==========================================
Now wires together ALL phases:
  Phase 1: ImportanceScorer + MemoryStore (ChromaDB)
  Phase 2: Summarizer + HierarchicalMemory (compression)
  Phase 3: HybridRetriever + TokenBudgetOptimizer + RedisCache

Public API is unchanged. All new capabilities activate automatically.
"""

from typing import List, Optional, Dict
from src.memory.memory_types import Message, MemoryEntry, MemoryTier, MessageRole
from src.memory.importance_scorer import ImportanceScorer
from src.memory.memory_store import MemoryStore
from src.utils.logger import get_logger
from src.utils.token_counter import count_tokens

logger = get_logger(__name__)


class MemoryManager:
    """
    Central interface for the LLM Memory Compression System.

    Usage:
        manager = MemoryManager()
        manager.ingest_conversation(messages)
        context = manager.build_context("What deployment strategy did I use?")
        # Inject context into your LLM prompt
    """

    def __init__(
        self,
        persist_dir: str = "./data/embeddings/chroma",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        importance_threshold: float = 0.35,
        max_short_term_tokens: int = 2000,
        max_context_tokens: int = 1500,
        top_k: int = 5,
        # Phase 2
        llm_provider: str = "extractive",
        llm_model: str = "gpt-3.5-turbo",
        summarizer_mode: str = "bullet",
        use_hierarchical: bool = True,
        # Phase 3
        use_hybrid_retrieval: bool = True,
        use_cache: bool = True,
        cache_host: str = "localhost",
        cache_port: int = 6379,
        cache_ttl: int = 300,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4,
        redundancy_threshold: float = 0.85,
    ):
        self.max_short_term_tokens = max_short_term_tokens
        self.max_context_tokens = max_context_tokens
        self.top_k = top_k
        self.use_hierarchical = use_hierarchical
        self.use_hybrid_retrieval = use_hybrid_retrieval

        logger.info(f"Initializing MemoryManager (provider={llm_provider}, hybrid={use_hybrid_retrieval}, cache={use_cache})...")

        # Phase 1 components
        self.scorer = ImportanceScorer(
            embedding_model_name=embedding_model,
            importance_threshold=importance_threshold,
        )
        self.store = MemoryStore(
            persist_dir=persist_dir,
            embedding_model_name=embedding_model,
        )

        # Phase 2 components
        from src.compression.summarizer import Summarizer
        self.summarizer = Summarizer(provider=llm_provider, model=llm_model, mode=summarizer_mode)

        if use_hierarchical:
            from src.compression.hierarchical_memory import HierarchicalMemory
            self.hierarchical = HierarchicalMemory(
                store=self.store,
                scorer=self.scorer,
                summarizer=self.summarizer,
                short_term_token_limit=max_short_term_tokens,
                importance_threshold=importance_threshold,
            )
        else:
            self.hierarchical = None

        # Phase 3 components
        if use_hybrid_retrieval:
            from src.retrieval.hybrid_retriever import HybridRetriever
            self.retriever = HybridRetriever(
                store=self.store,
                top_k=top_k,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight,
            )
        else:
            self.retriever = None

        from src.retrieval.token_budget_optimizer import TokenBudgetOptimizer
        self.optimizer = TokenBudgetOptimizer(
            token_budget=max_context_tokens,
            redundancy_threshold=redundancy_threshold,
        )

        if use_cache:
            from src.retrieval.cache import RedisCache
            self.cache = RedisCache(
                host=cache_host, port=cache_port, ttl_seconds=cache_ttl
            )
        else:
            self.cache = None

        self._short_term_buffer: List[Message] = []
        logger.info("MemoryManager ready.")

    # ------------------------------------------------------------------ #
    #  Ingestion                                                           #
    # ------------------------------------------------------------------ #

    def add_message(self, role: str, content: str) -> MemoryEntry:
        """Add a message. Routes through hierarchical manager if enabled."""
        msg = Message(role=MessageRole(role), content=content)

        if self.use_hierarchical and self.hierarchical:
            entry = self.hierarchical.add_message(msg)
            # Index in hybrid retriever
            if self.retriever:
                self.retriever.index_entry(entry)
            return entry

        # Fallback: Phase 1 simple path
        self._short_term_buffer.append(msg)
        age = len(self._short_term_buffer) * 30
        score = self.scorer.score_message(msg, age)
        entry = MemoryEntry(
            tier=MemoryTier.SHORT_TERM,
            original_content=f"{role}: {content}",
            importance_score=score,
            final_score=score,
            source_messages=[msg.message_id],
            created_at=msg.timestamp,
        )
        self.store.add(entry)
        if self.retriever:
            self.retriever.index_entry(entry)
        total_tokens = sum(count_tokens(m.content) for m in self._short_term_buffer)
        if total_tokens > self.max_short_term_tokens:
            self._promote_simple()
        return entry

    def ingest_conversation(self, messages: List[Dict[str, str]]):
        """Ingest a full conversation at once."""
        logger.info(f"Ingesting {len(messages)} messages...")
        for msg_dict in messages:
            self.add_message(msg_dict["role"], msg_dict["content"])
        logger.info(f"Ingestion complete. {self.stats()}")

    # ------------------------------------------------------------------ #
    #  Context building — Phase 3 path                                    #
    # ------------------------------------------------------------------ #

    def build_context(self, query: str) -> str:
        """
        Build compressed, relevant context for the given query.
        Uses cache → hybrid retrieval → token budget optimization.
        """
        # 1. Cache check
        if self.cache:
            cached = self.cache.get(query, top_k=self.top_k)
            if cached is not None:
                logger.debug(f"Cache hit for query: '{query[:50]}'")
                return cached.get("context_text", "")

        # 2. Recent context from buffer
        recent = ""
        if self.use_hierarchical and self.hierarchical:
            buf = self.hierarchical._buffer[-3:]
            if buf:
                recent = "\n".join(f"{m.role.value}: {m.content}" for m in buf)
        elif self._short_term_buffer:
            buf = self._short_term_buffer[-3:]
            recent = "\n".join(f"{m.role.value}: {m.content}" for m in buf)

        # 3. Retrieval — hybrid or semantic-only
        if self.retriever and self.retriever.stats()["cached_entries"] > 0:
            results = self.retriever.retrieve(query, top_k=self.top_k * 2)
            # Convert to format optimizer expects
            from src.retrieval.hybrid_retriever import RetrievalResult
            opt_results = results
        else:
            # Fallback: ChromaDB-only retrieval
            raw = self.store.query_all_tiers(query, top_k=self.top_k * 2)
            from src.retrieval.hybrid_retriever import RetrievalResult
            from src.memory.memory_types import MemoryEntry
            opt_results = [
                RetrievalResult(
                    entry=r.entry,
                    semantic_score=r.similarity_score,
                    keyword_score=0.0,
                    fusion_score=r.similarity_score,
                    rank=r.rank,
                )
                for r in raw
            ]

        # 4. Token budget optimization
        optimized = self.optimizer.optimize(opt_results, query=query, recent_context=recent)

        # 5. Cache the result
        if self.cache and optimized.context_text:
            self.cache.set(query, {"context_text": optimized.context_text}, top_k=self.top_k)

        # 6. Update access counts
        for r in opt_results[:optimized.entries_included]:
            r.entry.touch()
            self.store.update(r.entry)

        logger.info(
            f"Context built: {optimized.entries_included} entries, "
            f"{optimized.total_tokens} tokens ({optimized.budget_used_pct:.0f}% budget), "
            f"redundancy_removed={optimized.redundancy_removed}"
        )
        return optimized.context_text

    def compress_and_summarize(self, text: str, max_tokens: int = 250) -> str:
        summary, ratio = self.summarizer.summarize(text, max_tokens)
        return summary

    # ------------------------------------------------------------------ #
    #  Session management                                                  #
    # ------------------------------------------------------------------ #

    def end_session(self):
        if self.use_hierarchical and self.hierarchical:
            self.hierarchical.end_session()
        else:
            self._promote_simple()
        if self.cache:
            self.cache.flush()
        logger.info("Session ended. Memory persisted, cache flushed.")

    def _promote_simple(self):
        entries = self.scorer.score_messages(self._short_term_buffer)
        important = self.scorer.filter_important(entries)
        for entry in important:
            entry.tier = MemoryTier.WORKING
            self.store.add(entry)
        self._short_term_buffer = self._short_term_buffer[-4:]

    # ------------------------------------------------------------------ #
    #  Stats                                                               #
    # ------------------------------------------------------------------ #

    def stats(self) -> Dict:
        base = {}
        if self.use_hierarchical and self.hierarchical:
            base = self.hierarchical.stats()
        else:
            store_stats = self.store.stats()
            base = {
                "short_term_buffer_messages": len(self._short_term_buffer),
                "store": store_stats,
                "total_stored": sum(store_stats.values()),
            }

        if self.retriever:
            base["retriever"] = self.retriever.stats()
        if self.cache:
            base["cache"] = self.cache.stats()

        return base

    def clear_session(self):
        if self.use_hierarchical and self.hierarchical:
            self.hierarchical._buffer = []
            self.hierarchical._buffer_tokens = 0
        else:
            self._short_term_buffer = []
        self.store.clear_tier(MemoryTier.SHORT_TERM)
        if self.cache:
            self.cache.flush()
        logger.info("Session cleared.")
