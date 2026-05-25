"""
Memory Manager — Phase 2 (upgraded)
=====================================
Now wires together:
  - HierarchicalMemory  (three-tier promotion with compression)
  - Summarizer          (LLM or extractive)
  - ImportanceScorer    (what to keep)
  - MemoryStore         (ChromaDB)

The public API is unchanged from Phase 1.
"""

import time
from typing import List, Optional, Dict

from src.memory.memory_types import (
    Message, MemoryEntry, MemoryTier, MemoryQueryResult, MessageRole
)
from src.memory.importance_scorer import ImportanceScorer
from src.memory.memory_store import MemoryStore
from src.utils.logger import get_logger
from src.utils.token_counter import count_tokens

logger = get_logger(__name__)


class MemoryManager:
    """
    High-level interface for the LLM Memory Compression System.

    Usage:
        manager = MemoryManager()
        manager.ingest_conversation(messages)
        context = manager.build_context("What deployment strategy did I use?")
    """

    def __init__(
        self,
        persist_dir: str = "./data/embeddings/chroma",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        importance_threshold: float = 0.35,
        max_short_term_tokens: int = 2000,
        max_context_tokens: int = 1500,
        top_k: int = 5,
        llm_provider: str = "extractive",
        llm_model: str = "gpt-3.5-turbo",
        summarizer_mode: str = "bullet",
        use_hierarchical: bool = True,
    ):
        self.max_short_term_tokens = max_short_term_tokens
        self.max_context_tokens = max_context_tokens
        self.top_k = top_k
        self.use_hierarchical = use_hierarchical

        logger.info(f"Initializing MemoryManager (provider={llm_provider})...")

        self.scorer = ImportanceScorer(
            embedding_model_name=embedding_model,
            importance_threshold=importance_threshold,
        )
        self.store = MemoryStore(
            persist_dir=persist_dir,
            embedding_model_name=embedding_model,
        )

        from src.compression.summarizer import Summarizer
        self.summarizer = Summarizer(
            provider=llm_provider,
            model=llm_model,
            mode=summarizer_mode,
        )

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

        self._short_term_buffer: List[Message] = []
        logger.info("MemoryManager ready.")

    def add_message(self, role: str, content: str) -> MemoryEntry:
        msg = Message(role=MessageRole(role), content=content)

        if self.use_hierarchical and self.hierarchical:
            return self.hierarchical.add_message(msg)

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

        total_tokens = sum(count_tokens(m.content) for m in self._short_term_buffer)
        if total_tokens > self.max_short_term_tokens:
            self._promote_simple()

        return entry

    def ingest_conversation(self, messages: List[Dict[str, str]]):
        logger.info(f"Ingesting {len(messages)} messages...")
        for msg_dict in messages:
            self.add_message(msg_dict["role"], msg_dict["content"])
        logger.info(f"Ingestion complete. {self.stats()}")

    def build_context(self, query: str) -> str:
        if self.use_hierarchical and self.hierarchical:
            return self.hierarchical.build_context(query, max_tokens=self.max_context_tokens)

        results = self.store.query_all_tiers(query, top_k=self.top_k)
        if not results:
            return ""

        lines = ["=== Relevant Memory Context ==="]
        total_tokens = 0
        for r in results:
            content = r.entry.active_content
            tokens = count_tokens(content)
            if total_tokens + tokens > self.max_context_tokens:
                break
            lines.append(f"[score={r.similarity_score:.2f}] {content}")
            total_tokens += tokens
            r.entry.touch()
            self.store.update(r.entry)
        lines.append("==============================")
        return "\n".join(lines)

    def compress_and_summarize(self, text: str, max_tokens: int = 250) -> str:
        summary, ratio = self.summarizer.summarize(text, max_tokens)
        logger.info(f"Direct summarize: ratio={ratio:.2f}")
        return summary

    def end_session(self):
        if self.use_hierarchical and self.hierarchical:
            self.hierarchical.end_session()
        else:
            self._promote_simple()

    def _promote_simple(self):
        entries = self.scorer.score_messages(self._short_term_buffer)
        important = self.scorer.filter_important(entries)
        for entry in important:
            entry.tier = MemoryTier.WORKING
            self.store.add(entry)
        self._short_term_buffer = self._short_term_buffer[-4:]

    def stats(self) -> Dict:
        if self.use_hierarchical and self.hierarchical:
            return self.hierarchical.stats()
        store_stats = self.store.stats()
        return {
            "short_term_buffer_messages": len(self._short_term_buffer),
            "store": store_stats,
            "total_stored": sum(store_stats.values()),
        }

    def clear_session(self):
        if self.use_hierarchical and self.hierarchical:
            self.hierarchical._buffer = []
            self.hierarchical._buffer_tokens = 0
        else:
            self._short_term_buffer = []
        self.store.clear_tier(MemoryTier.SHORT_TERM)
        logger.info("Session cleared. Long-term memory preserved.")
