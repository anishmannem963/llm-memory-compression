"""
Memory Manager — Central Orchestrator
======================================
Ties together:
  - ImportanceScorer      (what to keep)
  - MemoryStore           (where to keep it)
  - CompressionEngine     (how to compress it)  [Phase 2]
  - Retriever             (how to retrieve it)  [Phase 3]

In Phase 1, compression and retrieval are stubbed out with basic versions.
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
    High-level interface for the entire memory system.
    Use this class in your application — it handles all internal routing.

    Usage:
        manager = MemoryManager()
        manager.ingest_conversation(messages)
        context = manager.build_context("What deployment strategy did I use?")
        # Pass `context` as part of your LLM prompt
    """

    def __init__(
        self,
        persist_dir: str = "./data/embeddings/chroma",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        importance_threshold: float = 0.35,
        max_short_term_tokens: int = 2000,
        max_context_tokens: int = 1500,
        top_k: int = 5,
    ):
        self.max_short_term_tokens = max_short_term_tokens
        self.max_context_tokens = max_context_tokens
        self.top_k = top_k

        logger.info("Initializing MemoryManager...")

        self.scorer = ImportanceScorer(
            embedding_model_name=embedding_model,
            importance_threshold=importance_threshold,
        )
        self.store = MemoryStore(
            persist_dir=persist_dir,
            embedding_model_name=embedding_model,
        )

        # Short-term buffer: keeps raw recent messages in RAM
        self._short_term_buffer: List[Message] = []

        logger.info("MemoryManager ready.")

    # ------------------------------------------------------------------ #
    #  Ingestion                                                           #
    # ------------------------------------------------------------------ #

    def add_message(self, role: str, content: str) -> MemoryEntry:
        """
        Add a single message to the memory system.
        Automatically scores, stores, and manages tier promotion.
        """
        msg = Message(role=MessageRole(role), content=content)
        self._short_term_buffer.append(msg)

        # Score and store
        age = (len(self._short_term_buffer) - 1) * 30
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

        # If short-term buffer is getting large, promote to working memory
        total_tokens = sum(count_tokens(m.content) for m in self._short_term_buffer)
        if total_tokens > self.max_short_term_tokens:
            self._promote_to_working_memory()

        return entry

    def ingest_conversation(self, messages: List[Dict[str, str]]):
        """
        Ingest a full conversation history at once.
        messages = [{"role": "user", "content": "..."}, ...]
        """
        logger.info(f"Ingesting {len(messages)} messages...")
        for msg_dict in messages:
            self.add_message(msg_dict["role"], msg_dict["content"])
        logger.info(f"Ingestion complete. Store stats: {self.store.stats()}")

    # ------------------------------------------------------------------ #
    #  Context Building                                                    #
    # ------------------------------------------------------------------ #

    def build_context(self, query: str) -> str:
        """
        Given a query, retrieve the most relevant memories and
        format them as a context block to prepend to your LLM prompt.

        Returns a string ready to inject into: system_prompt + context + user_query
        """
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

            lines.append(
                f"[score={r.similarity_score:.2f}] {content}"
            )
            total_tokens += tokens
            r.entry.touch()
            self.store.update(r.entry)

        lines.append("==============================")

        context = "\n".join(lines)
        logger.info(
            f"Built context: {len(results)} memories retrieved, "
            f"~{total_tokens} tokens used"
        )
        return context

    def get_recent_context(self, last_n_messages: int = 6) -> str:
        """Return the most recent N messages from short-term buffer as plain text."""
        recent = self._short_term_buffer[-last_n_messages:]
        return "\n".join(
            f"{m.role.value}: {m.content}" for m in recent
        )

    # ------------------------------------------------------------------ #
    #  Tier Management                                                     #
    # ------------------------------------------------------------------ #

    def _promote_to_working_memory(self):
        """
        Move important short-term messages to working memory.
        In Phase 1: just re-tier the entries (no compression yet).
        Phase 2 will add actual compression here.
        """
        logger.info("Promoting short-term buffer to working memory...")

        # Score all buffered messages
        entries = self.scorer.score_messages(self._short_term_buffer)
        important = self.scorer.filter_important(entries)

        for entry in important:
            entry.tier = MemoryTier.WORKING
            self.store.add(entry)

        # Clear short-term buffer (keep last 4 messages for immediate context)
        self._short_term_buffer = self._short_term_buffer[-4:]
        logger.info(
            f"Promoted {len(important)} entries to working memory. "
            f"Short-term buffer trimmed to {len(self._short_term_buffer)} messages."
        )

    def promote_to_long_term(self, min_access_count: int = 2):
        """
        Move frequently-accessed working memory entries to long-term storage.
        Call this periodically (e.g. end of session).
        """
        working_results = self.store.query(
            "", tier=MemoryTier.WORKING, top_k=100
        )
        promoted = 0
        for r in working_results:
            if r.entry.access_count >= min_access_count:
                r.entry.tier = MemoryTier.LONG_TERM
                self.store.add(r.entry)
                self.store.delete(r.entry.entry_id, MemoryTier.WORKING)
                promoted += 1

        logger.info(f"Promoted {promoted} entries to long-term memory.")

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    def stats(self) -> Dict:
        store_stats = self.store.stats()
        return {
            "short_term_buffer_messages": len(self._short_term_buffer),
            "store": store_stats,
            "total_stored": sum(store_stats.values()),
        }

    def clear_session(self):
        """Clear short-term buffer (keeps long-term memory intact)."""
        self._short_term_buffer = []
        self.store.clear_tier(MemoryTier.SHORT_TERM)
        logger.info("Session cleared. Long-term memory preserved.")
