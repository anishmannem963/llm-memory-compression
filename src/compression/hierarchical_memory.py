"""
Hierarchical Memory — Phase 2 Core
=====================================
Manages the full lifecycle of memories across three tiers:

  SHORT_TERM  →  WORKING  →  LONG_TERM
  (raw text)     (compressed)  (embeddings)

Promotion Rules:
  - Short-term fills up (>2000 tokens) → important entries promoted to working
  - Working entries accessed ≥2 times → promoted to long-term
  - Low-importance entries → pruned/discarded

This is the "brain" of the system. Everything flows through here.
"""

import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from src.memory.memory_types import MemoryEntry, MemoryTier, Message, MessageRole
from src.memory.importance_scorer import ImportanceScorer
from src.memory.memory_store import MemoryStore
from src.compression.summarizer import Summarizer
from src.utils.logger import get_logger
from src.utils.token_counter import count_tokens

logger = get_logger(__name__)


@dataclass
class PromotionStats:
    short_to_working: int = 0
    working_to_long_term: int = 0
    pruned: int = 0
    tokens_before: int = 0
    tokens_after: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.tokens_before == 0:
            return 1.0
        return self.tokens_after / self.tokens_before

    @property
    def token_reduction_pct(self) -> float:
        return (1 - self.compression_ratio) * 100


class HierarchicalMemory:
    """
    Full three-tier memory manager with compression at each promotion step.

    Short-term: raw messages in RAM
    Working:    LLM-compressed summaries in ChromaDB
    Long-term:  semantic embeddings of important working memories

    The key insight: compression only happens at promotion boundaries,
    not on every message. This keeps the hot path (adding messages) fast.
    """

    def __init__(
        self,
        store: MemoryStore,
        scorer: ImportanceScorer,
        summarizer: Summarizer,
        short_term_token_limit: int = 2000,
        working_token_limit: int = 1000,
        importance_threshold: float = 0.35,
        working_summary_tokens: int = 250,
        long_term_summary_tokens: int = 150,
        min_access_for_promotion: int = 2,
    ):
        self.store = store
        self.scorer = scorer
        self.summarizer = summarizer

        self.short_term_token_limit = short_term_token_limit
        self.working_token_limit = working_token_limit
        self.importance_threshold = importance_threshold
        self.working_summary_tokens = working_summary_tokens
        self.long_term_summary_tokens = long_term_summary_tokens
        self.min_access_for_promotion = min_access_for_promotion

        # In-memory short-term buffer
        self._buffer: List[Message] = []
        self._buffer_tokens: int = 0

        # Stats tracking
        self.total_promotions = PromotionStats()

    # ------------------------------------------------------------------ #
    #  Adding messages                                                     #
    # ------------------------------------------------------------------ #

    def add_message(self, message: Message) -> MemoryEntry:
        """
        Add a message to short-term buffer.
        Automatically triggers promotion if buffer overflows.
        """
        self._buffer.append(message)
        msg_tokens = count_tokens(message.content)
        self._buffer_tokens += msg_tokens

        # Score and store in short-term
        age = len(self._buffer) * 30
        score = self.scorer.score_message(message, age)

        entry = MemoryEntry(
            tier=MemoryTier.SHORT_TERM,
            original_content=f"{message.role.value}: {message.content}",
            importance_score=score,
            final_score=score,
            source_messages=[message.message_id],
            created_at=message.timestamp,
        )
        self.store.add(entry)

        # Trigger promotion if buffer is full
        if self._buffer_tokens > self.short_term_token_limit:
            stats = self.promote_short_to_working()
            logger.info(
                f"Auto-promoted: {stats.short_to_working} entries, "
                f"{stats.token_reduction_pct:.1f}% token reduction"
            )

        return entry

    def add_turn(self, user_msg: str, assistant_msg: str) -> Tuple[MemoryEntry, MemoryEntry]:
        """Convenience: add a user/assistant turn pair."""
        u = Message(role=MessageRole.USER, content=user_msg)
        a = Message(role=MessageRole.ASSISTANT, content=assistant_msg)
        return self.add_message(u), self.add_message(a)

    # ------------------------------------------------------------------ #
    #  Promotion: Short-term → Working                                    #
    # ------------------------------------------------------------------ #

    def promote_short_to_working(self) -> PromotionStats:
        """
        Compress and promote important short-term messages to working memory.

        Steps:
          1. Score all buffered messages
          2. Filter to important ones
          3. Group them into chunks
          4. Summarize each chunk
          5. Store summaries as working memory entries
          6. Prune low-importance entries
          7. Keep last 4 messages in buffer for immediate context
        """
        stats = PromotionStats()

        if not self._buffer:
            return stats

        logger.info(f"Promoting {len(self._buffer)} short-term messages to working memory...")

        # Score messages
        entries = self.scorer.score_messages(self._buffer)
        important = self.scorer.filter_important(entries)
        stats.short_to_working = len(important)
        stats.tokens_before = sum(count_tokens(e.original_content) for e in important)

        if not important:
            logger.info("No important messages to promote.")
            self._buffer = self._buffer[-4:]
            self._buffer_tokens = sum(count_tokens(m.content) for m in self._buffer)
            return stats

        # Group important entries into chunks for summarization
        chunks = self._group_into_chunks(important, chunk_tokens=800)

        promoted_entries = []
        for chunk in chunks:
            chunk_text = "\n".join(e.original_content for e in chunk)
            chunk_tokens = count_tokens(chunk_text)

            # Summarize the chunk
            if chunk_tokens > self.working_summary_tokens:
                summary, ratio = self.summarizer.summarize(
                    chunk_text, max_tokens=self.working_summary_tokens
                )
            else:
                summary = chunk_text
                ratio = 1.0

            # Create working memory entry
            avg_importance = sum(e.importance_score for e in chunk) / len(chunk)
            working_entry = MemoryEntry(
                tier=MemoryTier.WORKING,
                original_content=chunk_text,
                compressed_content=summary,
                importance_score=avg_importance,
                final_score=avg_importance,
                source_messages=[mid for e in chunk for mid in e.source_messages],
                created_at=chunk[0].created_at,
            )
            self.store.add(working_entry)
            promoted_entries.append(working_entry)

            stats.tokens_after += count_tokens(summary)

        logger.info(
            f"Promoted {len(promoted_entries)} chunks to working memory. "
            f"Compression: {stats.tokens_before}→{stats.tokens_after} tokens "
            f"({stats.token_reduction_pct:.1f}% reduction)"
        )

        # Update global stats
        self.total_promotions.short_to_working += stats.short_to_working
        self.total_promotions.tokens_before += stats.tokens_before
        self.total_promotions.tokens_after += stats.tokens_after

        # Keep last 4 messages in buffer for continuity
        self._buffer = self._buffer[-4:]
        self._buffer_tokens = sum(count_tokens(m.content) for m in self._buffer)

        return stats

    # ------------------------------------------------------------------ #
    #  Promotion: Working → Long-term                                     #
    # ------------------------------------------------------------------ #

    def promote_working_to_long_term(self) -> PromotionStats:
        """
        Promote frequently-accessed working memories to long-term storage.
        Called at end of session or periodically.

        Working memories with access_count >= threshold get:
          - Further compressed (shorter summary)
          - Promoted to LONG_TERM tier
          - Deleted from WORKING tier
        """
        stats = PromotionStats()

        working_results = self.store.query(
            query_text="",   # empty query returns all (ChromaDB behavior)
            tier=MemoryTier.WORKING,
            top_k=500,
        )

        for result in working_results:
            entry = result.entry

            if entry.access_count >= self.min_access_for_promotion:
                # Compress further for long-term storage
                content = entry.active_content
                if count_tokens(content) > self.long_term_summary_tokens:
                    final_summary, _ = self.summarizer.summarize(
                        content, max_tokens=self.long_term_summary_tokens
                    )
                else:
                    final_summary = content

                lt_entry = MemoryEntry(
                    tier=MemoryTier.LONG_TERM,
                    original_content=entry.original_content,
                    compressed_content=final_summary,
                    importance_score=entry.importance_score,
                    final_score=entry.importance_score,
                    source_messages=entry.source_messages,
                    created_at=entry.created_at,
                    access_count=entry.access_count,
                )
                self.store.add(lt_entry)
                self.store.delete(entry.entry_id, MemoryTier.WORKING)
                stats.working_to_long_term += 1

        self.total_promotions.working_to_long_term += stats.working_to_long_term
        logger.info(f"Promoted {stats.working_to_long_term} entries to long-term memory.")
        return stats

    # ------------------------------------------------------------------ #
    #  Context building                                                    #
    # ------------------------------------------------------------------ #

    def build_context(self, query: str, max_tokens: int = 1500) -> str:
        """
        Build a compressed, relevant context block for the given query.

        Strategy:
          1. Always include recent short-term buffer (last 3 messages)
          2. Retrieve relevant working + long-term memories via semantic search
          3. Pack into token budget, most relevant first
        """
        lines = []
        token_budget = max_tokens

        # 1. Recent context (always included)
        recent = self._buffer[-3:] if self._buffer else []
        if recent:
            recent_text = "\n".join(f"{m.role.value}: {m.content}" for m in recent)
            recent_tokens = count_tokens(recent_text)
            if recent_tokens <= token_budget * 0.4:
                lines.append("--- Recent conversation ---")
                lines.append(recent_text)
                token_budget -= recent_tokens

        # 2. Semantic retrieval from working + long-term
        results = self.store.query_all_tiers(query, top_k=8)

        lines.append("--- Relevant memory ---")
        for r in results:
            content = r.entry.active_content
            tokens = count_tokens(content)
            if tokens <= token_budget:
                lines.append(f"[relevance={r.similarity_score:.2f}] {content}")
                token_budget -= tokens
                r.entry.touch()
                self.store.update(r.entry)
            if token_budget < 50:
                break

        if len(lines) <= 2:
            return ""

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    def _group_into_chunks(
        self, entries: List[MemoryEntry], chunk_tokens: int = 800
    ) -> List[List[MemoryEntry]]:
        """Group entries into chunks not exceeding chunk_tokens each."""
        chunks, current_chunk, current_tokens = [], [], 0
        for entry in entries:
            t = count_tokens(entry.original_content)
            if current_tokens + t > chunk_tokens and current_chunk:
                chunks.append(current_chunk)
                current_chunk, current_tokens = [], 0
            current_chunk.append(entry)
            current_tokens += t
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def stats(self) -> Dict:
        store_stats = self.store.stats()
        return {
            "buffer_messages": len(self._buffer),
            "buffer_tokens": self._buffer_tokens,
            "store": store_stats,
            "total_stored": sum(store_stats.values()),
            "total_promotions": {
                "short_to_working": self.total_promotions.short_to_working,
                "working_to_long_term": self.total_promotions.working_to_long_term,
                "overall_compression_ratio": round(self.total_promotions.compression_ratio, 3),
                "token_reduction_pct": round(self.total_promotions.token_reduction_pct, 1),
            },
        }

    def end_session(self):
        """
        Called when a conversation session ends.
        Flushes buffer, promotes working to long-term.
        """
        logger.info("Ending session — flushing memory tiers...")
        if self._buffer:
            self.promote_short_to_working()
        self.promote_working_to_long_term()
        self._buffer = []
        self._buffer_tokens = 0
        logger.info("Session ended. Long-term memory persisted.")
