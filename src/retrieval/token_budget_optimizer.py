"""
Token Budget Optimizer — Phase 3
===================================
Given a list of retrieved results and a token budget,
pack the maximum useful information into that budget.

Strategies used:
  1. Saliency ranking     — sort by fusion score, take highest first
  2. Redundancy removal   — drop entries too similar to already-selected ones
  3. Adaptive truncation  — trim entries that slightly exceed budget
  4. Context packing      — format entries compactly to save tokens

Why this matters:
  Even with compression, retrieved entries may exceed the context budget.
  Naive truncation (just cut at N tokens) loses the most important content.
  This optimizer ensures the BEST content fills the budget, not the FIRST content.

Example:
  Budget: 1500 tokens
  Retrieved: 8 entries totalling 3200 tokens
  After optimization: 5 entries, 1480 tokens, all high-relevance
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass

from src.retrieval.hybrid_retriever import RetrievalResult
from src.utils.logger import get_logger
from src.utils.token_counter import count_tokens, truncate_to_token_limit

logger = get_logger(__name__)


@dataclass
class OptimizedContext:
    """The final packed context ready to inject into an LLM prompt."""
    context_text: str
    total_tokens: int
    entries_included: int
    entries_dropped: int
    budget_used_pct: float
    redundancy_removed: int


class TokenBudgetOptimizer:
    """
    Packs the most useful retrieved content into a fixed token budget.
    """

    def __init__(
        self,
        token_budget: int = 1500,
        redundancy_threshold: float = 0.85,   # cosine sim above this = redundant
        min_entry_tokens: int = 20,            # drop entries shorter than this
        truncation_margin: int = 50,           # allow slight overshoot then truncate
    ):
        self.token_budget = token_budget
        self.redundancy_threshold = redundancy_threshold
        self.min_entry_tokens = min_entry_tokens
        self.truncation_margin = truncation_margin

    def optimize(
        self,
        results: List[RetrievalResult],
        query: str = "",
        recent_context: str = "",
    ) -> OptimizedContext:
        """
        Select and pack the best entries within the token budget.

        Args:
            results:         Hybrid retrieval results, already ranked by fusion score
            query:           Original query (used for formatting)
            recent_context:  Recent conversation turns to always include first
        """
        if not results:
            return OptimizedContext(
                context_text="",
                total_tokens=0,
                entries_included=0,
                entries_dropped=0,
                budget_used_pct=0.0,
                redundancy_removed=0,
            )

        # Reserve tokens for recent context
        budget = self.token_budget
        recent_tokens = 0
        if recent_context:
            recent_tokens = count_tokens(recent_context)
            budget -= recent_tokens
            budget = max(budget, 200)  # always leave at least 200 for memories

        # Step 1: Filter too-short entries
        valid = [r for r in results if count_tokens(r.entry.active_content) >= self.min_entry_tokens]
        dropped_short = len(results) - len(valid)

        # Step 2: Remove redundant entries (MMR-style)
        selected, redundancy_removed = self._remove_redundancy(valid)

        # Step 3: Pack into budget (greedy by fusion score, already sorted)
        packed: List[RetrievalResult] = []
        used_tokens = 0
        dropped_budget = 0

        for result in selected:
            content = result.entry.active_content
            tokens = count_tokens(content)

            if used_tokens + tokens <= budget:
                packed.append(result)
                used_tokens += tokens
            elif used_tokens + tokens <= budget + self.truncation_margin:
                # Slightly over budget — truncate this entry to fit
                remaining = budget - used_tokens
                if remaining >= self.min_entry_tokens:
                    truncated = truncate_to_token_limit(content, remaining)
                    result.entry.compressed_content = truncated
                    packed.append(result)
                    used_tokens += count_tokens(truncated)
                else:
                    dropped_budget += 1
            else:
                dropped_budget += 1

            if used_tokens >= budget * 0.95:
                break

        # Step 4: Format the context block
        context_text = self._format_context(packed, recent_context, query)
        total_tokens = count_tokens(context_text)

        logger.info(
            f"Budget optimizer: {len(packed)}/{len(results)} entries packed, "
            f"{total_tokens}/{self.token_budget} tokens used "
            f"({total_tokens/self.token_budget:.0%}), "
            f"redundancy_removed={redundancy_removed}"
        )

        return OptimizedContext(
            context_text=context_text,
            total_tokens=total_tokens,
            entries_included=len(packed),
            entries_dropped=dropped_budget + dropped_short,
            budget_used_pct=round(total_tokens / self.token_budget * 100, 1),
            redundancy_removed=redundancy_removed,
        )

    def _remove_redundancy(
        self, results: List[RetrievalResult]
    ) -> Tuple[List[RetrievalResult], int]:
        """
        Remove entries that are highly similar to already-selected entries.
        Uses simple word-overlap Jaccard similarity (no embedding needed).
        """
        if not results:
            return [], 0

        selected = [results[0]]
        removed = 0

        for candidate in results[1:]:
            is_redundant = False
            cand_words = set(candidate.entry.active_content.lower().split())

            for sel in selected:
                sel_words = set(sel.entry.active_content.lower().split())
                if not cand_words or not sel_words:
                    continue
                intersection = len(cand_words & sel_words)
                union = len(cand_words | sel_words)
                jaccard = intersection / union if union > 0 else 0.0
                if jaccard > self.redundancy_threshold:
                    is_redundant = True
                    break

            if is_redundant:
                removed += 1
            else:
                selected.append(candidate)

        return selected, removed

    def _format_context(
        self,
        results: List[RetrievalResult],
        recent_context: str,
        query: str,
    ) -> str:
        """Format the selected entries into a clean context block."""
        lines = []

        if recent_context:
            lines.append("--- Recent conversation ---")
            lines.append(recent_context.strip())

        if results:
            lines.append("--- Relevant memory ---")
            for r in results:
                score_str = f"[relevance={r.fusion_score:.2f}]"
                lines.append(f"{score_str} {r.entry.active_content.strip()}")

        return "\n".join(lines)

    def budget_analysis(self, results: List[RetrievalResult]) -> dict:
        """
        Analyze token distribution across retrieved results.
        Useful for tuning the budget.
        """
        if not results:
            return {}

        token_counts = [count_tokens(r.entry.active_content) for r in results]
        total = sum(token_counts)

        return {
            "total_retrieved_tokens": total,
            "budget": self.token_budget,
            "would_exceed_budget": total > self.token_budget,
            "entries": len(results),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "avg_tokens": round(total / len(token_counts)),
            "budget_coverage": round(self.token_budget / total * 100, 1) if total > 0 else 100.0,
        }
