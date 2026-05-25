"""
Importance Scorer — Phase 1 Core Component
==========================================
Ranks messages/memories by how important they are to retain.

Scoring is a weighted combination of:
  1. Semantic importance  — embedding similarity to "important" anchor phrases
  2. Keyword signals      — presence of task-critical keywords
  3. Recency              — newer messages score higher (exponential decay)
  4. Length signal        — very short messages (greetings) score lower

All scores are normalized to [0, 1].
"""

import re
import time
import math
from typing import List, Optional

from src.memory.memory_types import Message, MemoryEntry, MemoryTier
from src.utils.logger import get_logger
from src.utils.token_counter import count_tokens

logger = get_logger(__name__)


# Keywords that signal high importance
HIGH_IMPORTANCE_KEYWORDS = [
    # Technical / task
    "deploy", "kubernetes", "aws", "docker", "api", "database", "error",
    "bug", "fix", "implement", "architecture", "design", "requirement",
    "deadline", "schedule", "important", "critical", "urgent",
    # Decision-making
    "decided", "agreed", "confirmed", "approved", "rejected", "chosen",
    "strategy", "plan", "goal", "objective",
    # Personal context
    "my name", "i am", "i work", "my project", "our team", "we need",
]

# Patterns that signal LOW importance (small talk, filler)
LOW_IMPORTANCE_PATTERNS = [
    r"^(hi|hello|hey|thanks|thank you|ok|okay|sure|got it|sounds good)[.!?]?$",
    r"^(yes|no|maybe|perhaps)[.!?]?$",
    r"^(lol|haha|nice|cool|great|awesome)[.!?]?$",
]


class ImportanceScorer:
    """
    Scores messages by importance for memory retention decisions.
    Works without any GPU — uses embedding model if available,
    falls back to keyword-only scoring if sentence-transformers not installed.
    """

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_embeddings: bool = True,
        recency_decay: float = 0.95,
        importance_threshold: float = 0.35,
    ):
        self.recency_decay = recency_decay
        self.importance_threshold = importance_threshold
        self.embedding_model = None

        if use_embeddings:
            self._load_embedding_model(embedding_model_name)

        # Anchor sentences representing "important" content
        self._importance_anchors = [
            "Deploy the application to production server",
            "The system architecture requires these components",
            "We decided to implement this feature",
            "The deadline is next week",
            "Critical bug found in the authentication module",
            "My name is John and I work on the ML team",
            "The project requirements are as follows",
        ]
        self._anchor_embeddings = None
        if self.embedding_model is not None:
            self._anchor_embeddings = self.embedding_model.encode(
                self._importance_anchors, convert_to_tensor=False
            )
            logger.info(f"Loaded {len(self._importance_anchors)} importance anchors")

    def _load_embedding_model(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"Embedding model loaded: {model_name}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Using keyword-only scoring. Run: pip install sentence-transformers"
            )
        except Exception as e:
            logger.warning(f"Could not load embedding model: {e}. Using keyword scoring.")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def score_message(self, message: Message, conversation_age_seconds: float = 0.0) -> float:
        """
        Score a single message. Returns a float in [0, 1].
        Higher = more important to remember.
        """
        text = message.content.strip()

        # Hard filter: system messages always retained
        if message.role.value == "system":
            return 1.0

        # Hard filter: very short messages (greetings, filler)
        if self._is_low_importance(text):
            return 0.05

        # Component scores
        keyword_score  = self._keyword_score(text)
        semantic_score = self._semantic_score(text)
        length_score   = self._length_score(text)
        recency_score  = self._recency_score(conversation_age_seconds)

        # Weighted combination
        if self.embedding_model is not None:
            final = (
                0.40 * semantic_score +
                0.30 * keyword_score  +
                0.20 * recency_score  +
                0.10 * length_score
            )
        else:
            # No embeddings: rely more on keywords
            final = (
                0.55 * keyword_score +
                0.30 * recency_score +
                0.15 * length_score
            )

        return round(min(max(final, 0.0), 1.0), 4)

    def score_messages(
        self,
        messages: List[Message],
        return_entries: bool = True
    ) -> List[MemoryEntry]:
        """
        Score a list of messages. Returns MemoryEntry objects with scores set.
        Messages are scored relative to their position in the conversation
        (earlier = older = lower recency).
        """
        now = time.time()
        entries = []

        for i, msg in enumerate(messages):
            # Age: earlier messages are "older"
            age_seconds = (len(messages) - i) * 30  # assume 30s between messages
            score = self.score_message(msg, age_seconds)

            entry = MemoryEntry(
                tier=MemoryTier.SHORT_TERM,
                original_content=f"{msg.role.value}: {msg.content}",
                importance_score=score,
                recency_score=self._recency_score(age_seconds),
                final_score=score,
                source_messages=[msg.message_id],
                created_at=msg.timestamp,
            )
            entries.append(entry)

        logger.info(
            f"Scored {len(entries)} messages. "
            f"Above threshold ({self.importance_threshold}): "
            f"{sum(1 for e in entries if e.final_score >= self.importance_threshold)}"
        )
        return entries

    def filter_important(self, entries: List[MemoryEntry]) -> List[MemoryEntry]:
        """Return only entries above the importance threshold."""
        return [e for e in entries if e.final_score >= self.importance_threshold]

    # ------------------------------------------------------------------ #
    #  Internal scoring components                                         #
    # ------------------------------------------------------------------ #

    def _is_low_importance(self, text: str) -> bool:
        """Check if text matches known low-importance patterns."""
        text_lower = text.lower().strip()
        for pattern in LOW_IMPORTANCE_PATTERNS:
            if re.match(pattern, text_lower):
                return True
        # Very short: under 4 words
        if len(text_lower.split()) < 4:
            return True
        return False

    def _keyword_score(self, text: str) -> float:
        """Score based on presence of high-importance keywords."""
        text_lower = text.lower()
        hits = sum(1 for kw in HIGH_IMPORTANCE_KEYWORDS if kw in text_lower)
        # Normalize: 3+ keyword hits = max score
        return min(hits / 3.0, 1.0)

    def _semantic_score(self, text: str) -> float:
        """
        Score by cosine similarity to importance anchor embeddings.
        Returns 0.0 if embedding model not available.
        """
        if self.embedding_model is None or self._anchor_embeddings is None:
            return 0.0

        try:
            import numpy as np
            text_emb = self.embedding_model.encode([text])[0]

            # Cosine similarity to each anchor
            sims = []
            for anchor_emb in self._anchor_embeddings:
                norm_t = np.linalg.norm(text_emb)
                norm_a = np.linalg.norm(anchor_emb)
                if norm_t == 0 or norm_a == 0:
                    sims.append(0.0)
                else:
                    sims.append(float(np.dot(text_emb, anchor_emb) / (norm_t * norm_a)))

            # Use the max similarity to any anchor
            return max(sims)

        except Exception as e:
            logger.warning(f"Semantic scoring failed: {e}")
            return 0.0

    def _length_score(self, text: str) -> float:
        """
        Longer messages carry more information (up to a point).
        Score: 0 for very short, peaks at ~100 tokens, flat after.
        """
        tokens = count_tokens(text)
        if tokens < 10:
            return 0.1
        elif tokens < 50:
            return 0.4
        elif tokens < 100:
            return 0.7
        else:
            return 1.0

    def _recency_score(self, age_seconds: float) -> float:
        """
        Exponential decay based on age. Recent = high score.
        decay_factor^(age_in_minutes)
        """
        age_minutes = age_seconds / 60.0
        return math.pow(self.recency_decay, age_minutes)
