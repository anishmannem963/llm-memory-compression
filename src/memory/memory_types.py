from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum
import time
import uuid


class MemoryTier(Enum):
    SHORT_TERM  = "short_term"    # Raw recent turns, full text
    WORKING     = "working"       # Compressed recent context
    LONG_TERM   = "long_term"     # Semantic embeddings of older memories


class MessageRole(Enum):
    USER      = "user"
    ASSISTANT = "assistant"
    SYSTEM    = "system"


@dataclass
class Message:
    """A single conversation turn."""
    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @property
    def token_estimate(self) -> int:
        from src.utils.token_counter import count_tokens
        return count_tokens(self.content)

    def __repr__(self):
        preview = self.content[:60] + "..." if len(self.content) > 60 else self.content
        return f"Message({self.role.value}, '{preview}')"


@dataclass
class MemoryEntry:
    """
    A processed memory unit stored in one of the three memory tiers.
    Holds original content, optional compressed version, and metadata.
    """
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tier: MemoryTier = MemoryTier.SHORT_TERM

    # Content
    original_content: str = ""
    compressed_content: Optional[str] = None
    embedding: Optional[List[float]] = None

    # Scoring
    importance_score: float = 0.0
    recency_score: float = 1.0
    final_score: float = 0.0

    # Metadata
    source_messages: List[str] = field(default_factory=list)  # message_ids
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    tags: List[str] = field(default_factory=list)

    @property
    def active_content(self) -> str:
        """Return compressed version if available, else original."""
        return self.compressed_content or self.original_content

    @property
    def compression_ratio(self) -> Optional[float]:
        if self.compressed_content and self.original_content:
            return len(self.compressed_content) / len(self.original_content)
        return None

    def touch(self):
        """Update access metadata."""
        self.last_accessed = time.time()
        self.access_count += 1

    def __repr__(self):
        preview = self.active_content[:50] + "..."
        return (
            f"MemoryEntry(tier={self.tier.value}, "
            f"score={self.final_score:.2f}, '{preview}')"
        )


@dataclass
class MemoryQueryResult:
    """Result returned from memory retrieval."""
    entry: MemoryEntry
    similarity_score: float
    rank: int

    def __repr__(self):
        return f"QueryResult(rank={self.rank}, sim={self.similarity_score:.3f})"
