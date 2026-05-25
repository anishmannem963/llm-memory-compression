"""
Compression Engine — Phase 2 Core Component
============================================
Three compression strategies:
  A. Summarization    — LLM-based, converts 5000 tokens → ~200-token summary
  B. Embedding        — PCA dimensionality reduction on vectors
  C. Hierarchical     — combines A + B with tier-aware logic

In Phase 1, only basic truncation is used.
In Phase 2, plug in your LLM API and the full strategies activate.
"""

import time
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod

from src.memory.memory_types import MemoryEntry, MemoryTier
from src.utils.logger import get_logger
from src.utils.token_counter import count_tokens, truncate_to_token_limit

logger = get_logger(__name__)


# ------------------------------------------------------------------ #
#  Base class                                                          #
# ------------------------------------------------------------------ #

class BaseCompressor(ABC):
    @abstractmethod
    def compress(self, text: str, max_tokens: int = 300) -> Tuple[str, float]:
        """
        Compress text to target token count.
        Returns: (compressed_text, compression_ratio)
        """
        pass


# ------------------------------------------------------------------ #
#  Strategy A: Summarization Compressor                               #
# ------------------------------------------------------------------ #

class SummarizationCompressor(BaseCompressor):
    """
    Uses an LLM to summarize long text into a compact structured summary.
    Requires OPENAI_API_KEY or ANTHROPIC_API_KEY in your .env
    """

    SUMMARIZE_PROMPT = """You are a memory compression system for an AI assistant.
Compress the following conversation excerpt into a concise, factual summary.

Rules:
- Keep all key decisions, facts, entities, and technical details
- Remove greetings, filler, and repetition
- Use bullet points for clarity
- Target length: {max_tokens} tokens or less
- Start directly with the summary, no preamble

Conversation to compress:
{text}

Compressed summary:"""

    def __init__(self, llm_provider: str = "openai", model: str = "gpt-3.5-turbo"):
        self.llm_provider = llm_provider
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client:
            return self._client

        if self.llm_provider == "openai":
            try:
                from openai import OpenAI
                import os
                self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                raise ImportError("Run: pip install openai")

        elif self.llm_provider == "anthropic":
            try:
                import anthropic, os
                self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            except ImportError:
                raise ImportError("Run: pip install anthropic")

        return self._client

    def compress(self, text: str, max_tokens: int = 300) -> Tuple[str, float]:
        original_tokens = count_tokens(text)

        if original_tokens <= max_tokens:
            return text, 1.0  # Already small enough

        try:
            client = self._get_client()
            prompt = self.SUMMARIZE_PROMPT.format(text=text, max_tokens=max_tokens)

            if self.llm_provider == "openai":
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens + 50,
                    temperature=0.3,
                )
                compressed = response.choices[0].message.content.strip()

            elif self.llm_provider == "anthropic":
                response = client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens + 50,
                    messages=[{"role": "user", "content": prompt}],
                )
                compressed = response.content[0].text.strip()

            ratio = count_tokens(compressed) / original_tokens
            logger.info(
                f"Summarization: {original_tokens} → {count_tokens(compressed)} tokens "
                f"(ratio: {ratio:.2f})"
            )
            return compressed, ratio

        except Exception as e:
            logger.warning(f"LLM compression failed: {e}. Falling back to truncation.")
            truncated = truncate_to_token_limit(text, max_tokens)
            return truncated, count_tokens(truncated) / original_tokens


# ------------------------------------------------------------------ #
#  Strategy B: Embedding Compressor                                    #
# ------------------------------------------------------------------ #

class EmbeddingCompressor:
    """
    Compresses embedding vectors using PCA dimensionality reduction.
    Reduces from e.g. 384-dim → 128-dim, saving ~66% memory.
    """

    def __init__(self, n_components: int = 128):
        self.n_components = n_components
        self._pca = None

    def fit(self, embeddings: List[List[float]]):
        """Fit PCA on a set of embeddings."""
        try:
            import numpy as np
            from sklearn.decomposition import PCA

            arr = np.array(embeddings)
            n_components = min(self.n_components, arr.shape[0], arr.shape[1])
            self._pca = PCA(n_components=n_components)
            self._pca.fit(arr)
            logger.info(
                f"PCA fitted: {arr.shape[1]}→{n_components} dims, "
                f"variance explained: {self._pca.explained_variance_ratio_.sum():.2%}"
            )
        except ImportError:
            logger.warning("scikit-learn not installed. Run: pip install scikit-learn")

    def compress_embedding(self, embedding: List[float]) -> Optional[List[float]]:
        if self._pca is None:
            return embedding
        import numpy as np
        arr = np.array(embedding).reshape(1, -1)
        return self._pca.transform(arr)[0].tolist()

    def compress_batch(self, embeddings: List[List[float]]) -> List[List[float]]:
        if self._pca is None:
            return embeddings
        import numpy as np
        arr = np.array(embeddings)
        return self._pca.transform(arr).tolist()


# ------------------------------------------------------------------ #
#  Main Compression Engine                                             #
# ------------------------------------------------------------------ #

class CompressionEngine:
    """
    Unified interface for all compression strategies.
    Selects the right strategy based on content type and tier.
    """

    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: str = "gpt-3.5-turbo",
        pca_components: int = 128,
        target_ratio: float = 0.20,
    ):
        self.target_ratio = target_ratio
        self.summarizer = SummarizationCompressor(llm_provider, llm_model)
        self.embedding_compressor = EmbeddingCompressor(pca_components)

    def compress_entry(self, entry: MemoryEntry, max_tokens: int = 300) -> MemoryEntry:
        """
        Compress a MemoryEntry in place.
        Returns the same entry with compressed_content filled in.
        """
        if not entry.original_content:
            return entry

        original_tokens = count_tokens(entry.original_content)

        # Don't compress short entries
        if original_tokens <= max_tokens:
            logger.debug(f"Entry already small ({original_tokens} tokens), skipping")
            return entry

        compressed_text, ratio = self.summarizer.compress(
            entry.original_content, max_tokens=max_tokens
        )
        entry.compressed_content = compressed_text

        logger.info(
            f"Compressed entry {entry.entry_id[:8]}: "
            f"{original_tokens} → {count_tokens(compressed_text)} tokens "
            f"(ratio: {ratio:.2f})"
        )
        return entry

    def compress_batch(
        self, entries: List[MemoryEntry], max_tokens: int = 300
    ) -> Tuple[List[MemoryEntry], dict]:
        """Compress a list of entries. Returns entries + compression stats."""
        stats = {
            "total": len(entries),
            "compressed": 0,
            "skipped": 0,
            "total_original_tokens": 0,
            "total_compressed_tokens": 0,
        }

        for entry in entries:
            original_tokens = count_tokens(entry.original_content)
            stats["total_original_tokens"] += original_tokens

            if original_tokens > max_tokens:
                entry = self.compress_entry(entry, max_tokens)
                stats["compressed"] += 1
                stats["total_compressed_tokens"] += count_tokens(entry.active_content)
            else:
                stats["skipped"] += 1
                stats["total_compressed_tokens"] += original_tokens

        if stats["total_original_tokens"] > 0:
            stats["overall_ratio"] = (
                stats["total_compressed_tokens"] / stats["total_original_tokens"]
            )
        else:
            stats["overall_ratio"] = 1.0

        logger.info(
            f"Batch compression: {stats['compressed']}/{stats['total']} entries compressed, "
            f"overall ratio: {stats.get('overall_ratio', 1.0):.2f}"
        )
        return entries, stats
