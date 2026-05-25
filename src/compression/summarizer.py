"""
Summarizer — Phase 2
=====================
Converts long conversation chunks into compact structured summaries.

Three summarization modes:
  - 'bullet'     : structured bullet-point summary (default, most compact)
  - 'narrative'  : flowing prose summary (better for creative/emotional content)
  - 'entity'     : entity-focused extraction (best for technical/factual content)

Works with OpenAI, Anthropic, or a local LLM (via Ollama).
Falls back to extractive summarization (no API key needed) if no LLM is configured.
"""

import re
from typing import List, Tuple, Optional
from src.utils.logger import get_logger
from src.utils.token_counter import count_tokens, truncate_to_token_limit

logger = get_logger(__name__)


# ------------------------------------------------------------------ #
#  Prompt templates                                                    #
# ------------------------------------------------------------------ #

BULLET_PROMPT = """You are a memory compression system. Compress the conversation below into a concise structured summary.

Rules:
- Extract ONLY: decisions made, facts stated, tasks assigned, technical details, named entities (people/places/systems)
- Use short bullet points (one fact per bullet)
- Remove: greetings, filler phrases, repeated information, opinions without substance
- Target: {max_tokens} tokens or fewer
- Start directly with bullets, no preamble

Conversation:
{text}

Compressed summary (bullet points):"""

ENTITY_PROMPT = """You are a knowledge extraction system. Extract structured facts from the conversation below.

Output format — use exactly these sections (skip any that have no content):
PEOPLE: [names and roles mentioned]
SYSTEMS: [technical systems, tools, services mentioned]
DECISIONS: [decisions or agreements made]
TASKS: [action items or tasks assigned]
FACTS: [important factual statements]
DATES: [dates, deadlines, timeframes mentioned]

Target: {max_tokens} tokens or fewer. Skip empty sections.

Conversation:
{text}

Extracted facts:"""

NARRATIVE_PROMPT = """Summarize the following conversation in plain prose. Focus on what was discussed and decided.
Be concise. Target: {max_tokens} tokens or fewer. No preamble.

Conversation:
{text}

Summary:"""

PROMPTS = {
    "bullet":    BULLET_PROMPT,
    "entity":    ENTITY_PROMPT,
    "narrative": NARRATIVE_PROMPT,
}


class Summarizer:
    """
    LLM-backed text summarizer.
    Supports OpenAI, Anthropic, and local Ollama models.
    Falls back to extractive summarization if no LLM configured.
    """

    def __init__(
        self,
        provider: str = "openai",          # "openai" | "anthropic" | "ollama" | "extractive"
        model: str = "gpt-3.5-turbo",
        mode: str = "bullet",              # "bullet" | "entity" | "narrative"
        temperature: float = 0.2,
        ollama_host: str = "http://localhost:11434",
    ):
        self.provider = provider
        self.model = model
        self.mode = mode
        self.temperature = temperature
        self.ollama_host = ollama_host
        self._client = None

        if provider not in ("extractive",):
            self._init_client()

    def _init_client(self):
        """Lazy-initialize the LLM client."""
        import os
        try:
            if self.provider == "openai":
                from openai import OpenAI
                key = os.getenv("OPENAI_API_KEY", "")
                if not key:
                    logger.warning("OPENAI_API_KEY not set. Falling back to extractive summarization.")
                    self.provider = "extractive"
                    return
                self._client = OpenAI(api_key=key)

            elif self.provider == "anthropic":
                import anthropic
                key = os.getenv("ANTHROPIC_API_KEY", "")
                if not key:
                    logger.warning("ANTHROPIC_API_KEY not set. Falling back to extractive summarization.")
                    self.provider = "extractive"
                    return
                self._client = anthropic.Anthropic(api_key=key)

            elif self.provider == "ollama":
                # Ollama uses HTTP directly — no client lib needed
                import urllib.request
                urllib.request.urlopen(f"{self.ollama_host}/api/tags", timeout=2)
                logger.info(f"Ollama server found at {self.ollama_host}")

            logger.info(f"Summarizer initialized: provider={self.provider}, model={self.model}, mode={self.mode}")

        except Exception as e:
            logger.warning(f"Could not initialize {self.provider} client: {e}. Using extractive fallback.")
            self.provider = "extractive"

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def summarize(self, text: str, max_tokens: int = 250) -> Tuple[str, float]:
        """
        Summarize text to within max_tokens.
        Returns: (summary_text, compression_ratio)
        """
        original_tokens = count_tokens(text)

        if original_tokens <= max_tokens:
            return text, 1.0

        if self.provider == "extractive":
            summary = self._extractive_summarize(text, max_tokens)
        elif self.provider == "openai":
            summary = self._openai_summarize(text, max_tokens)
        elif self.provider == "anthropic":
            summary = self._anthropic_summarize(text, max_tokens)
        elif self.provider == "ollama":
            summary = self._ollama_summarize(text, max_tokens)
        else:
            summary = self._extractive_summarize(text, max_tokens)

        compressed_tokens = count_tokens(summary)
        ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0

        logger.info(
            f"Summarized: {original_tokens} → {compressed_tokens} tokens "
            f"({ratio:.1%} of original) | provider={self.provider} mode={self.mode}"
        )
        return summary, ratio

    def summarize_batch(
        self, texts: List[str], max_tokens: int = 250
    ) -> List[Tuple[str, float]]:
        """Summarize a list of texts."""
        results = []
        for i, text in enumerate(texts):
            logger.debug(f"Summarizing chunk {i+1}/{len(texts)}")
            results.append(self.summarize(text, max_tokens))
        return results

    def chunk_and_summarize(
        self, text: str, chunk_tokens: int = 1500, summary_tokens: int = 200
    ) -> str:
        """
        For very long texts: split into chunks, summarize each,
        then optionally summarize the summaries (hierarchical compression).
        """
        words = text.split()
        approx_words_per_chunk = int(chunk_tokens / 1.3)

        chunks = []
        for i in range(0, len(words), approx_words_per_chunk):
            chunk = " ".join(words[i:i + approx_words_per_chunk])
            if chunk.strip():
                chunks.append(chunk)

        if len(chunks) == 1:
            summary, _ = self.summarize(text, summary_tokens)
            return summary

        logger.info(f"Hierarchical compression: {len(chunks)} chunks")
        chunk_summaries = []
        for chunk in chunks:
            summary, _ = self.summarize(chunk, summary_tokens // len(chunks) + 50)
            chunk_summaries.append(summary)

        combined = "\n".join(chunk_summaries)

        # If combined summaries still too long, do a final pass
        if count_tokens(combined) > summary_tokens * 2:
            final, _ = self.summarize(combined, summary_tokens)
            return final

        return combined

    # ------------------------------------------------------------------ #
    #  Provider implementations                                            #
    # ------------------------------------------------------------------ #

    def _openai_summarize(self, text: str, max_tokens: int) -> str:
        prompt = PROMPTS[self.mode].format(text=text, max_tokens=max_tokens)
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens + 80,
                temperature=self.temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"OpenAI summarization failed: {e}. Using extractive fallback.")
            return self._extractive_summarize(text, max_tokens)

    def _anthropic_summarize(self, text: str, max_tokens: int) -> str:
        prompt = PROMPTS[self.mode].format(text=text, max_tokens=max_tokens)
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens + 80,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.warning(f"Anthropic summarization failed: {e}. Using extractive fallback.")
            return self._extractive_summarize(text, max_tokens)

    def _ollama_summarize(self, text: str, max_tokens: int) -> str:
        import json, urllib.request
        prompt = PROMPTS[self.mode].format(text=text, max_tokens=max_tokens)
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": max_tokens + 80},
        }).encode()
        try:
            req = urllib.request.Request(
                f"{self.ollama_host}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read())
                return result.get("response", "").strip()
        except Exception as e:
            logger.warning(f"Ollama summarization failed: {e}. Using extractive fallback.")
            return self._extractive_summarize(text, max_tokens)

    def _extractive_summarize(self, text: str, max_tokens: int) -> str:
        """
        No-API fallback: extract the most information-dense sentences.
        Uses TF-IDF-like heuristic — sentences with rare, long words score higher.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if not sentences:
            return truncate_to_token_limit(text, max_tokens)

        # Score each sentence
        all_words = text.lower().split()
        word_freq: dict = {}
        for w in all_words:
            word_freq[w] = word_freq.get(w, 0) + 1

        def sentence_score(sent: str) -> float:
            words = sent.lower().split()
            if len(words) < 4:
                return 0.0
            # Reward: rare words, longer words, keywords
            keyword_hits = sum(1 for w in words if w in {
                "deploy", "kubernetes", "aws", "api", "error", "bug", "fix",
                "implement", "deadline", "critical", "decided", "architecture",
                "requirement", "database", "system", "model", "train", "compress"
            })
            rarity = sum(1.0 / word_freq.get(w, 1) for w in words) / max(len(words), 1)
            length_bonus = min(len(words) / 20.0, 1.0)
            return rarity + keyword_hits * 0.3 + length_bonus * 0.2

        scored = sorted(
            enumerate(sentences),
            key=lambda x: sentence_score(x[1]),
            reverse=True
        )

        # Pick top sentences by score until we hit token budget
        selected = []
        token_count = 0
        for original_idx, sent in scored:
            t = count_tokens(sent)
            if token_count + t <= max_tokens:
                selected.append((original_idx, sent))
                token_count += t
            if token_count >= max_tokens * 0.9:
                break

        # Re-order by original position
        selected.sort(key=lambda x: x[0])
        return " ".join(s for _, s in selected) if selected else truncate_to_token_limit(text, max_tokens)
