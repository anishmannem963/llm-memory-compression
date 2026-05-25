import re

def count_tokens(text: str, method: str = "approx") -> int:
    """
    Count tokens in a string.
    
    method='approx'  — fast heuristic (words * 1.3), no dependencies
    method='tiktoken' — accurate GPT tokenizer (requires: pip install tiktoken)
    """
    if method == "tiktoken":
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            pass  # fall through to approx

    # Approx: ~1.3 tokens per word (standard heuristic for English)
    words = len(re.findall(r"\S+", text))
    return int(words * 1.3)


def truncate_to_token_limit(text: str, max_tokens: int) -> str:
    """Truncate text to stay within a token budget."""
    words = text.split()
    # Estimate: max_tokens / 1.3 words
    max_words = int(max_tokens / 1.3)
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " [TRUNCATED]"
