import yaml
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def load_config(config_path: str = None) -> dict:
    """Load YAML config and merge with environment variables."""
    if config_path is None:
        # Auto-find config relative to project root
        root = Path(__file__).parent.parent.parent
        config_path = root / "configs" / "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Inject env vars on top of YAML defaults
    config["api_keys"] = {
        "openai": os.getenv("OPENAI_API_KEY", ""),
        "anthropic": os.getenv("ANTHROPIC_API_KEY", ""),
    }
    config["models"]["default_llm"] = os.getenv(
        "DEFAULT_LLM", config["models"].get("default_llm", "openai")
    )
    config["vector_db"]["persist_dir"] = os.getenv(
        "CHROMA_PERSIST_DIR", config["vector_db"]["persist_dir"]
    )

    return config
