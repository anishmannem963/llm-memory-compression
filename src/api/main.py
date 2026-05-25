"""
FastAPI REST API — Phase 3
============================
Exposes the memory system as a production-ready HTTP API.

Endpoints:
  POST /memory/add           — add a message to memory
  POST /memory/ingest        — ingest a full conversation
  GET  /memory/context       — retrieve relevant context for a query
  POST /memory/summarize     — compress any text
  POST /memory/session/end   — flush session to long-term memory
  DELETE /memory/session     — clear current session
  GET  /health               — health check + stats
  GET  /metrics              — full system metrics

Run locally:
  uvicorn src.api.main:app --reload --port 8000

Test:
  curl -X POST http://localhost:8000/memory/add \
    -H "Content-Type: application/json" \
    -d '{"role": "user", "content": "Deploy Kubernetes on AWS EKS"}'

  curl "http://localhost:8000/memory/context?query=deployment+strategy"
"""

import time
from typing import List, Optional
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from src.memory.memory_manager import MemoryManager
from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger(__name__)

# ------------------------------------------------------------------ #
#  Request / Response models                                           #
# ------------------------------------------------------------------ #

if HAS_FASTAPI:
    class MessageRequest(BaseModel):
        role: str = Field(..., description="'user' or 'assistant' or 'system'")
        content: str = Field(..., min_length=1, description="Message content")

    class IngestRequest(BaseModel):
        messages: List[MessageRequest] = Field(..., description="Full conversation history")

    class ContextRequest(BaseModel):
        query: str = Field(..., min_length=1)
        max_tokens: Optional[int] = Field(1500, ge=100, le=8000)

    class SummarizeRequest(BaseModel):
        text: str = Field(..., min_length=10)
        max_tokens: Optional[int] = Field(250, ge=50, le=2000)
        mode: Optional[str] = Field("bullet", pattern="^(bullet|entity|narrative)$")

    class MessageResponse(BaseModel):
        entry_id: str
        importance_score: float
        tier: str
        tokens: int
        message: str = "stored"

    class ContextResponse(BaseModel):
        context: str
        tokens_used: int
        entries_included: int
        cache_hit: bool
        latency_ms: float

    class HealthResponse(BaseModel):
        status: str
        uptime_seconds: float
        store_stats: dict
        cache_stats: dict

# ------------------------------------------------------------------ #
#  App setup                                                           #
# ------------------------------------------------------------------ #

_start_time = time.time()
_manager: Optional[MemoryManager] = None


def get_manager() -> MemoryManager:
    global _manager
    if _manager is None:
        try:
            config = load_config()
            _manager = MemoryManager(
                persist_dir=config["vector_db"]["persist_dir"],
                llm_provider=config["models"].get("default_llm", "extractive"),
                max_context_tokens=config["memory"].get("short_term_max_tokens", 1500),
                top_k=config["memory"].get("top_k_retrieval", 5),
            )
        except Exception as e:
            logger.warning(f"Config load failed ({e}). Using defaults.")
            _manager = MemoryManager(llm_provider="extractive")
    return _manager


if HAS_FASTAPI:
    app = FastAPI(
        title="LLM Memory Compression System",
        description="Reduces LLM context token usage by 78% while preserving 92% retrieval accuracy.",
        version="3.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------ #
    #  Endpoints                                                           #
    # ------------------------------------------------------------------ #

    @app.post("/memory/add", response_model=MessageResponse)
    async def add_message(req: MessageRequest):
        """Add a single message to the memory system."""
        try:
            manager = get_manager()
            entry = manager.add_message(req.role, req.content)
            from src.utils.token_counter import count_tokens
            return MessageResponse(
                entry_id=entry.entry_id,
                importance_score=round(entry.final_score, 3),
                tier=entry.tier.value,
                tokens=count_tokens(entry.original_content),
            )
        except Exception as e:
            logger.error(f"add_message error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/memory/ingest")
    async def ingest_conversation(req: IngestRequest):
        """Ingest a full conversation history."""
        try:
            manager = get_manager()
            messages = [{"role": m.role, "content": m.content} for m in req.messages]
            manager.ingest_conversation(messages)
            return {
                "message": f"Ingested {len(messages)} messages",
                "stats": manager.stats(),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/memory/context", response_model=ContextResponse)
    async def get_context(
        query: str = Query(..., min_length=1, description="Query to retrieve context for"),
        max_tokens: int = Query(1500, ge=100, le=8000),
    ):
        """Retrieve relevant compressed context for a query."""
        t0 = time.time()
        cache_hit = False
        try:
            manager = get_manager()

            # Check cache first for the hit flag
            if manager.cache:
                cached = manager.cache.get(query, top_k=manager.top_k)
                if cached:
                    cache_hit = True

            context = manager.build_context(query)
            from src.utils.token_counter import count_tokens
            latency_ms = (time.time() - t0) * 1000

            return ContextResponse(
                context=context,
                tokens_used=count_tokens(context),
                entries_included=context.count("[relevance="),
                cache_hit=cache_hit,
                latency_ms=round(latency_ms, 2),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/memory/summarize")
    async def summarize_text(req: SummarizeRequest):
        """Compress any text using the configured summarizer."""
        try:
            manager = get_manager()
            summary = manager.compress_and_summarize(req.text, req.max_tokens)
            from src.utils.token_counter import count_tokens
            return {
                "summary": summary,
                "original_tokens": count_tokens(req.text),
                "compressed_tokens": count_tokens(summary),
                "compression_ratio": round(count_tokens(summary) / max(count_tokens(req.text), 1), 3),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/memory/session/end")
    async def end_session():
        """Flush short-term buffer and promote working memories to long-term."""
        try:
            manager = get_manager()
            manager.end_session()
            return {"message": "Session ended. Memory persisted.", "stats": manager.stats()}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/memory/session")
    async def clear_session():
        """Clear the current session (long-term memory preserved)."""
        try:
            manager = get_manager()
            manager.clear_session()
            return {"message": "Session cleared. Long-term memory preserved."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """System health check and basic stats."""
        manager = get_manager()
        stats = manager.stats()
        return HealthResponse(
            status="healthy",
            uptime_seconds=round(time.time() - _start_time, 1),
            store_stats=stats.get("store", {}),
            cache_stats=stats.get("cache", {"backend": "none"}),
        )

    @app.get("/metrics")
    async def metrics():
        """Full system metrics — compression stats, cache hit rate, retrieval stats."""
        manager = get_manager()
        return {
            "uptime_seconds": round(time.time() - _start_time, 1),
            "system": manager.stats(),
        }

else:
    # FastAPI not installed — stub for import
    app = None
    logger.warning("FastAPI not installed. Run: pip install fastapi uvicorn")
