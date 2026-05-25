"""
Phase 1 Demo — Run this to see the full system working end-to-end.

What this does:
1. Simulates a 20-message technical conversation
2. Scores each message for importance
3. Stores important memories in ChromaDB
4. Retrieves relevant context for a query
5. Prints compression + retrieval stats

Run:
    python scripts/demo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from src.memory import MemoryManager
from src.memory.importance_scorer import ImportanceScorer
from src.memory.memory_types import Message, MessageRole
from src.evaluation import Evaluator

console = Console()

# ------------------------------------------------------------------ #
#  Sample conversation — simulates a real dev/ML discussion           #
# ------------------------------------------------------------------ #

SAMPLE_CONVERSATION = [
    {"role": "user",      "content": "Hey, how are you?"},
    {"role": "assistant", "content": "I'm doing well! How can I help you today?"},
    {"role": "user",      "content": "I need help designing the architecture for my ML system."},
    {"role": "assistant", "content": "Sure! What kind of ML system are you building?"},
    {"role": "user",      "content": "I'm building an LLM memory compression system for my masters thesis at UF. The goal is to reduce context token usage by 70-80% while preserving retrieval accuracy above 90%."},
    {"role": "assistant", "content": "That's a great project. For the architecture, I'd recommend a three-tier memory system: short-term for recent turns, working memory for compressed summaries, and long-term vector storage using FAISS or ChromaDB."},
    {"role": "user",      "content": "ok"},
    {"role": "user",      "content": "We decided to deploy the system on AWS using ECS with auto-scaling. The database will be PostgreSQL with pgvector extension for hybrid search."},
    {"role": "assistant", "content": "Good choice. ECS with Fargate is very cost-effective. For pgvector, make sure to create an IVFFLAT index for large collections. You'll want at least a c5.xlarge for the embedding service to handle concurrent requests."},
    {"role": "user",      "content": "The deadline for the first demo is March 15th. We need the importance scorer and ChromaDB integration done by then."},
    {"role": "assistant", "content": "That's achievable. I'd prioritize the importance scorer first since everything depends on it. Use sentence-transformers/all-MiniLM-L6-v2 as your embedding model — it's fast and good enough for importance scoring without a GPU."},
    {"role": "user",      "content": "thanks"},
    {"role": "user",      "content": "My name is Arjun and I'm the lead developer on this project. My advisor is Dr. Sarah Chen at UF."},
    {"role": "assistant", "content": "Nice to meet you, Arjun! Having a clear project lead is important. Make sure to document the system architecture clearly for Dr. Chen's review."},
    {"role": "user",      "content": "The critical requirement is that the system must handle conversations of up to 50,000 tokens and compress them down to under 2,000 tokens for the LLM context window."},
    {"role": "assistant", "content": "That's a 40:1 compression ratio target. You'll need hierarchical compression: first pass summarizes large chunks, second pass filters by importance score, third pass packs the most relevant embeddings into the context window using MMR (Maximal Marginal Relevance) to avoid redundancy."},
    {"role": "user",      "content": "We also need to integrate with Kubernetes for the distributed deployment. We're using EKS on AWS."},
    {"role": "assistant", "content": "For EKS, I recommend separate node groups for the embedding service (GPU nodes) and the memory management service (CPU nodes). Use Redis as a shared cache layer between pods."},
    {"role": "user",      "content": "cool"},
    {"role": "user",      "content": "The evaluation metrics we need to track: compression ratio, retrieval precision@5, recall@5, and response latency under 200ms."},
    {"role": "assistant", "content": "Perfect set of metrics. Also add ROUGE-L to measure information preservation, and track hallucination rate by comparing responses with and without compressed context."},
]


def run_demo():
    console.rule("[bold blue]LLM Memory Compression System — Phase 1 Demo[/bold blue]")

    # ------------------------------------------------------------------ #
    #  Step 1: Score messages for importance                              #
    # ------------------------------------------------------------------ #
    console.print("\n[bold]Step 1: Importance Scoring[/bold]")
    console.print("Scoring each message (no GPU needed for this step)...\n")

    scorer = ImportanceScorer(use_embeddings=False)  # keyword-only, no model download

    messages = [
        Message(role=MessageRole(m["role"]), content=m["content"])
        for m in SAMPLE_CONVERSATION
    ]

    table = Table(title="Message Importance Scores", show_lines=True)
    table.add_column("Role", style="cyan", width=10)
    table.add_column("Content Preview", width=55)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Keep?", justify="center", width=7)

    entries = scorer.score_messages(messages)
    for msg, entry in zip(messages, entries):
        preview = msg.content[:55] + "..." if len(msg.content) > 55 else msg.content
        score = entry.final_score
        keep = "✓" if score >= scorer.importance_threshold else "✗"
        color = "green" if score >= scorer.importance_threshold else "dim"
        table.add_row(
            msg.role.value,
            preview,
            f"[{color}]{score:.2f}[/{color}]",
            f"[{color}]{keep}[/{color}]",
        )

    console.print(table)

    important = scorer.filter_important(entries)
    console.print(
        f"\n[green]Kept {len(important)}/{len(entries)} messages "
        f"(threshold: {scorer.importance_threshold})[/green]"
    )

    # ------------------------------------------------------------------ #
    #  Step 2: Initialize MemoryManager and ingest                        #
    # ------------------------------------------------------------------ #
    console.print("\n[bold]Step 2: Ingesting into MemoryManager[/bold]")

    try:
        manager = MemoryManager(
            persist_dir="./data/embeddings/chroma_demo",
        )
        manager.ingest_conversation(SAMPLE_CONVERSATION)

        stats = manager.stats()
        console.print(Panel(
            f"[green]Short-term buffer:[/green] {stats['short_term_buffer_messages']} messages\n"
            f"[green]Short-term store:[/green]  {stats['store'].get('short_term', 0)} entries\n"
            f"[green]Working memory:[/green]    {stats['store'].get('working', 0)} entries\n"
            f"[green]Long-term memory:[/green]  {stats['store'].get('long_term', 0)} entries\n"
            f"[green]Total stored:[/green]      {stats['total_stored']} entries",
            title="Memory Store Stats"
        ))

    except Exception as e:
        console.print(f"[yellow]MemoryManager requires ChromaDB.[/yellow]")
        console.print(f"[dim]Install with: pip install chromadb sentence-transformers[/dim]")
        console.print(f"[dim]Error: {e}[/dim]")
        console.print("\n[bold]Continuing demo with scoring only...[/bold]\n")

    # ------------------------------------------------------------------ #
    #  Step 3: Show compression stats                                      #
    # ------------------------------------------------------------------ #
    console.print("\n[bold]Step 3: Compression Statistics[/bold]")

    evaluator = Evaluator()
    metrics = evaluator.measure_compression(entries)

    console.print(Panel(
        f"[cyan]Original tokens:[/cyan]    {metrics.original_tokens}\n"
        f"[cyan]After filtering:[/cyan]    "
        f"{sum(1 for e in entries if e.final_score >= scorer.importance_threshold)} entries kept\n"
        f"[cyan]Token reduction:[/cyan]    {metrics.token_reduction_percent:.1f}% (keyword filter only)\n"
        f"\n[dim]Phase 2 will add LLM summarization for 70-80% reduction[/dim]",
        title="Compression Metrics (Phase 1)"
    ))

    # ------------------------------------------------------------------ #
    #  Step 4: Demo retrieval query                                        #
    # ------------------------------------------------------------------ #
    console.print("\n[bold]Step 4: Sample Retrieval Query[/bold]")

    test_queries = [
        "What deployment strategy did we decide on?",
        "What are the evaluation metrics we need to track?",
        "Who is the lead developer and what is the deadline?",
    ]

    for query in test_queries:
        console.print(f"\n[yellow]Query:[/yellow] {query}")
        # Keyword-based retrieval as Phase 1 fallback
        query_words = set(query.lower().split())
        scored = []
        for entry in entries:
            entry_words = set(entry.original_content.lower().split())
            overlap = len(query_words & entry_words) / max(len(query_words), 1)
            if overlap > 0.1:
                scored.append((entry, overlap))
        scored.sort(key=lambda x: x[1], reverse=True)

        if scored:
            top = scored[0][0]
            preview = top.original_content[:120] + "..." if len(top.original_content) > 120 else top.original_content
            console.print(f"[green]Best match (score={scored[0][1]:.2f}):[/green] {preview}")
        else:
            console.print("[dim]No match found[/dim]")

    # ------------------------------------------------------------------ #
    #  Summary                                                             #
    # ------------------------------------------------------------------ #
    console.rule("[bold green]Phase 1 Complete![/bold green]")
    console.print("""
[bold]What's working:[/bold]
  ✓ Message importance scoring (keyword-based + recency decay)
  ✓ Memory entry creation with metadata
  ✓ Three-tier memory architecture (data models)
  ✓ Compression metrics pipeline
  ✓ Basic keyword retrieval

[bold]Next steps (Phase 2):[/bold]
  → Install full dependencies: pip install -r requirements.txt
  → Add your API key to .env
  → Run: python scripts/run_with_embeddings.py
  → LLM-based summarization compression (70-80% token reduction)
  → ChromaDB semantic search
    """)


if __name__ == "__main__":
    run_demo()
