"""
Evaluator — Benchmarking & Metrics
====================================
Measures the system across three categories:
  1. Compression metrics   — how much did we shrink things?
  2. Quality metrics       — did we retain useful information?
  3. Performance metrics   — how fast is retrieval?

Run this after each phase to generate benchmark numbers for your resume.
"""

import time
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path

from src.memory.memory_types import MemoryEntry
from src.utils.logger import get_logger
from src.utils.token_counter import count_tokens

logger = get_logger(__name__)


@dataclass
class CompressionMetrics:
    original_tokens: int = 0
    compressed_tokens: int = 0
    compression_ratio: float = 1.0
    token_reduction_percent: float = 0.0
    entries_compressed: int = 0
    entries_total: int = 0


@dataclass
class RetrievalMetrics:
    precision_at_k: float = 0.0   # Fraction of retrieved that are relevant
    recall_at_k: float = 0.0      # Fraction of relevant that were retrieved
    mrr: float = 0.0              # Mean Reciprocal Rank
    latency_ms: float = 0.0       # Average retrieval latency
    top_k: int = 5


@dataclass
class QualityMetrics:
    rouge_l: float = 0.0
    rouge_1: float = 0.0
    consistency_score: float = 0.0   # Same answer before/after compression?


@dataclass
class BenchmarkReport:
    timestamp: str = ""
    compression: CompressionMetrics = field(default_factory=CompressionMetrics)
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    notes: str = ""

    def summary(self) -> str:
        return (
            f"\n{'='*55}\n"
            f"  BENCHMARK REPORT — {self.timestamp}\n"
            f"{'='*55}\n"
            f"  COMPRESSION\n"
            f"    Token reduction:    {self.compression.token_reduction_percent:.1f}%\n"
            f"    Compression ratio:  {self.compression.compression_ratio:.3f}\n"
            f"    Entries compressed: {self.compression.entries_compressed}/{self.compression.entries_total}\n"
            f"\n  RETRIEVAL\n"
            f"    Precision@{self.retrieval.top_k}:      {self.retrieval.precision_at_k:.3f}\n"
            f"    Recall@{self.retrieval.top_k}:         {self.retrieval.recall_at_k:.3f}\n"
            f"    MRR:                {self.retrieval.mrr:.3f}\n"
            f"    Latency:            {self.retrieval.latency_ms:.1f}ms\n"
            f"\n  QUALITY\n"
            f"    ROUGE-L:            {self.quality.rouge_l:.3f}\n"
            f"    ROUGE-1:            {self.quality.rouge_1:.3f}\n"
            f"{'='*55}\n"
        )

    def to_dict(self) -> dict:
        return asdict(self)


class Evaluator:
    """
    Runs benchmarks and saves results to disk.
    """

    def __init__(self, output_dir: str = "./data/processed/eval_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Compression metrics                                                 #
    # ------------------------------------------------------------------ #

    def measure_compression(self, entries: List[MemoryEntry]) -> CompressionMetrics:
        """Measure compression statistics over a set of entries."""
        total_orig = sum(count_tokens(e.original_content) for e in entries)
        total_comp = sum(count_tokens(e.active_content) for e in entries)
        compressed_count = sum(1 for e in entries if e.compressed_content is not None)

        ratio = total_comp / total_orig if total_orig > 0 else 1.0
        reduction = (1 - ratio) * 100

        return CompressionMetrics(
            original_tokens=total_orig,
            compressed_tokens=total_comp,
            compression_ratio=ratio,
            token_reduction_percent=reduction,
            entries_compressed=compressed_count,
            entries_total=len(entries),
        )

    # ------------------------------------------------------------------ #
    #  Retrieval metrics                                                   #
    # ------------------------------------------------------------------ #

    def measure_retrieval(
        self,
        memory_manager,
        test_queries: List[Dict],
    ) -> RetrievalMetrics:
        """
        Measure retrieval quality.

        test_queries format:
        [
            {
                "query": "What deployment strategy did we use?",
                "relevant_ids": ["entry_id_1", "entry_id_2"],   # ground truth
            },
            ...
        ]
        """
        from src.memory.memory_store import MemoryStore

        precisions, recalls, reciprocal_ranks, latencies = [], [], [], []
        top_k = memory_manager.top_k

        for test in test_queries:
            query = test["query"]
            relevant_ids = set(test.get("relevant_ids", []))

            t0 = time.time()
            results = memory_manager.store.query_all_tiers(query, top_k=top_k)
            latency_ms = (time.time() - t0) * 1000

            retrieved_ids = [r.entry.entry_id for r in results]

            # Precision@k
            if retrieved_ids:
                hits = sum(1 for rid in retrieved_ids if rid in relevant_ids)
                precisions.append(hits / len(retrieved_ids))
            else:
                precisions.append(0.0)

            # Recall@k
            if relevant_ids:
                hits = sum(1 for rid in retrieved_ids if rid in relevant_ids)
                recalls.append(hits / len(relevant_ids))
            else:
                recalls.append(1.0)

            # MRR
            rr = 0.0
            for rank, rid in enumerate(retrieved_ids, 1):
                if rid in relevant_ids:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)
            latencies.append(latency_ms)

        return RetrievalMetrics(
            precision_at_k=sum(precisions) / len(precisions) if precisions else 0,
            recall_at_k=sum(recalls) / len(recalls) if recalls else 0,
            mrr=sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0,
            latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            top_k=top_k,
        )

    # ------------------------------------------------------------------ #
    #  Quality metrics (ROUGE)                                             #
    # ------------------------------------------------------------------ #

    def measure_quality(
        self,
        original_texts: List[str],
        compressed_texts: List[str],
    ) -> QualityMetrics:
        """Measure how much information is preserved after compression."""
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

            r1_scores, rl_scores = [], []
            for orig, comp in zip(original_texts, compressed_texts):
                scores = scorer.score(orig, comp)
                r1_scores.append(scores["rouge1"].fmeasure)
                rl_scores.append(scores["rougeL"].fmeasure)

            return QualityMetrics(
                rouge_1=sum(r1_scores) / len(r1_scores),
                rouge_l=sum(rl_scores) / len(rl_scores),
            )

        except ImportError:
            logger.warning("rouge-score not installed. Run: pip install rouge-score")
            return QualityMetrics()

    # ------------------------------------------------------------------ #
    #  Full benchmark run                                                  #
    # ------------------------------------------------------------------ #

    def run_benchmark(
        self,
        entries: List[MemoryEntry],
        memory_manager=None,
        test_queries: Optional[List[Dict]] = None,
        notes: str = "",
    ) -> BenchmarkReport:
        """Run all metrics and return a full BenchmarkReport."""
        from datetime import datetime

        report = BenchmarkReport(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            notes=notes,
        )

        # Compression
        report.compression = self.measure_compression(entries)

        # Quality (for entries that have both original and compressed)
        compressed_entries = [e for e in entries if e.compressed_content]
        if compressed_entries:
            report.quality = self.measure_quality(
                [e.original_content for e in compressed_entries],
                [e.compressed_content for e in compressed_entries],
            )

        # Retrieval
        if memory_manager and test_queries:
            report.retrieval = self.measure_retrieval(memory_manager, test_queries)

        # Print and save
        print(report.summary())
        self._save_report(report)
        return report

    def _save_report(self, report: BenchmarkReport):
        fname = f"benchmark_{report.timestamp.replace(' ', '_').replace(':', '-')}.json"
        path = self.output_dir / fname
        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Benchmark saved to {path}")
