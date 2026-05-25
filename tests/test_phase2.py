"""
Phase 2 Tests — Compression Engine
Run with: pytest tests/test_phase2.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from src.compression.summarizer import Summarizer
from src.compression.embedding_compressor import (
    PCACompressor, ScalarQuantizer, EmbeddingCompressorPipeline
)
from src.utils.token_counter import count_tokens


# ------------------------------------------------------------------ #
#  Summarizer tests                                                    #
# ------------------------------------------------------------------ #

class TestSummarizer:

    def test_extractive_short_text_unchanged(self):
        summarizer = Summarizer(provider="extractive")
        short = "Deploy the Kubernetes cluster on AWS."
        result, ratio = summarizer.summarize(short, max_tokens=200)
        assert ratio == 1.0
        assert result == short

    def test_extractive_long_text_compressed(self):
        summarizer = Summarizer(provider="extractive")
        long_text = " ".join(["Deploy the Kubernetes cluster on AWS using ECS."] * 50)
        result, ratio = summarizer.summarize(long_text, max_tokens=100)
        assert count_tokens(result) <= 120  # small tolerance
        assert ratio < 1.0

    def test_extractive_preserves_keywords(self):
        summarizer = Summarizer(provider="extractive")
        text = (
            "The system deadline is April 30th. "
            "We will deploy on AWS EKS using Kubernetes. "
            "The architecture requires Redis for caching. " * 20
        )
        result, _ = summarizer.summarize(text, max_tokens=80)
        # Key facts should appear in the summary
        assert any(kw in result.lower() for kw in ["april", "aws", "kubernetes", "redis"])

    def test_all_modes_run(self):
        for mode in ("bullet", "entity", "narrative"):
            summarizer = Summarizer(provider="extractive", mode=mode)
            text = "Deploy Kubernetes on AWS. Deadline is March 15th. Team lead is Arjun. " * 10
            result, ratio = summarizer.summarize(text, max_tokens=80)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_chunk_and_summarize(self):
        summarizer = Summarizer(provider="extractive")
        # 2000-token text
        long_text = "Deploy the Kubernetes cluster on AWS using ECS with auto-scaling. " * 200
        result = summarizer.chunk_and_summarize(
            long_text, chunk_tokens=500, summary_tokens=150
        )
        assert count_tokens(result) <= 400
        assert isinstance(result, str)

    def test_batch_summarize(self):
        summarizer = Summarizer(provider="extractive")
        texts = [
            "Deploy Kubernetes on AWS ECS. " * 30,
            "The deadline is March 15th for the project. " * 30,
            "Redis is used for caching in the architecture. " * 30,
        ]
        results = summarizer.summarize_batch(texts, max_tokens=80)
        assert len(results) == 3
        for summary, ratio in results:
            assert ratio < 1.0
            assert len(summary) > 0


# ------------------------------------------------------------------ #
#  PCA Compressor tests                                                #
# ------------------------------------------------------------------ #

class TestPCACompressor:

    def _make_vectors(self, n=200, dim=384):
        rng = np.random.default_rng(42)
        return rng.standard_normal((n, dim)).astype(np.float32)

    def test_fit_reduces_dimensions(self):
        vecs = self._make_vectors()
        pca = PCACompressor(n_components=128)
        pca.fit(vecs)
        assert pca.output_dim == 128
        assert pca.input_dim == 384

    def test_transform_shape(self):
        vecs = self._make_vectors()
        pca = PCACompressor(n_components=128)
        compressed = pca.fit_transform(vecs)
        assert compressed.shape == (200, 128)

    def test_variance_explained_reasonable(self):
        vecs = self._make_vectors()
        pca = PCACompressor(n_components=128)
        pca.fit(vecs)
        # For random data, 128/384 dims should explain ~33% variance
        assert pca.explained_variance_ratio > 0.20

    def test_single_vector_transform(self):
        vecs = self._make_vectors()
        pca = PCACompressor(n_components=64)
        pca.fit(vecs)
        single = vecs[0]
        result = pca.transform_single(single)
        assert result.shape == (64,)

    def test_not_fitted_raises(self):
        pca = PCACompressor()
        vecs = self._make_vectors(10)
        with pytest.raises(RuntimeError, match="fitted"):
            pca.transform(vecs)


# ------------------------------------------------------------------ #
#  Scalar Quantizer tests                                             #
# ------------------------------------------------------------------ #

class TestScalarQuantizer:

    def _make_vecs(self, n=100, dim=128):
        rng = np.random.default_rng(0)
        return rng.standard_normal((n, dim)).astype(np.float32)

    def test_quantize_dtype(self):
        vecs = self._make_vecs()
        sq = ScalarQuantizer()
        quantized = sq.fit_quantize(vecs)
        assert quantized.dtype == np.uint8

    def test_quantize_shape_preserved(self):
        vecs = self._make_vecs(50, 64)
        sq = ScalarQuantizer()
        q = sq.fit_quantize(vecs)
        assert q.shape == (50, 64)

    def test_dequantize_close_to_original(self):
        vecs = self._make_vecs()
        sq = ScalarQuantizer()
        q = sq.fit_quantize(vecs)
        restored = sq.dequantize(q)
        # Should be close but not exact (lossy)
        mse = float(np.mean((vecs - restored) ** 2))
        assert mse < 0.01  # very small reconstruction error

    def test_memory_reduction(self):
        vecs = self._make_vecs(1000, 128)
        sq = ScalarQuantizer()
        q = sq.fit_quantize(vecs)
        assert q.nbytes == vecs.nbytes // 4  # uint8 = 1 byte vs float32 = 4 bytes


# ------------------------------------------------------------------ #
#  Token counter tests                                                 #
# ------------------------------------------------------------------ #

class TestTokenCounter:

    def test_approx_scaling(self):
        short = "hello world"
        long_text = short * 100
        assert count_tokens(long_text) > count_tokens(short) * 50

    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_reasonable_estimate(self):
        # ~100 words should be ~130 tokens
        text = "word " * 100
        tokens = count_tokens(text)
        assert 100 <= tokens <= 160
