"""
Tests for ImportanceScorer — run with: pytest tests/
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.memory.importance_scorer import ImportanceScorer
from src.memory.memory_types import Message, MessageRole


@pytest.fixture
def scorer():
    return ImportanceScorer(use_embeddings=False)


def make_msg(content, role="user"):
    return Message(role=MessageRole(role), content=content)


class TestImportanceScorer:

    def test_greeting_scores_low(self, scorer):
        msg = make_msg("Hi")
        score = scorer.score_message(msg)
        assert score < 0.2, f"Greeting should score low, got {score}"

    def test_technical_content_scores_high(self, scorer):
        msg = make_msg("Deploy the Kubernetes cluster on AWS using ECS with auto-scaling.")
        score = scorer.score_message(msg)
        assert score > 0.4, f"Technical content should score high, got {score}"

    def test_system_message_always_retained(self, scorer):
        msg = make_msg("You are a helpful assistant.", role="system")
        score = scorer.score_message(msg)
        assert score == 1.0

    def test_filler_scores_low(self, scorer):
        for filler in ["ok", "thanks", "sure", "lol"]:
            msg = make_msg(filler)
            score = scorer.score_message(msg)
            assert score < 0.2, f"'{filler}' should score low, got {score}"

    def test_deadline_info_scores_high(self, scorer):
        msg = make_msg("The deadline for the project is March 15th. This is critical.")
        score = scorer.score_message(msg)
        assert score > 0.35

    def test_filter_important(self, scorer):
        messages = [
            make_msg("Hi"),
            make_msg("Deploy the system to AWS ECS with Kubernetes orchestration."),
            make_msg("ok"),
            make_msg("The architecture requires PostgreSQL with pgvector for semantic search."),
        ]
        entries = scorer.score_messages(messages)
        important = scorer.filter_important(entries)
        assert len(important) == 2

    def test_recency_decay(self, scorer):
        msg = make_msg("Deploy the system to production AWS cluster.")
        recent_score = scorer.score_message(msg, conversation_age_seconds=0)
        old_score = scorer.score_message(msg, conversation_age_seconds=3600)
        assert recent_score >= old_score, "Recent messages should score >= older ones"

    def test_scores_in_range(self, scorer):
        messages = [make_msg(f"Message number {i} with some content here.") for i in range(10)]
        entries = scorer.score_messages(messages)
        for entry in entries:
            assert 0.0 <= entry.final_score <= 1.0
