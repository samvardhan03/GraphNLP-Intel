"""Unit tests for domain adapters: finance, email, feedback, incidents."""

from __future__ import annotations

import networkx as nx
import pytest


# ── FinanceAdapter tests ────────────────────────────────────────────────────


class TestFinanceAdapter:
    def setup_method(self):
        from graphnlp.adapters.finance import FinanceAdapter
        self.adapter = FinanceAdapter()

    def test_domain(self):
        assert self.adapter.domain == "finance"

    def test_entity_types(self):
        assert "ORG" in self.adapter.entity_types
        assert "AMOUNT" in self.adapter.entity_types

    def test_preprocess_currency(self):
        text = "Revenue was $1,234.56 and costs were $789.00."
        result = self.adapter.preprocess(text)
        assert "1234.56 USD" in result
        assert "789.00 USD" in result
        assert "$" not in result

    def test_preprocess_abbreviations(self):
        text = "Q1 earnings showed strong EPS growth with positive ROI."
        result = self.adapter.preprocess(text)
        assert "first quarter" in result
        assert "earnings per share" in result
        assert "return on investment" in result

    def test_postprocess_competitor_edges(self):
        G = nx.DiGraph()
        G.add_node("Apple", type="ORG", mention_count=3)
        G.add_node("Google", type="ORG", mention_count=2)
        G.add_node("AI Market", type="MISC", mention_count=1)

        G.add_edge("Apple", "AI Market", predicate="competes_in", confidence=0.9, weight=0.7)
        G.add_edge("Google", "AI Market", predicate="competes_in", confidence=0.9, weight=0.7)

        result = self.adapter.postprocess(G)
        # Should add COMPETITOR_OF edge
        has_competitor = (
            result.has_edge("Apple", "Google") or result.has_edge("Google", "Apple")
        )
        assert has_competitor

    def test_entity_schema(self):
        schema = self.adapter.entity_schema()
        assert schema["domain"] == "finance"
        assert len(schema["entity_types"]) == len(self.adapter.entity_types)


# ── EmailAdapter tests ──────────────────────────────────────────────────────


class TestEmailAdapter:
    def setup_method(self):
        from graphnlp.adapters.email import EmailAdapter
        self.adapter = EmailAdapter()

    def test_domain(self):
        assert self.adapter.domain == "email"

    def test_preprocess_strips_html(self):
        text = "<html><body><p>Hello <b>World</b></p></body></html>"
        result = self.adapter.preprocess(text)
        assert "<html>" not in result
        assert "<p>" not in result
        assert "Hello" in result
        assert "World" in result

    def test_preprocess_strips_headers(self):
        text = "From: user@example.com\nTo: other@example.com\nActual content here."
        result = self.adapter.preprocess(text)
        assert "From:" not in result
        assert "Actual content here" in result

    def test_preprocess_strips_quoted_replies(self):
        text = "My reply.\n> Previous message.\n>> Older message."
        result = self.adapter.preprocess(text)
        assert "My reply" in result
        assert "> Previous" not in result

    def test_monthly_spend_summary(self):
        from graphnlp.adapters.email import EmailAdapter

        G = nx.DiGraph()
        G.add_node("John", type="SENDER", mention_count=1)
        G.add_node("Amazon", type="MERCHANT", mention_count=3)
        G.add_node("$234.56", type="MONEY", label="$234.56", mention_count=1)

        G.add_edge("$234.56", "Amazon", predicate="paid_to", confidence=0.9, weight=0.8)

        result = EmailAdapter.monthly_spend_summary(G)
        assert isinstance(result, dict)


# ── FeedbackAdapter tests ───────────────────────────────────────────────────


class TestFeedbackAdapter:
    def test_preprocess_ratings(self):
        from graphnlp.adapters.feedback import FeedbackAdapter

        adapter = FeedbackAdapter()
        text = "I give this product 4/5 stars!"
        result = adapter.preprocess(text)
        assert "[RATING:4]" in result

    def test_preprocess_star_emojis(self):
        from graphnlp.adapters.feedback import FeedbackAdapter

        adapter = FeedbackAdapter()
        text = "★★★★★ Amazing product!"
        result = adapter.preprocess(text)
        assert "[RATING:5]" in result


# ── IncidentAdapter tests ───────────────────────────────────────────────────


class TestIncidentAdapter:
    def test_preprocess_severity(self):
        from graphnlp.adapters.incidents import IncidentAdapter

        adapter = IncidentAdapter()
        text = "P0 incident affecting production database"
        result = adapter.preprocess(text)
        assert "[SEV:CRITICAL]" in result

    def test_preprocess_severity_medium(self):
        from graphnlp.adapters.incidents import IncidentAdapter

        adapter = IncidentAdapter()
        text = "SEV2 alert: disk space running low"
        result = adapter.preprocess(text)
        assert "[SEV:MEDIUM]" in result

    def test_preprocess_dedup_lines(self):
        from graphnlp.adapters.incidents import IncidentAdapter

        adapter = IncidentAdapter()
        text = "Error line\nError line\nError line\nDifferent line"
        result = adapter.preprocess(text)
        lines = [l for l in result.split("\n") if l.strip()]
        assert len(lines) == 2


# ── Adapter registry tests ──────────────────────────────────────────────────


class TestAdapterRegistry:
    def test_get_generic(self):
        from graphnlp.adapters.base import get_adapter

        adapter = get_adapter("generic")
        assert adapter.domain == "generic"

    def test_get_finance(self):
        from graphnlp.adapters.base import get_adapter
        # Import to trigger registration
        import graphnlp.adapters.finance  # noqa: F401

        adapter = get_adapter("finance")
        assert adapter.domain == "finance"

    def test_get_email(self):
        from graphnlp.adapters.base import get_adapter
        import graphnlp.adapters.email  # noqa: F401

        adapter = get_adapter("email")
        assert adapter.domain == "email"

    def test_get_unknown_falls_back(self):
        from graphnlp.adapters.base import get_adapter

        adapter = get_adapter("nonexistent_domain")
        assert adapter.domain == "generic"
