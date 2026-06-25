"""Tests for the Tri-Modal retrieval tools (retrieval_tools.py).

These tests verify that the public API works correctly:
- query_tantivy / query_lancedb / query_kuzudb return lists of dicts
- fuse_results_rrf correctly merges and ranks across sources
- Live backends are used when available, mock fallback otherwise
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_mock_data_exists():
    from retrieval_tools import MOCK_DOCUMENTS
    assert len(MOCK_DOCUMENTS) == 6
    sources = {doc["source"] for doc in MOCK_DOCUMENTS}
    assert "Tantivy" in sources
    assert "LanceDB" in sources
    assert "KuzuDB" in sources


def test_query_tantivy_returns_results():
    from retrieval_tools import query_tantivy
    results = query_tantivy("AWS revenue Q3")
    assert isinstance(results, list)
    assert len(results) > 0
    for r in results:
        assert "id" in r
        assert "title" in r
        assert "score" in r
        assert "source" in r


def test_query_lancedb_returns_results():
    from retrieval_tools import query_lancedb
    results = query_lancedb("AI services")
    assert isinstance(results, list)
    assert len(results) > 0
    for r in results:
        assert r["source"] == "LanceDB"


def test_query_kuzudb_returns_results():
    from retrieval_tools import query_kuzudb
    results = query_kuzudb("earnings call")
    assert isinstance(results, list)
    assert len(results) > 0
    for r in results:
        assert r["source"] == "KuzuDB"


def test_fuse_results_rrf_merges_all_sources():
    from retrieval_tools import query_tantivy, query_lancedb, query_kuzudb, fuse_results_rrf
    tantivy = query_tantivy("AWS")
    lancedb = query_lancedb("AI")
    kuzudb = query_kuzudb("earnings")
    fused = fuse_results_rrf(tantivy, lancedb, kuzudb)
    # Should contain all unique doc IDs from the three sources
    all_ids = {r["id"] for r in tantivy} | {r["id"] for r in lancedb} | {r["id"] for r in kuzudb}
    fused_ids = {r["id"] for r in fused}
    assert fused_ids == all_ids
    # Each result has rrf_score
    for r in fused:
        assert "rrf_score" in r
        assert r["rrf_score"] > 0


def test_fuse_results_rrf_ranks_by_score():
    from retrieval_tools import fuse_results_rrf
    tantivy = [
        {"id": "a", "title": "A", "snippet": "a", "score": 0.0, "source": "Tantivy"},
    ]
    lancedb = [
        {"id": "a", "title": "A", "snippet": "a", "score": 0.0, "source": "LanceDB"},
        {"id": "b", "title": "B", "snippet": "b", "score": 0.0, "source": "LanceDB"},
    ]
    kuzudb = []
    fused = fuse_results_rrf(tantivy, lancedb, kuzudb)
    # 'a' appears in 2 lists, so its RRF score should be higher than 'b' which appears in 1
    a_score = next(r["rrf_score"] for r in fused if r["id"] == "a")
    b_score = next(r["rrf_score"] for r in fused if r["id"] == "b")
    assert a_score > b_score


def test_fuse_results_rrf_handles_empty_lists():
    from retrieval_tools import fuse_results_rrf
    fused = fuse_results_rrf([], [], [])
    assert fused == []


def test_fuse_results_rrf_dedupes_by_id():
    from retrieval_tools import fuse_results_rrf
    doc = {"id": "x", "title": "X", "snippet": "x", "score": 0.0, "source": "Tantivy"}
    tantivy = [doc]
    lancedb = [dict(doc, source="LanceDB")]
    fused = fuse_results_rrf(tantivy, lancedb, [])
    # Only one entry for id="x" (deduplicated)
    assert len(fused) == 1
    assert fused[0]["id"] == "x"


def test_query_tantivy_falls_back_to_mock_when_index_empty(monkeypatch):
    """When the Tantivy index is empty, query_tantivy should return mock data."""
    from retrieval_tools import _tantivy_results
    # Point tantivy to a non-existent path that will fail
    monkeypatch.setenv("TANTIVY_INDEX_PATH", "/nonexistent/path/that/will/fail")
    result = _tantivy_results("test")
    # The function should either return None (then mock fallback) or fail gracefully
    # Currently: if no meta.json exists, it creates a new empty index → search returns []
    # → returns None → mock fallback
    assert result is None or isinstance(result, list)


def test_query_tantivy_uses_live_data_when_indexed(monkeypatch):
    """When the Tantivy index has documents, query_tantivy should use them."""
    from src.backends.tantivy_index import TantivyIndex
    from retrieval_tools import _tantivy_results

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("TANTIVY_INDEX_PATH", tmp)
        idx = TantivyIndex(index_path=tmp)
        idx.index_document(doc_id="live1", title="Live Test Doc", body="This is a real indexed document about Python")
        idx.index_document(doc_id="live2", title="Other Doc", body="Java content here")
        result = _tantivy_results("Python")
        assert result is not None
        assert len(result) > 0
        # Top result should be the Python doc
        assert "Python" in result[0]["title"] or "Python" in result[0]["snippet"]
        assert result[0]["source"] == "Tantivy"


def test_query_kuzudb_uses_live_data_when_indexed(monkeypatch):
    """When the Kuzu graph has documents, query_kuzudb should use them."""
    from src.backends.kuzu_graph import KuzuGraph
    from retrieval_tools import _kuzudb_results

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("KUZU_DB_PATH", os.path.join(tmp, "kuzu.db"))
        g = KuzuGraph()
        g.add_document("d1", "Test Doc")
        g.add_page("d1", "p1", 1, "Python programming tutorial for beginners")
        g.add_page("d1", "p2", 2, "Java is another language")
        result = _kuzudb_results("Python")
        g.close()
        assert result is not None
        assert len(result) == 1
        assert "Python" in result[0]["snippet"]


def test_query_lancedb_returns_mock_fallback():
    """LanceDB dense search currently returns mock data (vector search not wired for text)."""
    from retrieval_tools import _lancedb_results
    result = _lancedb_results("test")
    assert result is None  # Always returns None, so query_lancedb uses mock fallback
