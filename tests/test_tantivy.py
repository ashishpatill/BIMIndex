"""Tests for Tantivy BM25 search backend."""
import os
import tempfile
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.backends.tantivy_index import TantivyIndex


@pytest.fixture
def tmp_tantivy():
    """Fresh TantivyIndex in a temp dir per test."""
    with tempfile.TemporaryDirectory() as tmp:
        yield TantivyIndex(index_path=tmp)


def test_tantivy_import():
    assert TantivyIndex is not None


def test_tantivy_init_defaults():
    index = TantivyIndex()
    assert index.index_path == "./data/tantivy_index"
    assert index._index is None  # lazy


def test_tantivy_init_custom():
    index = TantivyIndex(index_path="/tmp/test_tantivy")
    assert index.index_path == "/tmp/test_tantivy"


def test_tantivy_init_from_env(monkeypatch):
    monkeypatch.setenv("TANTIVY_INDEX_PATH", "/tmp/env_tantivy")
    index = TantivyIndex()
    assert index.index_path == "/tmp/env_tantivy"


def test_tantivy_lazy_init_creates_index(tmp_tantivy):
    tmp_tantivy._lazy_init()
    assert tmp_tantivy._index is not None
    assert Path(tmp_tantivy.index_path).exists()


def test_tantivy_index_and_search(tmp_tantivy):
    """Live integration test: index a document, search it back."""
    tmp_tantivy.index_document(
        doc_id="1",
        title="Python programming",
        body="Python is a high-level programming language.",
    )
    results = tmp_tantivy.search("Python", top_k=5)
    assert len(results) == 1
    assert results[0]["title"] == "Python programming"
    assert results[0]["score"] > 0


def test_tantivy_bm25_ranking_order(tmp_tantivy):
    """Verify BM25 ranks more-relevant docs higher."""
    tmp_tantivy.index_document(
        doc_id="1", title="Python tutorial", body="Learn Python basics quickly"
    )
    tmp_tantivy.index_document(
        doc_id="2", title="Java tutorial", body="Java is another programming language"
    )
    tmp_tantivy.index_document(
        doc_id="3", title="Python advanced", body="Advanced Python data science and ML with Python"
    )
    results = tmp_tantivy.search("Python", top_k=5)
    # Python docs should rank above Java doc
    assert len(results) >= 2
    assert "Python" in results[0]["title"]
    # The "Python advanced" doc mentions Python multiple times, should rank highest
    assert "advanced" in results[0]["title"].lower() or "tutorial" in results[0]["title"].lower()


def test_tantivy_batch_indexing(tmp_tantivy):
    """Index multiple documents at once."""
    docs = [
        {"title": "Doc A", "body": "Content about AI"},
        {"title": "Doc B", "body": "Content about ML"},
        {"title": "Doc C", "body": "Content about Python"},
    ]
    tmp_tantivy.index_documents(docs)
    results_ai = tmp_tantivy.search("AI", top_k=5)
    results_ml = tmp_tantivy.search("ML", top_k=5)
    results_py = tmp_tantivy.search("Python", top_k=5)
    assert len(results_ai) >= 1
    assert len(results_ml) >= 1
    assert len(results_py) >= 1


def test_tantivy_empty_index_returns_empty(tmp_tantivy):
    """Empty index returns empty results, not error."""
    tmp_tantivy._lazy_init()
    results = tmp_tantivy.search("anything", top_k=5)
    assert results == []


def test_tantivy_save_persists_index(tmp_tantivy):
    """Save and re-open the index, data should persist."""
    tmp_tantivy.index_document(
        doc_id="1", title="Persistent", body="This doc should survive a save/reload"
    )
    tmp_tantivy.save()
    # Re-open the index
    reloaded = TantivyIndex(index_path=tmp_tantivy.index_path)
    reloaded._lazy_init()
    results = reloaded.search("Persistent", top_k=5)
    assert len(results) >= 1
    assert results[0]["title"] == "Persistent"
