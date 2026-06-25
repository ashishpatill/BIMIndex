"""Tests for KuzuDB graph backend."""
import os
import tempfile
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.backends.kuzu_graph import KuzuGraph


@pytest.fixture
def tmp_kuzu():
    """Fresh KuzuGraph in a temp dir per test."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "kuzu.db")
        g = KuzuGraph(db_path=db_path)
        yield g
        g.close()


def test_kuzu_import():
    assert KuzuGraph is not None


def test_kuzu_init_defaults():
    g = KuzuGraph()
    assert g.db_path == "./data/kuzu_db/kuzu.db"
    assert g._db is None


def test_kuzu_init_custom():
    g = KuzuGraph(db_path="/tmp/test_kuzu.db")
    assert g.db_path == "/tmp/test_kuzu.db"


def test_kuzu_init_from_env(monkeypatch):
    monkeypatch.setenv("KUZU_DB_PATH", "/tmp/env_kuzu.db")
    g = KuzuGraph()
    assert g.db_path == "/tmp/env_kuzu.db"


def test_kuzu_lazy_init_creates_schema(tmp_kuzu):
    """Lazy init creates the Document/Page/Token schema."""
    tmp_kuzu._lazy_init()
    assert tmp_kuzu._db is not None
    assert tmp_kuzu._conn is not None
    result = tmp_kuzu._conn.execute("CALL show_tables() RETURN name")
    table_names = [r[0] for r in result]
    assert "Document" in table_names
    assert "Page" in table_names
    assert "Token" in table_names


def test_kuzu_add_and_query_document(tmp_kuzu):
    """Add a document and query it back."""
    tmp_kuzu.add_document("d1", "Test Document")
    import kuzu
    conn = kuzu.Connection(tmp_kuzu._db)
    result = conn.execute("MATCH (d:Document) RETURN d.id, d.title")
    rows = list(result)
    assert len(rows) == 1
    assert rows[0] == ["d1", "Test Document"]


def test_kuzu_add_page_and_traverse(tmp_kuzu):
    """Add pages and traverse the document structure."""
    tmp_kuzu.add_document("d1", "Python Guide")
    tmp_kuzu.add_page("d1", "p1", 1, "Introduction to Python")
    tmp_kuzu.add_page("d1", "p2", 2, "Python is a programming language")
    pages = tmp_kuzu.traverse_document("d1")
    assert len(pages) == 2
    assert pages[0]["page"] == 1
    assert pages[1]["page"] == 2


def test_kuzu_search_by_text(tmp_kuzu):
    """Full-text search for pages containing query text."""
    tmp_kuzu.add_document("d1", "Python Guide")
    tmp_kuzu.add_page("d1", "p1", 1, "Introduction to Python programming")
    tmp_kuzu.add_page("d1", "p2", 2, "Java is a different language")
    tmp_kuzu.add_page("d1", "p3", 3, "Advanced Python data science")
    results = tmp_kuzu.search_by_text("Python", top_k=5)
    assert len(results) == 2
    # Both Python pages should be returned
    for r in results:
        assert "Python" in r["text"]


def test_kuzu_multiple_documents(tmp_kuzu):
    """Multiple documents can be added without conflicts."""
    tmp_kuzu.add_document("d1", "Doc 1")
    tmp_kuzu.add_document("d2", "Doc 2")
    tmp_kuzu.add_page("d1", "p1", 1, "Content in doc 1")
    tmp_kuzu.add_page("d2", "p2", 1, "Content in doc 2")
    p1_pages = tmp_kuzu.traverse_document("d1")
    p2_pages = tmp_kuzu.traverse_document("d2")
    assert len(p1_pages) == 1
    assert len(p2_pages) == 1
    assert p1_pages[0]["text"] == "Content in doc 1"
    assert p2_pages[0]["text"] == "Content in doc 2"


def test_kuzu_search_no_results(tmp_kuzu):
    """Searching for non-existent text returns empty list."""
    tmp_kuzu.add_document("d1", "Test")
    tmp_kuzu.add_page("d1", "p1", 1, "Some text")
    results = tmp_kuzu.search_by_text("nonexistent_term_xyz", top_k=5)
    assert results == []


def test_kuzu_traverse_empty_document(tmp_kuzu):
    """Traversing a non-existent document returns empty list."""
    pages = tmp_kuzu.traverse_document("nonexistent_doc")
    assert pages == []
