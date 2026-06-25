"""Tests for LanceDB dense retrieval backend."""
import json
import tempfile
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.backends.lancedb_index import LanceDBIndex


@pytest.fixture
def tmp_lancedb():
    """Fresh LanceDBIndex in a temp dir per test."""
    with tempfile.TemporaryDirectory() as tmp:
        yield LanceDBIndex(db_path=tmp)


def test_lancedb_import():
    assert LanceDBIndex is not None


def test_lancedb_init_defaults():
    index = LanceDBIndex()
    assert index.db_path == "./data/lancedb"
    assert index._db is None


def test_lancedb_init_custom():
    index = LanceDBIndex(db_path="/tmp/test_lancedb")
    assert index.db_path == "/tmp/test_lancedb"


def test_lancedb_init_from_env(monkeypatch):
    monkeypatch.setenv("LANCEDB_PATH", "/tmp/env_lancedb")
    index = LanceDBIndex()
    assert index.db_path == "/tmp/env_lancedb"


def test_lancedb_create_table(tmp_lancedb):
    """Create a table with the right schema."""
    table = tmp_lancedb.create_table("test", dimension=4)
    assert table is not None
    assert "vector" in table.schema.names


def test_lancedb_add_and_search(tmp_lancedb):
    """Add embeddings and search for nearest neighbor."""
    tmp_lancedb.create_table("test", dimension=4)
    np.random.seed(42)
    vectors = [np.random.rand(4).tolist() for _ in range(5)]
    ids = [str(i) for i in range(5)]
    texts = [f"document number {i}" for i in range(5)]
    tmp_lancedb.add_embeddings("test", ids, vectors, texts)

    # Search with a known vector — should return itself as the top hit
    results = tmp_lancedb.search("test", vectors[0], top_k=3)
    assert len(results) == 3
    assert results[0]["id"] == "0"
    assert results[0]["text"] == "document number 0"
    assert results[0]["score"] < 0.001  # Near-zero distance to itself


def test_lancedb_metadata_round_trip(tmp_lancedb):
    """Metadata should serialize/deserialize as JSON."""
    tmp_lancedb.create_table("test", dimension=4)
    np.random.seed(42)
    metadata = [{"source": "doc1.pdf", "page": 1}, {"source": "doc2.pdf", "page": 5}]
    tmp_lancedb.add_embeddings(
        "test",
        ids=["0", "1"],
        vectors=[np.random.rand(4).tolist() for _ in range(2)],
        texts=["a", "b"],
        metadatas=metadata,
    )
    results = tmp_lancedb.search("test", np.random.rand(4).tolist(), top_k=2)
    assert "metadata" in results[0]
    # metadata field is a JSON string
    if isinstance(results[0]["metadata"], str):
        meta = json.loads(results[0]["metadata"])
    else:
        meta = results[0]["metadata"]
    assert "source" in meta


def test_lancedb_create_index_with_sufficient_data(tmp_lancedb):
    """Index creation requires >=256 rows for PQ training."""
    dim = 16
    tmp_lancedb.create_table("big", dimension=dim)
    np.random.seed(42)
    n = 300
    vectors = [np.random.rand(dim).tolist() for _ in range(n)]
    ids = [str(i) for i in range(n)]
    texts = [f"doc {i}" for i in range(n)]
    tmp_lancedb.add_embeddings("big", ids, vectors, texts)
    success = tmp_lancedb.create_index("big", num_partitions=4, num_sub_vectors=4)
    assert success is True


def test_lancedb_search_returns_sorted_results(tmp_lancedb):
    """Search results should be sorted by distance ascending."""
    dim = 8
    tmp_lancedb.create_table("sorted", dimension=dim)
    np.random.seed(42)
    vectors = [np.random.rand(dim).tolist() for _ in range(20)]
    tmp_lancedb.add_embeddings("sorted", [str(i) for i in range(20)], vectors, [f"d{i}" for i in range(20)])
    results = tmp_lancedb.search("sorted", vectors[5], top_k=10)
    # Self (vectors[5]) should be top hit
    assert results[0]["id"] == "5"
    # Distances should be ascending
    for i in range(1, len(results)):
        assert results[i]["score"] >= results[i - 1]["score"]


def test_lancedb_search_empty_table(tmp_lancedb):
    """Searching an empty table returns empty list."""
    tmp_lancedb.create_table("empty", dimension=4)
    results = tmp_lancedb.search("empty", [0.1, 0.2, 0.3, 0.4], top_k=5)
    assert results == []


def test_lancedb_top_k_limit(tmp_lancedb):
    """top_k should limit the number of returned results."""
    dim = 4
    tmp_lancedb.create_table("topk", dimension=dim)
    np.random.seed(42)
    vectors = [np.random.rand(dim).tolist() for _ in range(10)]
    tmp_lancedb.add_embeddings(
        "topk",
        [str(i) for i in range(10)],
        vectors,
        [f"d{i}" for i in range(10)],
    )
    results = tmp_lancedb.search("topk", np.random.rand(dim).tolist(), top_k=3)
    assert len(results) == 3
