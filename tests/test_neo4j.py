"""Tests for Neo4j graph backend.

These tests use mocking since a real Neo4j server is not available in CI.
The neo4j driver itself is verified to be importable.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.backends.neo4j_graph import Neo4jGraph


def test_neo4j_import():
    assert Neo4jGraph is not None


def test_neo4j_init_defaults():
    graph = Neo4jGraph()
    assert graph.uri == "bolt://localhost:7687"
    assert graph.user == "neo4j"
    assert graph.password == "password"
    assert graph.database == "neo4j"
    assert graph._driver is None


def test_neo4j_init_custom():
    graph = Neo4jGraph(
        uri="bolt://custom:7687",
        user="admin",
        password="secret",
        database="testdb",
    )
    assert graph.uri == "bolt://custom:7687"
    assert graph.user == "admin"
    assert graph.password == "secret"
    assert graph.database == "testdb"


def test_neo4j_init_from_env(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://env:7687")
    monkeypatch.setenv("NEO4J_USER", "env_user")
    monkeypatch.setenv("NEO4J_PASSWORD", "env_pass")
    graph = Neo4jGraph()
    assert graph.uri == "bolt://env:7687"
    assert graph.user == "env_user"
    assert graph.password == "env_pass"


def test_neo4j_driver_is_importable():
    """Verify the neo4j Python driver is installed and importable."""
    from neo4j import GraphDatabase
    assert GraphDatabase is not None
    assert hasattr(GraphDatabase, "driver")


def test_neo4j_connect_creates_driver():
    """Mock the GraphDatabase.driver and verify it's called with right args."""
    graph = Neo4jGraph(
        uri="bolt://test:7687",
        user="test_user",
        password="test_pass",
    )
    with patch("neo4j.GraphDatabase") as mock_gd:
        mock_driver = MagicMock()
        mock_gd.driver.return_value = mock_driver
        graph._connect()
        mock_gd.driver.assert_called_once_with(
            "bolt://test:7687", auth=("test_user", "test_pass")
        )
        assert graph._driver is mock_driver


def test_neo4j_search_uses_cypher_contains():
    """Verify search issues a Cypher CONTAINS query with right params."""
    graph = Neo4jGraph(
        uri="bolt://test:7687",
        user="u",
        password="p",
    )
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.__iter__.return_value = iter([{"id": "c1", "text": "hello", "page": 1}])
    mock_session.run.return_value = mock_result
    mock_driver.session.return_value.__enter__.return_value = mock_session
    graph._driver = mock_driver

    results = graph.search("hello", top_k=5)
    assert len(results) == 1
    assert results[0]["id"] == "c1"
    mock_session.run.assert_called_once()
    call_args = mock_session.run.call_args
    assert "CONTAINS" in call_args.args[0]
    assert call_args.kwargs.get("query") == "hello" or (len(call_args.args) > 1 and call_args.args[1].get("query") == "hello")


def test_neo4j_traverse_picks_arrow_direction():
    """Verify traverse uses the right arrow direction for forward/backward/both."""
    graph = Neo4jGraph(uri="bolt://test:7687", user="u", password="p")
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_session.run.return_value = MagicMock(__iter__=lambda self: iter([]))
    mock_driver.session.return_value.__enter__.return_value = mock_session
    graph._driver = mock_driver

    graph.traverse("c1", direction="forward", max_depth=2)
    cypher = mock_session.run.call_args.args[0]
    assert "->" in cypher

    graph.traverse("c1", direction="backward", max_depth=2)
    cypher = mock_session.run.call_args.args[0]
    assert "<-" in cypher

    graph.traverse("c1", direction="both", max_depth=2)
    cypher = mock_session.run.call_args.args[0]
    # Both direction uses "-" which is unconstrained
    assert "->" not in cypher.split("OPTIONAL MATCH")[0].split("(")[1]


def test_neo4j_close_closes_driver():
    """close() should call driver.close() and reset state."""
    graph = Neo4jGraph(uri="bolt://test:7687", user="u", password="p")
    mock_driver = MagicMock()
    graph._driver = mock_driver
    graph.close()
    mock_driver.close.assert_called_once()
    assert graph._driver is None


def test_neo4j_create_document_graph_writes_cypher():
    """create_document_graph should issue MERGE + MATCH for nodes and relationships."""
    graph = Neo4jGraph(uri="bolt://test:7687", user="u", password="p")
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    graph._driver = mock_driver

    nodes = [
        {"id": "c1", "text": "First chunk", "page": 1, "position": 0},
        {"id": "c2", "text": "Second chunk", "page": 1, "position": 1},
    ]
    edges = [{"source": "c1", "target": "c2", "type": "NEXT"}]
    graph.create_document_graph("doc1", nodes, edges)
    # Should have run multiple Cypher statements
    assert mock_session.run.call_count >= 4  # 1 doc + 2 nodes + 1 edge


def test_neo4j_import_error_when_driver_missing():
    """If neo4j driver is not installed, _connect should raise ImportError."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "neo4j":
            raise ImportError("No module named 'neo4j'")
        return real_import(name, *args, **kwargs)

    graph = Neo4jGraph(uri="bolt://test:7687", user="u", password="p")
    with patch("builtins.__import__", side_effect=mock_import):
        with pytest.raises(ImportError):
            graph._connect()
