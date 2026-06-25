"""Tests for PageIndex tree-structured navigation."""
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.modes.page_index import PageIndex, PageIndexNode


def test_page_index_node_creation():
    node = PageIndexNode("n1", "Introduction", "section", 1)
    assert node.id == "n1"
    assert node.label == "Introduction"
    assert node.type == "section"
    assert node.page_number == 1
    assert node.children == []
    assert node.parent is None
    assert node.content == ""


def test_page_index_init():
    index = PageIndex()
    assert index.root is None
    assert index._nodes == {}


def test_build_from_document():
    index = PageIndex()
    pages = [
        {
            "number": 1,
            "text": "Page 1 content",
            "sections": [
                {"heading": "Intro", "text": "Intro text"},
            ],
        },
        {
            "number": 2,
            "text": "Page 2 content",
            "sections": [
                {"heading": "Method", "text": "Method text"},
                {"heading": "Results", "text": "Results text"},
            ],
        },
    ]
    index.build_from_document("doc1", pages)
    assert index.root is not None
    assert index.root.id == "doc1"
    assert index.root.type == "document"
    assert len(index.root.children) == 2
    assert len(index._nodes) == 6  # root + 2 pages + 3 sections


def test_get_node():
    index = PageIndex()
    index.build_from_document("doc1", [{"number": 1, "text": "Content", "sections": []}])
    node = index.get_node("doc1")
    assert node is not None
    assert node.id == "doc1"
    assert index.get_node("nonexistent") is None


def test_search_by_heading():
    index = PageIndex()
    index.build_from_document("doc1", [
        {"number": 1, "text": "Intro page", "sections": [{"heading": "Introduction", "text": "Welcome"}]},
        {"number": 2, "text": "Method page", "sections": [{"heading": "Methodology", "text": "Approach"}]},
    ])
    results = index.search_by_heading("Introduction")
    assert len(results) == 1
    assert results[0].label == "Introduction"


def test_search_by_heading_case_insensitive():
    index = PageIndex()
    index.build_from_document("doc1", [
        {"number": 1, "text": "", "sections": [{"heading": "INTRODUCTION", "text": "Start"}]},
    ])
    results = index.search_by_heading("introduction")
    assert len(results) == 1


def test_search_by_page():
    index = PageIndex()
    index.build_from_document("doc1", [
        {"number": 1, "text": "Page 1", "sections": [{"heading": "A", "text": "a"}]},
        {"number": 2, "text": "Page 2", "sections": [{"heading": "B", "text": "b"}, {"heading": "C", "text": "c"}]},
    ])
    results = index.search_by_page(2)
    assert len(results) == 3  # page node + 2 section nodes


def test_get_path():
    index = PageIndex()
    index.build_from_document("doc1", [
        {"number": 1, "text": "Page 1", "sections": [{"heading": "Intro", "text": "Hello"}]},
    ])
    section_id = f"doc1_p1_Intro"
    path = index.get_path(section_id)
    assert len(path) == 3
    assert path[0] == "Document doc1"
    assert path[1] == "Page 1"
    assert path[2] == "Intro"


def test_get_path_nonexistent():
    index = PageIndex()
    path = index.get_path("no_such_node")
    assert path == []


def test_traverse_from_root():
    index = PageIndex()
    index.build_from_document("doc1", [
        {"number": 1, "text": "Pg1", "sections": []},
        {"number": 2, "text": "Pg2", "sections": []},
    ])
    flat = index.traverse()
    assert len(flat) == 3  # root + 2 pages
    assert flat[0]["id"] == "doc1"


def test_traverse_from_node():
    index = PageIndex()
    index.build_from_document("doc1", [
        {"number": 1, "text": "Page 1", "sections": [{"heading": "Intro", "text": "Hi"}]},
    ])
    flat = index.traverse("doc1_p1")
    assert len(flat) >= 2  # page + section


def test_traverse_empty():
    index = PageIndex()
    assert index.traverse() == []


def test_search_text_overlap():
    index = PageIndex()
    index.build_from_document("doc1", [
        {"number": 1, "text": "Machine learning basics", "sections": []},
        {"number": 2, "text": "Deep learning advanced", "sections": [{"heading": "CNN", "text": "Convolutional neural networks"}]},
    ])
    results = index.search("learning", top_k=5)
    assert len(results) >= 1
    assert results[0]["score"] > 0


def test_search_label_boost():
    index = PageIndex()
    index.build_from_document("doc1", [
        {"number": 1, "text": "Other content", "sections": [{"heading": "Introduction", "text": "Some intro"}]},
    ])
    results = index.search("Introduction", top_k=5)
    assert len(results) == 1
    assert results[0]["score"] >= 2.0


def test_search_top_k():
    index = PageIndex()
    index.build_from_document("doc1", [
        {"number": i, "text": f"Content about topic {i}", "sections": []}
        for i in range(1, 20)
    ])
    results = index.search("topic", top_k=3)
    assert len(results) <= 3


def test_search_no_match():
    index = PageIndex()
    index.build_from_document("doc1", [
        {"number": 1, "text": "Alpha", "sections": [{"heading": "Beta", "text": "Gamma"}]},
    ])
    results = index.search("Omega", top_k=5)
    assert results == []


def test_search_empty_index():
    index = PageIndex()
    results = index.search("anything", top_k=5)
    assert results == []
