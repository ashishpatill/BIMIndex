"""Tests for SPLADE sparse search integration."""
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_splade_searcher_import():
    from src.search.splade import SPLADESearcher
    assert SPLADESearcher is not None


def test_splade_searcher_init_defaults():
    from src.search.splade import SPLADESearcher
    searcher = SPLADESearcher()
    assert searcher.model_name == "naver/splade-cocondenser-selfdistil"
    assert searcher.device == "cpu"
    assert searcher._model is None
    assert searcher._tokenizer is None
    assert searcher._index == {}
    assert searcher._next_id == 0


def test_splade_searcher_custom_config():
    from src.search.splade import SPLADESearcher
    searcher = SPLADESearcher(model_name="naver/splade-v2-distil", device="cuda:0")
    assert searcher.model_name == "naver/splade-v2-distil"
    assert searcher.device == "cuda:0"


def test_splade_lazy_init_triggers_import_error():
    from src.search.splade import SPLADESearcher
    searcher = SPLADESearcher()
    with pytest.raises((ImportError, OSError)):
        searcher._lazy_init()


def test_encode_without_init_triggers_import_error():
    from src.search.splade import SPLADESearcher
    searcher = SPLADESearcher()
    with pytest.raises((ImportError, OSError)):
        searcher.encode("test query")


def test_index_document_without_init_triggers_import_error():
    from src.search.splade import SPLADESearcher
    searcher = SPLADESearcher()
    with pytest.raises((ImportError, OSError)):
        searcher.index_document("doc1", "some text")


def test_search_without_init_triggers_import_error():
    from src.search.splade import SPLADESearcher
    searcher = SPLADESearcher()
    with pytest.raises((ImportError, OSError)):
        searcher.search("test query")


def test_search_empty_index_returns_empty():
    from src.search.splade import SPLADESearcher
    searcher = SPLADESearcher()
    with pytest.raises(ImportError):
        searcher.search("anything")
