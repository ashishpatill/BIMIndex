"""Tests for cross-encoder reranking integration."""
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_cross_encoder_reranker_import():
    from src.rerank.cross_encoder import CrossEncoderReranker
    assert CrossEncoderReranker is not None


def test_reranker_init_defaults():
    from src.rerank.cross_encoder import CrossEncoderReranker
    reranker = CrossEncoderReranker()
    assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert reranker.device == "cpu"
    assert reranker._model is None


def test_reranker_custom_config():
    from src.rerank.cross_encoder import CrossEncoderReranker
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2",
        device="cuda:0",
    )
    assert reranker.model_name == "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    assert reranker.device == "cuda:0"


def test_reranker_lazy_init_triggers_import_error():
    from src.rerank.cross_encoder import CrossEncoderReranker
    reranker = CrossEncoderReranker()
    with pytest.raises(ImportError, match="sentence_transformers not installed"):
        reranker._lazy_init()


def test_rerank_empty_candidates_returns_empty():
    from src.rerank.cross_encoder import CrossEncoderReranker
    reranker = CrossEncoderReranker()
    result = reranker.rerank("test query", [])
    assert result == []


def test_rerank_without_init_triggers_import_error():
    from src.rerank.cross_encoder import CrossEncoderReranker
    reranker = CrossEncoderReranker()
    with pytest.raises(ImportError, match="sentence_transformers not installed"):
        reranker.rerank("test query", [{"text": "some doc"}])


def test_rerank_with_scores_empty_docs_returns_empty():
    from src.rerank.cross_encoder import CrossEncoderReranker
    reranker = CrossEncoderReranker()
    result = reranker.rerank_with_scores("test query", [])
    assert result == []


def test_rerank_with_scores_triggers_import_error():
    from src.rerank.cross_encoder import CrossEncoderReranker
    reranker = CrossEncoderReranker()
    with pytest.raises(ImportError, match="sentence_transformers not installed"):
        reranker.rerank_with_scores("test query", ["doc1", "doc2"])


def test_rerank_prefers_text_key_over_content():
    from src.rerank.cross_encoder import CrossEncoderReranker
    reranker = CrossEncoderReranker()
    candidates = [{"text": "hello", "content": "world"}]
    with pytest.raises(ImportError, match="sentence_transformers not installed"):
        reranker.rerank("query", candidates)


def test_rerank_with_top_k():
    from src.rerank.cross_encoder import CrossEncoderReranker
    reranker = CrossEncoderReranker()
    candidates = [{"text": f"doc {i}"} for i in range(10)]
    with pytest.raises(ImportError, match="sentence_transformers not installed"):
        reranker.rerank("query", candidates, top_k=3)
