"""Cross-encoder reranking for retrieval result refinement."""

from typing import Optional


class CrossEncoderReranker:
    """Cross-encoder model for reranking candidate documents."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _lazy_init(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device)
        except ImportError:
            raise ImportError("sentence_transformers not installed.")

    def rerank(self, query: str, candidates: list[dict], top_k: Optional[int] = None) -> list[dict]:
        """Rerank candidate documents by cross-encoder relevance scoring."""
        if not candidates:
            return []
        self._lazy_init()

        pairs = [(query, c.get("text", c.get("content", ""))) for c in candidates]
        scores = self._model.predict(pairs)

        scored = list(zip(scores, candidates))
        scored.sort(key=lambda x: x[0], reverse=True)

        if top_k:
            scored = scored[:top_k]

        return [
            {**candidate, "rerank_score": float(score)}
            for score, candidate in scored
        ]

    def rerank_with_scores(self, query: str, documents: list[str]) -> list[tuple[str, float]]:
        """Return documents with scores, no dict wrapping needed."""
        if not documents:
            return []
        self._lazy_init()
        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs)

        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: float(x[1]), reverse=True)
        return [(doc, float(score)) for doc, score in scored]
