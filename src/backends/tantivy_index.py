"""Tantivy BM25 search backend for vectorless lexical retrieval."""

import os
from pathlib import Path
from typing import Optional


class TantivyIndex:
    """Tantivy-based BM25 search index."""

    def __init__(self, index_path: Optional[str] = None):
        self.index_path = index_path or os.getenv("TANTIVY_INDEX_PATH", "./data/tantivy_index")
        self._index = None
        self._schema_builder = None

    def _lazy_init(self):
        if self._index is not None:
            return
        try:
            import tantivy

            path = Path(self.index_path)
            path.mkdir(parents=True, exist_ok=True)

            schema_builder = tantivy.SchemaBuilder()
            schema_builder.add_text_field("title", stored=True)
            schema_builder.add_text_field("body", stored=True)
            schema = schema_builder.build()

            self._index = tantivy.Index(path=str(path), schema=schema)
        except ImportError:
            raise ImportError("tantivy not installed. Run: pip install tantivy")

    def index_document(self, doc_id: str, title: str, body: str):
        """Index a single document."""
        import tantivy

        self._lazy_init()
        writer = self._index.writer()
        writer.add_document(
            tantivy.Document(title=title, body=body)
        )
        writer.commit()
        self._index.reload()

    def index_documents(self, documents: list[dict]):
        """Index multiple documents in batch."""
        import tantivy

        self._lazy_init()
        writer = self._index.writer()
        for doc in documents:
            writer.add_document(
                tantivy.Document(
                    title=doc.get("title", ""),
                    body=doc.get("body", doc.get("text", "")),
                )
            )
        writer.commit()
        self._index.reload()

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Search using BM25 scoring."""
        self._lazy_init()
        searcher = self._index.searcher()
        query_obj = self._index.parse_query(query, ["title", "body"])

        results = searcher.search(query_obj, top_k)
        hits = []
        for score, doc_address in results.hits:
            doc = searcher.doc(doc_address)
            hits.append({
                "title": doc["title"][0],
                "body": doc["body"][0],
                "score": score,
            })
        return hits

    def save(self):
        """Persist index to disk."""
        if self._index:
            try:
                writer = self._index.writer()
                writer.commit()
                self._index.reload()
            except Exception:
                pass
