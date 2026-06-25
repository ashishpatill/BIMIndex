"""KuzuDB graph backend with HippoRAG 2 patterns."""

import os
from pathlib import Path
from typing import Optional


class KuzuGraph:
    """KuzuDB graph database for document retrieval with HippoRAG 2 patterns."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.getenv("KUZU_DB_PATH", "./data/kuzu_db/kuzu.db")
        self._db = None
        self._conn = None

    def _lazy_init(self):
        if self._db is not None and self._conn is not None:
            return
        try:
            import kuzu

            path = Path(self.db_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            self._db = kuzu.Database(str(path))
            self._conn = kuzu.Connection(self._db)
            self._init_schema()
        except ImportError:
            raise ImportError("kuzu not installed. Run: pip install kuzu")

    def _init_schema(self):
        """Create graph schema: Document -> Page -> Token nodes with relationships."""
        self._conn.execute(
            "CREATE NODE TABLE IF NOT EXISTS Document (id STRING, title STRING, PRIMARY KEY (id))"
        )
        self._conn.execute(
            "CREATE NODE TABLE IF NOT EXISTS Page (id STRING, number INT64, text STRING, PRIMARY KEY (id))"
        )
        self._conn.execute(
            "CREATE NODE TABLE IF NOT EXISTS Token (id STRING, value STRING, type STRING, PRIMARY KEY (id))"
        )
        self._conn.execute("CREATE REL TABLE IF NOT EXISTS CONTAINS (FROM Document TO Page)")
        self._conn.execute("CREATE REL TABLE IF NOT EXISTS HAS_TOKEN (FROM Page TO Token)")
        self._conn.execute("CREATE REL TABLE IF NOT EXISTS NEXT_PAGE (FROM Page TO Page)")
        self._conn.execute("CREATE REL TABLE IF NOT EXISTS REFERENCES (FROM Token TO Page)")

    def add_document(self, doc_id: str, title: str):
        self._lazy_init()
        self._conn.execute(
            "MERGE (d:Document {id: $id, title: $title})", {"id": doc_id, "title": title}
        )

    def add_page(self, doc_id: str, page_id: str, number: int, text: str):
        self._lazy_init()
        self._conn.execute(
            "MERGE (p:Page {id: $id, number: $num, text: $text})",
            {"id": page_id, "num": number, "text": text},
        )
        self._conn.execute(
            "MATCH (d:Document {id: $did}), (p:Page {id: $pid}) MERGE (d)-[:CONTAINS]->(p)",
            {"did": doc_id, "pid": page_id},
        )

    def search_by_text(self, query_text: str, top_k: int = 10) -> list[dict]:
        """Search pages containing query text."""
        self._lazy_init()
        results = self._conn.execute(
            "MATCH (p:Page) WHERE CONTAINS(p.text, $query) RETURN p.id, p.number, p.text LIMIT $k",
            {"query": query_text, "k": top_k},
        )
        return [{"id": r[0], "page": r[1], "text": r[2]} for r in results]

    def traverse_document(self, doc_id: str) -> list[dict]:
        """Get all pages in a document with their connections."""
        self._lazy_init()
        results = self._conn.execute(
            "MATCH (d:Document {id: $id})-[:CONTAINS]->(p:Page) RETURN p.id, p.number, p.text ORDER BY p.number",
            {"id": doc_id},
        )
        return [{"id": r[0], "page": r[1], "text": r[2]} for r in results]

    def close(self):
        """Close the connection and database."""
        if self._conn is not None:
            self._conn.close() if hasattr(self._conn, 'close') else None
            self._conn = None
        self._db = None
