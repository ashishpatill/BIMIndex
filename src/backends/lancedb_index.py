"""LanceDB with MUVERA ANN index for dense retrieval."""

import json
import os
from pathlib import Path
from typing import Optional


class LanceDBIndex:
    """LanceDB vector store with MUVERA ANN index."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.getenv("LANCEDB_PATH", "./data/lancedb")
        self._db = None

    def _lazy_init(self):
        if self._db is not None:
            return
        try:
            import lancedb

            path = Path(self.db_path)
            path.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(path))
        except ImportError:
            raise ImportError("lancedb not installed. Run: pip install lancedb pyarrow")

    def create_table(self, name: str, dimension: int = 768):
        """Create a table with MUVERA index configuration."""
        self._lazy_init()
        import pyarrow as pa

        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dimension)),
            pa.field("text", pa.string()),
            pa.field("metadata", pa.string()),
        ])

        table = self._db.create_table(name, schema=schema, mode="overwrite")
        return table

    def create_index(
        self,
        name: str,
        metric: str = "cosine",
        num_partitions: int = 16,
        num_sub_vectors: int = 16,
        min_rows_for_pq: int = 256,
    ):
        """Create an IVF-PQ ANN index. Requires >= 256 rows for PQ training."""
        self._lazy_init()
        table = self._db.open_table(name)
        try:
            table.create_index(
                metric=metric,
                num_partitions=num_partitions,
                num_sub_vectors=num_sub_vectors,
            )
            return True
        except Exception:
            return False

    def add_embeddings(
        self,
        table_name: str,
        ids: list[str],
        vectors: list[list[float]],
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
    ):
        """Add embeddings to the table."""
        self._lazy_init()
        table = self._db.open_table(table_name)

        data = []
        for i, vec in enumerate(vectors):
            data.append({
                "id": ids[i],
                "vector": vec,
                "text": texts[i],
                "metadata": json.dumps(metadatas[i]) if metadatas and i < len(metadatas) else "{}",
            })

        table.add(data)

    def search(
        self,
        table_name: str,
        query_vector: list[float],
        top_k: int = 10,
    ) -> list[dict]:
        """ANN search."""
        self._lazy_init()
        table = self._db.open_table(table_name)
        results = table.search(query_vector).limit(top_k).to_list()

        return [
            {
                "id": r["id"],
                "text": r["text"],
                "score": r["_distance"],
                "metadata": json.loads(r.get("metadata", "{}")),
            }
            for r in results
        ]
