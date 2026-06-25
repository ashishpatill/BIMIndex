"""Neo4j graph backend for graph traversal retrieval."""

import os
from typing import Optional


class Neo4jGraph:
    """Neo4j graph database client for document retrieval graphs."""
    
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "neo4j",
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.database = database
        self._driver = None
    
    def _connect(self):
        if self._driver is None:
            try:
                from neo4j import GraphDatabase
                self._driver = GraphDatabase.driver(
                    self.uri, auth=(self.user, self.password)
                )
            except ImportError:
                raise ImportError(
                    "Neo4j driver not installed. Run: pip install neo4j"
                )
    
    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None
    
    def create_document_graph(self, doc_id: str, nodes: list[dict], edges: list[dict]):
        """Create document graph nodes and relationships."""
        self._connect()
        with self._driver.session(database=self.database) as session:
            # Create document root
            session.run(
                "MERGE (d:Document {id: $id}) RETURN d",
                id=doc_id
            )
            # Create chunk nodes
            for node in nodes:
                session.run(
                    """
                    MATCH (d:Document {id: $doc_id})
                    MERGE (c:Chunk {id: $chunk_id})
                    SET c.text = $text, c.page = $page, c.position = $position
                    MERGE (d)-[:CONTAINS]->(c)
                    """,
                    doc_id=doc_id,
                    chunk_id=node["id"],
                    text=node.get("text", ""),
                    page=node.get("page", 0),
                    position=node.get("position", 0),
                )
            # Create relationships
            for edge in edges:
                session.run(
                    """
                    MATCH (a:Chunk {id: $source_id})
                    MATCH (b:Chunk {id: $target_id})
                    MERGE (a)-[:$rel_type]->(b)
                    """,
                    source_id=edge["source"],
                    target_id=edge["target"],
                    rel_type=edge.get("type", "NEXT"),
                )
    
    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Search graph using Cypher query."""
        self._connect()
        with self._driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (c:Chunk)
                WHERE c.text CONTAINS $query
                RETURN c.id AS id, c.text AS text, c.page AS page
                LIMIT $top_k
                """,
                query=query,
                top_k=top_k
            )
            return [dict(r) for r in result]
    
    def traverse(self, chunk_id: str, direction: str = "both", max_depth: int = 2) -> list[dict]:
        """Traverse graph from a starting chunk."""
        self._connect()
        direction_map = {
            "forward": "->",
            "backward": "<-",
            "both": "-",
        }
        arrow = direction_map.get(direction, "-")
        with self._driver.session(database=self.database) as session:
            query = f"""
                MATCH (start:Chunk {{id: $chunk_id}})
                OPTIONAL MATCH path = (start){arrow}[*1..$max_depth]{arrow}()
                UNWIND nodes(path) AS n
                WHERE n:Chunk AND n.id <> $chunk_id
                RETURN DISTINCT n.id AS id, n.text AS text, n.page AS page
                LIMIT 50
            """
            result = session.run(query, chunk_id=chunk_id, max_depth=max_depth)
            return [dict(r) for r in result]
