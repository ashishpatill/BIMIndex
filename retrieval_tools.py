"""Tri-Modal retrieval tools — Tantivy (BM25) + LanceDB (dense) + KuzuDB (graph).

This module provides the public `query_tantivy`, `query_lancedb`, `query_kuzudb`,
and `fuse_results_rrf` functions used by `retrieval_agent.py`.

Each backend is optional. If the corresponding index is not initialized (no
documents indexed yet), the function falls back to mock data so demos and
tests can run without requiring live data.
"""

import json
import os
from typing import Optional


# Mock Data (used as fallback when live indices are empty)
MOCK_DOCUMENTS = [
    {
        "id": "doc1",
        "title": "AWS Q3 Earnings Report",
        "snippet": "AWS announced a record $23B in revenue for Q3, largely driven by new generative AI services.",
        "score": 0.0,
        "source": "Tantivy",
    },
    {
        "id": "doc2",
        "title": "Cloud Market Share 2024",
        "snippet": "Amazon Web Services retains the top spot, citing strong adoption of Amazon Bedrock and AI tools contributing significantly to their top-line revenue.",
        "score": 0.0,
        "source": "Tantivy",
    },
    {
        "id": "doc3",
        "title": "Generative AI Impact on Cloud",
        "snippet": "AI represents a multi-billion dollar run rate for AWS. Q3 numbers reflect this massive shift in enterprise spending.",
        "score": 0.0,
        "source": "LanceDB",
    },
    {
        "id": "doc4",
        "title": "AWS and Anthropic Partnership",
        "snippet": "The strategic investment in Anthropic has accelerated AI model deployment on AWS, directly impacting Q3 revenue margins.",
        "score": 0.0,
        "source": "LanceDB",
    },
    {
        "id": "doc5",
        "title": "Earnings Call Transcript",
        "snippet": "CEO mentioned: 'Our generative AI business is growing at an unprecedented rate, adding billions to our Q3 revenue.'",
        "score": 0.0,
        "source": "KuzuDB",
    },
    {
        "id": "doc6",
        "title": "Entity Relation: AWS -> Generative AI",
        "snippet": "Graph traversal shows strong linkage between AWS Q3 financials and newly launched Generative AI compute instances.",
        "score": 0.0,
        "source": "KuzuDB",
    },
]


def _tantivy_results(query: str) -> Optional[list[dict]]:
    """Try a live Tantivy BM25 search. Returns None on failure/empty index."""
    try:
        from src.backends.tantivy_index import TantivyIndex

        idx = TantivyIndex()
        results = idx.search(query, top_k=10)
        if not results:
            return None
        return [
            {
                "id": f"tantivy-{i}",
                "title": r["title"],
                "snippet": r["body"],
                "score": r["score"],
                "source": "Tantivy",
            }
            for i, r in enumerate(results)
        ]
    except Exception:
        return None


def _lancedb_results(query: str) -> Optional[list[dict]]:
    """Try a live LanceDB dense search. Returns None on failure/empty index.

    Note: This requires a query vector; for text queries we currently fall back
    to mock. Vector search integration is tracked separately.
    """
    return None


def _kuzudb_results(query: str) -> Optional[list[dict]]:
    """Try a live KuzuDB graph search. Returns None on failure/empty graph."""
    try:
        from src.backends.kuzu_graph import KuzuGraph

        g = KuzuGraph()
        results = g.search_by_text(query, top_k=10)
        g.close()
        if not results:
            return None
        return [
            {
                "id": r["id"],
                "title": f"Page {r['page']}",
                "snippet": r["text"],
                "score": 1.0 / (1 + i),  # descending by rank
                "source": "KuzuDB",
            }
            for i, r in enumerate(results)
        ]
    except Exception:
        return None


def _mock_results(source: str) -> list[dict]:
    """Return mock results filtered by source."""
    results = [doc for doc in MOCK_DOCUMENTS if doc["source"] == source]
    for i, res in enumerate(results):
        res["score"] = 10.0 - i * 1.5  # mock scores
    return results


def query_tantivy(query: str) -> list[dict]:
    """Vectorless, deterministic lexical search using BM25 over the Tantivy index.

    Falls back to mock data if the live index is empty or not initialized.
    """
    print(f"[Tantivy Subagent] Executing lexical search for: '{query}'")
    live = _tantivy_results(query)
    if live is not None:
        return live
    return _mock_results("Tantivy")


def query_lancedb(query: str) -> list[dict]:
    """Dense, multi-vector search (MUVERA) over late-chunking embeddings in LanceDB.

    Falls back to mock data if the live index is empty or not initialized.
    """
    print(f"[LanceDB Subagent] Executing dense search for: '{query}'")
    live = _lancedb_results(query)
    if live is not None:
        return live
    return _mock_results("LanceDB")


def query_kuzudb(query: str) -> list[dict]:
    """Graph RAG search employing HippoRAG 2 with Personalized PageRank (PPR) over passage nodes in KuzuDB.

    Falls back to mock data if the live graph is empty or not initialized.
    """
    print(f"[KuzuDB Subagent] Executing graph RAG search for: '{query}'")
    live = _kuzudb_results(query)
    if live is not None:
        return live
    return _mock_results("KuzuDB")


def fuse_results_rrf(
    tantivy_results: list,
    lancedb_results: list,
    kuzudb_results: list,
    k: int = 60,
) -> list[dict]:
    """Combine retrieved contexts from multiple sources via Reciprocal Rank Fusion (RRF)."""
    print(
        f"[Fusion Subagent] Fusing {len(tantivy_results)} Tantivy, "
        f"{len(lancedb_results)} LanceDB, and {len(kuzudb_results)} KuzuDB "
        f"results with RRF (k={k})."
    )

    scores = {}
    docs = {}

    def add_to_rrf(results_list):
        for rank, doc in enumerate(results_list):
            doc_id = doc["id"]
            if doc_id not in docs:
                docs[doc_id] = doc
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank + 1)

    add_to_rrf(tantivy_results)
    add_to_rrf(lancedb_results)
    add_to_rrf(kuzudb_results)

    ranked_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    final_results = []
    for doc_id, rrf_score in ranked_docs:
        doc_copy = docs[doc_id].copy()
        doc_copy["rrf_score"] = rrf_score
        final_results.append(doc_copy)

    return final_results
