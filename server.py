"""BIMIndex tri-modal retrieval FastAPI server (T-ROOT-1).

Wraps `retrieval_tools.py` (Tantivy / LanceDB / KuzuDB + RRF fusion) behind a
clean HTTP API consumed by BIMAgent (POST) and BIMWeb (GET).

Endpoints
---------
- GET  /health
- GET  /search/{mode}?q=...&top_k=...      (BIMWeb direct-index UI)
- POST /search/{mode}   {query, top_k}     (BIMAgent skill)
- POST /fuse            {query, top_k}     (tri-modal RRF fusion)
- POST /ingest          {documents}        (index docs into Tantivy live)
- GET  /stats                              (index stats)

Modes: vectorless (Tantivy/BM25) | dense (LanceDB) | graph (KuzuDB)

Run: PYTHONPATH=. python -m uvicorn server:app --port 8001
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))

from retrieval_tools import (  # noqa: E402
    fuse_results_rrf,
    query_kuzudb,
    query_lancedb,
    query_tantivy,
)

_MODE_FUNCS = {
    "vectorless": query_tantivy,
    "dense": query_lancedb,
    "graph": query_kuzudb,
}

app = FastAPI(title="BIMIndex", version="0.1.0", description="Tri-Modal Retrieval Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)


class IngestRequest(BaseModel):
    documents: list[dict[str, Any]] = Field(..., description="List of {title, body/text}")


def _run_search(mode: str, query: str, top_k: int) -> list[dict[str, Any]]:
    func = _MODE_FUNCS.get(mode)
    if func is None:
        raise HTTPException(status_code=400, detail=f"Unknown mode '{mode}'. Use: vectorless, dense, graph")
    results = func(query)
    return results[:top_k] if results else []


@app.get("/health")
async def health():
    return {"status": "ok", "service": "bimindex", "modes": list(_MODE_FUNCS)}


@app.api_route("/search/{mode}", methods=["GET", "POST"])
async def search(
    mode: str,
    req: Optional[SearchRequest] = None,
    q: str = Query(default="", description="Search query (GET)"),
    top_k: int = Query(default=10, ge=1, le=100, description="Top-K (GET)"),
):
    """Search a single mode. Accepts POST JSON body (BIMAgent) or GET query params (BIMWeb)."""
    if req is not None:
        query, k = req.query, req.top_k
    else:
        query, k = q, top_k
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    results = _run_search(mode, query, k)
    return {"results": results, "mode": mode, "query": query, "total": len(results)}


@app.post("/fuse")
async def fuse(req: SearchRequest):
    """Run all three modes in parallel conceptually and fuse via Reciprocal Rank Fusion."""
    t = query_tantivy(req.query)[: req.top_k]
    l = query_lancedb(req.query)[: req.top_k]
    k = query_kuzudb(req.query)[: req.top_k]
    fused = fuse_results_rrf(t, l, k)[: req.top_k]
    return {
        "query": req.query,
        "results": fused,
        "total": len(fused),
        "per_mode": {"vectorless": len(t), "dense": len(l), "graph": len(k)},
    }


@app.post("/ingest")
async def ingest(req: IngestRequest):
    """Index documents into the live Tantivy backend (BM25)."""
    try:
        from src.backends.tantivy_index import TantivyIndex

        idx = TantivyIndex()
        idx.index_documents(req.documents)
        idx.save()
        return {"status": "ok", "indexed": len(req.documents), "backend": "tantivy"}
    except ImportError:
        return {"status": "degraded", "indexed": 0, "error": "tantivy not installed; documents not persisted"}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}")


@app.get("/stats")
async def stats():
    """Return per-mode availability (whether live backends respond)."""
    out: dict[str, Any] = {}
    for mode in _MODE_FUNCS:
        try:
            r = _run_search(mode, "stat", 1)
            out[mode] = {"available": True, "sample_results": len(r)}
        except Exception as e:  # noqa: BLE001
            out[mode] = {"available": False, "error": str(e)}
    return {"stats": out}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("BIMINDEX_PORT", "8001"))
    uvicorn.run("server:app", host="127.0.0.1", port=port, reload=False)
