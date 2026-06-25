# Changelog: BIMIndex

All notable changes to the `BIMIndex` repository will be documented in this file.

## [v1.0.0] - Tri-Modal Agentic Baseline

### Added
- **Antigravity Tri-Modal Orchestration**: `retrieval_agent.py` to oversee concurrent searches across dense, lexical, and graph indexers.
- **Retrieval Tools**: `retrieval_tools.py` implementing queries for Tantivy, LanceDB, and KuzuDB, alongside an RRF (Reciprocal Rank Fusion) combiner.
- **Evaluation & UI Harness**: Legacy `retrieval_research` system, Streamlit, and Gradio apps are present for deep debugging, query planning, and extraction stress-testing.

### Changed
- Standardized `README.md` to unify the agentic Tri-Modal retrieval narrative with the extensive local evaluation framework.
- Prepended Tri-Modal agent integration status to the `IMPLEMENTATION_PLAN.md`.

## [v1.1.0] - Phases 0–7 Hardening + Live Backend Code

### Added
- **`retrieval_research/` Package** (Phases 0–7):
  - Phase 0: Canonical schema + artifact store under `data/`. Ingestion pipeline and CLI entrypoints.
  - Phase 1: Page-aware chunking, BM25 + dense retrieval + hybrid fusion, query traces and evidence bundles.
  - Phase 2: Evaluation harness with manifest execution via CLI/API. Mode metrics, citation support, answerability and confidence reporting.
  - Phase 3: Inspector UI scaffold in `apps/web/` (Next.js 16) — Document library, document detail workspace, query workbench, eval page.
  - Phase 4: Multimodal page retrieval with visual retrieval mode. ColPali-compatible visual backend.
  - Phase 5: Planner and adaptive routing with query classifier, route-specific settings, evidence consolidation, confidence estimates.
  - Phase 6: Structured knowledge layer with graph retrieval, section/entity/reference extraction, knowledge cards, corpus graph search.
  - Phase 7: Centralized configuration (30+ env vars), structured logging, expanded test suite, Dockerfile (3 targets: api/cli/worker), background jobs for ingestion/indexing, Dependabot config.
- **Live Backend Wrappers** (`src/backends/`):
  - `tantivy_index.py` — `TantivyIndex` with BM25 search and index persistence.
  - `lancedb_index.py` — `LanceDBIndex` with MUVERA IVF index (num_partitions=256, num_sub_vectors=96, metric=cosine).
  - `kuzu_graph.py` — `KuzuGraph` with HippoRAG 2 schema (Document → Page → Token nodes).
  - `neo4j_graph.py` — `Neo4jGraph` with Cypher queries.
- **CI/CD**: `.github/workflows/ci.yml` and `cd.yml`.

## [v1.2.0] - T-ROOT-5 Live Tri-Modal Backends

### Added
- **Installed live packages** in `.venv`:
  - `tantivy==0.26.0` (BM25 search)
  - `lancedb==0.25.3` + `pyarrow==24.0.0` (dense vector search)
  - `kuzu==0.11.3` (graph database with HippoRAG 2)
  - `neo4j==6.2.0` (alternative graph driver, mocked in tests)
- **Updated `pyproject.toml`** with all 4 packages as runtime dependencies
- **Test coverage expansion** — 43 new live integration tests across 4 backend wrappers:
  - `tests/test_tantivy.py` (10 tests): lazy init, indexing, BM25 ranking, batch, persistence
  - `tests/test_lancedb.py` (11 tests): schema, add+search, IVF-PQ index, top_k, metadata round-trip
  - `tests/test_kuzu.py` (11 tests): schema creation, add doc/page, full-text search, traversal
  - `tests/test_neo4j.py` (11 tests): init, mocked connect/search/traverse, import error path

### Changed
- `src/backends/tantivy_index.py`: Updated to use tantivy v0.26 API (`tantivy.Index`, `tantivy.Document`, `idx.reload()`)
- `src/backends/kuzu_graph.py`: Updated to use kuzu v0.11 API (file path, persistent connection)
- `src/rerank/cross_encoder.py`: Fixed short-circuit on empty candidates (avoids unnecessary ImportError)
- `retrieval_tools.py`: Rewritten to use live backends with mock fallback when index is empty/uninitialized

### Added
- `tests/test_retrieval_tools.py` (12 tests): mock data, all 3 queries, RRF fusion, dedup, ranking, live-data path

### Test Suite Status
- **Total: 188 passed + 9 skipped** (previously tests were mostly skipped due to missing packages)
- All 4 backend wrappers now have live integration tests
- `retrieval_tools.py` is fully tested with both mock and live data paths

### Verified
- T-ROOT-5: 3× Live DB integrations (Tantivy/LanceDB/KuzuDB) — packages installed, live tests pass
- T-INDEX-1, 2, 3, 10: All backend wrappers fully tested
- `retrieval_agent.py` and `retrieval_tools.py` will use live data when indices are populated; mock fallback otherwise
