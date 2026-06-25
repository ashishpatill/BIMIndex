# Implementation Plan: BIMIndex

**Last updated: 2026-06-25** — 7 of 10 TASKS.md items complete. Critical gap: `tantivy`/`lancedb`/`kuzu` Python packages not installed in `.venv` — `src/backends/` code exists but cannot be exercised.

## Current Focus: Tri-Modal Agentic Retrieval (v1.0 Integration)

The current priority is establishing the robust Tri-Modal retrieval agent leveraging the Google Antigravity SDK.

### Completed Agentic Integrations
- `[x]` Antigravity Subagent Orchestration (`retrieval_agent.py`).
- `[x]` Mock tool definitions for Lexical (Tantivy), Dense (LanceDB), and Graph (KuzuDB) searches (`retrieval_tools.py`).
- `[x]` Reciprocal Rank Fusion (RRF) implementation for combining tri-modal results.
- `[x]` Graceful tool error handling and fallback hooks.

### Live Backend Code (Packages Not Installed)
- `[x]` `src/backends/tantivy_index.py` (78 lines) — `TantivyIndex` with BM25 search.
- `[x]` `src/backends/lancedb_index.py` (85 lines) — `LanceDBIndex` with MUVERA IVF index.
- `[x]` `src/backends/kuzu_graph.py` (78 lines) — `KuzuGraph` with HippoRAG 2 schema.
- `[x]` `src/backends/neo4j_graph.py` (112 lines) — `Neo4jGraph` with Cypher queries.

### Pending Tri-Modal Work (CRITICAL)
- `[ ]` **Install DB packages** (`uv add tantivy lancedb pyarrow kuzu neo4j`) and update `pyproject.toml`.
- `[ ]` Replace mock `retrieval_tools.py` data with live connections to Tantivy, LanceDB, and KuzuDB instances.
- `[ ]` Integrate `ColQwen2.5` multi-vector routing.
- `[ ]` Hook up Graph extraction data from the Phase 6 pipeline into the KuzuDB ingestion path.

---

## Legacy Roadmap & Eval Harness (Completed)

This plan turns `retrieval_roadmap.md` into an implementation sequence for the underlying evaluation and ingestion systems.

## Progress status (session checkpoint)

Last updated: 2026-05-10 (planner calibration + graph stress-testing completed)

Current milestone: **v0.3 (phase 7 — hardening)**

Completed (all phases 0–7):

- Phase 0 foundation: Canonical schema + artifact store under `data/`. Ingestion pipeline and CLI entrypoints. Package structure + smoke tests.
- Phase 1 text retrieval baseline: Page-aware chunking. BM25 + dense retrieval + hybrid fusion. Query traces and evidence bundles.
- Phase 2 evaluation harness: Eval manifest execution via CLI/API. Mode metrics, citation support, answerability and confidence reporting.
- Phase 3 inspector UI: Custom Next.js UI scaffold in `apps/web`. Document library, document detail workspace, query workbench, eval page. FastAPI backend surface in `retrieval_research/api.py`.
- Phase 4 multimodal page retrieval: Visual retrieval mode through CLI/API/UI. Baseline visual page index with OCR-independent image/layout profile signals. Optional ColPali-compatible visual backend and `int8` compression path. Visual routing integrated into planner traces, eval reports, query/eval UI diagnostics. Reproducible weak-OCR visual fixture and eval manifest builder via `scripts/build_visual_phase4_fixture.py`. Weak-OCR visual fixture validation: visual and planner page hit rate at `1.000`.
- Phase 5 planner and adaptive routing: Query classifier and rule-based route selection. Human-readable `route_explanation` in retrieval traces. Route-specific planner settings recorded in traces. Evidence consolidation with redundancy/conflict annotations in `planner_merge`. Confidence + answerability estimates added to knowledge cards and eval outputs. Planner-vs-static comparison summary. Planner merge controls support `score_max` and `route_vote`, plus optional query-overlap reranking. Default query mode is `planner`. Larger manifest templates for planner sweep benchmarking. Default planner merge optimizes for fixture MRR with `score_max` plus soft query-overlap reranking.
- Phase 6 structured knowledge layer: First graph-style retrieval path (`graph`) with chunk-neighborhood expansion. Section, entity, and reference links in addition to page/chunk neighborhood. Graph diagnostics in UI. Unresolved ambiguity notes and follow-up retrieval suggestions. `knowledge_graph.json` artifacts. Corpus graph search with cross-document traversal. Eval manifests can target `document_ids`. Planner can route multi-document corpus queries through corpus-level graph traversal. Graph extraction handles acronym definitions, quoted concepts, section aliases, numeric reference ranges, DOI/arXiv/URL references. Stress-tested on noisy OCR. Numeric range expansion. Section hierarchy aliases. 60+ OCR noise patterns.
- Phase 7 hardening: Centralized configuration system in `retrieval_research/config.py` with 30+ env-configurable settings. Structured logging module. Expanded test suite. Dockerfile with three targets (api/cli/worker). Background jobs for ingestion/indexing (file-based queue, JobStore, worker, API, CLI). Dependabot config. Error handling pass. Planner classifier calibration (vocabulary expansion + strong identifier detection). Broad visual benchmarks.

Remaining / next:

- Phase 5 follow-up: Run planner-vs-static eval comparisons on real-world mixed corpora. Monitor planner classification accuracy in production-like settings.
- Phase 6 follow-up: Validate graph extraction on real OCR output (Hybrid mode) from GLM-OCR/Gemini. Widen quality-tier corpus with actual scanned documents.
- Phase 7 follow-up: CI/CD pipeline with GitHub Actions (partially done — `ci.yml` and `cd.yml` exist).
- **Phase 8 (NEW)**: Install live DB packages (Tantivy/LanceDB/KuzuDB) and wire `src/backends/` to the active retrieval pipeline. This unblocks BIMAgent cross-repo integration (T-AGENT-3).

## Current baseline

The repository currently contains a working document parsing prototype plus a mature `retrieval_research/` package:

- `retrieval_research/`: Phases 0–7 complete. Production-grade BM25, dense, late interaction, graph, visual, and planner retrieval. FastAPI server, CLI, evaluation framework, evidence/knowledge card system, profiling, config management, job system, chunking.
- `src/backends/`: Live DB wrappers written but packages not installed. Test files wrap calls in try/except ImportError — they currently skip on missing packages.
- `retrieval_tools.py` + `retrieval_agent.py`: Mock/demo layer using `MOCK_DOCUMENTS`. Needs to be wired to `src/backends/` (or replaced with calls to `retrieval_research/`).
- `apps/web/`: Fully scaffolded Next.js 16 inspector UI with Dashboard, Documents, Query Workbench, Eval Runner pages.

## Target v0 product — SHIPPED

Build a local-first hard-document retrieval system that:
1. ✅ Ingests PDFs/images and preserves page-level provenance.
2. ✅ Normalizes OCR/refined output into a canonical document schema.
3. ✅ Chunks documents with page, section, layout, and confidence metadata.
4. ✅ Indexes chunks in lexical and dense retrieval backends.
5. ✅ Answers queries with grounded evidence and retrieval traces.
6. ✅ Evaluates retrieval and answer quality on a reproducible corpus.
7. ✅ Exposes the workflow through a practical inspector UI.

## Phase 8: Live Tri-Modal Backends (NEW)

**Goal**: Replace the mock `retrieval_tools.py` layer with real Tantivy/LanceDB/KuzuDB backends and integrate them with the orchestrator.

### Deliverables
- `uv add tantivy lancedb pyarrow kuzu neo4j` — install Python packages
- Update `pyproject.toml` to declare all 4 packages as runtime deps
- Add live integration tests that exercise real DB connections (not just structural checks)
- Update `retrieval_tools.py` to call `src/backends/*.py` instead of returning mock data
- Wire `retrieval_agent.py` to use the live backends via the RRF fusion
- Verify `pytest tests/ -v` passes with all 79+ tests, including live DB tests
- Document the live backend setup in README

### Acceptance criteria
- All 3 DB integration tests pass independently with real connections.
- `retrieval_tools.py` returns real results from indexed documents.
- RRF fusion combines results from 3 live backends.
- The BIMAgent `BIMIndexSearchSkill` can hit real endpoints.

## Near-term milestone: v1.0 (Live Tri-Modal)

Scope: Wire up real Tantivy, LanceDB, and KuzuDB; deprecate the mock layer.

Definition of done:
- `uv add tantivy lancedb pyarrow kuzu` succeeds
- `pytest tests/test_tantivy.py tests/test_lancedb.py tests/test_kuzu.py -v` all pass with real connections
- `retrieval_tools.py` calls `src/backends/*.py`
- `retrieval_agent.py` runs a query and gets fused results from 3 live backends
- BIMAgent's `BIMIndexSearchSkill` can dispatch queries to live BIMIndex endpoints
