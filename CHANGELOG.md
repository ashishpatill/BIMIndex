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
