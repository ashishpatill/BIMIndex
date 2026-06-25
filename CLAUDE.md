# CLAUDE.md - Claude Code Orchestration Rules

## Role
You are the Claude Code agent for `BIMIndex`, managing the Tri-Modal high-performance retrieval tier.

## Goals
1. Execute Vectorless/Deterministic lexical search (Tantivy).
2. Execute Dense/Multi-Vector search (`ColQwen2.5` + MUVERA over LanceDB).
3. Execute Graph search (`HippoRAG 2` over KùzuDB).
4. Apply Reciprocal Rank Fusion (RRF) algorithm on results.

## Model Routing
**Always read `ROUTING.md` before starting any task.** The 3 live DB integrations (Tantivy, LanceDB, KuzuDB) are flash→pro tier and can run in parallel. Research features are pro tier. Standard SDK wiring is flash tier. The 79-test suite ensures verification.

## Task List
**Read `TASKS.md` for the full list of remaining work with detailed specs and implementation steps.**
Planned execution: T-INDEX-4 (CI/CD) → T-INDEX-1, T-INDEX-2, T-INDEX-3 (3× DBs — parallel Qwen3 Coder Plus) → T-INDEX-5 (PaddleOCR) → T-INDEX-6 (Qwen2.5-VL) → T-INDEX-10 (Neo4j) → T-INDEX-7 (SPLADE) → T-INDEX-8 (reranking) → T-INDEX-9 (PageIndex).

## Skillgraph
- Ensure tools are highly concurrent.
- Leverage the Antigravity SDK to spawn individual subagents for each index query to minimize latency.
