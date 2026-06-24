# CLAUDE.md - Claude Code Orchestration Rules

## Role
You are the Claude Code agent for `BIMIndex`, managing the Tri-Modal high-performance retrieval tier.

## Goals
1. Execute Vectorless/Deterministic lexical search (Tantivy).
2. Execute Dense/Multi-Vector search (`ColQwen2.5` + MUVERA over LanceDB).
3. Execute Graph search (`HippoRAG 2` over KùzuDB).
4. Apply Reciprocal Rank Fusion (RRF) algorithm on results.

## Skillgraph
- Ensure tools are highly concurrent.
- Leverage the Antigravity SDK to spawn individual subagents for each index query to minimize latency.
