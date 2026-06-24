---
name: tri-modal-retrieval
description: "Executes parallel searches across Tantivy (lexical), LanceDB (dense), and KuzuDB (graph), and fuses results via RRF."
---

# Tri-Modal Retrieval Skill

This skill enables the agent to orchestrate the BIMIndex retrieval tier by using parallel subagents.

## Available Databases
1. **Tantivy**: BM25 lexical inverted index.
2. **LanceDB**: Dense multi-vector search (MUVERA).
3. **KuzuDB**: HippoRAG 2 Graph search with Personalized PageRank.

## Workflow
1. When asked to perform a tri-modal search, launch subagents or utilize the built-in python tools `query_tantivy`, `query_lancedb`, and `query_kuzudb` to search all three databases in parallel.
2. Collect the result lists from all three searches.
3. Pass all three lists into the `fuse_results_rrf` tool to combine and rank them.
4. Return the fused and ranked documents to the user as the final answer.
