import asyncio
from google.antigravity import Agent, LocalAgentConfig, types
from google.antigravity.hooks import hooks

from retrieval_tools import query_tantivy, query_lancedb, query_kuzudb, fuse_results_rrf

# ==========================================
# BIMIndex: Tri-Modal Retrieval Agent
# ==========================================

@hooks.on_session_start
async def on_start():
    print("[BIMIndex] Connecting to KuzuDB, LanceDB, and Tantivy.")

@hooks.on_tool_error
async def on_error(data: Exception):
    print(f"[BIMIndex] Retrieval tool failed: {data}. Invoking fallback retriever skill.")
    return None

async def run_retrieval_agent():
    config = LocalAgentConfig(
        capabilities=types.CapabilitiesConfig(enable_subagents=True, enable_tools=True),
        skills_paths=["./skills"],
        hooks=[on_start, on_error],
        tools=[query_tantivy, query_lancedb, query_kuzudb, fuse_results_rrf]
    )

    async with Agent(config) as agent:
        print("[BIMIndex] Tri-Modal Retrieval System Ready.")
        # Goal: Execute concurrent tri-modal search
        response = await agent.chat(
            "Execute a parallel Tri-Modal search for 'AWS AI Revenue Q3'. "
            "Use subagents to query Tantivy (Lexical), LanceDB (Dense/MUVERA), and KuzuDB (HippoRAG 2). "
            "Combine results using RRF (Reciprocal Rank Fusion)."
        )
        print(await response.text())

if __name__ == "__main__":
    asyncio.run(run_retrieval_agent())
