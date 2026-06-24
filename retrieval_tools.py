import json

# Mock Data
MOCK_DOCUMENTS = [
    {
        "id": "doc1",
        "title": "AWS Q3 Earnings Report",
        "snippet": "AWS announced a record $23B in revenue for Q3, largely driven by new generative AI services.",
        "score": 0.0,
        "source": "Tantivy"
    },
    {
        "id": "doc2",
        "title": "Cloud Market Share 2024",
        "snippet": "Amazon Web Services retains the top spot, citing strong adoption of Amazon Bedrock and AI tools contributing significantly to their top-line revenue.",
        "score": 0.0,
        "source": "Tantivy"
    },
    {
        "id": "doc3",
        "title": "Generative AI Impact on Cloud",
        "snippet": "AI represents a multi-billion dollar run rate for AWS. Q3 numbers reflect this massive shift in enterprise spending.",
        "score": 0.0,
        "source": "LanceDB"
    },
    {
        "id": "doc4",
        "title": "AWS and Anthropic Partnership",
        "snippet": "The strategic investment in Anthropic has accelerated AI model deployment on AWS, directly impacting Q3 revenue margins.",
        "score": 0.0,
        "source": "LanceDB"
    },
    {
        "id": "doc5",
        "title": "Earnings Call Transcript",
        "snippet": "CEO mentioned: 'Our generative AI business is growing at an unprecedented rate, adding billions to our Q3 revenue.'",
        "score": 0.0,
        "source": "KuzuDB"
    },
    {
        "id": "doc6",
        "title": "Entity Relation: AWS -> Generative AI",
        "snippet": "Graph traversal shows strong linkage between AWS Q3 financials and newly launched Generative AI compute instances.",
        "score": 0.0,
        "source": "KuzuDB"
    }
]

def query_tantivy(query: str) -> list[dict]:
    """
    Executes a vectorless, deterministic lexical search using BM25 over the Tantivy index.
    
    Args:
        query: The search string (e.g. 'AWS AI Revenue Q3').
        
    Returns:
        A list of retrieved document chunks matching the lexical query.
    """
    print(f"[Tantivy Subagent] Executing lexical search for: '{query}'")
    # Return mock results relevant to Tantivy
    results = [doc for doc in MOCK_DOCUMENTS if doc["source"] == "Tantivy"]
    for i, res in enumerate(results):
        res["score"] = 10.0 - i * 1.5  # Mock BM25 scores
    return results

def query_lancedb(query: str) -> list[dict]:
    """
    Executes a dense, multi-vector search (MUVERA) over late-chunking embeddings in LanceDB.
    
    Args:
        query: The search string.
        
    Returns:
        A list of retrieved document chunks matching the dense vector query.
    """
    print(f"[LanceDB Subagent] Executing dense search for: '{query}'")
    # Return mock results relevant to LanceDB
    results = [doc for doc in MOCK_DOCUMENTS if doc["source"] == "LanceDB"]
    for i, res in enumerate(results):
        res["score"] = 0.95 - i * 0.05  # Mock cosine similarity scores
    return results

def query_kuzudb(query: str) -> list[dict]:
    """
    Executes Graph RAG search employing HippoRAG 2 with Personalized PageRank (PPR) over passage nodes in KuzuDB.
    
    Args:
        query: The search string.
        
    Returns:
        A list of retrieved document chunks matching the graph structure.
    """
    print(f"[KuzuDB Subagent] Executing graph RAG search for: '{query}'")
    # Return mock results relevant to KuzuDB
    results = [doc for doc in MOCK_DOCUMENTS if doc["source"] == "KuzuDB"]
    for i, res in enumerate(results):
        res["score"] = 0.88 - i * 0.1  # Mock PPR scores
    return results

def fuse_results_rrf(tantivy_results: list, lancedb_results: list, kuzudb_results: list, k: int = 60) -> list[dict]:
    """
    Combines retrieved contexts from multiple sources via Reciprocal Rank Fusion (RRF).
    
    Args:
        tantivy_results: List of dicts retrieved by Tantivy.
        lancedb_results: List of dicts retrieved by LanceDB.
        kuzudb_results: List of dicts retrieved by KuzuDB.
        k: The RRF constant (default 60).
        
    Returns:
        A combined, deduped, and re-ranked list of documents.
    """
    print(f"[Fusion Subagent] Fusing {len(tantivy_results)} Tantivy, {len(lancedb_results)} LanceDB, and {len(kuzudb_results)} KuzuDB results with RRF (k={k}).")
    
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
    
    # Sort by RRF score descending
    ranked_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    
    final_results = []
    for doc_id, rrf_score in ranked_docs:
        doc_copy = docs[doc_id].copy()
        doc_copy["rrf_score"] = rrf_score
        final_results.append(doc_copy)
        
    return final_results
