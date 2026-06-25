# BIMIndex — Model Routing Guide

## RouteFusion Offload Scoring

```
offload_score = (blast_radius × 3 + ambiguity × 2 + quality_sensitivity × 2) / verification_strength
```

| Axis | 1 | 2 | 3 |
|------|---|---|---|
| blast_radius | local | module | system |
| ambiguity | low | medium | high |
| quality_sensitivity | low | medium | high |
| verification_strength | weak | moderate | strong |

| Score | Tier | Pattern |
|-------|------|---------|
| < 3 | free | Free model |
| 3–5 | flash | DeepSeek V4 Flash alone |
| 5–7 | flash→pro | Flash writes, Pro verifies |
| > 7 | pro | DeepSeek V4 Pro from scratch |

## Provider Setup

| Provider | Access | Models |
|----------|--------|--------|
| **OpenCode Zen** | Built-in (free) | `opencode/deepseek-v4-flash-free` |
| **OpenRouter** | Connected via opencode | `openrouter/...` |
| **Local Ollama** | `ollama pull` | `nanbeige4.1-3b` |

> OpenRouter key is configured in opencode. If API calls fail, run `/connect` → OpenRouter.

## Available Models (Use These Exact Model IDs)

| Model | Model ID | Cost/M | Ctx | License | Best at |
|-------|----------|--------|-----|---------|---------|
| DeepSeek V4 Flash *(free)* | `opencode/deepseek-v4-flash-free` | $0 | 1M | MIT | Free tier: trivial/docs |
| DeepSeek V4 Flash *(paid)* | `openrouter/deepseek/deepseek-v4-flash` | $0.09 | 1M | MIT | Bounded implementation, DB SDK |
| DeepSeek V4 Pro | `openrouter/deepseek/deepseek-v4-pro` | $0.435 | 1M | MIT | Planning, debugging, retrieval arch |
| Qwen3 Coder Plus | `openrouter/qwen/qwen3-coder-plus` | $0.65 | 1M | Apache 2.0 | Complex coding (I.90), TC (TC.92) |
| GLM-5.2 | `openrouter/z-ai/glm-5.2` | $0.15 | 1M | MIT | Cross-repo, 1M context reading |
| Qwen3.7 Plus | `openrouter/qwen/qwen3.7-plus` | $0.32 | 1M | Apache 2.0 | Stronger Pro alternative |
| MiMo V2.5 Pro | `openrouter/xiaomi/mimo-v2.5-pro` | $0.435 | 1M | Proprietary | Terminal/debug loops |
| Nex N2 Pro | `openrouter/nex-agi/nex-n2-pro` | $0.50 | 262K | Proprietary | Fast impl, Flash fallback |
| Phi-4 | `openrouter/microsoft/phi-4` | $0.07 | 16K | MIT | Small tasks, test generation |
| Gemini 3.5 Flash | `openrouter/google/gemini-3.5-flash` | $0.0375 | 1M | Proprietary | Cheap trivial throughput |
| Nanbeige 4.1 3B | *(local Ollama)* | $0 | — | Apache 2.0 | Private, sensitive material |

## Provider Selection Rules

| Scenario | Use |
|----------|-----|
| Trivial task, no sensitivity | `opencode/deepseek-v4-flash-free` (Zen free) |
| SDK wiring, DB integration | `openrouter/deepseek/deepseek-v4-flash` (OpenRouter) |
| Retrieval research, algorithms | `openrouter/deepseek/deepseek-v4-pro` (OpenRouter) |
| 3 DB integrations (parallel) | 3× `openrouter/qwen/qwen3-coder-plus` parallel + 1× Pro verify |
| Exposed credentials / secrets | `nanbeige4.1-3b` (local — never to API) |

## Scaffolding

### Confirm Model Access
```bash
opencode run -m openrouter/deepseek/deepseek-v4-flash "test"
opencode run -m openrouter/deepseek/deepseek-v4-pro "test"
```

### Run Tests to Verify
```bash
PYTHONPATH=. pytest tests/ -v    # 79 tests verify correctness
```

## Task-to-Model Routing

| Task | Offload | Tier | Model | Notes |
|------|---------|------|-------|-------|
| **Live Tantivy integration** | 7.0 | flash→pro | Qwen3 Coder Plus write, V4 Pro verify | Complex SDK → Python binding. Run with 79 tests. |
| **Live LanceDB/MUVERA** | 7.0 | flash→pro | Qwen3 Coder Plus write, V4 Pro verify | Same pattern. Parallelize with Tantivy + KuzuDB. |
| **Live KuzuDB/HippoRAG 2** | 7.0 | flash→pro | Qwen3 Coder Plus write, V4 Pro verify | Same pattern. 3 DBs in parallel agents. |
| CI/CD pipeline | 7.0 | flash | DeepSeek V4 Flash | GitHub Actions YAML. Override: Flash only. |
| PaddleOCR | 4.7 | flash | DeepSeek V4 Flash | Standard OCR integration |
| Qwen2.5-VL pipeline | 6.0 | flash→pro | Qwen3 Coder Plus write, V4 Pro verify | Vision-language integration |
| SPLADE sparse search | 7.0 | flash→pro | DeepSeek V4 Pro write + verify | Research-grade NLP — debug tier |
| Cross-encoder reranking | 7.0 | flash→pro | DeepSeek V4 Pro write + verify | Same as SPLADE |
| PageIndex / vectorless | 8.0 | pro | DeepSeek V4 Pro | Novel retrieval design — architecture first |
| Neo4j graph layer | 4.7 | flash | DeepSeek V4 Flash | Standard graph DB client |

## Decision Matrix

| If you need to... | Use this model | Why |
|-------------------|----------------|-----|
| Design a retrieval mode | DeepSeek V4 Pro | P.95 for architecture |
| Integrate a vector DB | Qwen3 Coder Plus (write) + Pro (verify) | Complex SDK, 79 tests verify |
| Wire a standard DB client | DeepSeek V4 Flash | Bounded, well-understood |
| Research a retrieval algorithm | DeepSeek V4 Pro | D.92 for research/debug |
| Set up CI/CD | DeepSeek V4 Flash | Standard YAML |
| Run terminal experiments | MiMo V2.5 Pro | TC.90 for shell loops |
| Handle secure/infra code | Nanbeige-3B (local) | Private, no data leakage |
| Review retrieval changes | DeepSeek V4 Pro | R.88, test suite must pass |
