# AORUS MASTER 16 AM6H — RAG Product Spec Q&A

基於 RAG（Retrieval-Augmented Generation）的 GIGABYTE AORUS MASTER 16 AM6H 筆電產品規格問答系統。使用本地 LLM 推理，無需雲端 API。

## Quick Start

```bash
# 1. 安裝依賴
uv sync

# 2. 下載模型 (~2.74 GB)
huggingface-cli download unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf --local-dir models/

# 3. 建立向量索引
uv run python scripts/build_index.py

# 4. 啟動互動式問答
uv run python scripts/run.py

# 5. 執行評測
uv run python scripts/run.py --evaluate
```

## 系統架構

```
User Query (中/英文)
  → Embedding (sentence-transformers, CPU)
  → FAISS Search (top-k chunks)
  → Prompt Assembly (context + query)
  → LLM Generation (llama.cpp, streaming)
  → Answer
```

## 模型選擇理由

### LLM: Qwen3.5-4B-Instruct Q4_K_M

| 考量 | 說明 |
|------|------|
| **為什麼 Qwen3.5** | 2026/03 最新模型，sub-5B 最強。中文能力遠超同級（Llama、Gemma、Phi） |
| **為什麼 4B** | 在 4GB VRAM 限制下，4B + Q4 量化（~2.74GB）是品質與大小的最佳平衡 |
| **為什麼 Q4_K_M** | K-quant 中等品質，比 Q4_0 好，比 Q5 省空間。適合消費級硬體 |
| **VRAM 使用** | 模型 ~2.74GB + KV cache ~0.3GB ≈ 3GB，符合 ≤4GB 限制 |

### Embedding: paraphrase-multilingual-MiniLM-L12-v2

- 支援 50+ 語言（含中英文），384 維向量
- 僅 ~120MB，跑在 CPU 上，不佔 VRAM
- 搭配 FAISS IndexFlatIP（正規化內積 = cosine similarity）

## 評測結果

> 待實際執行後填入

| 指標 | 結果 |
|------|------|
| Avg TTFT (首字延遲) | TBD |
| Avg TPS (生成速度) | TBD |
| 測試用例數 | 10 |

## 技術限制

- **禁止使用** LangChain / LlamaIndex，全部用純 Python 實作
- 使用 `llama-cpp-python` 作為推理引擎
- 使用 `uv` 管理套件環境
