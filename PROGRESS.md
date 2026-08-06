# Arabic Poetry RAG — Progress

## Goal
Build a semantic verse-finder for classical Arabic poetry (موزون) — a bilingual Arabic/English portfolio RAG project targeting AI/ML engineering roles, where a user describes an occasion or feeling in natural Arabic and gets back the most fitting classical verse with poet, era, and a plain-Arabic gloss of archaic vocabulary.

## Why this idea
Chosen over generic PDF-chat and compliance-bot clones because: (1) domain is genuinely underserved — people search for the right verse at weddings, condolences, graduations and currently get keyword search or group-chat guesses; (2) RAG confirmed as top 2026 portfolio anchor across independent hiring research; (3) Arabic NLP adds real depth and differentiation; (4) the asymmetric embedding split (embed corpus once locally, query at serving time on CPU) solves a real deployment constraint and is explainable in interviews. Fine-tuning was considered and rejected — a non-technical reviewer cannot distinguish a well-tuned model from a poorly-tuned one by reading a README.

## Phase roadmap
- [x] Phase 1 — Environment, dataset, and tooling setup
- [x] Phase 2 — Data profiling and cleaning decisions
- [x] Phase 3 — Embedding sanity check (voyage-4-nano on a small sample)
- [x] Phase 4 — Full corpus embed + vector store (Chroma/LanceDB)
- [x] Phase 5 — Hybrid retrieval (dense + BM25) and reranking
- [x] Phase 6 — Evaluation: 25-query golden set, recall@k harness
- [x] Phase 7 — Multi-tool orchestration layer (search_verses, analyze_meter, gloss_vocabulary)
- [x] Phase 8 — FastAPI backend (Streaming/SSE) + RTL-aware web frontend + deployment (Oracle ARM Local + OpenRouter API Dual-Toggle)
- [ ] Phase 9 — Interview-prep consolidation and mock interview

## Friction log
- Phase 1 (Jul 22) — uv not on PowerShell PATH in agent's shell; fixed by locating it at C:\Users\braaj\AppData\Local\hermes\bin\uv.exe
- Phase 2 (Jul 29) — sys.stdout.reconfigure() fails inside Jupyter (OutStream has no reconfigure); only needed when running as a .py script in PowerShell — remove that call in notebook cells
- Phase 2 (Jul 29) — Initial comment in preprocess.py incorrectly claimed that joining sadr+ajuz with a space would reunite mid-word hemistich splits (e.g. الخَل + قَ → الخلق). It does not — the space remains. Fix: Updated `raw_index` to include both the spaced version (`sadr + " " + ajuz`) and the unspaced version (`sadr + ajuz`). If it was a split word, this reunites it perfectly for BM25. If it wasn't, it creates a harmless compound token nobody will ever search for. Zero NLP overhead.
- Phase 3 (Aug 1) — suggested repo `lm-kit/voyage-4-nano-GGUF` for GGUF download; that repo does not exist. Correct repo is `jsonMartin/voyage-4-nano-gguf` (all lowercase). Additionally, that GGUF requires a separate `voyage-4-nano-linear.pt` linear projection file to recover the correct 1024-dim → 2048-dim embedding output. Decided to implement a pure numpy/zipfile loader for the .pt file to avoid torch entirely.
- Phase 3 (Aug 2) — First sanity check run called llm.embed() once per verse sequentially; KV cache leaked across calls, causing verses [2] and [3] to produce identical vectors ([2]&[3]=1.0). All pairs involving [2] or [3] were therefore meaningless. Fix: pass all verses as a list to llm.embed() in one batch call — processes them in a fresh context.
- Phase 4 (Aug 2) — Hard power loss during full corpus embedding corrupted ChromaDB's internal pickle/HNSW files (`unsupported opcode '\0'`). Because we designed `embed_corpus.py` to use deterministic SHA-256 content hashes for document IDs, the fix was simply to delete the corrupted `chromadb` folder and restart from scratch. No need for complex DB surgery; the pipeline is fully reproducible.
- Phase 4 (Aug 3) — `llama-cpp-python` crashed with `NULL pointer access` in `decode_batch` at 27% (after ~920k verses). Root cause: batch embedding an empty string (or a string of spaces that tokenizes to 0 tokens) causes the underlying C library to fail during sequence processing. Fix: added a strict filter (`len(c["text_index"].strip()) > 1`) before embedding to drop effectively empty verses.
- Phase 4 (Aug 3) — ChromaDB compaction thread crashed at 848k vectors (`Failed to apply logs to the hnsw segment writer`). Root cause: Chroma's local backend struggles to run its WAL-to-HNSW compaction under the heavy write load of GPU batch inserts at the 2.5 million scale. Fix: Swapped the vector database to LanceDB, which writes directly to an on-disk columnar format (Lance) without a fragile background compaction loop.
- Phase 6 (Aug 4) - Semantic search evaluation returned 0% Recall@5, with verses matching lexically instead of semantically. Root cause: We explicitly passed `pooling_type=2` (CLS token) to `llama-cpp-python`, but the Voyage-4-nano base model (Qwen 3) uses Mean Pooling (`pooling_type=1`). We extracted random noise instead of the semantic embedding. Fix: Updated `embed_corpus.py` and `retriever.py` to use `pooling_type=1` (Mean Pooling) and re-embedded the corpus. Semantic recall immediately jumped to expected levels, proving the vector store was healthy.
- Phase 7 (Aug 5) - The agent hallucinated gibberish poetry instead of calling the `search_verses` tool. Root cause: The script hardcoded `chat_format="qwen"`, but we loaded an `OmniCoder-Claude` model. Even after fixing the chat format, we discovered OmniCoder outputs Claude-style XML tags (`<tool_call><function=search_verses>...`) instead of the standard OpenAI JSON that `llama-cpp-python` parses natively. Fix: Because we built a native ReAct loop instead of relying on LangChain, we retained full control. We simply added a 15-line custom Regex fallback parser in `agent.py` to catch the XML, parse the arguments, and inject them seamlessly into our execution loop.
- Phase 7 (Aug 5) - Jinja template crashed with `TypeError: Can only get item pairs from a mapping` when appending the tool response. Root cause: The OmniCoder chat template does not support OpenAI's strict `role: tool` schema. Fix: Bypassed the limitation by appending the tool output as a standard `role: user` message.
- Phase 7 (Aug 5) - Agent continued to hallucinate fake poetic meters (e.g., "المتدارم") despite strict system prompts instructing it not to guess. Root cause: Modern free verse snippets in the dataset (like "والحب الجميل") had empty meter fields, and the uncensored model ignored the prompt to fill the void. Fix: Modified the `search_verses` tool to filter out fragments (`len >= 6` words), forcing the retrieval of classical Amoudi poetry. Because classical verses have complete `poem_meter` metadata in LanceDB, the agent finally received hard data and stopped hallucinating.
- Phase 8 (Aug 6) — OpenRouter toggle returned `404 No endpoints found for anthropic/claude-3.5-sonnet`. Root cause: Anthropic retired the claude-3.5-sonnet endpoint on OpenRouter; the model ID became a dead route rather than auto-upgrading. Fix: Updated the hard-coded `model=` value in `agent.py` to `anthropic/claude-sonnet-5`, the current available Claude Sonnet on OpenRouter. Lesson: pin provider-hosted model slugs as named constants so future deprecations are a one-line update, not a debugging hunt.
- Phase 8 (Aug 6) — Switching the backend toggle had no visible feedback; users toggled and kept chatting, confused why responses still came from the old model (conversation context stays server-side across model switches). Fix: Added a pill-shaped glassmorphism toast (CSS `position: fixed`, spring `cubic-bezier` entry animation, 6-second auto-dismiss) that slides in from the top and includes an inline "تحديث الآن" button wired to `location.reload()`.
- Phase 8 (Aug 6) — OpenRouter / Qwen 3.7 streamed `<think>...</think>` inline inside `delta.content` (plain text, not a separate token stream). The frontend's turn-start state machine never advanced past turn 0, so the thinking drawer received both the reasoning and the final answer, and the answer area stayed empty. Fix: Added a real-time three-state parser (`before_think → in_think → after_think`) in `agent.py` that buffers incoming chunks with a partial-tag lookahead, emits thinking content on turn 0, sends `turn_start: 1` on `</think>`, and routes the rest to the answer area on turn 1 — no frontend changes required.

## Companion docs
- `concepts.html` — 11 entries
- `interview_qa.html` — 17 questions
- `walkthrough.html` — 3 phase beats

## Mock interview weak spots
_To be filled after first live mock interview._
