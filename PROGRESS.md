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
- [ ] Phase 7 — Multi-tool orchestration layer (search_verses, analyze_meter, gloss_vocabulary)
- [ ] Phase 8 — FastAPI backend + RTL-aware web frontend + deployment
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
- Phase 6 (Aug 4) - Semantic search evaluation returned 0% Recall@5, with verses matching lexically instead of semantically. Root cause: We explicitly passed `pooling_type=2` (CLS token) to `llama-cpp-python`, but the Voyage-4-nano base model (Qwen 3) uses Mean Pooling (`pooling_type=1`). We extracted random noise instead of the semantic embedding. Fix: Update `embed_corpus.py` and `retriever.py` to use `pooling_type=1` and re-embed the corpus in the next session.

## Companion docs
- `concepts.html` — 10 entries
- `interview_qa.html` — 15 questions
- `walkthrough.html` — 3 phase beats

## Mock interview weak spots
_To be filled after first live mock interview._
