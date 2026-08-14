"""
OpenRouter multi-key failover.

OpenRouter is an aggregator in front of many upstream providers. A key that is
not bring-your-own-provider draws on the provider's SHARED POOL, and when that
pool's quota is exhausted the upstream answers with HTTP 429 carrying
provider_error_code 'insufficient_quota' and limit_source
'upstream_provider_shared_pool'. That condition is per-key: two keys can be
dead while a third works fine, so the correct mitigation is to retry the same
request with a different key — transparently, so the end user never sees the
error.

Rules:
  - Keys are read from every env var whose name starts with "opentouter_api"
    (the project's legacy spelling), e.g. opentouter_api / opentouter_api2 /
    opentouter_api3. They are tried in sorted name order.
  - ONLY a 429 triggers rotation. A 400 or 5xx is not a key problem and will
    not succeed with a different key, so it is propagated unchanged.
  - The index of the last key that succeeded is remembered in-process, so a
    healthy key is tried first and a dead key costs exactly one failed request
    per outage, not one per request.
  - Both the sync embedding path and the async streaming chat path use this
    module, so a quota hit can't break either half of the pipeline.
"""

import os
import threading

BASE_URL = "https://openrouter.ai/api/v1"

_lock = threading.Lock()
_key_index = 0


def openrouter_keys(api_key: str = None) -> list:
    """All configured keys, in sorted env-var order. Falls back to a single
    explicitly-passed key when no env vars are set (local dev)."""
    keys = [
        os.environ[k].strip().strip("'\"")
        for k in sorted(os.environ)
        if k.upper().startswith("OPENTOUTER_API") and os.environ[k].strip()
    ]
    if not keys and api_key:
        keys = [api_key]
    return keys


def is_quota_error(exc) -> bool:
    """Only 429 responses indicate a per-key quota/rate problem worth rotating
    over. Anything else (400, 401, 5xx, connection errors) is not key-scoped."""
    return getattr(exc, "status_code", None) == 429


def _current_index(keys):
    with _lock:
        return _key_index % len(keys)


def _mark_ok(idx):
    global _key_index
    with _lock:
        _key_index = idx


def embed_with_failover(model: str, input_texts, api_key: str = None):
    """OpenAI-compatible embeddings with transparent key rotation on 429."""
    import openai

    keys = openrouter_keys(api_key)
    if not keys:
        raise RuntimeError("No OpenRouter API key configured (opentouter_api* env vars).")

    n = len(keys)
    start = _current_index(keys)
    last_error = None
    for i in range(n):
        idx = (start + i) % n
        try:
            client = openai.OpenAI(base_url=BASE_URL, api_key=keys[idx])
            response = client.embeddings.create(model=model, input=input_texts)
            _mark_ok(idx)
            return response
        except openai.APIStatusError as e:
            if is_quota_error(e):
                last_error = e
                print(f"[openrouter_failover] key {idx + 1}/{n} rate-limited (429); rotating to next key.")
                continue
            raise
    raise last_error


async def async_chat_stream(model: str, messages: list, tools=None,
                            api_key: str = None, **kwargs):
    """Async generator of streaming chat chunks with transparent key rotation.

    Yields raw SDK chunks; a 429 raised on creation OR mid-stream triggers a
    full retry of the request with the next key. Usage:

        stream = async_chat_stream(model="qwen/qwen3.7-flash", messages=..., tools=...)
        async for chunk in stream:
            ...
    """
    import openai

    keys = openrouter_keys(api_key)
    if not keys:
        raise RuntimeError("No OpenRouter API key configured (opentouter_api* env vars).")

    n = len(keys)
    start = _current_index(keys)
    last_error = None
    for i in range(n):
        idx = (start + i) % n
        try:
            client = openai.AsyncOpenAI(base_url=BASE_URL, api_key=keys[idx])
            kwargs.setdefault("model", model)
            kwargs.setdefault("messages", messages)
            kwargs.setdefault("stream", True)
            if tools is not None:
                kwargs["tools"] = tools
            stream = await client.chat.completions.create(**kwargs)
            async for chunk in stream:
                yield chunk
            _mark_ok(idx)
            return
        except openai.APIStatusError as e:
            if is_quota_error(e):
                last_error = e
                print(f"[openrouter_failover] key {idx + 1}/{n} rate-limited (429); rotating to next key.")
                continue
            raise
    raise last_error
