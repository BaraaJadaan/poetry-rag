import json
import sys
import os
import asyncio
import re
from retriever import HybridRetriever

# ── Config ────────────────────────────────────────────────────────────────────
# You can update this to point exactly to where your Qwen 3.5 9B GGUF is stored.
# e.g. "./models/qwen-3.5-9b-instruct-q4_k_m.gguf"
AGENT_MODEL_PATH = os.getenv("AGENT_MODEL_PATH", "./models/Qwen3.5-2B-MTP-Q4_K_M.gguf")

print("Initializing Retriever...")
retriever = HybridRetriever()

llm = None
if os.getenv("CLOUD_DEPLOYMENT", "false").lower() != "true":
    print(f"\nLoading Agent Model ({AGENT_MODEL_PATH})...")
    try:
        from llama_cpp import Llama

        # Load the local model only in local mode. Cloud mode uses OpenRouter and
        # should not require the platform-specific llama-cpp-python package.
        llm = Llama(
            model_path=AGENT_MODEL_PATH,
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False
        )
    except Exception as e:
        print(f"Warning: Could not load Agent Model at {AGENT_MODEL_PATH}: {e}")
        print("Install llama-cpp-python separately and provide a local GGUF file for local mode.")


# ── Tool Definitions (Python Functions) ───────────────────────────────────────

# Free-verse / fragment rows in the cloud table have one empty hemistich
# (text_display starts with " ***" or ends with "*** ") or are single half-lines
# without a separator at all. They embed into a tight cluster that attracts short
# abstract queries and buries real verses, so they are excluded from retrieval.
# TRIM() is required because stored rows carry trailing whitespace after the
# separator, which would otherwise defeat exact LIKE patterns.
FULL_VERSE_FILTER = (
    "TRIM(text_display) LIKE '%***%' "
    "AND TRIM(text_display) NOT LIKE '***%' "
    "AND TRIM(text_display) NOT LIKE '%***'"
)

def _verse_word_count(text) -> int:
    """Count real words: strips '***', punctuation, and standalone diacritics
    so short free-verse lines can't masquerade as full classical verses."""
    cleaned = re.sub(r"[^\w\u0600-\u06FF\s]", " ", str(text).replace("*", " "))
    return len([w for w in cleaned.split() if w.strip()])

def search_verses(query: str, limit: int = 3) -> str:
    """Searches the vector database for poetry verses matching the query."""
    print(f"  [Tool Execution] search_verses(query='{query}', limit={limit})")
    # Fetch a wide window to account for filtering, then keep only full verses.
    results = retriever.search_hybrid(query, limit=limit * 20, filter_sql=FULL_VERSE_FILTER)
    if results.empty:
        return "No verses found matching the query."
    
    # Filter out short fragments (free verse usually has very short lines, classical Amoudi has 6+ words)
    results = results[results['text_display'].astype(str).apply(lambda s: _verse_word_count(s) >= 6)]
    results = results.head(limit)
    
    if results.empty:
        return "Found some verses, but they were too short to be full classical verses. Try a different query."
    
    formatted = []
    for i, row in results.iterrows():
        display = row.get("text_display") or row.get("text_index")
        poet = row.get("poet_name", "Unknown")
        meter = row.get("poem_meter")
        
        verse_str = f"Verse: {display} | Poet: {poet}"
        if meter and str(meter).strip() and str(meter).strip().lower() != "nan":
            verse_str += f" | Meter: {meter}"
            
        formatted.append(verse_str)
    
    return "\n".join(formatted)

def analyze_meter(verse: str) -> str:
    """Analyzes the poetic meter (بحر) of a given verse."""
    print(f"  [Tool Execution] analyze_meter(verse='{verse}')")
    # In a full implementation, this might call a dedicated ML model or deterministic algorithm (like pyarabic/tashaphyne).
    # For now, we return a mock or ask the LLM to deduce it.
    return "Meter analysis requires a dedicated Arabic NLP library. (Placeholder)"

def gloss_vocabulary(word: str) -> str:
    """Provides the classical Arabic definition of an archaic word."""
    print(f"  [Tool Execution] gloss_vocabulary(word='{word}')")
    # Similarly, this could query an Almaany/Lisan-Al-Arab SQL database.
    return f"Definition lookup for '{word}' (Placeholder: meaning depends on context)."


# ── The Tool Map & Schema ─────────────────────────────────────────────────────

# Map tool names to the actual python functions
AVAILABLE_TOOLS = {
    "search_verses": search_verses,
    "analyze_meter": analyze_meter,
    "gloss_vocabulary": gloss_vocabulary
}

# The JSON schema we pass to the LLM so it knows what tools exist
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_verses",
            "description": "Searches for Arabic poetry verses based on a concept, theme, or feeling.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query in Arabic (e.g. 'الفخر والشجاعة' or 'الشوق')."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of verses to return (default 3)."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_meter",
            "description": "Analyzes the poetic meter (العروض/البحر) of a specific Arabic verse.",
            "parameters": {
                "type": "object",
                "properties": {
                    "verse": {
                        "type": "string",
                        "description": "The Arabic verse to analyze."
                    }
                },
                "required": ["verse"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "gloss_vocabulary",
            "description": "Looks up the definition of an archaic or difficult classical Arabic word.",
            "parameters": {
                "type": "object",
                "properties": {
                    "word": {
                        "type": "string",
                        "description": "The classical Arabic word to define."
                    }
                },
                "required": ["word"]
            }
        }
    }
]


async def generate_response_stream(messages, use_openrouter=False, api_key=None):
    if messages and messages[0].get("role") != "system":
        system_prompt = (
            "أنت باحث متخصص في الشعر العربي الكلاسيكي. قواعدك الصارمة:\n"
            "\n"
            "1. لا تقتبس أبياتاً من ذاكرتك أبداً — يجب استخدام أداة search_verses أولاً دائماً.\n"
            "2. عند البحث: اكتب جملة طبيعية كاملة بالعربية تصف الموقف أو الشعور أو المعنى الذي يبحث عنه المستخدم، مثال صحيح: 'أجلس وحدي في الليل وأشعر بالحزن الشديد' — ولا تبحث عن أبيات بعينها أو شعراء بعينهم.\n"
            "3. لا تقدم للمستخدم أبداً أي أبيات غير مكتملة أو مجتزأة (يجب أن يكون البيت مكتملاً تماماً).\n"
            "4. كل تفكيرك وتحليلك يجب أن يكون داخل وسوم <think>...</think> فقط، باللغة العربية حصراً.\n"
            "5. بعد وسوم <think> مباشرة، اكتب إجابتك النهائية فقط باللغة العربية. يجب عليك اقتباس الأبيات كما وردت نصاً مع ذكر اسم الشاعر، ولا تقم بتلخيصها.\n"
            "6. إذا كان المعنى المطلوب لا يكتمل ببيت واحد، فاقتبس أكثر من بيت واحد (2-3 أبيات) مكتملة حتى يكتمل المعنى.\n"
            "7. يمكنك استدعاء أداة search_verses أكثر من مرة بأسئلة بحث مختلفة إذا كانت النتائج غير كافية، ثم اختر الأفضل منها.\n"
        )
        if not use_openrouter:
            system_prompt += (
                "5. صيغة استدعاء الأداة:\n"
                "<tool_call><function=search_verses><parameter=query>كلمات المعنى هنا</parameter></function></tool_call>\n"
                "\n"
                "مثال صحيح: البحث عن شعر الشوق → query='الشوق الفراق الحنين'\n"
                "مثال خاطئ: البحث عن 'قصيدة المتنبي المشهورة' — ممنوع."
            )
        messages.insert(0, {"role": "system", "content": system_prompt})
        
    turn_index = 0
    while True:
        full_content = ""
        tool_calls = []
        yield f"data: {json.dumps({'turn_start': turn_index})}\n\n"
        
        if use_openrouter:
            if not api_key:
                yield f"data: {json.dumps({'error': 'OpenRouter API key missing in .env'})}\n\n"
                return

            # On turns after the first, the loop already emitted turn_start: turn_index (>=1).
            # Re-emit turn_start: 0 so Qwen's second-pass <think> block routes to the
            # thinking drawer (currentTurn=0) rather than the answer area (currentTurn>=1).
            if turn_index > 0:
                yield f"data: {json.dumps({'turn_start': 0})}\n\n"

            from openrouter_failover import openrouter_keys, async_chat_stream
            if not openrouter_keys(api_key):
                yield f"data: {json.dumps({'error': 'OpenRouter API key missing in .env'})}\n\n"
                return

            try:
                # We restore tools=TOOLS_SCHEMA. Since we removed the XML tool prompt
                # instructions for OpenRouter, Qwen will use its native tool calling
                # capabilities properly, triggering delta.tool_calls. async_chat_stream
                # retries the request across OpenRouter API keys on 429, so shared-pool
                # quota hits never reach the user.
                stream = async_chat_stream(
                    model="qwen/qwen3.7-flash",
                    messages=messages,
                    tools=TOOLS_SCHEMA,
                    api_key=api_key,
                )

                # ── Real-time think-tag splitter ────────────────────────────────────
                # States:  before_think -> in_think -> after_think
                #   in_think content    -> turn 0 SSE (thinking drawer)
                #   after_think content -> turn 1 SSE (answer area)
                # Policy: content is REASONING only after the think opener is seen
                # (drawer until the close tag). Content that never opens the tag is
                # the ANSWER (turn 1) -- Qwen's post-tool pass often skips the tags
                # entirely, and routing that to the drawer hides the final answer.
                OR_THINK_OPEN  = '<think>'
                OR_THINK_CLOSE = '</think>'
                or_state = "before_think"
                or_buf   = ""
                answer_started = False

                async for chunk in stream:
                    delta = chunk.choices[0].delta

                    # Handle native tool_calls
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            if len(tool_calls) <= tc.index:
                                tool_calls.append({"id": tc.id, "type": "function",
                                                   "function": {"name": tc.function.name or "", "arguments": ""}})
                            if tc.function.arguments:
                                tool_calls[tc.index]["function"]["arguments"] += tc.function.arguments

                    if not delta.content:
                        continue

                    or_buf += delta.content
                    full_content += delta.content

                    while or_buf:
                        if or_state == "before_think":
                            idx = or_buf.find(OR_THINK_OPEN)
                            if idx == -1:
                                safe_len = max(0, len(or_buf) - len(OR_THINK_OPEN) + 1)
                                if safe_len:
                                    # No think opener yet -- this is the ANSWER.
                                    if not answer_started:
                                        yield f"data: {json.dumps({'turn_start': 1})}\n\n"
                                        answer_started = True
                                    yield f"data: {json.dumps({'content': or_buf[:safe_len]})}\n\n"
                                    or_buf = or_buf[safe_len:]
                                break
                            else:
                                if idx > 0:
                                    # Untagged preamble before the opener: answer.
                                    if not answer_started:
                                        yield f"data: {json.dumps({'turn_start': 1})}\n\n"
                                        answer_started = True
                                    yield f"data: {json.dumps({'content': or_buf[:idx]})}\n\n"
                                if answer_started:
                                    # Re-sync: tagged reasoning that follows a
                                    # preamble belongs in the thinking drawer.
                                    yield f"data: {json.dumps({'turn_start': 0})}\n\n"
                                or_buf = or_buf[idx + len(OR_THINK_OPEN):]
                                or_state = "in_think"
                                # turn_start: 0 is already in effect (loop top or re-emit above)

                        elif or_state == "in_think":
                            idx = or_buf.find(OR_THINK_CLOSE)
                            if idx == -1:
                                safe_len = max(0, len(or_buf) - len(OR_THINK_CLOSE) + 1)
                                if safe_len:
                                    yield f"data: {json.dumps({'content': or_buf[:safe_len]})}\n\n"
                                    or_buf = or_buf[safe_len:]
                                break
                            else:
                                if idx > 0:
                                    yield f"data: {json.dumps({'content': or_buf[:idx]})}\n\n"
                                or_buf = or_buf[idx + len(OR_THINK_CLOSE):]
                                or_state = "after_think"
                                yield f"data: {json.dumps({'turn_start': 1})}\n\n"

                        elif or_state == "after_think":
                            yield f"data: {json.dumps({'content': or_buf})}\n\n"
                            or_buf = ""
                            break

                    await asyncio.sleep(0)

                # Flush any tail: if the stream ended before a close tag, the
                # buffered text belongs to the state we are in. Untagged tail
                # (before_think) is the answer; tagged tail is reasoning.
                if or_buf.strip():
                    if or_state == "before_think" and not answer_started:
                        yield f"data: {json.dumps({'turn_start': 1})}\n\n"
                    yield f"data: {json.dumps({'content': or_buf})}\n\n"

                print(f"[agent] pass {turn_index} done: state={or_state}, content_chars={len(full_content)}, tool_calls={len(tool_calls)}")
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                return

        else:
            if not llm:
                yield f"data: {json.dumps({'error': 'Local model not loaded.'})}\n\n"
                return
            
            # Local llama.cpp streaming
            # Local llama.cpp streaming
            response = llm.create_chat_completion(
                messages=messages,
                stream=True,
                temperature=0.3,
                top_p=0.8,
                top_k=20,
                min_p=0.05,
                frequency_penalty=0.7,
                presence_penalty=0.7,
                repeat_penalty=1.12,
                stop=["</tool_call>", "</function>"]
            )
            for chunk in response:
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta:
                    text = delta["content"]
                    full_content += text
                    yield f"data: {json.dumps({'content': text})}\n\n"
                await asyncio.sleep(0.01)
                
            # Parse XML fallback for OmniCoder
            if not tool_calls and "<function=" in full_content:
                func_match = re.search(r"<function=([^>]+)>", full_content)
                if func_match:
                    func_name = func_match.group(1).strip()
                    param_blocks = re.findall(r"<parameter=([^>]+)>(.*?)</parameter>", full_content, re.DOTALL)
                    args = {}
                    for p_name, p_val in param_blocks:
                        val = p_val.strip()
                        if val.isdigit(): val = int(val)
                        args[p_name] = val
                    tool_calls = [{
                        "id": "call_xml_fallback",
                        "type": "function",
                        "function": {
                            "name": func_name,
                            "arguments": json.dumps(args)
                        }
                    }]
                    
        messages.append({"role": "assistant", "content": full_content or ""})
        turn_index += 1

        if tool_calls:
            for tool_call in tool_calls:
                func_name = tool_call["function"]["name"]
                try:
                    func_args = json.loads(tool_call["function"]["arguments"])
                except:
                    func_args = {}
                
                yield f"data: {json.dumps({'tool_executing': func_name, 'args': func_args})}\n\n"
                
                if func_name in AVAILABLE_TOOLS:
                    func = AVAILABLE_TOOLS[func_name]
                    tool_result = func(**func_args)
                else:
                    tool_result = f"Error: Tool {func_name} not found."
                
                messages.append({
                    "role": "user",
                    "content": f"[نتيجة الأداة '{func_name}':\n{tool_result}\nاكتب إجابتك النهائية الآن باللغة العربية فقط. قم باقتباس الأبيات نصياً مع ذكر اسم الشاعر كما وردت في النتيجة دون أي مقدمات، ولا تقم بتلخيصها. تنبيه: لا تقم باستدعاء الأداة مرة أخرى، اكتفِ بتقديم الإجابة.]"
                })
            continue
        else:
            yield f"data: {json.dumps({'done': True})}\n\n"
            break

if __name__ == "__main__":
    print("Agent is now run via the FastAPI server. Please run 'uvicorn app:app --reload'")
