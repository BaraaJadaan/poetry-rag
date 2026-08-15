const messagesDiv = document.getElementById('messages');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const stopBtn = document.getElementById('stop-btn');
const suggestionsBar = document.getElementById('suggestions');

const configuredApiBase = window.POETRY_RAG_CONFIG?.apiBaseUrl || '';
const API_BASE_URL = configuredApiBase.replace(/\/+$/, '');
const apiUrl = (path) => `${API_BASE_URL}${path}`;

// Deployed on Pages (API_BASE_URL set) → always OpenRouter; same-origin local dev → local model.
const useOpenRouter = API_BASE_URL.length > 0;

let conversation = [];
let currentAbortController = null;

const CACHE_KEY = 'poetry_rag_chats';    // last-10 Q&A pairs

marked.setOptions({ breaks: true, gfm: true });

// ── Quick suggestions ─────────────────────────────────────────────────────────
const SUGGESTIONS = [
    { label: 'عزة النفس والكرامة', prompt: 'أبيات عن عزة النفس والكرامة والترفع عن الدنايا' },
    { label: 'الشوق والغربة', prompt: 'شعر عن الشوق والحنين وألم الغربة والبعد عن الوطن' },
    { label: 'فخر وشجاعة المتنبي', prompt: 'أبيات للمتنبي في الفخر والشجاعة وركوب الخيل' },
    { label: 'الحكمة وتجارب الأيام', prompt: 'أجمل ما قيل في الحكمة وتجارب الدهر والأيام' },
    { label: 'الصبر عند الشدائد', prompt: 'أبيات عن الصبر وحسن الظن وتجاوز الشدائد والمحن' },
    { label: 'الصداقة والوفاء', prompt: 'شعر عن الصداقة الصادقة ووفاء الأصحاب وإخلاصهم' },
    { label: 'وصف الطبيعة والربيع', prompt: 'أبيات بديعة في وصف جمال الطبيعة والرياض والأزهار' },
    { label: 'مرور العمر والشباب', prompt: 'أبيات عن مرور العمر والتحسر على أيام الشباب' },
    { label: 'الغزل العفيف', prompt: 'أجمل أبيات الغزل العفيف والرقة والوجد' }
];

function initSuggestions() {
    if (!suggestionsBar) return;
    suggestionsBar.innerHTML = '';

    let isDragging = false;
    let startX = 0;
    let scrollStart = 0;
    let dragThresholdPassed = false;

    SUGGESTIONS.forEach(item => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'suggestion-chip';
        btn.textContent = item.label;
        btn.title = item.prompt;

        btn.addEventListener('click', (e) => {
            if (dragThresholdPassed) return; // Prevent click if user dragged
            if (currentAbortController) return; // Ignore while streaming
            userInput.value = item.prompt;
            userInput.style.height = 'auto';
            userInput.style.height = Math.max(44, Math.min(160, userInput.scrollHeight)) + 'px';
            sendMessage();
        });

        suggestionsBar.appendChild(btn);
    });

    // ── Mouse Drag-to-Scroll for PC ──────────────────────────────────────────
    suggestionsBar.addEventListener('mousedown', (e) => {
        isDragging = true;
        dragThresholdPassed = false;
        startX = e.pageX;
        scrollStart = suggestionsBar.scrollLeft;
        suggestionsBar.classList.add('dragging');
    });

    window.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        const delta = e.pageX - startX;
        if (Math.abs(delta) > 5) {
            dragThresholdPassed = true;
        }
        suggestionsBar.scrollLeft = scrollStart - delta;
    });

    window.addEventListener('mouseup', () => {
        if (isDragging) {
            isDragging = false;
            suggestionsBar.classList.remove('dragging');
            // reset drag threshold after short delay so click listener gets current state
            setTimeout(() => { dragThresholdPassed = false; }, 50);
        }
    });

    // ── Mouse Wheel horizontal scroll for PC ─────────────────────────────────
    suggestionsBar.addEventListener('wheel', (e) => {
        if (e.deltaY !== 0 && Math.abs(e.deltaY) >= Math.abs(e.deltaX)) {
            e.preventDefault();
            // RTL scroll direction in modern browsers
            suggestionsBar.scrollLeft += e.deltaY;
        }
    }, { passive: false });
}
initSuggestions();

// ── Responsive placeholder ────────────────────────────────────────────────────
// Textarea placeholders never wrap — use a compact hint on small screens so
// the full sentence doesn't scroll/clip.
const PLACEHOLDER_LONG = 'اكتب وصفاً أو شعوراً للبحث عن أبيات تناسبه... (مثال: الشوق والفراق)';
const PLACEHOLDER_SHORT = 'صف شعورك لتجد أبياتاً تناسبه...';

function applyPlaceholder() {
    userInput.placeholder = window.innerWidth <= 640 ? PLACEHOLDER_SHORT : PLACEHOLDER_LONG;
}
applyPlaceholder();
window.addEventListener('resize', applyPlaceholder);

// ── Cache helpers ─────────────────────────────────────────────────────────────
function saveToCache(userText, answerText, thinkingText) {
    try {
        const cached = JSON.parse(localStorage.getItem(CACHE_KEY) || '[]');
        cached.push({ user: userText, answer: answerText, thinking: thinkingText || '' });
        if (cached.length > 10) cached.splice(0, cached.length - 10);
        localStorage.setItem(CACHE_KEY, JSON.stringify(cached));
    } catch (e) {}
}

function loadCachedChats() {
    let cached;
    try { cached = JSON.parse(localStorage.getItem(CACHE_KEY) || '[]'); }
    catch (e) { localStorage.removeItem(CACHE_KEY); return; }
    if (!cached.length) return;

    const sep = document.createElement('div');
    sep.className = 'cache-divider';
    sep.textContent = '— المحادثات السابقة —';
    messagesDiv.appendChild(sep);

    cached.forEach(({ user, answer, thinking }) => {
        const userEl = createMessageEl('user');
        userEl.textContent = user;
        // Do NOT push to `conversation` array. We want a fresh context!

        const astEl = createMessageEl('assistant');

        // Restore collapsed thinking drawer if we have thinking content
        if (thinking && thinking.trim()) {
            const drawerEl = document.createElement('div');
            drawerEl.className = 'thinking-drawer';
            const header = document.createElement('div');
            header.className = 'thinking-header';
            header.innerHTML = '<span>⚙️</span><span>تفكير (Thinking)</span><span class="chevron">▾</span>';
            const contentEl = document.createElement('div');
            contentEl.className = 'thinking-content'; // collapsed (no "open" class)
            contentEl.textContent = thinking.replace(/<\/?(?:think|thinking|thought|tool_call|function)[^>]*>/gi, '').trim();
            header.addEventListener('click', () => contentEl.classList.toggle('open'));
            drawerEl.appendChild(header);
            drawerEl.appendChild(contentEl);
            astEl.appendChild(drawerEl);
        }

        const answerEl = document.createElement('div');
        answerEl.className = 'markdown-body';
        answerEl.innerHTML = marked.parse(answer || '');
        astEl.appendChild(answerEl);
        // Do NOT push to `conversation` array. We want a fresh context!
    });

    const sep2 = document.createElement('div');
    sep2.className = 'cache-divider';
    sep2.textContent = '— الجلسة الحالية —';
    messagesDiv.appendChild(sep2);

    window.scrollTo({ top: document.body.scrollHeight });
}

// ── Refresh Toast ─────────────────────────────────────────────────────────────

function createMessageEl(role) {
    const el = document.createElement('div');
    el.className = `message ${role}`;
    messagesDiv.appendChild(el);
    return el;
}

userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
sendBtn.addEventListener('click', sendMessage);

// Auto-grow the textarea while typing, up to its max height
userInput.addEventListener('input', () => {
    userInput.style.height = 'auto';
    userInput.style.height = Math.max(44, Math.min(160, userInput.scrollHeight)) + 'px';
});

stopBtn.addEventListener('click', () => { if (currentAbortController) currentAbortController.abort(); });

function setStreaming(active) {
    sendBtn.style.display = active ? 'none' : 'flex';
    stopBtn.style.display = active ? 'flex' : 'none';
    if (suggestionsBar) {
        suggestionsBar.classList.toggle('disabled', active);
    }
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    userInput.value = '';
    userInput.style.height = '44px';
    setStreaming(true);

    const userEl = createMessageEl('user');
    userEl.textContent = text;
    conversation.push({ role: 'user', content: text });

    const astEl = createMessageEl('assistant');

    // ── Heartbeat while streaming ─────────────────────────────────────────────
    // The assistant bubble pulses gently (fade out → back in) for as long as
    // the response is running: waiting for the first token, tool execution,
    // and token streaming. Removed in `finally` when the stream ends.
    astEl.classList.add('pulsing');

    // ── Thinking drawer (lazy) ────────────────────────────────────────────────
    let thinkingContentEl = null;
    let thinkingRaw = '';   // accumulates pre-answer turn content (stripped of tags)

    function ensureThinkingDrawer() {
        if (thinkingContentEl) return;
        const drawerEl = document.createElement('div');
        drawerEl.className = 'thinking-drawer';
        const header = document.createElement('div');
        header.className = 'thinking-header';
        header.innerHTML = '<span>⚙️</span><span>تفكير (Thinking)</span><span class="chevron">▾</span>';
        thinkingContentEl = document.createElement('div');
        thinkingContentEl.className = 'thinking-content open';
        header.addEventListener('click', () => thinkingContentEl.classList.toggle('open'));
        drawerEl.appendChild(header);
        drawerEl.appendChild(thinkingContentEl);
        // Always insert the drawer BEFORE the answer area
        astEl.insertBefore(drawerEl, answerEl);
    }

    function appendThinking(text) {
        ensureThinkingDrawer();
        thinkingRaw += text;
        // Strip XML tags from display, keep text content
        thinkingContentEl.textContent = thinkingRaw
            .replace(/<\/?(?:think|thinking|thought|tool_call|function)[^>]*>/gi, '')
            .trim();
    }

    // ── Answer area ───────────────────────────────────────────────────────────
    const answerEl = document.createElement('div');
    answerEl.className = 'markdown-body';
    astEl.appendChild(answerEl);

    let answerRaw = '';

    function appendAnswer(text) {
        answerRaw += text;
        answerEl.innerHTML = marked.parse(answerRaw);
    }

    // ── State ─────────────────────────────────────────────────────────────────
    // currentTurn 0 = pre-tool reasoning → goes to thinking drawer
    // currentTurn 1+ = post-tool answer → goes to answer area
    let currentTurn = 0;
    let fullRawContent = '';

    currentAbortController = new AbortController();

    try {
        const response = await fetch(apiUrl('/chat'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ messages: conversation, use_openrouter: useOpenRouter }),
            signal: currentAbortController.signal
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            for (const line of decoder.decode(value, { stream: true }).split('\n')) {
                if (!line.startsWith('data: ')) continue;
                const dataStr = line.slice(6).trim();
                if (!dataStr) continue;

                let data;
                try { data = JSON.parse(dataStr); } catch { continue; }

                if (data.turn_start !== undefined) {
                    currentTurn = data.turn_start;
                    // When a new answer-turn starts, close the thinking drawer
                    if (currentTurn >= 1 && thinkingContentEl) {
                        thinkingContentEl.classList.remove('open');
                    }

                } else if (data.content) {
                    fullRawContent += data.content;
                    if (currentTurn === 0) {
                        appendThinking(data.content);
                    } else {
                        appendAnswer(data.content);
                    }

                } else if (data.tool_executing) {
                    appendThinking(`\n[جاري تنفيذ الأداة: ${data.tool_executing}]\n`);

                } else if (data.error) {
                    answerEl.innerHTML = `<p style="color:#f87171;"><strong>خطأ:</strong> ${data.error}</p>`;
                    console.error('SSE error from backend:', data.error);

                } else if (data.done) {
                    break;
                }
            }
            window.scrollTo({ top: document.body.scrollHeight });
        }

        // Final state: collapse thinking, do one clean markdown render
        if (thinkingContentEl) thinkingContentEl.classList.remove('open');
        if (answerRaw.trim()) answerEl.innerHTML = marked.parse(answerRaw);

        conversation.push({ role: 'assistant', content: fullRawContent });
        // Save even when only the thinking block arrived (e.g. the stream died
        // before the answer) so partial sessions survive a reload.
        if (answerRaw.trim() || thinkingRaw.trim()) saveToCache(text, answerRaw, thinkingRaw);

    } catch (err) {
        if (err.name === 'AbortError') {
            const note = document.createElement('p');
            note.className = 'stop-note';
            note.textContent = '⏹ تم الإيقاف.';
            astEl.appendChild(note);
            if (fullRawContent) conversation.push({ role: 'assistant', content: fullRawContent });
        } else {
            console.error(err);
            answerEl.textContent = 'Connection error.';
        }
    } finally {
        astEl.classList.remove('pulsing');
        setStreaming(false);
        currentAbortController = null;
        window.scrollTo({ top: document.body.scrollHeight });
    }
}

// ── Load cache on startup (after functions are defined) ──────────────────────
loadCachedChats();

// ── Clear cache button ───────────────────────────────────────────────────────
document.getElementById('clear-cache-btn').addEventListener('click', () => {
    localStorage.removeItem(CACHE_KEY);
    conversation.length = 0;          // reset in-memory context
    messagesDiv.innerHTML = '';       // clear rendered messages

    const notice = document.createElement('div');
    notice.className = 'cache-divider';
    notice.textContent = '— تم مسح السجل — ابدأ محادثة جديدة —';
    messagesDiv.appendChild(notice);
});
