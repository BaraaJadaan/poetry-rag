const messagesDiv = document.getElementById('messages');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const stopBtn = document.getElementById('stop-btn');
const backendToggle = document.getElementById('backend-toggle');

let conversation = [];
let currentAbortController = null;
let toastTimer = null;

const CACHE_KEY   = 'poetry_rag_chats';    // last-10 Q&A pairs
const TOGGLE_KEY  = 'poetry_rag_backend';  // 'local' | 'openrouter'

marked.setOptions({ breaks: true, gfm: true });

// ── Restore toggle preference ─────────────────────────────────────────────────
(function restoreToggle() {
    if (localStorage.getItem(TOGGLE_KEY) === 'openrouter') backendToggle.checked = true;
})();

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
            header.innerHTML = '<span>⛙️</span><span>تفكير (Thinking)</span><span class="chevron">▾</span>';
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
function showRefreshToast() {
    // Remove any existing toast immediately
    const existing = document.getElementById('refresh-toast');
    if (existing) existing.remove();
    if (toastTimer) { clearTimeout(toastTimer); toastTimer = null; }

    const toast = document.createElement('div');
    toast.id = 'refresh-toast';
    const label = backendToggle.checked ? ' API (إنترنت)' : 'محلي (Local)';
    toast.innerHTML = `
        <span>تم التبديل إلى <strong>${label}</strong> — تحتاج الصفحة إلى إعادة تحميل لتطبيق التغيير</span>
        <button class="toast-refresh-btn" onclick="location.reload()">تحديث الآن</button>
    `;
    document.body.appendChild(toast);

    // Auto-dismiss after 6 seconds
    toastTimer = setTimeout(() => {
        toast.classList.add('hide');
        toast.addEventListener('animationend', () => toast.remove(), { once: true });
        toastTimer = null;
    }, 6000);
}

backendToggle.addEventListener('change', () => {
    localStorage.setItem(TOGGLE_KEY, backendToggle.checked ? 'openrouter' : 'local');
    showRefreshToast();
});

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

stopBtn.addEventListener('click', () => { if (currentAbortController) currentAbortController.abort(); });

function setStreaming(active) {
    sendBtn.style.display = active ? 'none' : 'block';
    stopBtn.style.display = active ? 'block' : 'none';
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    userInput.value = '';
    setStreaming(true);

    const userEl = createMessageEl('user');
    userEl.textContent = text;
    conversation.push({ role: 'user', content: text });

    const astEl = createMessageEl('assistant');

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
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ messages: conversation, use_openrouter: backendToggle.checked }),
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
        if (answerRaw.trim()) saveToCache(text, answerRaw, thinkingRaw);

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
