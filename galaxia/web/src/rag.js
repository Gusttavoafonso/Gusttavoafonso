// Geração da resposta. Dois modos:
//  - "llm": um backend (server/app.py) recebe a pergunta + ids das estrelas vizinhas
//           e devolve a resposta do Claude em streaming;
//  - "extrativo": sem backend, o próprio navegador escolhe a frase mais relevante
//           de cada resumo (embedding de frases x embedding da pergunta).
import { cosine } from './search.js';

const API = (new URLSearchParams(location.search).get('api') ?? import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '');

let backend = null;

export async function detectBackend() {
  try {
    const r = await fetch(`${API}/api/health`, { signal: AbortSignal.timeout(2500) });
    const j = r.ok && r.headers.get('content-type')?.includes('json') ? await r.json() : null;
    backend = j?.ok ? j : null;
  } catch {
    backend = null;
  }
  return backend;
}

export const hasLLM = () => !!backend;

/** Resposta em streaming do LLM. `onText` recebe o texto acumulado. */
export async function askLLM(question, ids, onText, signal) {
  const r = await fetch(`${API}/api/ask`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ question, ids }),
    signal,
  });
  if (!r.ok || !r.body) throw new Error(`o servidor respondeu ${r.status}`);
  const reader = r.body.pipeThrough(new TextDecoderStream()).getReader();
  let text = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    text += value;
    onText(text);
  }
  return text;
}

const sentences = (t) => t.split(/(?<=[.!?])\s+(?=[A-Z"(])/).map((s) => s.trim()).filter((s) => s.length > 40);

/** Para cada fonte, a frase do resumo mais próxima da pergunta. */
export async function extractive(embedder, qvec, sources) {
  const all = [];
  sources.forEach((s, k) => sentences(s.abstract).slice(0, 8).forEach((text) => all.push({ k, text })));
  if (!all.length) return sources.map(() => null);
  const vecs = await embedder.embed(all.map((s) => s.text));
  const best = sources.map(() => null);
  all.forEach((s, j) => {
    const score = cosine(qvec, vecs[j]);
    if (!best[s.k] || score > best[s.k].score) best[s.k] = { text: s.text, score };
  });
  return best;
}

const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

/** Markdown mínimo -> HTML seguro, com citações [n] clicáveis. */
export function renderAnswer(text, nSources) {
  return esc(text)
    .split(/\n{2,}/)
    .map((p) => p.trim())
    .filter(Boolean)
    .map((p) => `<p>${p
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/^[-*] /gm, '• ')
      .replace(/\n/g, '<br>')
      .replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (m, list) => list.split(',').map((x) => {
        const n = +x;
        return n >= 1 && n <= nSources ? `<button class="cite" data-n="${n}">${n}</button>` : `[${x.trim()}]`;
      }).join(''))}</p>`)
    .join('');
}

export { esc };
