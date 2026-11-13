import '@fontsource/instrument-serif/400.css';
import '@fontsource/instrument-serif/400-italic.css';
import '@fontsource/ibm-plex-mono/400.css';
import '@fontsource/ibm-plex-mono/500.css';
import './style.css';
import * as THREE from 'three';
import { Galaxy } from './galaxy.js';
import { Embedder, VectorIndex, locate } from './search.js';
import { detectBackend, hasLLM, askLLM, extractive, renderAnswer, esc } from './rag.js';

const DATA = `${import.meta.env.BASE_URL}data/`;
const $ = (s) => document.querySelector(s);
const fmt = new Intl.NumberFormat('pt-BR');
const MONTHS = ['jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez'];
const EXAMPLES = [
  'How do diffusion models generate images?',
  'Can LLMs reason step by step?',
  'Privacy in federated learning',
  'Robots that learn from human demonstrations',
];
const TOP_K = 12;       // estrelas destacadas no mapa
const CONTEXT_K = 6;    // estrelas enviadas ao LLM / leitura extrativa

// ------------------------------------------------------------------ carga
async function fetchWithProgress(url, onBytes) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  const reader = r.body.getReader();
  const chunks = [];
  let size = 0;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    chunks.push(value);
    size += value.length;
    onBytes(value.length);
  }
  const out = new Uint8Array(size);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out.buffer;
}

async function loadCore() {
  const meta = await (await fetch(`${DATA}meta.json`)).json();
  const expected = meta.n * (12 + 2 + 2);
  let got = 0;
  const tick = (b) => {
    got += b;
    $('#loader-bar').style.width = `${Math.min(100, (got / expected) * 100).toFixed(1)}%`;
  };
  const [pos, cluster, day] = await Promise.all([
    fetchWithProgress(`${DATA}points.bin`, tick),
    fetchWithProgress(`${DATA}cluster.bin`, tick),
    fetchWithProgress(`${DATA}day.bin`, tick),
  ]);
  return { meta, n: meta.n, pos: new Float32Array(pos), cluster: new Int16Array(cluster), day: new Uint16Array(day) };
}

const data = await loadCore().catch((e) => {
  $('#loader-text').textContent = `não consegui carregar os dados (${e.message})`;
  throw e;
});
const { meta } = data;
const galaxy = new Galaxy($('#scene'), $('#labels'), data);
window.galaxy = galaxy;   // útil para depurar no console

// textos e índice vetorial chegam em segundo plano
const titlesReady = fetch(`${DATA}titles.json`).then((r) => r.json());
let titles = null;
titlesReady.then((t) => (titles = t));
let indexReady = null;
const loadIndex = () => (indexReady ??= fetch(`${DATA}emb.bin`)
  .then((r) => r.arrayBuffer())
  .then((e) => new VectorIndex(meta, new Int8Array(e))));
const embedder = new Embedder(meta.modelBrowser);

const shards = new Map();
function abstractOf(i) {
  const s = Math.floor(i / meta.shard);
  if (!shards.has(s)) shards.set(s, fetch(`${DATA}abs/${String(s).padStart(3, '0')}.json`).then((r) => r.json()));
  return shards.get(s).then((list) => list[i % meta.shard]);
}

const startDate = new Date(`${meta.start}T00:00:00Z`);
const dateOf = (day) => new Date(startDate.getTime() + day * 864e5);
const yearOf = (i) => dateOf(data.day[i]).getUTCFullYear();
const clusterOf = (i) => (data.cluster[i] >= 0 ? meta.clusters[data.cluster[i]] : null);
const dotOf = (i) => `<span class="dot" style="--c:${clusterOf(i)?.color ?? '#56648f'}"></span>`;

$('#stats').innerHTML = `${fmt.format(meta.n)} artigos · ${meta.clusters.length} constelações<br>${esc(meta.corpus)}`;

// ------------------------------------------------------------------ painel
const panel = $('#panel'), body = $('#panel-body');
let lastAnswer = null;      // para o "voltar à resposta"
let abortAnswer = null;

const wide = () => innerWidth > 900;
function openPanel(html) {
  body.innerHTML = html;
  body.scrollTop = 0;
  panel.hidden = false;
  document.body.classList.add('panel-open');
  galaxy.shiftTarget = wide() ? panel.offsetWidth / 2 : 0;
  galaxy.shiftYTarget = wide() ? 0 : innerHeight * 0.2;
}
// os rótulos das constelações desviam da interface
galaxy.reserved = () => {
  const boxes = ['#brand', '#ask', '#top-actions', '#timeline'];
  if (!panel.hidden) boxes.push('#panel');
  return boxes.map((sel) => {
    const r = $(sel).getBoundingClientRect();
    return [r.left - 10, r.top - 10, r.right + 10, r.bottom + 10];
  });
};
function closePanel() {
  panel.hidden = true;
  document.body.classList.remove('panel-open');
  galaxy.shiftTarget = galaxy.shiftYTarget = 0;
  abortAnswer?.abort();
  lastAnswer = null;
  stopTour();
  galaxy.resetView();
  galaxy.idleRotate = true;
  history.replaceState(null, '', location.pathname);
}
$('#panel-close').addEventListener('click', closePanel);

body.addEventListener('click', (e) => {
  const t = e.target.closest('[data-star],[data-cluster],[data-back],.cite');
  if (!t) return;
  if (t.matches('.cite')) {
    const src = body.querySelectorAll('.src')[+t.dataset.n - 1];
    src?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    src?.classList.add('hot');
    setTimeout(() => src?.classList.remove('hot'), 1600);
    if (lastAnswer) galaxy.select(lastAnswer.hits[+t.dataset.n - 1].i);
  } else if (t.dataset.star) showPaper(+t.dataset.star);
  else if (t.dataset.cluster) showCluster(+t.dataset.cluster);
  else if (t.dataset.back) restoreAnswer();
});
body.addEventListener('pointerover', (e) => {
  const t = e.target.closest('.src[data-star]');
  if (t) galaxy.select(+t.dataset.star);
});

// ------------------------------------------------------------------ artigo
async function showPaper(i, fly = true) {
  stopTour();
  await titlesReady;
  const c = clusterOf(i), d = dateOf(data.day[i]);
  galaxy.select(i);
  if (fly) galaxy.flyTo(galaxy.starPosition(i), 9, { duration: 1700 });
  const head = `
    ${lastAnswer ? '<p><button class="btn-line" data-back="1">← voltar à resposta</button></p>' : ''}
    <p class="kicker">estrela · ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}</p>
    <h2>${esc(titles.titles[i])}</h2>`;
  openPanel(`${head}<p class="status">buscando o resumo…</p>`);
  const a = await abstractOf(i);
  if (galaxy.selected !== i) return;
  openPanel(`${head}
    <p class="dim" style="font-size:12px">${esc(a.u)} · ${esc(a.c)}</p>
    ${c ? `<p><button class="btn-line" data-cluster="${c.id}">${dotOf(i)} &nbsp;${esc(c.name)}</button></p>` : '<p class="dim" style="font-size:12px">poeira interestelar: fora de qualquer constelação</p>'}
    <p class="abstract">${esc(a.a)}</p>
    <p><a class="link" href="https://arxiv.org/abs/${titles.ids[i]}" target="_blank" rel="noopener">abrir no arXiv ↗</a></p>`);
}

// ------------------------------------------------------------------ constelação
function drawSeries(canvas, series, color, upto = series.length) {
  const w = canvas.clientWidth, h = canvas.clientHeight, dpr = Math.min(devicePixelRatio, 2);
  canvas.width = w * dpr; canvas.height = h * dpr;
  const g = canvas.getContext('2d');
  g.scale(dpr, dpr);
  // suaviza com média móvel de 6 meses
  const sm = series.map((_, k) => { let s = 0, c = 0; for (let j = Math.max(0, k - 5); j <= k; j++) { s += series[j]; c++; } return s / c; });
  const max = Math.max(...sm, 1e-9);
  const X = (k) => (k / (sm.length - 1)) * w, Y = (v) => h - 2 - (v / max) * (h - 6);
  const path = (end) => { g.beginPath(); g.moveTo(0, h); for (let k = 0; k < end; k++) g.lineTo(X(k), Y(sm[k])); g.lineTo(X(end - 1), h); g.closePath(); };
  path(sm.length); g.fillStyle = 'rgba(143,216,255,.07)'; g.fill();
  const grad = g.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, color); grad.addColorStop(1, 'rgba(0,0,0,0)');
  path(Math.max(2, upto)); g.globalAlpha = 0.55; g.fillStyle = grad; g.fill(); g.globalAlpha = 1;
  g.beginPath(); for (let k = 0; k < Math.max(2, upto); k++) g.lineTo(X(k), Y(sm[k]));
  g.strokeStyle = color; g.lineWidth = 1.5; g.stroke();
}

const totalSeries = meta.months.map((_, m) => meta.clusters.reduce((s, c) => s + c.series[m], 0));

async function showCluster(id, fromTour = false) {
  if (!fromTour) stopTour();
  const c = meta.clusters[id];
  lastAnswer = null;
  galaxy.clearQuery();
  galaxy.select(null);
  galaxy.focusCluster(id);
  await titlesReady;
  const share = c.series.map((v, m) => (totalSeries[m] ? v / totalSeries[m] : 0));
  const peak = share.map((_, k) => share.slice(Math.max(0, k - 11), k + 1).reduce((a, b) => a + b, 0)).reduce((b, v, k, arr) => (v > arr[b] ? k : b), 0);
  openPanel(`
    <p class="kicker" style="color:${c.color}">constelação · ${fmt.format(c.size)} estrelas</p>
    <h2>${esc(c.name)}</h2>
    ${c.desc ? `<p class="abstract">${esc(c.desc)}</p>` : ''}
    <div class="tags">${c.keywords.map((k) => `<span class="tag">${esc(k)}</span>`).join('')}</div>
    <h3>fatia do corpus ao longo do tempo</h3>
    <canvas class="series"></canvas>
    <div class="axis"><span>${meta.months[0].slice(0, 4)}</span><span>auge em ${meta.months[peak].slice(0, 4)}</span><span>${meta.months.at(-1).slice(0, 4)}</span></div>
    <h3>estrelas mais centrais</h3>
    ${c.top.map((i, k) => `<button class="src" data-star="${i}"><span class="n">${k + 1}</span><span class="t">${esc(titles.titles[i])}</span><span class="m">${yearOf(i)}</span></button>`).join('')}`);
  drawSeries(body.querySelector('canvas.series'), share, c.color);
}
galaxy.onLabel = (id) => showCluster(id);

// ------------------------------------------------------------------ pergunta
const form = $('#ask-form'), input = $('#q'), go = $('#ask-go');
$('#chips').innerHTML = EXAMPLES.map((q) => `<button class="chip" type="button">${esc(q)}</button>`).join('');
$('#chips').addEventListener('click', (e) => {
  if (!e.target.matches('.chip')) return;
  input.value = e.target.textContent;
  ask(input.value);
});
input.addEventListener('focus', () => { embedder.warm().catch(() => {}); loadIndex(); }, { once: true });
form.addEventListener('submit', (e) => { e.preventDefault(); ask(input.value); });

function sourcesHTML(hits, quotes = []) {
  return hits.map((h, k) => `
    <button class="src" data-star="${h.i}">
      <span class="n">${k + 1}</span><span class="t">${esc(titles.titles[h.i])}</span>
      ${quotes[k] ? `<span class="q">“${esc(quotes[k].text)}”</span>` : ''}
      <span class="m">${dotOf(h.i)} ${esc(clusterOf(h.i)?.name ?? 'poeira interestelar')} · ${yearOf(h.i)}<span class="sim">${h.score.toFixed(2)}</span></span>
    </button>`).join('');
}

function restoreAnswer() {
  if (!lastAnswer) return;
  openPanel(lastAnswer.html);
  galaxy.select(null);
  galaxy.flyTo(lastAnswer.position, lastAnswer.distance, { duration: 1500 });
}

async function ask(question) {
  question = question.trim();
  if (!question || go.disabled) return;
  stopTour();
  setPlaying(false);
  setDay(meta.days);
  abortAnswer?.abort();
  const ctl = (abortAnswer = new AbortController());
  go.disabled = true;
  galaxy.idleRotate = false;
  history.replaceState(null, '', `?q=${encodeURIComponent(question)}`);
  const head = `<p class="kicker gold">pergunta</p><h2>${esc(question)}</h2>`;
  const status = (msg) => openPanel(`${head}<p class="status">${msg}</p>`);

  try {
    status('carregando o modelo de embeddings…');
    embedder.onProgress = (p) => {
      const el = body.querySelector('.status');
      if (el && p < 1) el.textContent = `baixando o modelo de embeddings (só na primeira vez)… ${(p * 100).toFixed(0)}%`;
    };
    const [index] = await Promise.all([loadIndex(), embedder.warm(), titlesReady]);
    status('transformando a pergunta em vetor…');
    const [qvec] = await embedder.embed([question]);
    const hits = index.search(qvec, TOP_K);

    // a pergunta vira uma estrela e a câmera voa até ela
    const position = new THREE.Vector3(...locate(hits, data.pos));
    const spread = Math.max(...hits.slice(0, 8).map((h) => galaxy.starPosition(h.i).distanceTo(position)));
    const distance = THREE.MathUtils.clamp(spread * 2.3, 11, 60);
    galaxy.select(null);
    galaxy.showQuery(position, hits.map((h) => h.i));
    galaxy.flyTo(position, distance, { duration: 2600 });

    const ctx = hits.slice(0, CONTEXT_K);
    const where = [...new Set(hits.map((h) => clusterOf(h.i)?.name).filter(Boolean))].slice(0, 3);
    const region = where.length ? `<p class="dim" style="font-size:12px">caiu perto de ${where.map((w) => `<em>${esc(w)}</em>`).join(', ')}</p>` : '';
    const frame = (answer, quotes) => `${head}${region}${answer}
      <h3>estrelas vizinhas · similaridade</h3>${sourcesHTML(hits, quotes)}`;
    openPanel(frame('<p class="status">lendo as estrelas vizinhas…</p>'));

    const abstracts = await Promise.all(ctx.map((h) => abstractOf(h.i)));
    if (ctl.signal.aborted) return;
    let answerHTML, quotes = [];
    if (hasLLM()) {
      const draw = (text, done) => {
        const el = body.querySelector('.answer');
        if (el) el.innerHTML = renderAnswer(text, ctx.length) + (done ? '' : '<span class="caret"></span>');
      };
      openPanel(frame('<div class="answer"><p class="status">gerando a resposta…</p></div>'));
      try {
        const text = await askLLM(question, ctx.map((h) => h.i), (t) => draw(t, false), ctl.signal);
        answerHTML = `<div class="answer">${renderAnswer(text, ctx.length)}</div>
          <p class="note">Resposta gerada por LLM usando apenas os ${ctx.length} resumos mais próximos como contexto.</p>`;
      } catch (err) {
        if (ctl.signal.aborted) return;
        answerHTML = `<p class="note">O LLM não respondeu (${esc(err.message)}). Abaixo, os trechos mais relevantes.</p>`;
      }
    }
    if (!answerHTML || !hasLLM()) {
      quotes = await extractive(embedder, qvec, ctx.map((h, k) => ({ abstract: abstracts[k].a })));
      answerHTML ??= '<p class="note">Modo estático: a busca semântica roda inteira no seu navegador e destaca o trecho mais relevante de cada estrela. Com o backend ligado, um LLM escreve a resposta citando as fontes.</p>';
    }
    if (ctl.signal.aborted) return;
    const html = frame(answerHTML, quotes);
    lastAnswer = { html, hits, position, distance };
    const keep = body.scrollTop;
    openPanel(html);
    body.scrollTop = keep;
  } catch (err) {
    console.error(err);
    openPanel(`${head}<p class="note">Não consegui responder: ${esc(err.message)}</p>`);
  } finally {
    go.disabled = false;
  }
}

// ------------------------------------------------------------------ hover / clique
const tip = $('#tooltip');
galaxy.onHover = (i, x, y) => {
  if (i < 0 || !titles) { tip.hidden = true; return; }
  tip.innerHTML = `${esc(titles.titles[i])}<small>${dotOf(i)} ${esc(clusterOf(i)?.name ?? 'poeira interestelar')} · ${yearOf(i)}</small>`;
  tip.hidden = false;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = `${Math.min(x + 16, innerWidth - w - 8)}px`;
  tip.style.top = `${y + 18 + h > innerHeight ? y - h - 12 : y + 18}px`;
};
let down = null;
$('#scene').addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; stopTour(); $('#hint').classList.add('gone'); });
$('#scene').addEventListener('pointerup', (e) => {
  if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) < 5 && galaxy.hovered >= 0) showPaper(galaxy.hovered);
  down = null;
});

// ------------------------------------------------------------------ linha do tempo
const slider = $('#time'), yearEl = $('#year'), playIcon = $('#play-icon'), spark = $('#spark');
let playing = false, day = meta.days, lastMonth = -1;

function monthIndex(d) {
  const dt = dateOf(d);
  return Math.min(meta.months.length - 1, (dt.getUTCFullYear() - startDate.getUTCFullYear()) * 12 + dt.getUTCMonth() - startDate.getUTCMonth());
}

function rising(m) {
  if (m < 18) return [];
  const sum = (s, a, b) => { let t = 0; for (let k = Math.max(0, a); k <= b; k++) t += s[k]; return t; };
  const totR = sum(totalSeries, m - 11, m), totP = sum(totalSeries, m - 47, m - 12);
  return meta.clusters
    .map((c) => {
      const r = sum(c.series, m - 11, m), p = sum(c.series, m - 47, m - 12);
      return { c, r, lift: (r / totR + 0.003) / (p / totP + 0.003) };
    })
    .filter((x) => x.r >= 15)
    .sort((a, b) => b.lift - a.lift)
    .slice(0, 3);
}

function setDay(d, flash = 0) {
  day = Math.max(0, Math.min(meta.days, d));
  galaxy.setDay(day, flash);
  slider.value = Math.round((day / meta.days) * 1000);
  const dt = dateOf(day), m = monthIndex(day);
  yearEl.innerHTML = `${dt.getUTCFullYear()}<small>${MONTHS[dt.getUTCMonth()]}</small>`;
  if (m !== lastMonth) {
    lastMonth = m;
    drawSeries(spark, totalSeries, '#f5d68a', m + 1);
    $('#rising-list').innerHTML = rising(m).map(({ c, lift }) =>
      `<button class="rise-item" data-cluster="${c.id}"><span class="dot" style="--c:${c.color}"></span>${esc(c.name)}<b>×${lift.toFixed(1)}</b></button>`).join('')
      || '<span class="dim">ainda é cedo para dizer</span>';
  }
}
function setPlaying(on) {
  playing = on;
  playIcon.setAttribute('d', on ? 'M7 5h4v14H7zM13 5h4v14h-4z' : 'M8 5v14l11-7z');
  if (!on) galaxy.setDay(day, 0);
}
$('#play').addEventListener('click', () => {
  stopTour();
  if (!playing && day >= meta.days - 1) setDay(0);
  setPlaying(!playing);
});
slider.addEventListener('input', () => { setPlaying(false); setDay((slider.value / 1000) * meta.days); });
$('#rising-list').addEventListener('click', (e) => {
  const t = e.target.closest('[data-cluster]');
  if (t) { setPlaying(false); showCluster(+t.dataset.cluster); }
});
let prev = performance.now();
galaxy.frameHooks.push((now) => {
  const dt = Math.min(250, now - prev);
  prev = now;
  if (!playing) return;
  setDay(day + (meta.days / 50000) * dt, 1);     // ~50 s para a história inteira
  if (day >= meta.days) setPlaying(false);
});
window.addEventListener('resize', () => { lastMonth = -1; setDay(day); });

// ------------------------------------------------------------------ passeio
let tour = null;
function stopTour() {
  if (!tour) return;
  clearTimeout(tour.timer);
  tour = null;
  $('#btn-tour').classList.remove('on');
}
function startTour() {
  const order = meta.clusters.slice(0, 14).map((c) => c.id).sort((a, b) => {
    const A = meta.clusters[a].centroid, B = meta.clusters[b].centroid;
    return Math.atan2(A[2], A[0]) - Math.atan2(B[2], B[0]);
  });
  tour = { k: 0, timer: 0 };
  $('#btn-tour').classList.add('on');
  setPlaying(false);
  setDay(meta.days);
  const next = () => {
    if (!tour) return;
    showCluster(order[tour.k++ % order.length], true);
    tour.timer = setTimeout(next, 7000);
  };
  next();
}
$('#btn-tour').addEventListener('click', () => (tour ? (stopTour(), closePanel()) : startTour()));

// ------------------------------------------------------------------ sobre
const about = $('#about');
$('#about-steps').innerHTML = `
  <li><div><b>Corpus</b>${fmt.format(meta.n)} artigos de Ciência da Computação do arXiv, de ${meta.start.slice(0, 4)} a ${meta.end.slice(0, 4)}, amostrados mês a mês.</div></li>
  <li><div><b>Embeddings</b>Título e resumo viram um vetor de ${meta.embDim} dimensões com <code>${esc(meta.model)}</code>.</div></li>
  <li><div><b>UMAP</b>Os vetores são reduzidos para 3D preservando vizinhanças: cada artigo vira uma estrela.</div></li>
  <li><div><b>HDBSCAN</b>Regiões densas viram ${meta.clusters.length} constelações; ${(meta.noise * 100).toFixed(0)}% dos artigos ficam como poeira, sem grupo. ${meta.namedBy === 'llm' ? 'Um LLM batiza cada uma a partir das palavras-chave e dos títulos centrais.' : 'Os nomes vêm das palavras-chave (c-TF-IDF).'}</div></li>
  <li><div><b>Pergunta → estrela</b>O mesmo modelo roda no seu navegador (ONNX). A pergunta é comparada com todos os artigos em um índice quantizado em int8 (recall@10 de ${(meta.recallAt10 * 100).toFixed(0)}% contra float32) e cai entre as vizinhas mais próximas.</div></li>
  <li><div><b>RAG</b>Os resumos vizinhos viram o contexto de um LLM, que responde citando as fontes.</div></li>`;
$('#about-foot').textContent = 'sentence-transformers · UMAP · HDBSCAN · Three.js · transformers.js · Claude';
$('#btn-about').addEventListener('click', () => about.showModal());
about.addEventListener('click', (e) => { if (e.target === about || e.target.matches('.dialog-close')) about.close(); });

// ------------------------------------------------------------------ início
setDay(meta.days);
galaxy.idleRotate = true;
await document.fonts.ready.catch(() => {});
$('#loader').classList.add('done');
galaxy.start();
detectBackend();
setTimeout(() => $('#hint').classList.add('gone'), 14000);

const initial = new URLSearchParams(location.search).get('q');
if (initial) {
  input.value = initial;
  setTimeout(() => ask(initial), 2600);
} else {
  // adianta o download do índice para a primeira pergunta sair rápida
  setTimeout(loadIndex, 5000);
}
