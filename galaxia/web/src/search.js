// Busca semântica 100% no navegador: embedding da pergunta (worker) + índice int8.

export class Embedder {
  constructor(model) {
    this.model = model;
    this.wasm = new URL(`${import.meta.env.BASE_URL}ort/`, document.baseURI).href;
    this.pending = new Map();
    this.seq = 0;
    this.onProgress = () => {};
  }

  #call(msg) {
    if (!this.worker) {
      this.worker = new Worker(new URL('./embed.worker.js', import.meta.url), { type: 'module' });
      this.worker.onmessage = ({ data }) => {
        if (data.type === 'progress') return this.onProgress(data.loaded / data.total);
        const p = this.pending.get(data.id);
        if (!p) return;
        this.pending.delete(data.id);
        data.ok ? p.resolve(data) : p.reject(new Error(data.error));
      };
    }
    return new Promise((resolve, reject) => {
      const id = ++this.seq;
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, model: this.model, wasm: this.wasm, ...msg });
    });
  }

  /** Começa a baixar o modelo (cacheado pelo navegador depois da 1ª vez). */
  warm() {
    return (this.ready ??= this.#call({ type: 'load' }).catch((e) => { this.ready = null; throw e; }));
  }

  /** textos -> Float32Array[] normalizados (384d). */
  async embed(texts) {
    const { vectors, dims } = await this.#call({ type: 'embed', texts });
    const d = dims[dims.length - 1];
    return texts.map((_, i) => vectors.subarray(i * d, (i + 1) * d));
  }
}

export class VectorIndex {
  constructor(meta, emb) {
    this.n = meta.n;
    this.d = meta.embDim;
    this.scale = meta.embScale;
    this.emb = emb;                                   // Int8Array n x d
  }

  /** Top-k por similaridade de cosseno (busca exata, força bruta sobre vetores int8). */
  search(q, k = 12) {
    const { n, d, emb } = this;
    const scores = new Float32Array(n);
    for (let i = 0, o = 0; i < n; i++, o += d) {
      let s = 0;
      for (let j = 0; j < d; j++) s += q[j] * emb[o + j];
      scores[i] = s;
    }
    const top = [];
    for (let i = 0; i < n; i++) {
      const s = scores[i];
      if (top.length < k || s > top[top.length - 1].s) {
        let p = top.length;
        while (p > 0 && top[p - 1].s < s) p--;
        top.splice(p, 0, { i, s });
        if (top.length > k) top.pop();
      }
    }
    return top.map(({ i, s }) => ({ i, score: s / this.scale }));
  }
}

export function cosine(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/**
 * Onde a pergunta "cai" no mapa: média das posições 3D das vizinhas, com peso
 * exponencial na similaridade (a UMAP não tem transformação inversa barata no
 * navegador; a vizinhança no espaço original é quem decide a posição).
 */
export function locate(hits, pos, take = 8, temperature = 0.035) {
  const top = hits.slice(0, take);
  let x = 0, y = 0, z = 0, w = 0;
  for (const h of top) {
    const wi = Math.exp((h.score - top[0].score) / temperature);
    x += pos[h.i * 3] * wi; y += pos[h.i * 3 + 1] * wi; z += pos[h.i * 3 + 2] * wi; w += wi;
  }
  return [x / w, y / w, z / w];
}
