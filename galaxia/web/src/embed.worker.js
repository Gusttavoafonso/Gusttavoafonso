// Web Worker: roda o modelo de embeddings (ONNX, via transformers.js) fora da thread da UI.
import { env, pipeline } from '@huggingface/transformers';

let extractor = null;

self.onmessage = async ({ data }) => {
  const { id, type } = data;
  try {
    if (!extractor) {
      env.backends.onnx.wasm.wasmPaths = data.wasm;   // runtime ONNX servido pelo próprio site
      const seen = new Map();
      extractor = pipeline('feature-extraction', data.model, {
        dtype: 'q8',
        progress_callback: (p) => {
          if (p.status !== 'progress' || !p.total) return;
          seen.set(p.file, [p.loaded, p.total]);
          let loaded = 0, total = 0;
          for (const [l, t] of seen.values()) { loaded += l; total += t; }
          self.postMessage({ type: 'progress', loaded, total });
        },
      });
    }
    const model = await extractor;
    if (type === 'load') return self.postMessage({ id, ok: true });
    const out = await model(data.texts, { pooling: 'mean', normalize: true });
    self.postMessage({ id, ok: true, dims: out.dims, vectors: out.data }, [out.data.buffer]);
  } catch (err) {
    extractor = null;
    self.postMessage({ id, ok: false, error: String(err?.message || err) });
  }
};
