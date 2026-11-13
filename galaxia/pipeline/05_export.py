"""Etapa 5: empacota tudo em arquivos estáticos para o site (web/public/data).

- points.bin    Float32  n x 3     posição 3D de cada estrela
- cluster.bin   Int16    n         constelação (-1 = poeira, sem cluster)
- day.bin       Uint16   n         dias desde a primeira data do corpus
- emb.bin       Int8     n x 384   índice vetorial quantizado (1 byte por dimensão)
- titles.json   ids e títulos
- abs/NNN.json  resumos em blocos, carregados sob demanda pelo RAG
- meta.json     constelações, linha do tempo e métricas

    python 05_export.py
"""
import colorsys
import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import config as C

NAMES = Path(__file__).parent / "cluster_names.json"


def clip_text(s, n):
    if len(s) <= n:
        return s
    cut = s[:n]
    dot = cut.rfind(". ")
    return cut[: dot + 1] if dot > n * 0.6 else cut.rsplit(" ", 1)[0] + "…"


def recall_at_k(full, approx, k=10, n_queries=400, seed=0):
    """Quanto do top-k exato (384d float) o índice comprimido recupera."""
    rng = np.random.default_rng(seed)
    q = rng.choice(len(full), n_queries, replace=False)
    hits = 0
    for i in q:
        a = np.argsort(-(full @ full[i]))[1:k + 1]
        b = np.argsort(-(approx @ approx[i]))[1:k + 1]
        hits += len(set(a) & set(b))
    return hits / (n_queries * k)


def main():
    emb = np.load(C.DATA / "emb.npy")
    xyz = np.load(C.DATA / "xyz.npy")
    labels = np.load(C.DATA / "labels.npy")
    docs = [json.loads(l) for l in open(C.DATA / "docs.jsonl")]
    cl = json.load(open(C.DATA / "clusters.json"))
    names = json.loads(NAMES.read_text()) if NAMES.exists() else {}
    n = len(docs)

    out = C.WEB_DATA
    shutil.rmtree(out, ignore_errors=True)
    (out / "abs").mkdir(parents=True)

    # --- geometria e atributos ---------------------------------------------
    xyz.astype("<f4").tofile(out / "points.bin")
    labels.astype("<i2").tofile(out / "cluster.bin")
    d0 = date.fromisoformat(docs[0]["date"])
    days = np.array([(date.fromisoformat(d["date"]) - d0).days for d in docs])
    days.astype("<u2").tofile(out / "day.bin")

    # --- índice vetorial quantizado -----------------------------------------
    # float32 -> int8: 4x menor e praticamente sem perda de ranking. (Reduzir com
    # PCA antes parecia tentador, mas a 96d o recall@10 caía para ~73%.)
    scale = 127.0 / np.abs(emb).max()
    q8 = np.clip(np.round(emb * scale), -127, 127).astype(np.int8)
    q8.tofile(out / "emb.bin")
    recall = recall_at_k(emb, q8.astype(np.float32))
    print(f"recall@10 do índice int8 vs. float32: {recall:.3f}")

    # --- textos --------------------------------------------------------------
    json.dump({"ids": [d["id"] for d in docs], "titles": [d["title"] for d in docs]},
              open(out / "titles.json", "w"), ensure_ascii=False, separators=(",", ":"))
    for s in range(0, n, C.SHARD):
        shard = [{"a": clip_text(d["abstract"], C.ABSTRACT_CHARS),
                  "u": ", ".join(d["authors"][:3]) + (" et al." if len(d["authors"]) > 3 else ""),
                  "c": d["cat"]} for d in docs[s:s + C.SHARD]]
        json.dump(shard, open(out / "abs" / f"{s // C.SHARD:03d}.json", "w"),
                  ensure_ascii=False, separators=(",", ":"))

    # --- constelações --------------------------------------------------------
    clusters = []
    for c in cl["clusters"]:
        x, y, z = c["centroid"]
        hue = (np.arctan2(z, x) / (2 * np.pi) + 0.58) % 1.0   # vizinhos têm cores próximas
        light = 0.62 + 0.10 * np.tanh(y / 40)
        r, g, b = colorsys.hls_to_rgb(hue, light, 0.78)
        nm = names.get(str(c["id"]))
        clusters.append({
            "id": c["id"],
            "name": nm["name"] if nm else ", ".join(c["keywords"][:3]),
            "desc": nm["description"] if nm else "",
            "keywords": c["keywords"][:8],
            "size": c["size"],
            "centroid": c["centroid"],
            "color": "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255)),
            "stars": c["stars"],
            "edges": c["edges"],
            "top": c["representatives"][:5],
            "series": c["series"],
        })

    meta = {
        "n": n,
        "corpus": "arXiv · Ciência da Computação",
        "start": docs[0]["date"],
        "end": docs[-1]["date"],
        "days": int(days.max()),
        "months": cl["months"],
        "model": C.MODEL,
        "modelBrowser": C.MODEL_BROWSER,
        "embDim": C.EMB_DIM,
        "embScale": float(scale),
        "shard": C.SHARD,
        "recallAt10": round(recall, 3),
        "noise": round(float(np.mean(labels < 0)), 3),
        "namedBy": "llm" if names else "ctfidf",
        "clusters": clusters,
    }
    json.dump(meta, open(out / "meta.json", "w"), ensure_ascii=False, separators=(",", ":"))
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"{n} estrelas, {len(clusters)} constelações, {total / 1e6:.1f} MB em {out}")


if __name__ == "__main__":
    main()
