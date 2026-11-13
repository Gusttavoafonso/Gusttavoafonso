"""Etapa 2: gera os embeddings (sentence-transformers) de título + resumo.

    python 02_embed.py
"""
import json

import numpy as np
from sentence_transformers import SentenceTransformer

import config as C


def main():
    docs = {}
    for f in sorted(C.RAW.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                docs[d["id"]] = d
    docs = sorted(docs.values(), key=lambda d: (d["date"], d["id"]))
    print(f"{len(docs)} documentos únicos")

    model = SentenceTransformer(C.MODEL, device="cpu")
    model.max_seq_length = 256
    texts = [f"{d['title']}. {d['abstract']}" for d in docs]
    emb = model.encode(
        texts, batch_size=128, normalize_embeddings=True,
        show_progress_bar=True, convert_to_numpy=True,
    ).astype(np.float32)

    C.DATA.mkdir(exist_ok=True)
    np.save(C.DATA / "emb.npy", emb)
    with open(C.DATA / "docs.jsonl", "w") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print("embeddings:", emb.shape)


if __name__ == "__main__":
    main()
