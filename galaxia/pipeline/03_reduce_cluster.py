"""Etapa 3: UMAP (3D para visualizar, 8D para agrupar) + HDBSCAN + c-TF-IDF.

Saídas em data/: xyz.npy, labels.npy, clusters.json

    python 03_reduce_cluster.py
"""
import json

import hdbscan
import numpy as np
import umap
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import CountVectorizer

import config as C

EXTRA_STOP = {
    "paper", "propose", "proposed", "based", "method", "methods", "approach", "results",
    "using", "use", "used", "show", "new", "problem", "model", "models", "data", "work",
    "present", "novel", "study", "performance", "algorithm", "algorithms", "existing",
    "different", "provide", "also", "however", "two", "one", "can", "task", "tasks",
    "framework", "analysis", "system", "systems", "learning", "network", "networks",
    "state", "art", "time", "large", "demonstrate", "experiments", "information",
}


def ctfidf_keywords(texts, labels, n_clusters, top=10):
    """Palavras-chave por cluster: TF por classe x IDF entre classes (c-TF-IDF)."""
    joined = [" ".join(t for t, l in zip(texts, labels) if l == c) for c in range(n_clusters)]
    stop = list(CountVectorizer(stop_words="english").get_stop_words() | EXTRA_STOP)
    cv = CountVectorizer(stop_words=stop, ngram_range=(1, 2), min_df=2, max_features=60000)
    tf = cv.fit_transform(joined).toarray().astype(np.float64)
    words = np.array(cv.get_feature_names_out())
    avg = tf.sum() / n_clusters
    idf = np.log(1 + avg / tf.sum(axis=0))
    score = tf / tf.sum(axis=1, keepdims=True) * idf
    out = []
    for c in range(n_clusters):
        picked = []
        for w in words[np.argsort(-score[c])]:
            # evita redundância ("neural" + "neural network")
            if not any(w in p or p in w for p in picked):
                picked.append(w)
            if len(picked) == top:
                break
        out.append(picked)
    return out


def constellation(points, k=7, seed=0):
    """Escolhe k estrelas do núcleo do cluster e liga por uma árvore geradora
    mínima: o desenho da 'constelação'. Devolve (índices locais, arestas)."""
    center = np.median(points, axis=0)
    d = np.linalg.norm(points - center, axis=1)
    core = np.argsort(d)[: max(k * 5, int(len(points) * 0.10))]
    chosen = [core[0]]
    dist = np.linalg.norm(points[core] - points[core[0]], axis=1)
    for _ in range(k - 1):  # farthest-point sampling
        nxt = int(np.argmax(dist))
        chosen.append(core[nxt])
        dist = np.minimum(dist, np.linalg.norm(points[core] - points[core[nxt]], axis=1))
    P = points[chosen]
    mst = minimum_spanning_tree(cdist(P, P)).tocoo()
    return [int(i) for i in chosen], [[int(a), int(b)] for a, b in zip(mst.row, mst.col)]


def main():
    emb = np.load(C.DATA / "emb.npy")
    docs = [json.loads(l) for l in open(C.DATA / "docs.jsonl")]
    n = len(docs)

    print("UMAP 3D...")
    xyz = umap.UMAP(n_components=3, random_state=C.SEED, **C.UMAP_3D).fit_transform(emb)
    xyz = xyz - np.median(xyz, axis=0)
    xyz = (xyz / np.percentile(np.linalg.norm(xyz, axis=1), 98) * 100).astype(np.float32)

    print("UMAP 8D + HDBSCAN...")
    low = umap.UMAP(random_state=C.SEED, **C.UMAP_CLUSTER).fit_transform(emb)
    raw = hdbscan.HDBSCAN(**C.HDBSCAN).fit_predict(low)

    # Reordena os clusters por tamanho (0 = maior); ruído continua -1.
    ids, counts = np.unique(raw[raw >= 0], return_counts=True)
    remap = {int(old): new for new, old in enumerate(ids[np.argsort(-counts)])}
    labels = np.array([remap.get(int(l), -1) for l in raw], dtype=np.int16)
    k = len(remap)
    print(f"{k} clusters, {np.mean(labels < 0):.1%} de ruído")

    texts = [f"{d['title']} {d['abstract']}" for d in docs]
    keywords = ctfidf_keywords(texts, labels, k)

    months = sorted({d["date"][:7] for d in docs})
    m_index = {m: i for i, m in enumerate(months)}
    clusters = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        centroid_e = emb[idx].mean(axis=0)
        centroid_e /= np.linalg.norm(centroid_e)
        rep = idx[np.argsort(-(emb[idx] @ centroid_e))[:12]]
        stars, edges = constellation(xyz[idx])
        series = np.zeros(len(months), dtype=int)
        for i in idx:
            series[m_index[docs[i]["date"][:7]]] += 1
        clusters.append({
            "id": c,
            "size": int(len(idx)),
            "keywords": keywords[c],
            "centroid": [round(float(v), 2) for v in np.median(xyz[idx], axis=0)],
            "representatives": [int(i) for i in rep],
            "rep_titles": [docs[i]["title"] for i in rep],
            "stars": [int(idx[s]) for s in stars],
            "edges": edges,
            "series": series.tolist(),
        })

    np.save(C.DATA / "xyz.npy", xyz)
    np.save(C.DATA / "labels.npy", labels)
    json.dump({"months": months, "clusters": clusters}, open(C.DATA / "clusters.json", "w"),
              ensure_ascii=False)
    for c in clusters[:60]:
        print(f"{c['id']:3d} {c['size']:5d}  {', '.join(c['keywords'][:6])}")


if __name__ == "__main__":
    main()
