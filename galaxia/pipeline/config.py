"""Configuração central do pipeline da Galáxia de Embeddings."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("GALAXIA_DATA", ROOT / "data"))   # artefatos intermediários (fora do git)
RAW = DATA / "raw"
WEB_DATA = ROOT / "web" / "public" / "data"   # o que o site consome

# --- Corpus -----------------------------------------------------------------
# Ciência da Computação no arXiv (todas as subáreas cs.*), amostrada mês a mês.
ARXIV_QUERY = "cat:cs.*"
START = (2012, 1)
END = (2026, 10)
# A amostra por mês cresce linearmente, acompanhando (de forma suavizada)
# o crescimento real do arXiv.
PER_MONTH_START = 170
PER_MONTH_END = 420

# --- Embeddings -------------------------------------------------------------
# O mesmo modelo roda no navegador (Xenova/all-MiniLM-L6-v2 via transformers.js),
# então a pergunta do usuário cai no MESMO espaço vetorial dos documentos.
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_BROWSER = "Xenova/all-MiniLM-L6-v2"
EMB_DIM = 384

# --- UMAP / HDBSCAN ---------------------------------------------------------
SEED = 42
UMAP_3D = dict(n_neighbors=30, min_dist=0.08, metric="cosine")
UMAP_CLUSTER = dict(n_components=8, n_neighbors=30, min_dist=0.0, metric="cosine")
HDBSCAN = dict(min_cluster_size=110, min_samples=12, cluster_selection_method="eom")

# --- Exportação -------------------------------------------------------------
ABSTRACT_CHARS = 700
SHARD = 500
