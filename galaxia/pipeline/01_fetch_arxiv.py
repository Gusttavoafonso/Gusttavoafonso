"""Etapa 1: baixa uma amostra estratificada por mês do arXiv (API oficial).

Cada mês é dividido em duas quinzenas para a amostra não ficar concentrada
nos primeiros dias. O script é retomável: cada mês vira um arquivo em
data/raw/ e meses já baixados são pulados.

    python 01_fetch_arxiv.py
"""
import calendar
import json
import re
import sys
import time
import xml.etree.ElementTree as ET

import requests

import config as C

API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
PAUSE = 3.5  # a API pede no mínimo 3 s entre requisições
session = requests.Session()
session.headers["User-Agent"] = "galaxia-embeddings/1.0 (research sample)"


def months():
    y, m = C.START
    while (y, m) <= C.END:
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def quota(i, n):
    """Amostra do i-ésimo mês (rampa linear)."""
    t = i / max(n - 1, 1)
    return round(C.PER_MONTH_START + t * (C.PER_MONTH_END - C.PER_MONTH_START))


def clean(s):
    return re.sub(r"\s+", " ", s or "").strip()


def query(d0, d1, n):
    params = {
        "search_query": f"{C.ARXIV_QUERY} AND submittedDate:[{d0}0000 TO {d1}2359]",
        "start": 0,
        "max_results": n,
        "sortBy": "submittedDate",
        "sortOrder": "ascending",
    }
    wait = 20
    for attempt in range(12):
        try:
            r = session.get(API, params=params, timeout=120)
            if r.status_code == 200 and "<feed" in r.text:
                root = ET.fromstring(r.text)
                entries = root.findall("a:entry", NS)
                total = int(root.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults").text)
                # A API às vezes devolve página vazia mesmo havendo resultados.
                if entries or total == 0:
                    return entries
            print(f"    tentativa {attempt + 1}: HTTP {r.status_code}, aguardando {wait}s", flush=True)
        except (requests.RequestException, ET.ParseError) as e:
            print(f"    tentativa {attempt + 1}: {type(e).__name__}, aguardando {wait}s", flush=True)
        time.sleep(wait)
        wait = min(wait * 1.6, 240)
    raise RuntimeError(f"arXiv não respondeu para {d0}-{d1}")


def parse(e):
    url = e.find("a:id", NS).text
    arxiv_id = re.sub(r"v\d+$", "", url.rsplit("/abs/", 1)[-1])
    return {
        "id": arxiv_id,
        "title": clean(e.find("a:title", NS).text),
        "abstract": clean(e.find("a:summary", NS).text),
        "date": e.find("a:published", NS).text[:10],
        "cat": e.find("x:primary_category", NS).attrib["term"],
        "authors": [clean(a.find("a:name", NS).text) for a in e.findall("a:author", NS)][:6],
    }


def main():
    C.RAW.mkdir(parents=True, exist_ok=True)
    all_months = list(months())
    for i, (y, m) in enumerate(all_months):
        out = C.RAW / f"{y}-{m:02d}.jsonl"
        if out.exists():
            continue
        n = quota(i, len(all_months))
        last = calendar.monthrange(y, m)[1]
        halves = [(f"{y}{m:02d}01", f"{y}{m:02d}15"), (f"{y}{m:02d}16", f"{y}{m:02d}{last}")]
        docs = {}
        for d0, d1 in halves:
            for e in query(d0, d1, n // 2):
                d = parse(e)
                if len(d["abstract"]) > 200:
                    docs[d["id"]] = d
            time.sleep(PAUSE)
        tmp = out.with_suffix(".tmp")
        tmp.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in docs.values()))
        tmp.rename(out)
        print(f"{y}-{m:02d}: {len(docs)} artigos ({i + 1}/{len(all_months)})", flush=True)


if __name__ == "__main__":
    sys.exit(main())
