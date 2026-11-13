"""Backend opcional do RAG: recebe a pergunta e os ids das estrelas vizinhas
(a busca semântica já aconteceu no navegador) e devolve, em streaming, uma
resposta do Claude fundamentada nos resumos desses artigos.

    export ANTHROPIC_API_KEY=...
    uvicorn app:app --port 8000

O servidor nunca aceita texto de contexto vindo do cliente: ele mesmo lê os
resumos pelos ids, então o endpoint não serve de proxy genérico para o LLM.
Se web/dist existir, o site também é servido daqui (um único deploy).
"""
import json
import os
import time
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "web" / "dist"
DATA = Path(os.environ.get("GALAXIA_WEB_DATA", ROOT / "web" / "public" / "data"))
MODEL = os.environ.get("GALAXIA_MODEL", "claude-opus-5-5")
ORIGINS = os.environ.get("GALAXIA_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
RATE_LIMIT, RATE_WINDOW = 20, 600      # perguntas por IP a cada 10 minutos
MAX_SOURCES = 8

SYSTEM = """Você responde perguntas dentro de um mapa 3D de artigos científicos do arXiv. \
A cada pergunta você recebe os resumos dos artigos mais próximos dela no espaço de embeddings.

Responda usando apenas o que esses resumos sustentam e cite cada afirmação com o número \
do artigo entre colchetes, como [2] ou [1][4]. Se os resumos não bastarem para responder, \
diga isso e conte o que eles cobrem. Escreva no idioma da pergunta, em dois ou três \
parágrafos curtos de texto corrido, sem títulos nem listas: a resposta aparece em um \
painel lateral estreito."""

app = FastAPI(title="Galáxia de Embeddings · RAG")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST"], allow_headers=["content-type"])
client = anthropic.AsyncAnthropic()
hits: dict[str, deque] = defaultdict(deque)


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=300)
    ids: list[int] = Field(min_length=1, max_length=MAX_SOURCES)


@lru_cache(maxsize=1)
def corpus():
    meta = json.loads((DATA / "meta.json").read_text())
    titles = json.loads((DATA / "titles.json").read_text())["titles"]
    return meta, titles


@lru_cache(maxsize=64)
def shard(k: int):
    return json.loads((DATA / "abs" / f"{k:03d}.json").read_text())


def context(ids: list[int]) -> str:
    meta, titles = corpus()
    blocks = []
    for n, i in enumerate(ids, 1):
        if not 0 <= i < meta["n"]:
            raise HTTPException(422, "id de artigo inexistente")
        doc = shard(i // meta["shard"])[i % meta["shard"]]
        blocks.append(f'<artigo n="{n}">\n<titulo>{titles[i]}</titulo>\n<resumo>{doc["a"]}</resumo>\n</artigo>')
    return "\n\n".join(blocks)


def throttle(ip: str):
    now, q = time.monotonic(), hits[ip]
    while q and now - q[0] > RATE_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        raise HTTPException(429, "muitas perguntas em pouco tempo; tente de novo em alguns minutos")
    q.append(now)


@app.get("/api/health")
def health():
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    return {"ok": has_key, "model": MODEL}


@app.post("/api/ask")
async def ask(body: Ask, request: Request):
    throttle(request.client.host if request.client else "?")
    prompt = f"{context(body.ids)}\n\nPergunta: {body.question}"

    async def generate():
        try:
            async with client.beta.messages.stream(
                model=MODEL,
                max_tokens=16000,
                system=SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": "low"},          # resposta curta: prioriza latência
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",                       # se o modelo recusar, outro assume
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                final = await stream.get_final_message()
                if final.stop_reason == "refusal":
                    yield "\n\nO modelo preferiu não responder a esta pergunta."
        except anthropic.RateLimitError:
            yield "\n\nO LLM está com muitas requisições agora. Tente de novo em instantes."
        except anthropic.APIStatusError as e:
            yield f"\n\nO LLM devolveu um erro ({e.status_code})."
        except anthropic.APIConnectionError:
            yield "\n\nNão consegui falar com o LLM."

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8",
                             headers={"X-Accel-Buffering": "no", "Cache-Control": "no-store"})


if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="site")
