"""Etapa 4: um LLM batiza cada constelação.

Para cada cluster o modelo recebe as palavras-chave (c-TF-IDF) e os títulos
mais centrais, e devolve um nome curto e uma descrição de uma linha, em
português. O resultado fica em pipeline/cluster_names.json (versionado), então
rodar de novo só nomeia clusters que ainda não têm nome.

    export ANTHROPIC_API_KEY=...
    python 04_name_clusters.py            # --force para renomear tudo
"""
import json
import sys
from pathlib import Path

import anthropic
from pydantic import BaseModel

import config as C

NAMES = Path(__file__).parent / "cluster_names.json"
MODEL = "claude-opus-5-5"
BATCH = 20

SYSTEM = """Você nomeia agrupamentos de artigos científicos para um mapa 3D em que \
cada agrupamento é uma constelação. Para cada agrupamento, escreva em português do Brasil:
- name: o tema em 2 a 4 palavras, como um rótulo de mapa (ex.: "Modelos de Difusão", \
"Robótica e Controle"). Mantenha em inglês os termos técnicos que a comunidade não traduz.
- description: uma frase de até 20 palavras dizendo do que tratam os artigos.
Nomes de agrupamentos diferentes precisam ser distinguíveis entre si."""


class Name(BaseModel):
    id: int
    name: str
    description: str


class Names(BaseModel):
    clusters: list[Name]


def main():
    force = "--force" in sys.argv
    clusters = json.load(open(C.DATA / "clusters.json"))["clusters"]
    names = {} if force or not NAMES.exists() else json.loads(NAMES.read_text())
    todo = [c for c in clusters if str(c["id"]) not in names]
    client = anthropic.Anthropic()

    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        blocks = []
        for c in batch:
            titles = "\n".join(f"  - {t}" for t in c["rep_titles"])
            blocks.append(f"Agrupamento {c['id']} ({c['size']} artigos)\n"
                          f"Palavras-chave: {', '.join(c['keywords'])}\nTítulos centrais:\n{titles}")
        taken = ", ".join(v["name"] for v in names.values()) or "(nenhum)"
        response = client.messages.parse(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM,
            messages=[{"role": "user", "content":
                       f"Nomes já usados: {taken}\n\n" + "\n\n".join(blocks)}],
            output_format=Names,
        )
        if response.stop_reason == "refusal" or response.parsed_output is None:
            raise SystemExit(f"O modelo não devolveu nomes (stop_reason={response.stop_reason})")
        for item in response.parsed_output.clusters:
            names[str(item.id)] = {"name": item.name, "description": item.description}
        NAMES.write_text(json.dumps(names, ensure_ascii=False, indent=1))
        print(f"{len(names)}/{len(clusters)} constelações nomeadas")


if __name__ == "__main__":
    main()
