import os, json, requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = "index_store"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "gemma3:1b"
TOP_K = 5


def query_ollama(model, prompt):
    url = "http://localhost:11434/api/generate"
    data = {"model": model, "prompt": prompt, "stream": False}
    resp = requests.post(url, json=data)
    return resp.json()["response"].strip()


def load_resources():
    index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    model = SentenceTransformer(EMB_MODEL)
    return index, meta, model


def retrieve(query, index, meta, model, k=TOP_K):
    q_emb = model.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb).astype("float32")
    _, I = index.search(q_emb, k)
    return [meta[i] for i in I[0]]


def format_context(items):
    blocks = []
    for it in items:
        blocks.append(f"[{it['title']} pág {it['page']}] {it['text']}")
    return "\n\n".join(blocks)


def main():
    index, meta, emb_model = load_resources()
    print("Sistema RAG local listo con Ollama. Escribí tu pregunta:")
    while True:
        q = input("\n> ")
        if q.strip().lower() in ("exit", "quit"):
            break
        chunks = retrieve(q, index, meta, emb_model)
        context = format_context(chunks)
        prompt = f"""Actuá como un asistente que SOLO puede responder usando la información provista en el CONTEXTO.

Reglas estrictas:
1. Si la información necesaria para responder NO se encuentra en el CONTEXTO, respondé exactamente: "No se encuentra en el material indexado."
2. No inventes, no completes con conocimientos externos ni con suposiciones.
3. No uses información general o ejemplos que no estén explícitamente en el CONTEXTO.
4. Respondé de forma breve, precisa y en español neutro.
5. Citá siempre la fuente y página entre paréntesis al final de cada párrafo o idea, con el formato: (fuente: [título del documento], pág. X).

PREGUNTA: {q}

CONTEXTO:
{context}

RESPUESTA:"""
        ans = query_ollama(OLLAMA_MODEL, prompt)
        print("\n=== RESPUESTA ===\n", ans)


if __name__ == "__main__":
    main()
