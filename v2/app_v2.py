#!/usr/bin/env python3
"""
app_v2.py
---------
Front-end Streamlit para consulta híbrida (semántica + lexical) sobre PDFs indexados localmente.
Incluye:
- Cache de recursos y cache de consultas frecuentes (memoización).
- Modos de búsqueda: SEMÁNTICA, LEXICAL y HÍBRIDA.
- Límite de contexto por caracteres para prompt.
- UI con métricas simples de tiempo.
"""
from __future__ import annotations

import json
from statistics import mode
import time
from functools import lru_cache
from typing import Dict, Iterable, List, Tuple

import faiss  # type: ignore
import numpy as np
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer


# ===========================
# Configuración
# ===========================
INDEX_DIR = "index_store"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "gemma3:1b"
TOP_K = 8
MAX_CONTEXT_CHARS = 6000


# ===========================
# Carga de recursos (cacheados)
# ===========================
@st.cache_resource(show_spinner=False)
def load_faiss_and_meta():
    index = faiss.read_index(f"{INDEX_DIR}/faiss.index")
    with open(f"{INDEX_DIR}/meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta


@st.cache_resource(show_spinner=False)
def load_inverted_index():
    with open(f"{INDEX_DIR}/inverted_index.json", "r", encoding="utf-8") as f:
        inverted = json.load(f)
    return inverted


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(EMB_MODEL)


# ===========================
# Búsquedas
# ===========================
def semantic_retrieve(
    query: str,
    index,
    meta: List[Dict],
    emb_model,
    k: int = TOP_K,
    initial_k: int | None = None,
) -> List[Dict]:
    """
    Recupera por similitud semántica usando FAISS.
    """
    if initial_k is None:
        initial_k = min(20, len(meta))

    q_emb = emb_model.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb).astype("float32")
    scores, I = index.search(q_emb, initial_k)

    results: List[Dict] = []
    total_chars = 0
    for score, idx in zip(scores[0], I[0]):
        if idx < 0:
            continue
        doc = meta[idx]
        total_chars += len(doc["text"])
        if total_chars > MAX_CONTEXT_CHARS:
            break
        results.append(doc)
        if len(results) >= k:
            break
    return results


def lexical_retrieve(
    query: str, inverted: Dict[str, List[int]], meta: List[Dict], k: int = TOP_K
) -> List[Dict]:
    """
    Recupera por términos exactos usando índice invertido.
    Operador AND simple entre tokens de la consulta.
    """
    tokens = [t.strip().lower() for t in query.split() if t.strip()]
    if not tokens:
        return []

    # Intersección de postings (AND)
    postings_lists = [set(inverted.get(tok, [])) for tok in tokens]
    if not postings_lists:
        return []

    ids = set.intersection(*postings_lists) if postings_lists else set()
    # Orden simple por longitud de texto asc/desc (heurística); podría mejorarse con tf-idf
    ranked = sorted(ids, key=lambda i: -len(meta[i]["text"]))
    results = [meta[i] for i in ranked[:k]]
    return results


def hybrid_retrieve(
    query: str, index, inverted, meta, emb_model, k: int = TOP_K
) -> List[Dict]:
    """
    Combina resultados semánticos y lexicals, evitando duplicados.
    """
    sem = semantic_retrieve(query, index, meta, emb_model, k=k)
    lex = lexical_retrieve(query, inverted, meta, k=k)

    # Merge único preservando orden: primero semánticos, luego los lexicals no incluidos
    seen = set()
    merged: List[Dict] = []
    for item in sem + lex:
        key = (item["title"], item["page"], item["text"][:64])
        if key not in seen:
            merged.append(item)
            seen.add(key)
        if len(merged) >= k:
            break
    return merged


# ===========================
# Memoización de consultas
# ===========================
@lru_cache(maxsize=128)
def cached_search(query: str, mode: str) -> List[Dict]:
    index, meta = load_faiss_and_meta()
    inverted = load_inverted_index()
    emb_model = load_embedding_model()

    if mode == "Semántica":
        return semantic_retrieve(query, index, meta, emb_model, k=TOP_K)
    if mode == "Lexical":
        return lexical_retrieve(query, inverted, meta, k=TOP_K)
    return hybrid_retrieve(query, index, inverted, meta, emb_model, k=TOP_K)


# ===========================
# LLM local (Ollama)
# ===========================
def query_ollama(prompt: str) -> str:
    url = "http://localhost:11434/api/generate"
    r = requests.post(
        url,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["response"].strip()


def build_prompt(q: str, context: str) -> str:
    return f"""
Eres un asistente que solo puede usar la INFORMACIÓN explícita presente en el CONTEXTO para responder.

INSTRUCCIONES (seguir al pie de la letra):
1) Antes de responder, busca en el CONTEXTO si hay información que responda la PREGUNTA.
2) Si encuentras información suficiente para responder, redacta una respuesta breve y clara en español usando SOLO el CONTEXTO. 
   - Después de cualquier afirmación factual, añade la cita de la fuente entre corchetes en el formato: [título pág X].
   - Si la información útil proviene de varias entradas, puedes citar varias fuentes.
3) Si NO encuentras información suficiente en el CONTEXTO para responder la pregunta, responde exactamente (sin comillas):
   No se encuentra en el material indexado.
4) No inventes, no añadas explicaciones externas, ni uses conocimientos previos. Usa únicamente lo que aparece en el CONTEXTO.
5) Si la información en el CONTEXTO es parcial, responde con lo que haya y usa las citas correspondientes; no rellenes con supuestos.
6) Responde en español, de forma concisa (máximo 3-5 frases).

PREGUNTA:
{q}

CONTEXTO:
{context}

RESPUESTA:
"""


# ===========================
# UI
# ===========================
def main() -> None:
    st.set_page_config(page_title="Asistente local con PDFs (V2)", layout="centered")
    st.title("🧠 RAG Local V2 - Búsqueda Híbrida sobre tus PDFs")

    # ----- Configuración lateral -----
    with st.sidebar:
        st.markdown("### ⚙️ Configuración")
        mode = st.radio(
            "Modo de búsqueda", ["Híbrida", "Semántica", "Lexical"], index=0
        )
        k = st.slider("Resultados (k)", 3, 20, TOP_K)
        st.caption("Los recursos e índices se cargan una vez y se cachean.")

    # ----- Carga de recursos -----
    index, meta = load_faiss_and_meta()
    inverted = load_inverted_index()
    emb_model = load_embedding_model()
    st.success("Índices y modelo cargados ✅")

    # ----- Entrada de usuario -----
    question = st.text_input("Escribí tu pregunta:", "")
    buscar = st.button("🔎 Buscar")

    # Detectar si se presionó Enter (Streamlit recarga y mantiene el texto)
    enter_pressed = question and st.session_state.get("last_query") != question

    # Ejecutar búsqueda solo si se hace click o se presiona Enter
    if question and (buscar or enter_pressed):
        st.session_state["last_query"] = question  # Guarda la última consulta
        t0 = time.perf_counter()

        with st.spinner("Buscando información..."):
            items = cached_search(question, mode)[:k]
            context = "\n\n".join(
                [f"[{it['title']} pág {it['page']}] {it['text']}" for it in items]
            )[:MAX_CONTEXT_CHARS]
            prompt = build_prompt(question, context)
            answer = query_ollama(prompt)

        dt = (time.perf_counter() - t0) * 1000

        # ----- Mostrar resultados -----
        st.subheader("🗣️ Respuesta")
        st.write(answer)

        st.subheader("📚 Fuentes consultadas")
        for it in items:
            st.markdown(f"- **{it['title']}** (pág {it['page']})")

        st.caption(f"⏱️ Tiempo total: {dt:.1f} ms | Modo: {mode}")


if __name__ == "__main__":
    main()
