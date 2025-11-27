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
OLLAMA_MODEL = "qwen2.5:3b"
TOP_K = 8
MAX_CONTEXT_CHARS = 10000
# Nota: eliminamos truncado por tokens; usamos truncado por caracteres


# ===========================
# Carga de recursos (cacheados)
# ===========================
@st.cache_resource(show_spinner=False)
def load_faiss_and_meta():
    try:
        index = faiss.read_index(f"{INDEX_DIR}/faiss.index")
    except Exception as e:
        raise RuntimeError(f"Error cargando FAISS desde '{INDEX_DIR}/faiss.index': {e}")
    try:
        with open(f"{INDEX_DIR}/meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Error leyendo meta.json en '{INDEX_DIR}': {e}")
    return index, meta


@st.cache_resource(show_spinner=False)
def load_inverted_index():
    try:
        with open(f"{INDEX_DIR}/inverted_index.json", "r", encoding="utf-8") as f:
            inverted = json.load(f)
    except FileNotFoundError:
        # No existe índice léxico: devolvemos dict vacío y lo señalamos en la UI
        return {}
    except Exception as e:
        raise RuntimeError(f"Error leyendo inverted_index.json en '{INDEX_DIR}': {e}")
    return inverted


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    try:
        return SentenceTransformer(EMB_MODEL)
    except Exception as e:
        raise RuntimeError(f"Error cargando el modelo de embeddings '{EMB_MODEL}': {e}")


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
# Construcción de contexto (truncado por tokens estimados)
# ===========================
def build_context_from_items_by_chars(items: List[Dict], max_chars: int = MAX_CONTEXT_CHARS) -> Tuple[str, List[Dict]]:
    """Construye un contexto concatenando entradas hasta alcanzar `max_chars` caracteres.

    Devuelve el contexto (string) y la lista de items usados.
    """
    selected: List[Dict] = []
    total_chars = 0
    parts: List[str] = []

    for it in items:
        part = f"[{it['title']} pág {it['page']}] {it['text']}"
        if total_chars + len(part) > max_chars:
            if not selected:
                selected.append(it)
                parts.append(part[: max_chars])
            break
        selected.append(it)
        parts.append(part)
        total_chars += len(part)

    context = "\n\n".join(parts)
    return context, selected


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
    try:
        r = requests.post(
            url,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error conectando con Ollama en {url}: {e}")
    except ValueError:
        raise RuntimeError("Respuesta inválida desde Ollama: no se pudo decodificar JSON")


def build_prompt(q: str, context: str) -> str:
     return f"""
Eres un asistente en español. Solo puedes usar la INFORMACIÓN presente en el CONTEXTO para responder.

Reglas (seguir estrictamente):
1) Busca evidencia en el CONTEXTO que responda la PREGUNTA.
2) Si HAY evidencia suficiente: responde en máximo 3 frases en español. Usa SOLO información del CONTEXTO (puedes parafrasear). Después de cada afirmación factual añade la cita entre corchetes: [Título pág X].
3) Si NO hay evidencia suficiente: responde exactamente (sin comillas):
    No se encuentra en el material indexado.
4) No inventes, no añadas nada externo ni supongas detalles.
5) Si la evidencia es parcial, da la respuesta con lo que exista y cita las fuentes.

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
        show_debug = st.checkbox("Mostrar contexto y prompt (debug)", value=False)
        include_all_index = st.checkbox("Incluir todo el índice en el contexto", value=False)
        ignore_context_limit = False
        if include_all_index:
            ignore_context_limit = st.checkbox(
                "Ignorar límite de contexto (puede exceder la ventana del modelo)", value=False
            )
        st.caption("Los recursos e índices se cargan una vez y se cachean.")

    # ----- Carga de recursos -----
    try:
        index, meta = load_faiss_and_meta()
        inverted = load_inverted_index()
        emb_model = load_embedding_model()
        st.success("Índices y modelo cargados ✅")
    except RuntimeError as e:
        st.error(f"Error al cargar recursos: {e}")
        st.stop()

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
            items = cached_search(question, mode)[: k * 3]

            # Construir contexto basado en caracteres o incluir TODO el índice si se solicita
            if include_all_index:
                # Concatenar todo el meta (índice) en orden; advertir si se ignora el límite
                if ignore_context_limit:
                    st.warning(
                        "Has elegido incluir todo el índice sin límite: puede generar prompts muy largos y fallos en el modelo."
                    )
                    parts = [f"[{it['title']} pág {it['page']}] {it['text']}" for it in meta]
                    context = "\n\n".join(parts)
                    used_items = list(meta)
                else:
                    # Usar el builder por caracteres para incluir tanto como quepa
                    context, used_items = build_context_from_items_by_chars(meta, max_chars=MAX_CONTEXT_CHARS)
            else:
                # Construir contexto sólo a partir de los items recuperados
                context, used_items = build_context_from_items_by_chars(items, max_chars=MAX_CONTEXT_CHARS)
            
            # --- Filtro anti-alucinación ---
            if len(used_items) == 0 or len(context.strip()) < 50:
                st.subheader("🗣️ Respuesta")
                st.write("No se encuentra en el material indexado.")
                return
            prompt = build_prompt(question, context)

            # Mostrar contexto/prompt en UI si el usuario lo solicita (debug)
            if st.session_state.get("show_debug") is None:
                st.session_state["show_debug"] = show_debug
            if show_debug:
                with st.expander("🔍 Contexto (truncado) / Prompt enviado", expanded=True):
                    st.text_area("Contexto (truncado):", value=context, height=200)
                    st.text_area("Prompt enviado:", value=prompt, height=300)

            try:
                answer = query_ollama(prompt)
            except RuntimeError as e:
                st.error(f"Error en la consulta al LLM: {e}")
                answer = "No se pudo obtener respuesta del modelo (ver mensaje de error)."

        dt = (time.perf_counter() - t0) * 1000

        # ----- Mostrar resultados -----
        st.subheader("🗣️ Respuesta")
        st.write(answer)

        st.subheader("📚 Fuentes consultadas")
        for it in (used_items if 'used_items' in locals() else items)[:k]:
            st.markdown(f"- **{it['title']}** (pág {it['page']})")

        st.caption(f"⏱️ Tiempo total: {dt:.1f} ms | Modo: {mode}")


if __name__ == "__main__":
    main()
