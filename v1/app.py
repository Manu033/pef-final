import streamlit as st
import json, requests, faiss, numpy as np
from sentence_transformers import SentenceTransformer

# Configuración
INDEX_DIR = "index_store"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "gemma3:1b"
TOP_K = 8


@st.cache_resource
def load_resources():
    index = faiss.read_index(f"{INDEX_DIR}/faiss.index")
    with open(f"{INDEX_DIR}/meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    emb_model = SentenceTransformer(EMB_MODEL)
    return index, meta, emb_model


def retrieve(query, index, meta, emb_model, k=TOP_K):
    # Encode query and search
    q_emb = emb_model.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb).astype("float32")

    # Get more results initially
    initial_k = min(20, len(meta))
    scores, I = index.search(q_emb, initial_k)

    # Filter results with a score threshold
    results = []
    total_chars = 0
    max_chars = 6000  # Aproximadamente 1000 tokens

    for score, idx in zip(scores[0], I[0]):
        if total_chars > max_chars:
            break

        doc = meta[idx]
        total_chars += len(doc["text"])
        results.append(doc)

    return results


def query_ollama(prompt):
    url = "http://localhost:11434/api/generate"
    r = requests.post(
        url, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    )
    return r.json()["response"].strip()


def build_prompt(q, context):
    return f"""
Sos un asistente que debe responder usando el CONTEXTO proporcionado.
- Usá SOLO la información del CONTEXTO
- Si no hay información relevante, respondé "No se encuentra en el material indexado"
- Citá las fuentes usando [título pág X]
- Sé conciso y claro

PREGUNTA:
{q}

CONTEXTO:
{context}

RESPUESTA:
"""


def main():
    st.set_page_config(page_title="Asistente local con PDFs", layout="centered")
    st.title("🧠 RAG Local - Preguntas sobre tus PDFs")

    index, meta, emb_model = load_resources()
    st.success("Índice cargado correctamente ✅")

    question = st.text_input("Escribí tu pregunta:", "")
    if question:
        with st.spinner("Buscando información..."):
            items = retrieve(question, index, meta, emb_model)
            context = "\n\n".join(
                [f"[{it['title']} pág {it['page']}] {it['text']}" for it in items]
            )

            prompt = build_prompt(question, context)
            answer = query_ollama(prompt)

        st.subheader("🗣️ Respuesta")
        st.write(answer)

        st.subheader("📚 Fuentes consultadas")
        for it in items:
            st.markdown(f"- **{it['title']}** (pág {it['page']})")


if __name__ == "__main__":
    main()
