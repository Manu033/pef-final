import os, json, fitz, faiss, numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

DATA_DIR = "data_pdfs"
OUT_DIR = "index_store"

# Modelo de embeddings liviano
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Parámetros de segmentación
CHUNK_SIZE = 800
OVERLAP = 200


def extract_text_from_pdf(path):
    """Extrae texto de todas las páginas de un PDF."""
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").replace("\n", " ")
        pages.append({"page": i + 1, "text": text})
    return pages


def chunk_text(text):
    """Divide texto en fragmentos (chunks)."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + CHUNK_SIZE)
        chunks.append(text[start:end])
        start += CHUNK_SIZE - OVERLAP
    return chunks


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = SentenceTransformer(EMB_MODEL)
    corpus = []

    print(">> Construyendo corpus desde PDFs...")
    for pdf in Path(DATA_DIR).glob("*.pdf"):
        print(f"   Procesando {pdf.name}")
        for page in extract_text_from_pdf(str(pdf)):
            for chunk in chunk_text(page["text"]):
                corpus.append({"title": pdf.stem, "page": page["page"], "text": chunk})

    print(f">> Total de fragmentos: {len(corpus)}")

    texts = [c["text"] for c in corpus]
    embeddings = model.encode(texts, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, os.path.join(OUT_DIR, "faiss.index"))
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)

    print("✅ Índice creado y guardado en", OUT_DIR)


if __name__ == "__main__":
    main()
