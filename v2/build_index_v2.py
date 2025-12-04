#!/usr/bin/env python3
"""
build_index_v2.py
-----------------
Construye:
- Índice semántico FAISS
- Índice léxico invertido
- meta.json con chunks por página

Compatible con app_v2.py
"""

from __future__ import annotations
import os
import json
from pathlib import Path
from typing import List, Dict, Iterable

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import re

from concurrent.futures import ProcessPoolExecutor   # ← AÑADIDO

# ===========================
# CONFIG
# ===========================
INDEX_DIR = "index_store"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
BATCH_SIZE = 64


# ===========================
# CHUNKING
# ===========================
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> Iterable[str]:
    text = text.strip()
    if len(text) <= size:
        yield text
        return

    step = size - overlap
    i = 0
    while i < len(text):
        yield text[i : i + size]
        i += step


# ===========================
# ÍNDICE INVERTIDO
# ===========================
def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-ZáéíóúÁÉÍÓÚñÑ]+", text.lower())
    return tokens


def build_inverted_index(meta: List[Dict]) -> Dict[str, List[int]]:
    inverted: Dict[str, set] = {}

    for i, entry in enumerate(meta):
        tokens = tokenize(entry["text"])
        for tok in tokens:
            if tok not in inverted:
                inverted[tok] = set()
            inverted[tok].add(i)

    return {word: sorted(list(ids)) for word, ids in inverted.items()}


# ===========================
# STREAMING / GENERATORS
# ===========================
def stream_pdf_pages(pdf_path: str):
    """Generator que devuelve (page_num, text) sin cargar todo el PDF en RAM."""
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            yield page_num, text


def stream_pdf_lines(pdf_path: str):
    for page_num, text in stream_pdf_pages(pdf_path):
        for line in (text or "").splitlines():
            line = line.strip()
            if line:
                yield page_num, line


# ===========================
# EXTRAER PDF → META
# ===========================
def extract_chunks_from_pdf(pdf_path: str) -> List[Dict]:
    """Función que se ejecuta en paralelo por cada PDF."""
    title = Path(pdf_path).stem
    meta_entries = []

    for page_num, text in stream_pdf_pages(pdf_path):
        text = text.strip()
        if not text:
            continue

        for ch in chunk_text(text):
            meta_entries.append({
                "title": title,
                "page": page_num,
                "text": ch
            })

    return meta_entries


# ===========================
# BATCH ENCODING
# ===========================
def encode_in_batches(model: SentenceTransformer, texts: List[str], batch_size: int = BATCH_SIZE, normalize: bool = True):
    if not texts:
        dim = model.get_sentence_embedding_dimension()
        return np.empty((0, dim), dtype="float32")

    embeddings_parts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, normalize_embeddings=normalize)
        emb = np.array(emb, dtype="float32")
        embeddings_parts.append(emb)
        print(f"  -> encoded batch {i//batch_size + 1} ({len(batch)} items)")

    return np.vstack(embeddings_parts)


# ===========================
# MAIN INDEX BUILDER
# ===========================
def build_index(pdf_dir: str = "pdfs"):
    os.makedirs(INDEX_DIR, exist_ok=True)

    print("Cargando modelo de embeddings...")
    model = SentenceTransformer(EMB_MODEL)

    print(f"Buscando PDFs en {pdf_dir} ...")
    pdf_files = sorted(Path(pdf_dir).glob("*.pdf"))

    if not pdf_files:
        raise RuntimeError(f"No hay PDFs en la carpeta {pdf_dir}")

    print("Procesando PDFs en paralelo...")
    all_meta: List[Dict] = []

    # ← CONCURRENCIA REAL AQUÍ
    with ProcessPoolExecutor() as executor:
        results = executor.map(extract_chunks_from_pdf, map(str, pdf_files))

        for entries in results:
            all_meta.extend(entries)

    print(f"Total de chunks: {len(all_meta)}")

    with open(f"{INDEX_DIR}/meta.json", "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=2)

    # --------------------------
    # FAISS
    # --------------------------
    print("Generando embeddings en batches...")
    texts = [m["text"] for m in all_meta]
    embeddings = encode_in_batches(model, texts, batch_size=BATCH_SIZE, normalize=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    print("Construyendo índice FAISS...")
    index.add(embeddings)

    faiss.write_index(index, f"{INDEX_DIR}/faiss.index")

    # --------------------------
    # INVERTED INDEX
    # --------------------------
    print("Construyendo índice léxico...")
    inv = build_inverted_index(all_meta)

    with open(f"{INDEX_DIR}/inverted_index.json", "w", encoding="utf-8") as f:
        json.dump(inv, f, ensure_ascii=False, indent=2)

    print("\n🎉 Índices generados correctamente en index_store/")


# ===========================
# ENTRYPOINT
# ===========================
if __name__ == "__main__":
    import sys
    
    if "--profile" in sys.argv:
        import cProfile
        import pstats
        from io import StringIO
        
        print("🔍 Ejecutando con profiling...\n")
        profiler = cProfile.Profile()
        profiler.enable()
        
        build_index("../data_pdfs")
        
        profiler.disable()
        
        s = StringIO()
        stats = pstats.Stats(profiler, stream=s)
        stats.strip_dirs()
        stats.sort_stats('cumulative')
        stats.print_stats(25)
        
        print("\n" + "="*80)
        print("📊 PROFILING - Top 25 funciones por tiempo acumulado")
        print("="*80)
        print(s.getvalue())
    else:
        build_index("../data_pdfs")
