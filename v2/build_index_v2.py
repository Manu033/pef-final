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
import pdfplumber
import re

# ===========================
# CONFIG
# ===========================
INDEX_DIR = "index_store"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200


# ===========================
# CHUNKING
# ===========================
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> Iterable[str]:
    """
    Divide el texto en chunks solapados.
    Si el texto es más corto que size → devuelve un solo chunk.
    """
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
    """Tokenización básica (minúsculas + elimina no letras)."""
    tokens = re.findall(r"[a-zA-ZáéíóúÁÉÍÓÚñÑ]+", text.lower())
    return tokens


def build_inverted_index(meta: List[Dict]) -> Dict[str, List[int]]:
    """
    Construye un índice invertido:
        palabra → [ids de chunks donde aparece]
    IDs sin duplicados.
    """
    inverted: Dict[str, set] = {}

    for i, entry in enumerate(meta):
        tokens = tokenize(entry["text"])
        for tok in tokens:
            if tok not in inverted:
                inverted[tok] = set()
            inverted[tok].add(i)

    # convertimos sets a listas
    return {word: sorted(list(ids)) for word, ids in inverted.items()}


# ===========================
# EXTRAER PDF → META
# ===========================
def extract_chunks_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Devuelve lista de:
        { "title": str, "page": int, "text": str }
    por cada chunk.
    """
    title = Path(pdf_path).stem
    meta_entries = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text() or ""
            raw_text = raw_text.strip()

            if not raw_text:
                continue

            for ch in chunk_text(raw_text):
                meta_entries.append({
                    "title": title,
                    "page": page_num,
                    "text": ch
                })

    return meta_entries


# ===========================
# MAIN INDEX BUILDER
# ===========================
def build_index(pdf_dir: str = "pdfs"):
    """
    Lee todos los PDFs en pdf_dir, genera:
        - meta.json
        - faiss.index
        - inverted_index.json
    """
    os.makedirs(INDEX_DIR, exist_ok=True)

    print("Cargando modelo de embeddings...")
    model = SentenceTransformer(EMB_MODEL)

    all_meta: List[Dict] = []

    print(f"Buscando PDFs en {pdf_dir} ...")
    pdf_files = sorted(Path(pdf_dir).glob("*.pdf"))

    if not pdf_files:
        raise RuntimeError(f"No hay PDFs en la carpeta {pdf_dir}")

    # Extraer chunks
    for pdf_path in pdf_files:
        print(f"Procesando: {pdf_path.name}")
        entries = extract_chunks_from_pdf(str(pdf_path))
        all_meta.extend(entries)

    print(f"Total de chunks: {len(all_meta)}")
    
    # Guardar metadata
    with open(f"{INDEX_DIR}/meta.json", "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=2)

    # ------------------------------
    # FAISS INDEX
    # ------------------------------
    print("Generando embeddings...")
    texts = [m["text"] for m in all_meta]
    embeddings = model.encode(texts, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    print("Construyendo índice FAISS...")
    index.add(embeddings)

    faiss.write_index(index, f"{INDEX_DIR}/faiss.index")

    # ------------------------------
    # INVERTED INDEX
    # ------------------------------
    print("Construyendo índice léxico...")
    inv = build_inverted_index(all_meta)

    with open(f"{INDEX_DIR}/inverted_index.json", "w", encoding="utf-8") as f:
        json.dump(inv, f, ensure_ascii=False, indent=2)

    print("\n🎉 Índices generados correctamente en index_store/")


# ===========================
# ENTRYPOINT
# ===========================
if __name__ == "__main__":
    build_index("../data_pdfs")
