#!/usr/bin/env python3
"""
build_index_v2.py
-----------------
Construye un índice híbrido para búsqueda semántica y por palabras clave desde PDFs.

Características nuevas respecto a la versión anterior:
- Generadores para recorrer PDFs página a página sin cargar todo en memoria.
- Batching para generar embeddings por lotes.
- Concurrencia (multiprocessing) para extraer texto de múltiples PDFs en paralelo.
- Índice invertido (hashing) para búsqueda lexical clásica (términos).
- Docstrings y estilo PEP 8.
- Opción de profiling con cProfile.
"""
from __future__ import annotations

import argparse
import cProfile
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Tuple

import faiss  # type: ignore
import fitz  # PyMuPDF  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer


# ===========================
# Configuración
# ===========================
DATA_DIR = "../data_pdfs"
OUT_DIR = "index_store"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Segmentación de texto
CHUNK_SIZE = 800
OVERLAP = 200

# Batching para embeddings
BATCH_SIZE = 32


# ===========================
# Utilidades
# ===========================
def extract_pages_from_pdf(path: str) -> Generator[Tuple[int, str], None, None]:
    """
    Generador que produce (page_number, text) para cada página del PDF.

    Se evita construir una lista completa en memoria.
    """
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            yield (i + 1, page.get_text("text").replace("\n", " "))


def chunk_text(
    text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP
) -> Iterable[str]:
    """
    Divide un texto en fragmentos (chunks) con solapamiento.

    Args:
        text: Texto de entrada.
        size: Tamaño máximo del chunk.
        overlap: Cantidad de caracteres compartidos entre chunks consecutivos.

    Yields:
        Fragmentos de texto.
    """
    if size <= 0:
        raise ValueError("size must be > 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be in [0, size)")

    start = 0
    n = len(text)
    step = size - overlap
    while start < n:
        end = min(n, start + size)
        yield text[start:end]
        start += step


def build_inverted_index(corpus: List[Dict]) -> Dict[str, List[int]]:
    """
    Crea un índice invertido simple (hashing) {termino -> [ids_de_fragmento]}.
    """
    inverted: Dict[str, set] = defaultdict(set)
    for i, doc in enumerate(corpus):
        # tokenización mínima; puede reemplazarse por una más sofisticada
        for word in doc["text"].split():
            token = word.strip().lower()
            if token:
                inverted[token].add(i)

    # Convertimos sets a listas ordenadas para reducir tamaño en disco
    return {term: sorted(list(ids)) for term, ids in inverted.items()}


def encode_in_batches(
    model: SentenceTransformer, texts: List[str], batch_size: int = BATCH_SIZE
) -> np.ndarray:
    """
    Genera embeddings en lotes para reducir uso de RAM.
    """
    emb_list: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        emb_batch = model.encode(batch, normalize_embeddings=True)
        emb_list.append(np.asarray(emb_batch, dtype="float32"))
    return np.vstack(emb_list) if emb_list else np.zeros((0, 384), dtype="float32")


def process_pdf_to_corpus(pdf_path: Path) -> List[Dict]:
    """
    Procesa un PDF a una lista de fragmentos indexables.
    """
    items: List[Dict] = []
    for page_num, text in extract_pages_from_pdf(str(pdf_path)):
        for chunk in chunk_text(text):
            items.append({"title": pdf_path.stem, "page": page_num, "text": chunk})
    return items


def main(profile: bool = False) -> None:
    """
    Punto de entrada: construye índices FAISS e invertido.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    pdf_paths = list(Path(DATA_DIR).glob("*.pdf"))
    if not pdf_paths:
        print(f"[WARN] No se encontraron PDFs en {DATA_DIR}.")
        return

    print(">> Extrayendo texto (paralelo)...")
    corpus: List[Dict] = []
    # Concurrencia: múltiples PDFs en paralelo
    with ProcessPoolExecutor() as ex:
        futures = {ex.submit(process_pdf_to_corpus, p): p for p in pdf_paths}
        for fut in as_completed(futures):
            part = fut.result()
            corpus.extend(part)
            print(f"   + {len(part):4d} chunks de {futures[fut].name}")

    print(f">> Total de fragmentos: {len(corpus)}")

    print(">> Cargando modelo de embeddings...")
    model = SentenceTransformer(EMB_MODEL)

    print(">> Generando embeddings por lotes...")
    texts = [c["text"] for c in corpus]
    embeddings = encode_in_batches(model, texts, BATCH_SIZE)

    print(">> Construyendo índice FAISS (inner product)...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    print(">> Construyendo índice invertido (hashing de términos)...")
    inverted = build_inverted_index(corpus)

    print(">> Guardando artefactos...")
    faiss.write_index(index, os.path.join(OUT_DIR, "faiss.index"))
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)

    with open(os.path.join(OUT_DIR, "inverted_index.json"), "w", encoding="utf-8") as f:
        json.dump(inverted, f, ensure_ascii=False)

    with open(os.path.join(OUT_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "emb_model": EMB_MODEL,
                "chunk_size": CHUNK_SIZE,
                "overlap": OVERLAP,
                "batch_size": BATCH_SIZE,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("✅ Índices creados en", OUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Construye índices híbridos para PDFs."
    )
    parser.add_argument(
        "--profile", action="store_true", help="Habilita cProfile durante la ejecución."
    )
    args = parser.parse_args()

    if args.profile:
        with cProfile.Profile() as pr:
            main(profile=True)
        import pstats

        stats = pstats.Stats(pr).sort_stats(pstats.SortKey.TIME)
        stats.print_stats(20)
    else:
        main()
