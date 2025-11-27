# tests/test_search.py
"""
Tests mínimos para funciones utilitarias.
Ejecutar con:  pytest -q
"""
import builtins
import json
from pathlib import Path

import numpy as np
import pytest

# Para importar desde el directorio raíz del proyecto durante los tests
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from build_index_v2 import chunk_text, build_inverted_index


def test_chunk_text_basic():
    text = "a" * 1000
    chunks = list(chunk_text(text, size=200, overlap=50))
    # Cada chunk no debe superar 200
    assert all(len(c) <= 200 for c in chunks)
    # Debe haber solapamiento y más de un chunk
    assert len(chunks) > 1
    # El primero y el segundo comparten 50 caracteres
    assert chunks[0][-50:] == chunks[1][:50]


def test_chunk_text_shorter_than_size():
    text = "hola"
    chunks = list(chunk_text(text, size=10, overlap=2))
    # Texto más corto que el tamaño debe devolver el texto entero en un solo chunk
    assert chunks == [text]


def test_inverted_index():
    corpus = [
        {"title": "doc", "page": 1, "text": "Hola, mundo hola"},
        {"title": "doc", "page": 2, "text": "Mundo vector inverso."},
    ]
    inv = build_inverted_index(corpus)
    assert "hola" in inv and "mundo" in inv
    # 'hola' aparece solo en el primer doc (aunque se repita)
    assert set(inv["hola"]) == {0}
    # 'mundo' aparece en ambos
    assert set(inv["mundo"]) == {0, 1}


def test_inverted_index_no_duplicate_ids():
    corpus = [{"title": "doc", "page": 1, "text": "hola hola hola"}]
    inv = build_inverted_index(corpus)
    # Asegurar que no haya IDs duplicados en el posting
    assert set(inv.get("hola", [])) == {0}
