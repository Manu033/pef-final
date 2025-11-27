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
    # Debe haber solapamiento
    assert len(chunks) > 1
    # El primero y el segundo comparten 50 caracteres
    assert chunks[0][-50:] == chunks[1][:50]


def test_inverted_index():
    corpus = [
        {"title": "doc", "page": 1, "text": "Hola mundo hola"},
        {"title": "doc", "page": 2, "text": "Mundo vector inverso"},
    ]
    inv = build_inverted_index(corpus)
    assert "hola" in inv and "mundo" in inv
    # 'hola' aparece en el primer doc dos veces pero solo una entrada de id
    assert inv["hola"] == [0]
    # 'mundo' aparece en ambos
    assert inv["mundo"] == [0, 1]
