# RAG Local V2

Mejoras incluidas:
- Índice FAISS + índice invertido (hashing de términos).
- Generadores, batching y procesamiento paralelo.
- Memoización de consultas frecuentes.
- Docstrings/PEP8 y tests básicos con `pytest`.
- UI Streamlit con modos Semántica / Lexical / Híbrida.

## Uso

1) Instalar dependencias:
```
pip install -r requirements.txt
```

2) Construir índices desde PDFs ubicados en `data_pdfs/`:
```
python build_index_v2.py
# o con profiling
python build_index_v2.py --profile
```

3) Ejecutar la app:
```
streamlit run app_v2.py
```

> Asegurate de tener Ollama corriendo localmente y el modelo configurado en `app_v2.py`.
