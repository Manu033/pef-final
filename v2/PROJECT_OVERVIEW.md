# Proyecto — Resumen y documentación (V2)

Este documento describe la versión `v2` del proyecto (excluye `v1`) y explica los archivos principales, su propósito y funciones clave. Está pensado para que sirva como guion para una presentación y como referencia rápida para entender la arquitectura y el flujo de datos.

---

## Resumen general
Proyecto RAG (Retrieval-Augmented Generation) local para consultar PDFs.

Flujo principal:

1. Extraer texto de PDFs y dividirlo en fragments (chunks).
2. Construir índices:
   - Índice semántico FAISS (embeddings).
   - Índice invertido léxico (token → docIDs).
3. UI Streamlit (`app_v2.py`) para recibir preguntas en lenguaje natural, recuperar fragments relevantes, construir un `context` y enviar un `prompt` a un LLM local (ej. Ollama / Qwen).

Casos de uso: Q&A sobre apuntes/PDFs, asistencia para presentaciones, generación de resúmenes locales.

---

## Estructura (carpeta `v2`)

- `app_v2.py` — Aplicación Streamlit; orquesta retrieval, construcción del prompt y consulta al LLM.
- `build_index_v2.py` — Script que procesa PDFs, chunkifica, genera embeddings y construye FAISS e índice invertido.
- `count_tokens_estimate.py` — Script auxiliar para estimar número de tokens del índice (heurístico de caracteres)
- `test_search.py` — Tests unitarios (pytest) para `chunk_text` y `build_inverted_index`.
- `requirements.txt` — Dependencias del proyecto.
- `README.md` — Guía rápida de uso.
- `index_store/` — Artefactos generados por `build_index_v2.py`:
  - `faiss.index` — índice FAISS (binario).
  - `meta.json` — lista de fragments indexados (title, page, text).
  - `inverted_index.json` — índice léxico (token → [ids]).
  - `config.json` — parámetros usados para construir el índice.

---

## Descripción por archivo

### `app_v2.py`

**Propósito**: Interfaz y orquestador RAG.

Funciones/Componentes clave:

- `load_faiss_and_meta()` — carga `faiss.index` y `meta.json` (cacheado con `@st.cache_resource`). Lanza `RuntimeError` con mensajes claros si falla.
- `load_inverted_index()` — lee `inverted_index.json` si existe; si no existe devuelve `{}`.
- `load_embedding_model()` — carga `SentenceTransformer` para generar embeddings de la query cuando es necesario.
- `semantic_retrieve(query, index, meta, emb_model, k, initial_k)` — busca por similitud en FAISS; devuelve hasta `k` fragments respetando `MAX_CONTEXT_CHARS`.
- `lexical_retrieve(query, inverted, meta, k)` — busca por tokens exactos usando el índice invertido.
- `hybrid_retrieve(...)` — combina semántica + léxica evitando duplicados.
- `build_context_from_items_by_chars(items, max_chars)` — concatena fragments hasta `max_chars` caracteres; devuelve `context` y `used_items`.
- `cached_search(query, mode)` — memoiza búsquedas (LRU cache) para acelerar queries repetidas.
- `build_prompt(q, context)` — prompt estructurado (en español) que obliga al LLM a usar solo el `context` o responder exactamente: "No se encuentra en el material indexado." si no hay evidencia.
- `query_ollama(prompt)` — POST a `http://localhost:11434/api/generate` (modelo local). Maneja excepciones HTTP y JSON.

UI / opciones relevantes:

- Modo de búsqueda: `Híbrida`, `Semántica`, `Lexical`.
- `Resultados (k)` — cuántos resultados mostrar.
- `Mostrar contexto y prompt (debug)` — muestra el `context` y el `prompt` antes de la llamada al LLM.
- `Incluir todo el índice en el contexto` — opción para concatenar todo `meta.json` al prompt (peligro de exceder la ventana del modelo).
- Lógica anti-alucinación: si no hay fragments usados o el contexto es muy corto, la app devuelve "No se encuentra en el material indexado." sin llamar al LLM.

Limitaciones y recomendaciones:

- Enviar todo el índice puede exceder la ventana del LLM → preferir retrieval+resumen o aumentar `MAX_CONTEXT_CHARS` con precaución.
- `build_prompt` es deliberadamente estricto para evitar respuestas inventadas.

---

### `build_index_v2.py`

**Propósito**: Construir los índices a partir de PDFs.

Funciones/Componentes clave:

- `extract_pages_from_pdf(path)` — generador que extrae texto por página con PyMuPDF (`fitz`).
- `chunk_text(text, size, overlap)` — divide el texto en chunks con solapamiento, validando parámetros.
- `build_inverted_index(corpus)` — tokeniza (regex `\\w+` y `.lower()`) y genera `{token: sorted([doc_ids])}` usando sets internos para evitar duplicados.
- `encode_in_batches(model, texts, batch_size)` — genera embeddings por lotes.
- `process_pdf_to_corpus(pdf_path)` — convierte cada página en varios fragments.
- `main()` — orquesta todo: extrae textos en paralelo, genera embeddings, construye FAISS (`IndexFlatIP`) y guarda artefactos en `index_store`.

Puntos importantes:

- FAISS se crea con `IndexFlatIP` y asume embeddings normalizados.
- `meta.json` es la fuente canónica de fragments que la app usa para construir contexto.

---

### `count_tokens_estimate.py`

Script auxiliar que suma caracteres en `meta.json` y aplica heurística `1 token ≈ 4 caracteres` para estimar cuántos tokens ocupa todo el índice. Útil para evaluar si "todo el índice" cabe en la ventana del LLM.

Uso:
```
python count_tokens_estimate.py
```

---

### `test_search.py`

Tests unitarios con `pytest` que cubren:

- `chunk_text` — comprobaciones de solapamiento y caso en que el texto es más corto que el chunk.
- `build_inverted_index` — comprobación de normalización de tokens y ausencia de IDs duplicados.

Ejecutar:
```
pytest -q
```

---

### `requirements.txt`

Contiene las dependencias necesarias para ejecutar y desarrollar la aplicación (Streamlit, faiss-cpu, pymupdf, sentence-transformers, numpy, pytest, requests).

Instalación:
```
pip install -r requirements.txt
```

---

## Flujo de datos (alto nivel)

1. `build_index_v2.py` procesa PDFs → `corpus` (fragments) → embeddings → FAISS + inverted index → guarda `meta.json`, `faiss.index`, `inverted_index.json`.
2. `app_v2.py` carga recursos cacheados y espera queries.
3. Usuario pregunta → `cached_search` recupera fragments → `build_context_from_items_by_chars` arma `context` (o se incluye todo `meta` si el usuario lo pide) → `build_prompt` crea prompt → `query_ollama` envía al LLM → respuesta mostrada con fuentes.

---

## Riesgos y mejoras sugeridas (para slides de "Mejoras futuras")

- Usar un summarizer por fragmento para poder enviar más información sin exceder la ventana.
- Normalizar diacríticos (remover tildes) en `build_inverted_index` para búsquedas más tolerantes.
- Añadir reranker (BM25 o similar) para mejorar el ranking léxico.
- Integrar métricas (precisión de retrieval, BLEU/ROUGE para resúmenes) y tests de regresión.
- Añadir CI que ejecute `pytest` y verifique que `index_store` existe / se puede cargar.

---

## Comandos útiles

Construir índice:
```
python build_index_v2.py
```

Correr la app:
```
streamlit run app_v2.py
```

Ejecutar tests:
```
pytest -q
```

Estimar tokens:
```
python count_tokens_estimate.py
```

---

## Sugerencia de estructura para la presentación (diapositivas)

1. Título y objetivo del proyecto.
2. Visión general y caso de uso.
3. Arquitectura (diagrama: PDFs → Index build → index_store → app_v2 → LLM).
4. Detalle de `build_index_v2.py` (chunking, embeddings, FAISS, inverted index).
5. Detalle de `app_v2.py` (retrieval modes, construcción de contexto, prompt, interacción con LLM).
6. Prompt y medidas anti-alucinación (regla de "No se encuentra...").
7. Limitaciones y mejoras propuestas.
8. Demo (capturas o ejecución en vivo).
9. Tests y reproducibilidad.

