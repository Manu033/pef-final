import json
from pathlib import Path

META = Path("index_store/meta.json")  # ajustá si tu meta.json está en otra ruta
CHARS_PER_TOKEN = 4.0  # heurística (ajustá a 3.5 si querés ser más permisivo)

if not META.exists():
    print("No encontré", META)
    raise SystemExit(1)

meta = json.loads(META.read_text(encoding="utf-8"))

total_chars = 0
for doc in meta:
    text = doc.get("text", "")
    total_chars += len(text)

est_tokens = int(total_chars / CHARS_PER_TOKEN)
print(f"Documentos: {len(meta)}")
print(f"Caracteres totales: {total_chars:,}")
print(f"Estimación de tokens (1 token ≈ {CHARS_PER_TOKEN} chars): {est_tokens:,}")