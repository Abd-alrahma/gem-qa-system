"""
build_index.py
==============
Builds the FAISS vector index from chunks.json.
Uses multilingual-e5-base which supports Arabic properly.

Run AFTER scraper.py:
    python build_index.py

Output:
    faiss_index.index
"""

import json
import sys
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHUNKS_FILE = "chunks.json"
INDEX_FILE  = "faiss_index.index"
EMBED_MODEL = "intfloat/multilingual-e5-base"   # supports Arabic + English

print("=" * 50)
print("  GEM Index Builder")
print("=" * 50)

print(f"\nLoading chunks from {CHUNKS_FILE}…")
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"  {len(chunks)} chunks loaded")

print(f"\nLoading embedding model: {EMBED_MODEL}")
print("  (First run downloads ~1GB — wait for it)")
model = SentenceTransformer(EMBED_MODEL)

print("\nEncoding chunks…")
# multilingual-e5 requires 'passage: ' prefix for indexing
prefixed = ["passage: " + c for c in chunks]
embeddings = model.encode(
    prefixed,
    show_progress_bar=True,
    normalize_embeddings=True,
    batch_size=32,
)
embeddings = np.array(embeddings, dtype="float32")

print("\nBuilding FAISS index…")
dim   = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)   # cosine similarity
index.add(embeddings)
faiss.write_index(index, INDEX_FILE)
print(f"  Saved {INDEX_FILE} — {index.ntotal} vectors, dim={dim}")

print("\nSelf-test…")
test_questions = [
    "ما هو المتحف المصري الكبير؟",
    "أين يقع المتحف المصري الكبير؟",
    "كم عدد القطع الأثرية في المتحف؟",
    "كيف أزور المتحف؟",
]
for q in test_questions:
    q_emb = model.encode(["query: " + q], normalize_embeddings=True)
    D, I  = index.search(np.array(q_emb, dtype="float32"), k=1)
    score = round(float(D[0][0]), 3)
    chunk_preview = chunks[I[0][0]][:80]
    print(f"  Q: {q}")
    print(f"  → score={score}  {chunk_preview}…")
    print()

print("Done! Run:  uvicorn api:app --reload")
