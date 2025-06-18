import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# === Load chunk data ===
with open("content_chunks.json", "r", encoding="utf-8") as f:
    chunk_data = json.load(f)

print(f"✅ Loaded {len(chunk_data)} markdown chunks.")

# === Load embedding model ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Compute embeddings ===
embeddings = []
metadata = []

for chunk in chunk_data:
    text = chunk["chunk"]
    emb = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    embeddings.append(emb)
    metadata.append(chunk)

# === Save metadata ===
with open("embedding_md_data.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

# === Build FAISS index ===
embeddings_np = np.vstack(embeddings).astype("float32")
index = faiss.IndexFlatIP(embeddings_np.shape[1])
index.add(embeddings_np)

# === Save FAISS index ===
faiss.write_index(index, "faiss_md_index.idx")

print("✅ Markdown FAISS index saved as faiss_md_index.idx")
