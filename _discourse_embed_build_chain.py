# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "sentence_transformers",
#   "faiss-cpu"
# ]
# ///


import json
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
from pathlib import Path

def clean_text(text):
    return " ".join(text.strip().split())

# === Load and combine data ===
topics = {}
for json_file in Path("discourse_posts_updated").glob("*.json"):
    with open(json_file, "r", encoding="utf-8") as f:
        topic_data = json.load(f)
        topic_id = topic_data["topic_id"]
        topics[topic_id] = {
            "topic_title": topic_data.get("topic_title", ""),
            "posts": topic_data["posts"]
        }

print(f"Loaded {len(topics)} topics from {len(list(Path('discourse_posts_updated').glob('*.json')))} JSON files.")

# === Sort posts within each topic ===
for topic_id in topics:
    topics[topic_id]["posts"].sort(key=lambda p: p["post_number"])

# === Prepare embedding model ===
model = SentenceTransformer("all-MiniLM-L6-v2")

def build_reply_map(posts):
    reply_map = defaultdict(list)
    posts_by_number = {}
    for post in posts:
        posts_by_number[post["post_number"]] = post
        parent = post.get("reply_to_post_number")
        reply_map[parent].append(post)
    return reply_map, posts_by_number

def enumerate_chains(post_num, reply_map, current_chain, all_chains):
    current_chain.append(post_num)
    children = reply_map.get(post_num, [])
    if not children:
        all_chains.append(list(current_chain))
    else:
        for child in children:
            enumerate_chains(child["post_number"], reply_map, current_chain, all_chains)
    current_chain.pop()

def all_contiguous_subchains(chain):
    n = len(chain)
    subchains = []
    for i in range(n):
        for j in range(i+1, n+1):
            subchains.append(chain[i:j])
    return subchains

all_texts = []
all_metadata = []

for topic_id, topic_data in tqdm(topics.items()):
    posts = topic_data["posts"]
    topic_title = topic_data["topic_title"]
    reply_map, posts_by_number = build_reply_map(posts)
    root_posts = reply_map[None]
    all_chains = []
    for root_post in root_posts:
        enumerate_chains(root_post["post_number"], reply_map, [], all_chains)
    all_subchains = set()
    for chain in all_chains:
        for subchain in all_contiguous_subchains(chain):
            all_subchains.add(tuple(subchain))
    for subchain in sorted(all_subchains):
        chain_posts = [posts_by_number[num] for num in subchain]
        combined_text = f"Topic title: {topic_title}\n\n"
        combined_text += "\n\n---\n\n".join(clean_text(p["content"]) for p in chain_posts)
        # Optionally, add image descriptions
        img_descs = [clean_text(p["image_description"]) for p in chain_posts if p.get("image_description")]
        if img_descs:
            combined_text += "\n\n---\n\n" + "\n\n---\n\n".join(img_descs)
        all_texts.append(combined_text)
        all_metadata.append({
            "type": "contiguous_subchain",
            "topic_id": topic_id,
            "topic_title": topic_title,
            "post_numbers": [p["post_number"] for p in chain_posts],
            "combined_text": combined_text,
            "url": [p["post_url"] for p in chain_posts]
        })

print("Encoding all subchains...")
embeddings = []
for text in tqdm(all_texts):
    emb = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    embeddings.append(emb)

embeddings_np = np.vstack(embeddings).astype("float32")

# === Save results ===
embedding_data = []
for meta, emb in zip(all_metadata, embeddings_np):
    embedding_data.append(meta)

with open("embedding_data_all_subchains.json", "w", encoding="utf-8") as f:
    json.dump(embedding_data, f, indent=2)

index = faiss.IndexFlatIP(embeddings_np.shape[1])
index.add(embeddings_np)
faiss.write_index(index, "faiss_index_all_subchains.idx")

print("✅ Embeddings and FAISS index for all subchains saved.")
