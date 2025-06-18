import json
from pathlib import Path
import re

def clean_text(text):
    return " ".join(text.strip().split())

def chunk_markdown(content, original_url, chunk_size=500):
    """
    Split content into chunks of roughly `chunk_size` characters,
    ensuring each chunk includes the original_url.
    """
    words = content.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(" ".join(current_chunk)) >= chunk_size:
            chunks.append({
                "chunk": clean_text(" ".join(current_chunk)),
                "original_url": original_url
            })
            current_chunk = []

    if current_chunk:
        chunks.append({
            "chunk": clean_text(" ".join(current_chunk)),
            "original_url": original_url
        })

    return chunks

def extract_original_url_and_content(md_text):
    yaml_match = re.search(r"---\s*(.*?)\s*---", md_text, re.DOTALL)
    if yaml_match:
        yaml_block = yaml_match.group(1)
        content_without_yaml = md_text[yaml_match.end():]
        url_match = re.search(r'original_url:\s*"(.*?)"', yaml_block)
        original_url = url_match.group(1) if url_match else ""
    else:
        original_url = ""
        content_without_yaml = md_text
    return original_url, content_without_yaml

# === Process all md files ===
all_chunks = []
md_files = [*Path("markdown_files").glob("*.md"), *Path("markdown_files").rglob("*.md")]

for md_path in md_files:
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    original_url, content = extract_original_url_and_content(md_text)
    chunks = chunk_markdown(content, original_url)
    all_chunks.extend(chunks)

print(f"✅ Processed {len(md_files)} files into {len(all_chunks)} chunks.")

# === Save ===
with open("content_chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2)

print("✅ Saved chunks to chunks.json")
