# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "sentence_transformers",
#   "faiss-cpu",
#   "requests",
#   "httpx",
#   "fastapi",
#   "uvicorn",
#   "pytesseract",
#   "slugify",
#   "pillow",
#   "requests",
#   "python-dotenv",
#   "python-slugify"
# ]
# ///

from networkx import general_random_intersection_graph
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import httpx
import json
import base64
from io import BytesIO
from PIL import Image
import pytesseract
from slugify import slugify 
import mimetypes
from fastapi.responses import JSONResponse
from slugify import slugify

from collections import defaultdict


# === Constants ===
from dotenv import load_dotenv
import os

# Load env variables from .env
load_dotenv()

AIPROXY_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
AIPROXY_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjI0ZjEwMDA2NTZAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.JogfYgwBHNTixeIqVUq-Pdh3xgnQhi7AC9h6_0nixvU"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not AIPROXY_URL or not AIPROXY_TOKEN:
    raise RuntimeError("Missing AIPROXY_URL or AIPROXY_TOKEN in environment variables.")

# === Load resources ===
print("🔹 Loading FAISS index...")
index = faiss.read_index("faiss_combined_index_subchain.idx")

print("🔹 Loading metadata...")
with open("embedding_combined_data_latest.json", "r", encoding="utf-8") as f:
    embedding_data = json.load(f)

print("🔹 Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Utility ===
def get_image_mimetype(base64_string):
    image_data = base64.b64decode(base64_string)
    try:
        img = Image.open(BytesIO(image_data))
        img_type = img.format.lower()  # e.g. 'jpeg', 'png'
        mime_type = f'image/{img_type}'
        extension = mimetypes.guess_extension(mime_type) or f".{img_type}"
    except Exception as e:
        print(f"Image type detection failed: {e}")
        mime_type = 'application/octet-stream'
        extension = '.bin'
    return mime_type, extension, image_data

# Dummy placeholder for your Gemini integration
def get_image_description(image_path):
    # Replace with actual Gemini API call
    return "This is a placeholder image description."


def query_faiss(query, top_k=5):
    query_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = index.search(query_emb, top_k)

    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < len(embedding_data):
            results.append({
                "score": float(score),
                **embedding_data[idx]
            })

    # Build a map: url -> list of all entries containing that url in their 'url' list
    url_to_entries = defaultdict(list)
    for entry in embedding_data:
        urls = entry.get("url", [])
        for url in urls:
            url_to_entries[url].append(entry)

    final_results = []
    seen = set()  # to avoid duplicates, store (url, topic_id, post_numbers tuple)

    for res in results:
        base_score = res["score"]
        urls = res.get("url", [])
        if not isinstance(urls, list):
            urls = [urls]

        for url in urls:
            for entry in url_to_entries.get(url, []):
                # Create a deduplication key based on url + topic_id + post_numbers
                key = (
                    url,
                    entry.get("topic_id"),
                    tuple(entry.get("post_numbers", []))
                )
                if key in seen:
                    continue
                seen.add(key)

                # Compose the result item
                final_results.append({
                    "score": base_score,
                    "text_snippet": entry.get("combined_text", "")[:1000], 
                    "topic_title": entry.get("topic_title", ""),
                    "url": url,
                    "post_numbers": entry.get("post_numbers", [])
                })

    return final_results

def filter_maximal_chains(results):
    # Extract sets of post_numbers for each result
    postnum_sets = [set(r["post_numbers"]) for r in results]

    filtered = []
    for i, r in enumerate(results):
        c_set = postnum_sets[i]
        is_subset = False
        for j, other_set in enumerate(postnum_sets):
            if i != j and c_set < other_set:  # proper subset
                is_subset = True
                break
        if not is_subset:
            filtered.append(r)
    return filtered

def filter_last_per_first_post(results):
    latest_entries = {}
    for entry in results:
        post_numbers = entry.get("post_numbers", [])
        if not post_numbers:
            continue  # skip if no post numbers
        first_post = post_numbers[0]
        # Always overwrite so the last entry with this first_post is kept
        latest_entries[first_post] = entry
    # Return entries sorted by first post number or just the values
    return list(latest_entries.values())



def generate_llm_response(query, context_texts):
    context = "\n\n---\n\n".join(context_texts)
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on forum discussions."
            },
            {
                "role": "user",
                "content": f"""<instructions>
You are a helpful assistant that answers questions based on forum discussions. 
First, think step-by-step about the relevant context. 
Provide your reasoning clearly, showing how each piece of the forum content contributes to your answer. 
Finally, give the answer in a concise and accurate way and if there are any reccomendations given in the context data also recommend it to user.
</instructions>
<context>
{context}
</context>
<question>
{query}
</question>
<format>
Respond in Markdown format:
- Use ### Reasoning as a section header for your reasoning
- Use ### Final Answer as a section header for your final answer
</format>
"""
            }
        ],
        "temperature": 0.3,
        "max_tokens": 400
    }

    response = httpx.post(AIPROXY_URL, headers=headers, json=payload, timeout=25.0)
    if response.status_code == 200:
        print("successfully got response")
        return response.json()['choices'][0]['message']['content']
    else:
        return JSONResponse({"answer": f"test", "links": []})
        raise HTTPException(status_code=500, detail=f"AIPipe Error: {response.text}")

def answer(question, image):
    if image:
        mime_type, extension, image_data = get_image_mimetype(image)
        filename = f"output_image{extension}"
        with open(filename, "wb") as f:
            f.write(image_data)
        image_response = get_image_description(filename)
        question = f"{question}\nImage description: {image_response}"

    results = query_faiss(question, top_k=10)
    with open("faiss_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("FAISS results saved to faiss_results.json")

    maximal_results = filter_maximal_chains(results)
    with open("maximal_faiss_results.json", "w", encoding="utf-8") as f:
        json.dump(maximal_results, f, indent=2, ensure_ascii=False)
    
    filtered_results = filter_last_per_first_post(results)
    with open("filtered_faiss_results.json", "w", encoding="utf-8") as f:
        json.dump(filtered_results, f, indent=2, ensure_ascii=False)
    
    print("FAISS results saved to maiximal_faiss_results.json")
    results = filtered_results
    url_text_list = []
    for item in results:
        url = item.get("url", "")
        text = item.get("text_snippet", "")
        url_text_list.append({
            "url": url,
            "text": text
        })
    context_texts = []
    for entry in url_text_list:
        url = entry.get("url", "")
        text = entry.get("text", "")
        combined = f"{url} {text}"
        context_texts.append(combined)

    output_file = "context_texts_new.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        for line in context_texts:
            f.write(line + "\n")  # Each combined entry on a new line

    print(f"Saved {len(context_texts)} entries to {output_file}")
    # Get answer text from model
    response_text = generate_llm_response(question, context_texts)
    response_text = response_text or "No answer generated."

    # Save to a file
    with open("llm_response_new.txt", "w", encoding="utf-8") as f:
        f.write(response_text)

    print("LLM response saved to llm_response.txt")

    result = {
        "answer": response_text,
        "links": url_text_list
    }

    # Save to file
    with open("answer_result_new.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    # Return as API response
    return JSONResponse(result)



# === FastAPI app ===
app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "FastAPI is running!"}

class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

@app.post("/api")
def api_answer(request: QueryRequest):
    try:
        return answer(request.question, request.image)
    except Exception as e:
        return JSONResponse({"answer": f"api Error: {str(e)}", "links": []})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,port=8080)
# answer("If a student scores 10/10 on GA4 as well as a bonus, how would it appear on the dashboard?",None)