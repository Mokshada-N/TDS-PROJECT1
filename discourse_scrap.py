# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "rich",
#  "beautifulsoup4",
#   "google-genai",
# "Pillow",
# "tqdm"
# ]
# ///

import os
import httpx
from google import genai
import requests
from bs4 import BeautifulSoup
import json
from PIL import Image
import io
from datetime import datetime, timezone
from tqdm import tqdm
import time
from google.genai.errors import ServerError
import random
# Use your Discourse session token (_t)
DISCOURSE_COOKIE = "L10Ji5G%2BJoO%2F9FWkxFmfrjb%2BtIDMSbGM%2BCEpVCxcRAzfLFXFXLG4IHQZN2n7Cevoc3FXq0NkKYZ9EwAKCImX0dSUUKYetBThqHff6YMG2vgoYOcSDraVSY%2BAtje2RSV4YkvIx65lIn3LdxudzbAtkH7aB3F9tVekfMLBuHj5nvPwtNHkt18hXCTeIFcDZ0Wfy7sgHgpk3glKMzDkFyBObTnq0LVs090CygdSba3wsrgIrAeafA%2BX43EAHpDuzXg92GFyidNyPPbQtpQEDy9N%2FrIy0JijGRj5CezCWmRy08q%2B7%2Bpx3wvsbRDhHtvE8GBc--0akkJq%2Fdp5LWe3IE--Q839QSiOEQSeK5oIrGfDbA%3D%3D"

# Setup session
session = requests.Session()
session.cookies.set("_t", DISCOURSE_COOKIE, domain="discourse.onlinedegree.iitm.ac.in")
session.headers.update({"User-Agent": "Mozilla/5.0"})

BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"

GEMINI_API_KEY = "AIzaSyB6mU36fAgnDd0yzGCA615BpHqsruQMbxo"

client = genai.Client(api_key=GEMINI_API_KEY)
import time
import base64
import io
from PIL import Image
from google.genai import Client
from google.genai.errors import ClientError

# Initialize Gemini client
client = Client()

# Cache for image descriptions
image_description_cache = {}

# Rate limit settings
REQUEST_INTERVAL = 4.5  # seconds between requests (~15 req/min free tier)

def get_image_description(image_url):
    if image_url in image_description_cache:
        return image_description_cache[image_url]
    
    try:
        # Download the image
        response = requests.get(image_url)
        response.raise_for_status()

        # Convert to image
        image = Image.open(io.BytesIO(response.content))

        # Convert image to base64 for Gemini API
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Call Gemini API
        gemini_response = None
        while gemini_response is None:
            try:
                gemini_response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[
                        {"role": "user", "parts": [{"inline_data": {"mime_type": "image/png", "data": img_base64}}]},
                        {"role": "user", "parts": [{"text": "Describe the image in detail."}]}
                    ]
                )
            except ClientError as e:
                if e.status_code == 429:
                    retry_delay = extract_retry_delay(e)
                    print(f"Rate limit hit. Sleeping for {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise

        description = gemini_response.candidates[0].content.parts[0].text
        image_description_cache[image_url] = description

        # Rate limit
        time.sleep(REQUEST_INTERVAL)

        return description
    
    except Exception as ex:
        print(f"Failed to process image: {ex}")
        return None

def extract_retry_delay(client_error):
    """Extract retry delay from ClientError if available."""
    try:
        for detail in client_error.error.get("details", []):
            if detail.get("@type") == "type.googleapis.com/google.rpc.RetryInfo":
                retry_delay_str = detail.get("retryDelay", "10s")
                # convert '44s' -> 44
                return int(retry_delay_str.replace("s", ""))
    except:
        pass
    return 10  # default fallback delay

ct = 0
def extract_image_url_from_post(post):
    global ct
    cooked_html = post.get("cooked", "")
    soup = BeautifulSoup(cooked_html, "html.parser")
    img_tag = soup.find("img")

    if img_tag and img_tag.get("src"):
        if "user_avatar" in img_tag["src"] or "emoji" in img_tag["src"] or "avatar" in img_tag["src"]:
            return None
        ct += 1
        return img_tag["src"]
    
    return None

def get_topic_ids(category_slug="courses/tds-kb", category_id=34):
    topics = []
    for page in tqdm(range(0, 20)):  # Adjust if you want more pages
        url = f"{BASE_URL}/c/{category_slug}/{category_id}.json?page={page}"
        r = session.get(url)
        if r.status_code != 200:
            break
        data = r.json()
        new_topics = data["topic_list"]["topics"]
        if not new_topics:
            break
        topics.extend(new_topics)
    return topics


def get_posts_in_topic(topic_id):
    print("Getting posts")
    base_topic_url = f"{BASE_URL}/t/{topic_id}.json"
    r = session.get(base_topic_url)
    print(base_topic_url)
    print(r.status_code)
    if r.status_code != 200:
        return []

    data = r.json()

    # Get all post IDs in the topic
    all_post_ids = data.get("post_stream", {}).get("stream", [])
    loaded_posts = data.get("post_stream", {}).get("posts", [])
    topic_slug = data.get("slug")
    print(topic_slug)
    # Extract already-loaded post IDs
    loaded_ids = {post["id"] for post in loaded_posts}
    remaining_ids = [pid for pid in all_post_ids if pid not in loaded_ids]

    posts = loaded_posts.copy()

    # Batch size for additional post requests (same as browser usually does)
    BATCH_SIZE = 20

    for i in tqdm(range(0, len(remaining_ids), BATCH_SIZE), desc=f"Fetching posts for topic {topic_id}", leave=False):
        batch_ids = remaining_ids[i:i+BATCH_SIZE]
        params = [("post_ids[]", pid) for pid in batch_ids]
        url = f"{BASE_URL}/t/{topic_id}/posts.json"
        r = session.get(url, params=params)

        if r.status_code != 200:
            print(f"Failed to fetch post batch: {batch_ids}")
            continue

        batch_data = r.json()
        posts.extend(batch_data.get("post_stream", {}).get("posts", []))
    # Parse and clean up
    return [
                {
                    "topic_id": topic_id,
                    "post_id": post["id"],
                    "username": post["username"],
                    "created_at": post["created_at"],
                    # Append image description to content if it exists
                    "content": BeautifulSoup(post["cooked"], "html.parser").get_text(),
                    "post_number": post["post_number"],
                    "reply_to_post_number" : post["reply_to_post_number"],
                    "post_url": f"{BASE_URL}/t/{topic_slug}/{topic_id}/{post['post_number']}",
                    "image_description": get_image_description(extract_image_url_from_post(post)) if extract_image_url_from_post(post) else None,
                    "topic_url" : f"{BASE_URL}/t/{topic_slug}/{topic_id}"
                }
                for post in posts
            ]
# Create directory for saving posts

os.makedirs("discourse_posts_new_submission", exist_ok=True)

# Modified section at the bottom of the script
topics = get_topic_ids()
# target_topics = [163147]
# topics = [t for t in topics if t["id"] in target_topics]

# Save topics metadata first
with open("discourse_posts_new_submission/topics.json", "w", encoding="utf-8") as f:
    json.dump([{"id": t["id"], "title": t["title"]} for t in topics], f, indent=2)

# Process each topic
for topic in tqdm(topics):
    print("Processing topic:", topic["title"] , "ID:", topic["id"])
    # Create filename-safe topic title
    safe_title = "".join([c if c.isalnum() else "_" for c in topic["title"]])
    filename = f"discourse_posts_new_submission/topic_{topic['id']}_{safe_title}.json"
    
    # Skip if file already exists
    if os.path.exists(filename):
        continue
        
    # Parse created_at as timezone-aware datetime (UTC)
    created_at = datetime.fromisoformat(topic["created_at"].replace("Z", "+00:00"))
    
    # Check date range
    if created_at >= datetime(2025, 1, 1, tzinfo=timezone.utc) and created_at <= datetime(2025, 4, 14, tzinfo=timezone.utc):
        posts = get_posts_in_topic(topic["id"])
        print("Got Posts")
        if posts == []:
            continue
        # Save posts for this topic
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({
                "topic_id": topic["id"],
                "topic_title": topic["title"],
                "posts": posts
            }, f, indent=2, ensure_ascii=False)
        time.sleep(2)

print(f"Processed {len(topics)} topics.")
print(ct)