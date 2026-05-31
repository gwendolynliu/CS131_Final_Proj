import os
import json
import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image

# ---- paths ----
POEMS_CSV = "data_poems/filtered_poems.csv"
TEST_IMAGES_DIR = "test_images"
CACHE_DIR = "cache"
EMBEDDINGS_FILE = os.path.join(CACHE_DIR, "poem_embeddings.npy")
IDS_FILE = os.path.join(CACHE_DIR, "poem_ids.json")

# CLIP maxes out at 77 tokens -- poems are longer but only the first ~60 words
# get embedded. that's a known limitation we're living with for now.
CLIP_MAX_TOKENS = 77

# ---- load CLIP ----
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(image_input)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze(0).cpu().numpy()


def get_text_embedding(text):
    # truncate=True clips to 77 tokens instead of throwing an error
    tokens = clip.tokenize([text], truncate=True).to(device)
    with torch.no_grad():
        emb = model.encode_text(tokens)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze(0).cpu().numpy()


def build_poem_cache():
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(IDS_FILE):
        return

    os.makedirs(CACHE_DIR, exist_ok=True)
    poem_df = pd.read_csv(POEMS_CSV)
    embeddings = []
    poem_ids = []

    for _, row in poem_df.iterrows():
        emb = get_text_embedding(str(row["text"]))
        embeddings.append(emb)
        poem_ids.append(int(row["id"]))


    np.save(EMBEDDINGS_FILE, np.array(embeddings))
    with open(IDS_FILE, "w") as f:
        json.dump(poem_ids, f)


def retrieve_poems(image_path, top_k=5):
    embeddings = np.load(EMBEDDINGS_FILE)  # shape (n_poems, 512)
    with open(IDS_FILE) as f:
        poem_ids = json.load(f)

    poem_df = pd.read_csv(POEMS_CSV).set_index("id")

    img_emb = get_image_embedding(image_path)  # shape (512,)

    # both embeddings are normalized so dot product == cosine similarity
    scores = embeddings @ img_emb  # shape (n_poems,)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        pid = poem_ids[idx]
        row = poem_df.loc[pid]
        results.append({
            "poem_id": pid,
            "title": row["title"],
            "author": row["author"],
            "text": str(row["text"]),
            "score": float(scores[idx]),
        })

    return results

# ---- build cache (skipped if already done) ----
build_poem_cache()


if __name__ == "__main__":
    # ---- demo ----
    demo_images = [
        ("1_beach_sunny.jpg",          "bright / joyful"),
        ("17_abandoned_classroom.jpg", "dark / melancholic"),
        ("5_city_crowded.jpg",         "urban / energetic"),
    ]

    print("\n" + "=" * 60)
    print("Track 1: CLIP Semantic Matching")
    print("=" * 60)

    for filename, mood in demo_images:
        image_path = os.path.join(TEST_IMAGES_DIR, filename)
        print(f"\nImage : {filename}  ({mood})")
        print("-" * 50)

        results = retrieve_poems(image_path, top_k=5)

        for i, poem in enumerate(results, 1):
            print(f"  {i}. \"{poem['title']}\" by {poem['author']}  [score: {poem['score']:.4f}]")
            print()
