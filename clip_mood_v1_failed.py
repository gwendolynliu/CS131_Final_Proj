# V1 approach: scored both images and poems against mood prompts using CLIP
# text-text similarity. Failed because CLIP text embeddings cluster too tightly
# — all poem mood vectors had pairwise cosine similarity 0.965–1.000 (std 0.001),
# making rankings meaningless. Replaced by V2 (CLIP + SBERT hybrid) in
# clip_mood.py. Kept for documentation purposes.

import os
import json
import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image

# ---- paths ----
POEMS_CSV = "data_poems/filtered_poems.csv"
MOOD_PROMPTS_FILE = "mood_prompts.json"
TEST_IMAGES_DIR = "test_images"
CACHE_DIR = "cache"
MOOD_VECS_FILE = os.path.join(CACHE_DIR, "poem_mood_vectors_v1.npy")
MOOD_IDS_FILE = os.path.join(CACHE_DIR, "mood_poem_ids_v1.json")

# ---- load CLIP ----
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

with open(MOOD_PROMPTS_FILE) as f:
    mood_prompts = json.load(f)

# encode all mood prompts once
mood_tokens = clip.tokenize(mood_prompts).to(device)
with torch.no_grad():
    mood_text_embs = model.encode_text(mood_tokens)
    mood_text_embs = mood_text_embs / mood_text_embs.norm(dim=-1, keepdim=True)
mood_text_embs = mood_text_embs.cpu().numpy()  # (17, 512)


def prompt_to_label(prompt):
    return prompt.split("feels ")[1].split(",")[0]


def get_image_mood_vector(image_path):
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        img_emb = model.encode_image(image_input)
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    img_emb = img_emb.squeeze(0).cpu().numpy()

    scores = mood_text_embs @ img_emb  # (17,)
    scores = scores / np.linalg.norm(scores)
    return scores


def get_poem_mood_vector(text):
    # encode poem text with CLIP, then project onto mood prompt directions
    tokens = clip.tokenize([text], truncate=True).to(device)
    with torch.no_grad():
        text_emb = model.encode_text(tokens)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    text_emb = text_emb.squeeze(0).cpu().numpy()

    scores = mood_text_embs @ text_emb  # (17,)
    scores = scores / np.linalg.norm(scores)
    return scores


def build_mood_vector_cache():
    if os.path.exists(MOOD_VECS_FILE) and os.path.exists(MOOD_IDS_FILE):
        return

    os.makedirs(CACHE_DIR, exist_ok=True)
    poem_df = pd.read_csv(POEMS_CSV)
    all_vecs = []
    poem_ids = []

    for _, row in poem_df.iterrows():
        vec = get_poem_mood_vector(str(row["text"]))
        all_vecs.append(vec)
        poem_ids.append(int(row["id"]))

    np.save(MOOD_VECS_FILE, np.array(all_vecs))
    with open(MOOD_IDS_FILE, "w") as f:
        json.dump(poem_ids, f)


def retrieve_poems_by_mood_v1(image_path, top_k=5):
    all_vecs = np.load(MOOD_VECS_FILE)
    with open(MOOD_IDS_FILE) as f:
        poem_ids = json.load(f)
    poem_df = pd.read_csv(POEMS_CSV).set_index("id")

    image_vec = get_image_mood_vector(image_path)

    # cosine sim between image mood vector and each poem mood vector
    # (both normalized, so dot product == cosine sim)
    scores = all_vecs @ image_vec  # (n_poems,)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        pid = poem_ids[idx]
        row = poem_df.loc[pid]
        results.append({
            "title": row["title"],
            "author": row["author"],
            "score": float(scores[idx]),
        })

    return results, image_vec


# ---- build cache ----
build_mood_vector_cache()


# ---- demo ----
demo_images = [
    ("1_beach_sunny.jpg",          "bright / joyful"),
    ("17_abandoned_classroom.jpg", "dark / melancholic"),
    ("5_city_crowded.jpg",         "urban / energetic"),
]

print("\n" + "=" * 60)
print("Track 3 V1: Mood-Based Matching (CLIP text-text, broken)")
print("=" * 60)

for filename, mood in demo_images:
    image_path = os.path.join(TEST_IMAGES_DIR, filename)
    print(f"\nImage : {filename}  ({mood})")
    print("-" * 50)

    results, image_vec = retrieve_poems_by_mood_v1(image_path, top_k=5)

    print("  Mood distribution (image):")
    ranked = sorted(zip(mood_prompts, image_vec), key=lambda x: -x[1])
    for prompt, score in ranked:
        print(f"    {score:.4f}  {prompt_to_label(prompt)}")

    print("\n  Top 5 poems:")
    for i, poem in enumerate(results, 1):
        print(f"  {i}. \"{poem['title']}\" by {poem['author']}  [score: {poem['score']:.4f}]")
    print()
