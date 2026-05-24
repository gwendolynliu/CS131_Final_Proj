import os
import json
import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image
from sentence_transformers import SentenceTransformer
from sbert_retrieval_v4_expanded import retrieve_poems as sbert_retrieve

# ---- paths ----
POEMS_CSV = "data_poems/filtered_poems.csv"
MOOD_PROMPTS_FILE = "mood_prompts.json"
TEST_IMAGES_DIR = "test_images"
CACHE_DIR = "cache"
SBERT_CHUNKS_FILE = os.path.join(CACHE_DIR, "poem_sbert_chunks.npy")
SBERT_CHUNK_MAP_FILE = os.path.join(CACHE_DIR, "poem_sbert_chunk_map.json")
SBERT_IDS_FILE = os.path.join(CACHE_DIR, "sbert_poem_ids.json")

# top 5 moods go into the query; 3-tier intensity mapping by rank
TOP_K_MOODS = 5
# words per chunk -- keeps us safely under SBERT's 256-token limit for poetry
CHUNK_WORDS = 180

# ---- load CLIP (for image mood scoring) ----
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

# ---- load SBERT (for poem retrieval) ----
sbert = SentenceTransformer("all-MiniLM-L6-v2")


# extract the primary mood label from a prompt like
# "a photo that feels desolate, bleak, and abandoned" → "desolate"
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


def build_mood_query(mood_scores):
    # rank all moods by score and take the top 5
    ranked = sorted(zip(mood_prompts, mood_scores), key=lambda x: -x[1])
    top5 = ranked[:TOP_K_MOODS]

    labels = [prompt_to_label(p) for p, _ in top5]

    # 3-tier intensity mapping by rank position:
    # rank 1 → "overwhelmingly", rank 2 → "deeply",
    # rank 3 → "somewhat", ranks 4-5 → "with hints of"
    overwhelming = labels[0]
    deep = labels[1]
    somewhat = labels[2]
    hints = labels[3:5]

    query = (
        f"something that feels overwhelmingly {overwhelming} and deeply {deep}, "
        f"somewhat {somewhat}, with hints of {hints[0]} and {hints[1]}"
    )
    return query


def chunk_poem(text):
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_WORDS):
        chunk = " ".join(words[i:i + CHUNK_WORDS])
        chunks.append(chunk)
    return chunks if chunks else [text]


def build_sbert_cache():
    if (os.path.exists(SBERT_CHUNKS_FILE)
            and os.path.exists(SBERT_CHUNK_MAP_FILE)
            and os.path.exists(SBERT_IDS_FILE)):
        return

    os.makedirs(CACHE_DIR, exist_ok=True)
    poem_df = pd.read_csv(POEMS_CSV)

    all_chunk_embs = []   # flat list of all chunk embeddings
    chunk_map = []        # per poem: (start_idx, end_idx) into all_chunk_embs
    poem_ids = []

    for _, row in poem_df.iterrows():
        chunks = chunk_poem(str(row["text"]))
        embs = sbert.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        # normalize each chunk embedding
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / np.where(norms == 0, 1, norms)

        start = len(all_chunk_embs)
        all_chunk_embs.extend(embs)
        end = len(all_chunk_embs)

        chunk_map.append([start, end])
        poem_ids.append(int(row["id"]))

    np.save(SBERT_CHUNKS_FILE, np.array(all_chunk_embs))
    with open(SBERT_CHUNK_MAP_FILE, "w") as f:
        json.dump(chunk_map, f)
    with open(SBERT_IDS_FILE, "w") as f:
        json.dump(poem_ids, f)


def retrieve_poems_by_mood(image_path, top_k=5):
    mood_scores = get_image_mood_vector(image_path)
    query = build_mood_query(mood_scores)
    results, expanded_query = sbert_retrieve(query, top_k)
    return results, expanded_query, mood_scores


# ---- build cache (skipped if already done) ----
build_sbert_cache()


if __name__ == "__main__":
    # ---- demo ----
    demo_images = [
        ("1_beach_sunny.jpg",          "bright / joyful"),
        ("17_abandoned_classroom.jpg", "dark / melancholic"),
        ("5_city_crowded.jpg",         "urban / energetic"),
    ]

    print("\n" + "=" * 60)
    print("Track 3: Mood-Based Matching (SBERT)")
    print("=" * 60)

    for filename, mood in demo_images:
        image_path = os.path.join(TEST_IMAGES_DIR, filename)
        print(f"\nImage : {filename}  ({mood})")
        print("-" * 50)

        results, query, mood_scores = retrieve_poems_by_mood(image_path, top_k=5)

        print("  Mood distribution:")
        ranked = sorted(zip(mood_prompts, mood_scores), key=lambda x: -x[1])
        for prompt, score in ranked:
            print(f"    {score:.4f}  {prompt}")

        print(f"\n  Query: \"{query}\"")

        print("\n  Top 5 poems:")
        for i, poem in enumerate(results, 1):
            print(f"  {i}. \"{poem['title']}\" by {poem['author']}  [score: {poem['score']:.4f}]")
            print()
