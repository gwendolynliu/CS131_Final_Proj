# V0: Raw SBERT retrieval with no baseline normalization. Centroid poems like
# "This poem is not addressed to you" dominate all queries because they sit near
# the center of SBERT embedding space. Kept for documentation.

import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ---- paths ----
POEMS_CSV = "data_poems/filtered_poems.csv"
CACHE_DIR = "cache"
SBERT_CHUNKS_FILE = os.path.join(CACHE_DIR, "poem_sbert_chunks.npy")
SBERT_CHUNK_MAP_FILE = os.path.join(CACHE_DIR, "poem_sbert_chunk_map.json")
SBERT_IDS_FILE = os.path.join(CACHE_DIR, "sbert_poem_ids.json")

sbert = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve_poems(query, top_k=5):
    all_chunk_embs = np.load(SBERT_CHUNKS_FILE)
    with open(SBERT_CHUNK_MAP_FILE) as f:
        chunk_map = json.load(f)
    with open(SBERT_IDS_FILE) as f:
        poem_ids = json.load(f)
    poem_df = pd.read_csv(POEMS_CSV).set_index("id")

    query_emb = sbert.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
    query_emb = query_emb / np.linalg.norm(query_emb)

    poem_scores = []
    for start, end in chunk_map:
        chunk_embs = all_chunk_embs[start:end]
        sims = chunk_embs @ query_emb
        poem_scores.append(float(np.max(sims)))

    poem_scores = np.array(poem_scores)
    top_indices = np.argsort(poem_scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        pid = poem_ids[idx]
        row = poem_df.loc[pid]
        results.append({
            "title": row["title"],
            "author": row["author"],
            "text": str(row["text"]),
            "score": float(poem_scores[idx]),
        })

    return results


if __name__ == "__main__":
    demo_queries = [
        ("1_beach_sunny.jpg  (bright / joyful)",
         "a poem that feels overwhelmingly sublime and deeply peaceful, "
         "somewhat lonely, with hints of surreal and contemplative"),
        ("17_abandoned_classroom.jpg  (dark / melancholic)",
         "a poem that feels overwhelmingly desolate and deeply mysterious, "
         "somewhat lonely, with hints of melancholic and dark"),
        ("5_city_crowded.jpg  (urban / energetic)",
         "a poem that feels overwhelmingly chaotic and deeply cozy, "
         "somewhat nostalgic, with hints of tense and surreal"),
    ]

    print("\n" + "=" * 60)
    print("SBERT Retrieval V0: Raw (no baseline)")
    print("=" * 60)

    for image_label, query in demo_queries:
        print(f"\nImage : {image_label}")
        print(f"Query : \"{query}\"")
        print("-" * 50)
        results = retrieve_poems(query, top_k=5)
        for i, poem in enumerate(results, 1):
            print(f"  {i}. \"{poem['title']}\" by {poem['author']}  [score: {poem['score']:.4f}]")
        print()
