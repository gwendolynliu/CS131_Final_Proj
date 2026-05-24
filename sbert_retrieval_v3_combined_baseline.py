# V3: Combined baseline — average of V1 (mood prompt) and V2 (corpus chunk) baselines.
# V1 strongly penalizes self-referential/centroid poems because its 17 mood prompts
# cluster in a narrow semantic region, spreading poem scores wide (std 0.074, Justice z=+2.05).
# V2 penalizes broadly popular poems across diverse topics but compresses the range
# (std 0.056, Justice z=+1.39), letting Justice slip back to #1.
# Averaging the two baselines combines both penalization signals.
# Kept for documentation.

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
SBERT_BASELINE_V1_FILE = os.path.join(CACHE_DIR, "sbert_poem_baseline_v1.npy")
SBERT_BASELINE_V2_FILE = os.path.join(CACHE_DIR, "sbert_poem_baseline_v2.npy")

sbert = SentenceTransformer("all-MiniLM-L6-v2")

baseline_v1 = np.load(SBERT_BASELINE_V1_FILE)
baseline_v2 = np.load(SBERT_BASELINE_V2_FILE)
baseline_v3 = (baseline_v1 + baseline_v2) / 2


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

    poem_scores = np.array(poem_scores) - baseline_v3
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
    with open(SBERT_IDS_FILE) as f:
        poem_ids_list = json.load(f)
    poem_df = pd.read_csv(POEMS_CSV).set_index("id")
    id_to_idx = {pid: i for i, pid in enumerate(poem_ids_list)}

    mean_v3 = np.mean(baseline_v3)
    std_v3 = np.std(baseline_v3)

    print("\n" + "=" * 60)
    print("V3 Combined Baseline Diagnostics")
    print("=" * 60)
    print(f"  mean: {mean_v3:.4f}   std: {std_v3:.4f}")
    print(f"  min:  {np.min(baseline_v3):.4f}   max: {np.max(baseline_v3):.4f}")

    targets = [
        ("Donald Justice", "Poem"),
        ("James Laughlin", "Technical Notes"),
    ]
    print()
    for author, title_frag in targets:
        matches = poem_df[(poem_df["author"].str.contains(author, na=False)) &
                          (poem_df["title"].str.contains(title_frag, na=False))]
        for pid, row in matches.iterrows():
            idx = id_to_idx[pid]
            b = baseline_v3[idx]
            z = (b - mean_v3) / std_v3
            print(f'  "{row["title"][:55]}"')
            print(f'    combined baseline: {b:.4f}  z-score: {z:+.2f}')
            print(f'    (V1: {baseline_v1[idx]:.4f}, V2: {baseline_v2[idx]:.4f})')

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
    print("SBERT Retrieval V3: Combined Baseline")
    print("=" * 60)

    for image_label, query in demo_queries:
        print(f"\nImage : {image_label}")
        print(f"Query : \"{query}\"")
        print("-" * 50)
        results = retrieve_poems(query, top_k=5)
        for i, poem in enumerate(results, 1):
            print(f"  {i}. \"{poem['title']}\" by {poem['author']}  [score: {poem['score']:.4f}]")
        print()
