# V2: SBERT retrieval with corpus-based baseline normalization. Samples 300 random
# poem chunks as baseline queries to estimate true corpus-wide centrality.  
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
SBERT_BASELINE_FILE = os.path.join(CACHE_DIR, "sbert_poem_baseline_v2.npy")

BASELINE_SAMPLE_SIZE = 300
BASELINE_SEED = 42

sbert = SentenceTransformer("all-MiniLM-L6-v2")


def build_corpus_baseline():
    if os.path.exists(SBERT_BASELINE_FILE):
        return

    all_chunk_embs = np.load(SBERT_CHUNKS_FILE)
    with open(SBERT_CHUNK_MAP_FILE) as f:
        chunk_map = json.load(f)

    # sample 300 random chunk embeddings as proxy queries for corpus centrality
    rng = np.random.default_rng(BASELINE_SEED)
    sample_idx = rng.choice(len(all_chunk_embs), size=BASELINE_SAMPLE_SIZE, replace=False)
    baseline_queries = all_chunk_embs[sample_idx]  # (300, 384), already normalized

    baselines = []
    for start, end in chunk_map:
        chunk_embs = all_chunk_embs[start:end]
        sims = baseline_queries @ chunk_embs.T   # (300, n_chunks)
        max_per_query = np.max(sims, axis=1)     # (300,)
        baselines.append(float(np.mean(max_per_query)))

    np.save(SBERT_BASELINE_FILE, np.array(baselines))


build_corpus_baseline()


def retrieve_poems(query, top_k=5):
    all_chunk_embs = np.load(SBERT_CHUNKS_FILE)
    with open(SBERT_CHUNK_MAP_FILE) as f:
        chunk_map = json.load(f)
    with open(SBERT_IDS_FILE) as f:
        poem_ids = json.load(f)
    poem_df = pd.read_csv(POEMS_CSV).set_index("id")
    baseline = np.load(SBERT_BASELINE_FILE)

    query_emb = sbert.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
    query_emb = query_emb / np.linalg.norm(query_emb)

    poem_scores = []
    for start, end in chunk_map:
        chunk_embs = all_chunk_embs[start:end]
        sims = chunk_embs @ query_emb
        poem_scores.append(float(np.max(sims)))

    poem_scores = np.array(poem_scores) - baseline
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
    print("SBERT Retrieval V2: Corpus Baseline")
    print("=" * 60)

    for image_label, query in demo_queries:
        print(f"\nImage : {image_label}")
        print(f"Query : \"{query}\"")
        print("-" * 50)
        results = retrieve_poems(query, top_k=5)
        for i, poem in enumerate(results, 1):
            print(f"  {i}. \"{poem['title']}\" by {poem['author']}  [score: {poem['score']:.4f}]")
        print()
