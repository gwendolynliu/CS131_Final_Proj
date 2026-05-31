# V4: Adds query expansion with theme words and diversity filtering to V3's combined
# baseline normalization. Builds on V3's combined V1+V2 baseline.
# Changed query phrasing from "a poem..." to "something..." (MOST USEFUL!!!)

import os
import json
import re
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

DIVERSITY_THRESHOLD = 0.85
CANDIDATE_POOL = 50

mood_expansions = {
    "joyful":        ["light", "energy", "celebration", "warmth", "movement"],
    "peaceful":      ["ease", "breath", "soft light", "settling", "quiet"],
    "melancholic":   ["sadness", "memory", "loss", "longing", "grief"],
    "lonely":        ["isolation", "absence", "solitude", "distance", "waiting"],
    "nostalgic":     ["memory", "childhood", "past", "return", "bittersweetness"],
    "tense":         ["pressure", "conflict", "urgency", "unease", "strain"],
    "mysterious":    ["uncertainty", "shadows", "strangeness", "fear", "mystery"],
    "romantic":      ["desire", "tenderness", "intimacy", "devotion", "passion"],
    "chaotic":       ["noise", "confusion", "frenzy", "disorder", "overwhelm"],
    "desolate":      ["abandonment", "emptiness", "ruins", "decay", "silence"],
    "sublime":       ["open space", "distance", "expanse", "wonder", "grandeur"],
    "cozy":          ["shelter", "comfort", "home", "softness", "belonging"],
    "contemplative": ["pausing", "looking inward", "slow attention", "noticing", "awareness"],
    "surreal":       ["dream", "distortion", "unreality", "hallucination", "otherworldly"],
    "dark":          ["shadow", "night", "dread", "weight", "obscurity"],
    "hopeful":       ["dawn", "possibility", "growth", "renewal", "aspiration"],
    "playful":       ["humor", "mischief", "lightness", "curiosity", "surprise"],
}

sbert = SentenceTransformer("all-MiniLM-L6-v2")

baseline_v1 = np.load(SBERT_BASELINE_V1_FILE)
baseline_v2 = np.load(SBERT_BASELINE_V2_FILE)
baseline_v3 = (baseline_v1 + baseline_v2) / 2


def expand_query(query):
    m = re.search(
        r"overwhelmingly (\w+) and deeply (\w+), somewhat (\w+), with hints of (\w+) and (\w+)",
        query
    )
    if not m:
        return query
    labels = list(m.groups())
    expansion_words = []
    for label in labels[:3]:
        expansion_words.extend(mood_expansions.get(label, []))
    if expansion_words:
        query += ", with themes of " + ", ".join(expansion_words)
    return query


def retrieve_poems(query, top_k=5, diversity=True):
    expanded_query = expand_query(query)

    all_chunk_embs = np.load(SBERT_CHUNKS_FILE)
    with open(SBERT_CHUNK_MAP_FILE) as f:
        chunk_map = json.load(f)
    with open(SBERT_IDS_FILE) as f:
        poem_ids = json.load(f)
    poem_df = pd.read_csv(POEMS_CSV).set_index("id")

    query_emb = sbert.encode([expanded_query], convert_to_numpy=True, show_progress_bar=False)[0]
    query_emb = query_emb / np.linalg.norm(query_emb)

    poem_scores = []
    poem_mean_embs = []
    for start, end in chunk_map:
        chunk_embs = all_chunk_embs[start:end]
        sims = chunk_embs @ query_emb
        poem_scores.append(float(np.max(sims)))
        if diversity:
            poem_mean_embs.append(np.mean(chunk_embs, axis=0))

    poem_scores = np.array(poem_scores) - baseline_v3

    if diversity:
        poem_mean_embs = np.array(poem_mean_embs)
        norms = np.linalg.norm(poem_mean_embs, axis=1, keepdims=True)
        poem_mean_embs = poem_mean_embs / np.where(norms == 0, 1, norms)

        top_candidates = np.argsort(poem_scores)[::-1][:CANDIDATE_POOL]
        selected_indices = []
        selected_embs = []
        for idx in top_candidates:
            emb = poem_mean_embs[idx]
            if all(float(emb @ sel) < DIVERSITY_THRESHOLD for sel in selected_embs):
                selected_indices.append(idx)
                selected_embs.append(emb)
            if len(selected_indices) == top_k:
                break
    else:
        selected_indices = list(np.argsort(poem_scores)[::-1][:top_k])

    results = []
    for idx in selected_indices:
        pid = poem_ids[idx]
        row = poem_df.loc[pid]
        results.append({
            "poem_id": pid,
            "title": row["title"],
            "author": row["author"],
            "text": str(row["text"]),
            "score": float(poem_scores[idx]),
        })

    return results, expanded_query


def score_poems(query, poem_ids):
    """Score a specific list of poem_ids against a query. Returns {poem_id: score}."""
    expanded_query = expand_query(query)

    all_chunk_embs = np.load(SBERT_CHUNKS_FILE)
    with open(SBERT_CHUNK_MAP_FILE) as f:
        chunk_map = json.load(f)
    with open(SBERT_IDS_FILE) as f:
        all_poem_ids = json.load(f)

    query_emb = sbert.encode([expanded_query], convert_to_numpy=True, show_progress_bar=False)[0]
    query_emb = query_emb / np.linalg.norm(query_emb)

    id_to_idx = {pid: i for i, pid in enumerate(all_poem_ids)}

    scores = {}
    for pid in poem_ids:
        if pid not in id_to_idx:
            continue
        idx = id_to_idx[pid]
        start, end = chunk_map[idx]
        chunk_embs = all_chunk_embs[start:end]
        sims = chunk_embs @ query_emb
        scores[pid] = float(np.max(sims)) - float(baseline_v3[idx])

    return scores


if __name__ == "__main__":
    demo_queries = [
        ("1_beach_sunny.jpg  (bright / joyful)",
         "something that feels overwhelmingly sublime and deeply peaceful, "
         "somewhat lonely, with hints of surreal and contemplative"),
        ("17_abandoned_classroom.jpg  (dark / melancholic)",
         "something that feels overwhelmingly desolate and deeply mysterious, "
         "somewhat lonely, with hints of melancholic and dark"),
        ("5_city_crowded.jpg  (urban / energetic)",
         "something that feels overwhelmingly chaotic and deeply cozy, "
         "somewhat nostalgic, with hints of tense and surreal"),
    ]

    print("\n" + "=" * 60)
    print("SBERT Retrieval V4: Query Expansion + Diversity Filtering")
    print("=" * 60)

    for image_label, query in demo_queries:
        print(f"\nImage : {image_label}")
        print("-" * 50)
        results, expanded_query = retrieve_poems(query, top_k=5)
        print(f"  Expanded query: \"{expanded_query}\"")
        print()
        print("  Top 5 poems:")
        for i, poem in enumerate(results, 1):
            print(f"  {i}. \"{poem['title']}\" by {poem['author']}  [score: {poem['score']:.4f}]")
        print()
