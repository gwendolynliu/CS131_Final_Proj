import os
import numpy as np
import pandas as pd

from clip_semantic import retrieve_poems as clip_retrieve
from blip_captions import retrieve_poems_by_caption
from clip_mood import retrieve_poems_by_mood
from clip_mood import get_mood_query as get_mood_query_t3
from handcrafted_features import retrieve_poems_by_handcrafted
from handcrafted_features import get_mood_query as get_mood_query_t4
from sbert_retrieval_v4_expanded import score_poems as sbert_score_poems

# ---- constants ----
POEMS_CSV = "data_poems/filtered_poems.csv"
TEST_IMAGES_DIR = "test_images"
INTEGRATION_TOP_K = 50   # results each track returns before combining
CREATIVE_POOL = 30       # top-n poems to sample from in creative mode
# changing CREATIVE_SEED produces different creative outputs while keeping results reproducible
CREATIVE_SEED = 42

MODES = {
    "content":  {1: 0.50, 2: 0.30, 3: 0.15, 4: 0.05},
    "mood":     {1: 0.15, 2: 0.10, 3: 0.60, 4: 0.15},
    "balanced": {1: 0.40, 2: 0.20, 3: 0.30, 4: 0.10},
    "creative": {1: 0.05, 2: 0.05, 3: 0.55, 4: 0.35},
}


# ---- per-track wrappers returning (poem_id, score) pairs ----

def track1_retrieve(image_path, top_k):
    results = clip_retrieve(image_path, top_k=top_k)
    return [(r["poem_id"], r["score"]) for r in results]


def track2_retrieve(image_path, top_k):
    results, _, _, _ = retrieve_poems_by_caption(image_path, top_k=top_k)
    return [(r["poem_id"], r["score"]) for r in results]


def track3_retrieve(image_path, top_k):
    # diversity=False: skip greedy diversity filtering, return raw top-k by score
    results, _, _ = retrieve_poems_by_mood(image_path, top_k=top_k, diversity=False)
    return [(r["poem_id"], r["score"]) for r in results]


def track4_retrieve(image_path, top_k):
    # diversity=False: skip greedy diversity filtering, return raw top-k by score
    results, *_ = retrieve_poems_by_handcrafted(image_path, top_k=top_k, diversity=False)
    return [(r["poem_id"], r["score"]) for r in results]


def normalize_scores(scores):
    if not scores:
        return []
    vals = [s for _, s in scores]
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return [(pid, 0.5) for pid, _ in scores]
    return [(pid, (s - lo) / (hi - lo)) for pid, s in scores]


def parallel_retrieve(image_path, mode="balanced", top_k=5):
    weights = MODES[mode]

    tracks = [
        normalize_scores(track1_retrieve(image_path, INTEGRATION_TOP_K)),
        normalize_scores(track2_retrieve(image_path, INTEGRATION_TOP_K)),
        normalize_scores(track3_retrieve(image_path, INTEGRATION_TOP_K)),
        normalize_scores(track4_retrieve(image_path, INTEGRATION_TOP_K)),
    ]

    # weighted sum — poems missing from a track's top-50 contribute 0 for that track
    combined = {}
    for track_scores, weight in zip(tracks, [weights[1], weights[2], weights[3], weights[4]]):
        for pid, score in track_scores:
            combined[pid] = combined.get(pid, 0.0) + weight * score

    ranked = sorted(combined.items(), key=lambda x: -x[1])

    if mode == "creative":
        pool = ranked[:CREATIVE_POOL]
        pool_ids = [pid for pid, _ in pool]
        pool_scores = np.array([s for _, s in pool], dtype=float)
        pool_scores = pool_scores / pool_scores.sum()  # normalize to probability distribution

        rng = np.random.default_rng(CREATIVE_SEED)
        chosen = rng.choice(len(pool), size=min(top_k, len(pool)), replace=False, p=pool_scores)
        # return chosen poems sorted by descending combined score
        selected = sorted([(pool_ids[i], pool[i][1]) for i in chosen], key=lambda x: -x[1])
    else:
        selected = ranked[:top_k]

    poem_df = pd.read_csv(POEMS_CSV).set_index("id")
    results = []
    for pid, score in selected:
        row = poem_df.loc[pid]
        results.append({
            "poem_id": pid,
            "title": row["title"],
            "author": row["author"],
            "text": str(row["text"]),
            "combined_score": score,
        })
    return results


def rerank_retrieve(image_path, top_k=5):
    # step 1: get content candidates from T1 and T2
    t1_raw = track1_retrieve(image_path, INTEGRATION_TOP_K)
    t2_raw = track2_retrieve(image_path, INTEGRATION_TOP_K)

    # union by poem_id — content tracks get them in the door
    candidate_ids = list({pid for pid, _ in t1_raw} | {pid for pid, _ in t2_raw})

    t1_norm = dict(normalize_scores(t1_raw))
    t2_norm = dict(normalize_scores(t2_raw))

    # step 2: score only the candidates with T3 and T4 (not the full corpus)
    t3_query = get_mood_query_t3(image_path)
    t4_query = get_mood_query_t4(image_path)

    t3_raw = sbert_score_poems(t3_query, candidate_ids)
    t4_raw = sbert_score_poems(t4_query, candidate_ids)

    t3_norm = dict(normalize_scores(list(t3_raw.items())))
    t4_norm = dict(normalize_scores(list(t4_raw.items())))

    # step 3: combine — mood decides final order
    combined = {}
    for pid in candidate_ids:
        combined[pid] = (
            0.3 * t3_norm.get(pid, 0.0) +
            0.2 * t4_norm.get(pid, 0.0) +
            0.3 * t1_norm.get(pid, 0.0) +
            0.2 * t2_norm.get(pid, 0.0)
        )

    ranked = sorted(combined.items(), key=lambda x: -x[1])[:top_k]

    poem_df = pd.read_csv(POEMS_CSV).set_index("id")
    results = []
    for pid, score in ranked:
        row = poem_df.loc[pid]
        results.append({
            "poem_id": pid,
            "title": row["title"],
            "author": row["author"],
            "text": str(row["text"]),
            "combined_score": score,
        })
    return results


if __name__ == "__main__":
    demo_images = [
        "1_beach_sunny.jpg",
        "17_abandoned_classroom.jpg",
        "5_city_crowded.jpg",
    ]

    print("\n" + "=" * 60)
    print("Integration Layer Demo")
    print("=" * 60)

    for filename in demo_images:
        image_path = os.path.join(TEST_IMAGES_DIR, filename)
        print(f"\nImage: {filename}")
        print("-" * 50)

        for mode in ["content", "mood", "balanced", "creative"]:
            print(f"\n  [{mode}]")
            results = parallel_retrieve(image_path, mode=mode, top_k=5)
            for i, poem in enumerate(results, 1):
                print(f"    {i}. \"{poem['title']}\" — {poem['author']}  [score: {poem['combined_score']:.4f}]")

        print(f"\n  [rerank]")
        results = rerank_retrieve(image_path, top_k=5)
        for i, poem in enumerate(results, 1):
            print(f"    {i}. \"{poem['title']}\" — {poem['author']}  [score: {poem['combined_score']:.4f}]")
