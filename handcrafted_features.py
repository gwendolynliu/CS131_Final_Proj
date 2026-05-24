import os
import json
import numpy as np
from PIL import Image, ImageFilter
from sbert_retrieval_v4_expanded import retrieve_poems as sbert_retrieve

# ---- paths ----
POEMS_CSV = "data_poems/filtered_poems.csv"
MOOD_PROMPTS_FILE = "mood_prompts.json"
TEST_IMAGES_DIR = "test_images"
CACHE_DIR = "cache"
EDGE_THRESHOLD = 20  # pixel brightness cutoff in PIL FIND_EDGES output
TOP_K_MOODS = 5

with open(MOOD_PROMPTS_FILE) as f:
    mood_prompts = json.load(f)


def prompt_to_label(prompt):
    return prompt.split("feels ")[1].split(",")[0]



def extract_color_temperature(image_path):
    arr = np.array(Image.open(image_path).convert("RGB"), dtype=float)
    r, b = arr[:, :, 0], arr[:, :, 2]
    # weight warm/cool signal by per-pixel saturation so gray pixels don't bias the score
    sat = np.max(arr, axis=2) - np.min(arr, axis=2)
    weighted = np.sum((r - b) * sat) / (np.sum(sat) + 1e-6)
    return float((weighted / 255 + 1) / 2)  # [-1, 1] → [0, 1]


def extract_brightness_contrast(image_path):
    arr = np.array(Image.open(image_path).convert("L"), dtype=float)
    brightness = float(np.mean(arr) / 255)
    contrast = float(min(np.std(arr) / 64, 1.0))
    return brightness, contrast


def extract_edge_density(image_path):
    img = Image.open(image_path).convert("L")
    edges = np.array(img.filter(ImageFilter.FIND_EDGES), dtype=float)
    return float(np.mean(edges > EDGE_THRESHOLD))


def map_features_to_mood(warm, brightness, contrast, edge_density):
    cool = 1 - warm
    dim = 1 - brightness

    scores = np.zeros(17)
    scores[0]  = 0.4*warm + 0.4*brightness + 0.2*edge_density              # joyful
    scores[1]  = 0.3*brightness + 0.4*(1-edge_density) + 0.3*(1-contrast)  # peaceful
    scores[2]  = 0.4*cool + 0.4*dim + 0.2*(1-edge_density)                 # melancholic
    scores[3]  = 0.3*dim + 0.4*cool + 0.3*(1-edge_density)                 # lonely
    scores[4]  = 0.5*warm + 0.3*brightness + 0.2*(1-contrast)              # nostalgic
    scores[5]  = 0.4*contrast + 0.3*edge_density + 0.3*dim                 # tense
    scores[6]  = 0.4*dim + 0.3*contrast + 0.3*(1-edge_density)             # mysterious
    scores[7]  = 0.4*warm + 0.2*brightness + 0.4*(1-contrast)              # romantic
    scores[8]  = 0.5*edge_density + 0.5*contrast                           # chaotic
    scores[9]  = 0.5*dim + 0.3*cool + 0.2*(1-edge_density)                 # desolate
    scores[10] = 0.4*brightness + 0.3*(1-edge_density) + 0.3*contrast      # sublime
    scores[11] = 0.3*warm + 0.3*brightness + 0.4*(1-edge_density)          # cozy
    scores[12] = 0.4*(1-edge_density) + 0.3*(1-contrast) + 0.3*brightness  # contemplative
    scores[13] = 0.5*contrast + 0.3*(1-edge_density) + 0.2*dim             # surreal
    scores[14] = 0.6*dim + 0.4*cool                                        # dark
    scores[15] = 0.4*brightness + 0.3*warm + 0.3*(1-edge_density)          # hopeful
    scores[16] = 0.4*warm + 0.3*edge_density + 0.3*brightness              # playful

    scores = scores / np.linalg.norm(scores)
    return scores


def build_mood_query(mood_scores):
    ranked = sorted(zip(mood_prompts, mood_scores), key=lambda x: -x[1])
    top5 = ranked[:TOP_K_MOODS]
    labels = [prompt_to_label(p) for p, _ in top5]
    query = (
        f"something that feels overwhelmingly {labels[0]} and deeply {labels[1]}, "
        f"somewhat {labels[2]}, with hints of {labels[3]} and {labels[4]}"
    )
    return query


def retrieve_poems_by_handcrafted(image_path, top_k=5):
    warm = extract_color_temperature(image_path)
    brightness, contrast = extract_brightness_contrast(image_path)
    edge_density = extract_edge_density(image_path)
    mood_scores = map_features_to_mood(warm, brightness, contrast, edge_density)
    query = build_mood_query(mood_scores)
    results, expanded_query = sbert_retrieve(query, top_k)
    return results, warm, brightness, contrast, edge_density, mood_scores, expanded_query


if __name__ == "__main__":
    # ---- demo ----
    demo_images = [
        ("1_beach_sunny.jpg",          "bright / joyful"),
        ("17_abandoned_classroom.jpg", "dark / melancholic"),
        ("5_city_crowded.jpg",         "urban / energetic"),
    ]

    print("\n" + "=" * 60)
    print("Track 4: Handcrafted CV Features")
    print("=" * 60)

    for filename, mood in demo_images:
        image_path = os.path.join(TEST_IMAGES_DIR, filename)
        print(f"\nImage : {filename}  ({mood})")
        print("-" * 50)

        results, warm, brightness, contrast, edge_density, mood_scores, query = \
            retrieve_poems_by_handcrafted(image_path, top_k=5)

        print("  Raw features:")
        print(f"    warm:         {warm:.3f}")
        print(f"    brightness:   {brightness:.3f}")
        print(f"    contrast:     {contrast:.3f}")
        print(f"    edge_density: {edge_density:.3f}")

        print("\n  Mood vector:")
        ranked_moods = sorted(zip(mood_prompts, mood_scores), key=lambda x: -x[1])
        for prompt, score in ranked_moods:
            print(f"    {score:.4f}  {prompt_to_label(prompt)}")

        print(f"\n  Query: \"{query}\"")

        print("\n  Top 5 poems:")
        for i, poem in enumerate(results, 1):
            print(f"  {i}. \"{poem['title']}\" by {poem['author']}  [score: {poem['score']:.4f}]")
            print()
