import os
import json

from clip_semantic import retrieve_poems as clip_retrieve
from blip_captions import retrieve_poems_by_caption
from clip_mood import retrieve_poems_by_mood
from handcrafted_features import retrieve_poems_by_handcrafted

# ---- constants ----
TEST_IMAGES_DIR = "test_images"
METADATA_FILE = "test_images_metadata.json"
OUTPUT_DIR = "milestone_results"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "demo_results.txt")
IMAGE_IDS = [1, 2, 5, 6, 15, 17, 18, 21, 25]

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(METADATA_FILE) as f:
    metadata = {m["id"]: m for m in json.load(f)}


def run_image(image_id):
    meta = metadata[image_id]
    image_path = os.path.join(TEST_IMAGES_DIR, meta["filename"])

    print(f"  Track 1...")
    t1 = clip_retrieve(image_path, top_k=1)

    print(f"  Track 2...")
    t2, consensus, _, _ = retrieve_poems_by_caption(image_path, top_k=1)

    print(f"  Track 3...")
    t3, t3_query, _ = retrieve_poems_by_mood(image_path, top_k=1)

    print(f"  Track 4...")
    t4, _, _, _, _, _, t4_query = retrieve_poems_by_handcrafted(image_path, top_k=1)

    lines = []
    lines.append(f"IMAGE {image_id}: {meta['description']}")
    lines.append(f"Expected moods: {', '.join(meta['expected_moods'])}")
    lines.append("")

    p = t1[0]
    lines.append(f"Track 1 (CLIP Semantic) — score: {p['score']:.4f}")
    lines.append(f"  \"{p['title']}\" — {p['author']}")
    for text_line in p["text"].splitlines():
        lines.append(f"  {text_line}")
    lines.append("")

    p = t2[0]
    lines.append(f"Track 2 (BLIP Caption) — score: {p['score']:.4f}")
    lines.append(f"  Caption: \"{consensus}\"")
    lines.append(f"  \"{p['title']}\" — {p['author']}")
    for text_line in p["text"].splitlines():
        lines.append(f"  {text_line}")
    lines.append("")

    p = t3[0]
    lines.append(f"Track 3 (Mood CLIP+SBERT) — score: {p['score']:.4f}")
    lines.append(f"  Query: \"{t3_query}\"")
    lines.append(f"  \"{p['title']}\" — {p['author']}")
    for text_line in p["text"].splitlines():
        lines.append(f"  {text_line}")
    lines.append("")

    p = t4[0]
    lines.append(f"Track 4 (Handcrafted CV) — score: {p['score']:.4f}")
    lines.append(f"  Query: \"{t4_query}\"")
    lines.append(f"  \"{p['title']}\" — {p['author']}")
    for text_line in p["text"].splitlines():
        lines.append(f"  {text_line}")
    lines.append("")

    lines.append("=" * 60)
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    all_output = []
    for image_id in IMAGE_IDS:
        print(f"\nImage {image_id}: {metadata[image_id]['description']}")
        all_output.append(run_image(image_id))

    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(all_output))

    print(f"\nSaved → {OUTPUT_FILE}")
