import json
import os
from integrate import parallel_retrieve, rerank_retrieve

TEST_IMAGES_DIR = "test_images"
OUTPUT_FILE = "cache/precomputed_results.json"
STRATEGIES = ["content", "mood", "balanced", "creative", "rerank"]

with open("test_images_metadata.json") as f:
    metadata = json.load(f)

results = {}
total = len(metadata) * len(STRATEGIES)
done = 0

for m in metadata:
    filename = m["filename"]
    image_path = os.path.join(TEST_IMAGES_DIR, filename)
    results[filename] = {}
    print(f"\n[{m['id']}/25] {filename}")

    for strategy in STRATEGIES:
        print(f"  {strategy}...", end=" ", flush=True)
        if strategy == "rerank":
            poems = rerank_retrieve(image_path, top_k=1)
        else:
            poems = parallel_retrieve(image_path, mode=strategy, top_k=1)
        poem = poems[0]
        results[filename][strategy] = {
            "title": poem["title"],
            "author": poem["author"],
            "text": poem["text"],
            "score": poem["combined_score"],
        }
        done += 1
        print(f"done  ({done}/{total})")

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to {OUTPUT_FILE}")
