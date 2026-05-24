# V1 approach: used YOLOv8 to detect COCO objects and encode detected labels as
# a CLIP text embedding for poem retrieval. Failed because YOLO's 80 COCO
# categories are too narrow for our image set — scenic images like beaches
# produced zero detections, and complex scenes reduced to sparse labels like
# 'chair'. Replaced by V2 using BLIP image captioning in blip_captions.py.
# Kept for documentation.

import os
import json
import numpy as np
import pandas as pd
import torch
import clip
from ultralytics import YOLO

# ---- paths ----
POEMS_CSV = "data_poems/filtered_poems.csv"
TEST_IMAGES_DIR = "test_images"
CACHE_DIR = "cache"
EMBEDDINGS_FILE = os.path.join(CACHE_DIR, "poem_embeddings.npy")
IDS_FILE = os.path.join(CACHE_DIR, "poem_ids.json")
YOLO_MODEL = "yolov8n.pt"
CONF_THRESHOLD = 0.3

# ---- load CLIP (reuses same cache as clip_semantic.py) ----
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# ---- load YOLO ----
yolo = YOLO(YOLO_MODEL)


def detect_objects(image_path, conf_threshold=CONF_THRESHOLD):
    results = yolo(image_path, verbose=False)[0]

    # keep highest-confidence detection per label
    best_conf = {}
    for box in results.boxes:
        conf = float(box.conf[0])
        label = results.names[int(box.cls[0])]
        if conf >= conf_threshold:
            if label not in best_conf or conf > best_conf[label]:
                best_conf[label] = conf

    return sorted(best_conf.items(), key=lambda x: -x[1])


def build_object_query(detections):
    labels = [label for label, _ in detections]
    return ", ".join(labels)


def retrieve_poems_by_objects(image_path, top_k=5):
    embeddings = np.load(EMBEDDINGS_FILE)
    with open(IDS_FILE) as f:
        poem_ids = json.load(f)
    poem_df = pd.read_csv(POEMS_CSV).set_index("id")

    detections = detect_objects(image_path)
    query = build_object_query(detections)

    if not query:
        return [], detections, query

    tokens = clip.tokenize([query], truncate=True).to(device)
    with torch.no_grad():
        query_emb = model.encode_text(tokens)
    query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)
    query_emb = query_emb.squeeze(0).cpu().numpy()

    scores = embeddings @ query_emb
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        pid = poem_ids[idx]
        row = poem_df.loc[pid]
        results.append({
            "title": row["title"],
            "author": row["author"],
            "text": str(row["text"]),
            "score": float(scores[idx]),
        })

    return results, detections, query


# ---- demo ----
demo_images = [
    ("1_beach_sunny.jpg",          "bright / joyful"),
    ("17_abandoned_classroom.jpg", "dark / melancholic"),
    ("5_city_crowded.jpg",         "urban / energetic"),
]

print("\n" + "=" * 60)
print("Track 2: Object-Based Matching (YOLO + CLIP)")
print("=" * 60)

all_results = []

for filename, mood in demo_images:
    image_path = os.path.join(TEST_IMAGES_DIR, filename)
    print(f"\nImage : {filename}  ({mood})")
    print("-" * 50)

    results, detections, query = retrieve_poems_by_objects(image_path, top_k=5)
    all_results.append((filename, detections, query, results))

    if not detections:
        print("  No objects detected above confidence threshold.")
        continue

    print("  Detected objects:")
    for label, conf in detections:
        print(f"    {conf:.2f}  {label}")

    print(f"\n  Query: \"{query}\"")

    print("\n  Top 5 poems:")
    for i, poem in enumerate(results, 1):
        print(f"  {i}. \"{poem['title']}\" by {poem['author']}  [score: {poem['score']:.4f}]")
        print()

