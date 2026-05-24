import os
import json
from collections import Counter
import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# ---- paths ----
POEMS_CSV = "data_poems/filtered_poems.csv"
TEST_IMAGES_DIR = "test_images"
CACHE_DIR = "cache"
EMBEDDINGS_FILE = os.path.join(CACHE_DIR, "poem_embeddings.npy")
IDS_FILE = os.path.join(CACHE_DIR, "poem_ids.json")
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-large"

CONSENSUS_THRESHOLD = 3   # phrase must appear in at least this many of 8 captions
CONSENSUS_MIN_WORDS = 2   # fall back to most common caption if consensus is shorter

# 6 nucleus sampling runs (varied top_p and temperature) + 2 beam search anchors
CAPTION_CONFIGS = [
    {"num_beams": 1, "do_sample": True, "top_p": 0.90, "temperature": 1.0},
    {"num_beams": 1, "do_sample": True, "top_p": 0.90, "temperature": 1.2},
    {"num_beams": 1, "do_sample": True, "top_p": 0.80, "temperature": 0.9},
    {"num_beams": 1, "do_sample": True, "top_p": 0.95, "temperature": 1.1},
    {"num_beams": 1, "do_sample": True, "top_p": 0.75, "temperature": 1.0},
    {"num_beams": 1, "do_sample": True, "top_p": 0.85, "temperature": 1.3},
    {"num_beams": 3, "do_sample": False},
    {"num_beams": 5, "do_sample": False},
]

STOP_WORDS = {
    "a", "an", "the", "of", "on", "in", "at", "with", "by", "near",
    "through", "and", "or", "that", "which", "is", "are", "was", "were",
    "some", "many", "several", "few", "its", "their", "this", "these",
    "to", "for", "from", "up", "down", "into", "onto", "over", "under",
    "there", "here", "it", "they", "he", "she", "we", "you", "i",
}

# ---- load CLIP ----
device = "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

# ---- load BLIP ----
blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME, use_safetensors=True)
blip_model.eval()


# kept for documentation — original single-caption approach before consensus filtering
def generate_caption_single(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(image, return_tensors="pt")
    with torch.no_grad():
        out = blip_model.generate(**inputs)
    return blip_processor.decode(out[0], skip_special_tokens=True)


def extract_phrases(caption):
    caption = caption.lower().strip()
    parts = caption.split(",")
    phrases = []
    for part in parts:
        tokens = part.split()
        chunk = []
        for tok in tokens:
            tok = tok.strip(".,!?;:")
            if tok in STOP_WORDS:
                if chunk:
                    phrases.append(" ".join(chunk))
                    chunk = []
            elif tok:
                chunk.append(tok)
        if chunk:
            phrases.append(" ".join(chunk))
    return [p.strip() for p in phrases if p.strip()]


def generate_caption_consensus(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(image, return_tensors="pt")

    raw_captions = []
    for config in CAPTION_CONFIGS:
        with torch.no_grad():
            out = blip_model.generate(**inputs, **config)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        raw_captions.append(caption)

    all_phrases = set()
    for cap in raw_captions:
        all_phrases.update(extract_phrases(cap))

    # count how many captions each phrase appears in (substring match on raw caption)
    phrase_counts = {
        phrase: sum(1 for cap in raw_captions if phrase in cap.lower())
        for phrase in all_phrases
    }

    consensus_phrases = [
        p for p, c in sorted(phrase_counts.items(), key=lambda x: -x[1])
        if c >= CONSENSUS_THRESHOLD
    ]

    consensus = ", ".join(consensus_phrases)
    if len(consensus.split()) < CONSENSUS_MIN_WORDS:
        consensus = Counter(raw_captions).most_common(1)[0][0]

    return consensus, raw_captions, phrase_counts


def retrieve_poems_by_caption(image_path, top_k=5):
    embeddings = np.load(EMBEDDINGS_FILE)
    with open(IDS_FILE) as f:
        poem_ids = json.load(f)
    poem_df = pd.read_csv(POEMS_CSV).set_index("id")

    consensus, raw_captions, phrase_counts = generate_caption_consensus(image_path)

    tokens = clip.tokenize([consensus], truncate=True).to(device)
    with torch.no_grad():
        caption_emb = clip_model.encode_text(tokens)
    caption_emb = caption_emb / caption_emb.norm(dim=-1, keepdim=True)
    caption_emb = caption_emb.squeeze(0).cpu().numpy()

    scores = embeddings @ caption_emb
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

    return results, consensus, raw_captions, phrase_counts


if __name__ == "__main__":
    # ---- demo ----
    demo_images = [
        ("1_beach_sunny.jpg",          "bright / joyful"),
        ("17_abandoned_classroom.jpg", "dark / melancholic"),
        ("5_city_crowded.jpg",         "urban / energetic"),
    ]

    print("\n" + "=" * 60)
    print("Track 2: Scene-Based Matching (BLIP + CLIP)")
    print("=" * 60)

    for filename, mood in demo_images:
        image_path = os.path.join(TEST_IMAGES_DIR, filename)
        print(f"\nImage : {filename}  ({mood})")
        print("-" * 50)

        results, consensus, raw_captions, phrase_counts = retrieve_poems_by_caption(image_path, top_k=5)

        print("  Raw captions:")
        for i, cap in enumerate(raw_captions, 1):
            print(f"    {i}. \"{cap}\"")

        print("\n  Phrase frequencies (* = above consensus threshold):")
        for phrase, count in sorted(phrase_counts.items(), key=lambda x: -x[1]):
            marker = " *" if count >= CONSENSUS_THRESHOLD else ""
            print(f"    {count}/8  {phrase}{marker}")

        print(f"\n  Consensus caption: \"{consensus}\"")

        print("\n  Top 5 poems:")
        for i, poem in enumerate(results, 1):
            print(f"  {i}. \"{poem['title']}\" by {poem['author']}  [score: {poem['score']:.4f}]")
        print()
