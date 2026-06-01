"""
llm_judge_eval.py

Runs an Anthropic/Claude LLM judge over the top-1 retrieved poem from each of:
  1. Track 1: CLIP Semantic
  2. Track 2: BLIP Caption
  3. Track 3: Mood CLIP+SBERT
  4. Track 4: Handcrafted CV Features
  5. Rerank Retrieve

For each image-poem pair, the judge scores:
  - semantic_fit: 1-5
  - emotional_fit: 1-5
  - overall_resonance: 1-5

Outputs:
  - llm_judge_results/judgments_detailed.csv
  - llm_judge_results/judgments_detailed.json
  - llm_judge_results/model_summary.csv
  - llm_judge_results/model_summary.json

Run from repo root:
  python llm_judge_eval.py

Before running:
  1. Fill in ANTHROPIC_API_KEY below, or set env var ANTHROPIC_API_KEY.
  2. Make sure your cache files exist.
  3. Make sure test_images/ and test_images_metadata.json exist.
"""

import base64
import csv
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from clip_semantic import retrieve_poems as clip_retrieve
from blip_captions import retrieve_poems_by_caption
from clip_mood import retrieve_poems_by_mood
from handcrafted_features import retrieve_poems_by_handcrafted
from integrate import rerank_retrieve


# =========================
# Config
# =========================

ANTHROPIC_API_KEY = "FILL IN LATER"  # or set environment variable ANTHROPIC_API_KEY
ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"

BASE_DIR = Path(__file__).resolve().parent
TEST_IMAGES_DIR = BASE_DIR / "test_images"
METADATA_FILE = BASE_DIR / "test_images_metadata.json"
OUTPUT_DIR = BASE_DIR / "llm_judge_results"

TOP_K_PER_MODEL = 1
MAX_POEM_CHARS = 3500
REQUEST_SLEEP_SECONDS = 0.5
MAX_RETRIES = 3

# Set to None to evaluate every image in test_images_metadata.json.
# Or use a subset for debugging, e.g. IMAGE_IDS = [1, 2, 5]
IMAGE_IDS: Optional[List[int]] = None


# =========================
# Retrieval wrappers
# =========================

def get_first_result(results: List[Dict[str, Any]], model_name: str, image_path: Path) -> Dict[str, Any]:
    if not results:
        raise RuntimeError(f"{model_name} returned no results for {image_path}")
    return results[0]


def run_track_1(image_path: Path) -> Dict[str, Any]:
    results = clip_retrieve(str(image_path), top_k=TOP_K_PER_MODEL)
    poem = get_first_result(results, "track_1_clip_semantic", image_path)
    return {
        "model_name": "track_1_clip_semantic",
        "retrieval_score": poem.get("score"),
        "retrieval_extra": "",
        "poem": poem,
    }


def run_track_2(image_path: Path) -> Dict[str, Any]:
    results, consensus_caption, raw_captions, phrase_counts = retrieve_poems_by_caption(
        str(image_path),
        top_k=TOP_K_PER_MODEL,
    )
    poem = get_first_result(results, "track_2_blip_caption", image_path)
    return {
        "model_name": "track_2_blip_caption",
        "retrieval_score": poem.get("score"),
        "retrieval_extra": f"Consensus caption: {consensus_caption}",
        "poem": poem,
    }


def run_track_3(image_path: Path) -> Dict[str, Any]:
    results, expanded_query, mood_scores = retrieve_poems_by_mood(
        str(image_path),
        top_k=TOP_K_PER_MODEL,
    )
    poem = get_first_result(results, "track_3_clip_sbert_mood", image_path)
    return {
        "model_name": "track_3_clip_sbert_mood",
        "retrieval_score": poem.get("score"),
        "retrieval_extra": f"Expanded mood query: {expanded_query}",
        "poem": poem,
    }


def run_track_4(image_path: Path) -> Dict[str, Any]:
    results, warm, brightness, contrast, edge_density, mood_scores, expanded_query = retrieve_poems_by_handcrafted(
        str(image_path),
        top_k=TOP_K_PER_MODEL,
    )
    poem = get_first_result(results, "track_4_handcrafted_cv", image_path)
    return {
        "model_name": "track_4_handcrafted_cv",
        "retrieval_score": poem.get("score"),
        "retrieval_extra": (
            f"Expanded handcrafted query: {expanded_query}; "
            f"warm={warm:.3f}, brightness={brightness:.3f}, "
            f"contrast={contrast:.3f}, edge_density={edge_density:.3f}"
        ),
        "poem": poem,
    }


def run_rerank(image_path: Path) -> Dict[str, Any]:
    results = rerank_retrieve(str(image_path), top_k=TOP_K_PER_MODEL)
    poem = get_first_result(results, "rerank_retrieve", image_path)
    return {
        "model_name": "rerank_retrieve",
        "retrieval_score": poem.get("combined_score"),
        "retrieval_extra": "Integrated rerank using content candidates from Track 1/2 and mood scoring from Track 3/4.",
        "poem": poem,
    }


MODEL_RUNNERS: List[Tuple[str, Callable[[Path], Dict[str, Any]]]] = [
    ("track_1_clip_semantic", run_track_1),
    ("track_2_blip_caption", run_track_2),
    ("track_3_clip_sbert_mood", run_track_3),
    ("track_4_handcrafted_cv", run_track_4),
    ("rerank_retrieve", run_rerank),
]


# =========================
# Anthropic judge
# =========================

def image_to_anthropic_block(image_path: Path) -> Dict[str, Any]:
    media_type, _ = mimetypes.guess_type(str(image_path))
    if media_type is None:
        media_type = "image/jpeg"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": encoded,
        },
    }


def truncate_text(text: Any, max_chars: int = MAX_POEM_CHARS) -> str:
    text = "" if text is None else str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED]"


def build_judge_prompt(
    image_meta: Dict[str, Any],
    model_name: str,
    retrieval_extra: str,
    poem: Dict[str, Any],
) -> str:
    title = poem.get("title", "")
    author = poem.get("author", "")
    poem_text = truncate_text(poem.get("text", ""))

    expected_moods = ", ".join(image_meta.get("expected_moods", []))

    return f"""
You are evaluating an image-to-poem retrieval system.

Your task:
Score this retrieved image-poem pair on three separate dimensions from 1 to 5.

Dimensions:
1. semantic_fit:
   Does the poem match the image's visible content, subject matter, setting, objects, people, or themes?
2. emotional_fit:
   Does the poem match the image's mood, atmosphere, affective tone, or emotional impression?
3. overall_resonance:
   Does the image-poem pairing feel meaningful, compelling, aesthetically convincing, or poetically resonant?

Scoring scale:
1 = very poor fit
2 = weak fit
3 = moderate / plausible fit
4 = strong fit
5 = excellent fit

Important:
- Keep the three dimensions separate.
- Do not reward a poem just because it is generally good.
- Judge the pairing between the provided image and this poem.
- Use the image itself as primary evidence.
- The metadata below is only context, not the ground truth.

Image metadata:
- image_id: {image_meta.get("id")}
- filename: {image_meta.get("filename")}
- category: {image_meta.get("category")}
- description: {image_meta.get("description")}
- expected_moods: {expected_moods}
- scene_type: {image_meta.get("scene_type")}

Retrieval model:
- model_name: {model_name}
- retrieval_extra: {retrieval_extra}

Retrieved poem:
Title: {title}
Author: {author}

Poem text:
\"\"\"
{poem_text}
\"\"\"

Return ONLY valid JSON with this exact schema:
{{
  "semantic_fit": 1,
  "emotional_fit": 1,
  "overall_resonance": 1,
  "rationale": "Brief 2-4 sentence explanation."
}}
""".strip()


def call_anthropic_judge(
    image_path: Path,
    image_meta: Dict[str, Any],
    model_name: str,
    retrieval_extra: str,
    poem: Dict[str, Any],
) -> Dict[str, Any]:
    api_key = os.environ.get("ANTHROPIC_API_KEY") or ANTHROPIC_API_KEY
    if not api_key or api_key == "FILL IN LATER":
        raise RuntimeError(
            "Missing Anthropic API key. Fill in ANTHROPIC_API_KEY in this file "
            "or set environment variable ANTHROPIC_API_KEY."
        )

    prompt = build_judge_prompt(image_meta, model_name, retrieval_extra, poem)

    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 600,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    image_to_anthropic_block(image_path),
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                raw = response.read().decode("utf-8")
                data = json.loads(raw)
                text = extract_text_from_anthropic_response(data)
                return parse_judge_json(text)

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Anthropic HTTP error {e.code}: {body}") from e
            time.sleep(2 ** attempt)

        except Exception:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2 ** attempt)

    raise RuntimeError("Unexpected Anthropic call failure.")


def extract_text_from_anthropic_response(data: Dict[str, Any]) -> str:
    parts = []
    for block in data.get("content", []):
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts).strip()


def parse_judge_json(text: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: extract first JSON object if the model wrapped it in extra text.
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse judge response as JSON:\n{text}")
        parsed = json.loads(match.group(0))

    for key in ["semantic_fit", "emotional_fit", "overall_resonance"]:
        if key not in parsed:
            raise ValueError(f"Judge response missing key {key}: {parsed}")
        parsed[key] = int(parsed[key])
        if not 1 <= parsed[key] <= 5:
            raise ValueError(f"Judge score out of range for {key}: {parsed[key]}")

    parsed["rationale"] = str(parsed.get("rationale", "")).strip()
    return parsed


# =========================
# Evaluation loop
# =========================

def load_metadata() -> List[Dict[str, Any]]:
    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)

    if IMAGE_IDS is not None:
        wanted = set(IMAGE_IDS)
        metadata = [m for m in metadata if m.get("id") in wanted]

    return metadata


def score_one_pair(
    image_meta: Dict[str, Any],
    image_path: Path,
    model_name: str,
    runner: Callable[[Path], Dict[str, Any]],
) -> Dict[str, Any]:
    retrieval = runner(image_path)
    poem = retrieval["poem"]

    judgment = call_anthropic_judge(
        image_path=image_path,
        image_meta=image_meta,
        model_name=model_name,
        retrieval_extra=retrieval.get("retrieval_extra", ""),
        poem=poem,
    )

    return {
        "image_id": image_meta.get("id"),
        "filename": image_meta.get("filename"),
        "image_description": image_meta.get("description"),
        "expected_moods": ", ".join(image_meta.get("expected_moods", [])),
        "scene_type": image_meta.get("scene_type"),
        "model_name": model_name,
        "poem_id": poem.get("poem_id"),
        "poem_title": poem.get("title"),
        "poem_author": poem.get("author"),
        "retrieval_score": retrieval.get("retrieval_score"),
        "retrieval_extra": retrieval.get("retrieval_extra", ""),
        "semantic_fit": judgment["semantic_fit"],
        "emotional_fit": judgment["emotional_fit"],
        "overall_resonance": judgment["overall_resonance"],
        "rationale": judgment["rationale"],
    }


def write_json(path: Path, rows: Any) -> None:
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    summary = (
        df.groupby("model_name")
        .agg(
            n_pairs=("overall_resonance", "count"),
            semantic_fit_mean=("semantic_fit", "mean"),
            emotional_fit_mean=("emotional_fit", "mean"),
            overall_resonance_mean=("overall_resonance", "mean"),
            semantic_fit_std=("semantic_fit", "std"),
            emotional_fit_std=("emotional_fit", "std"),
            overall_resonance_std=("overall_resonance", "std"),
        )
        .reset_index()
        .sort_values("overall_resonance_mean", ascending=False)
    )

    for col in summary.columns:
        if col.endswith("_mean") or col.endswith("_std"):
            summary[col] = summary[col].round(3)

    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    metadata = load_metadata()
    rows: List[Dict[str, Any]] = []

    total = len(metadata) * len(MODEL_RUNNERS)
    completed = 0

    print(f"Evaluating {len(metadata)} images x {len(MODEL_RUNNERS)} models = {total} image-poem pairs")
    print(f"Output directory: {OUTPUT_DIR}")

    for image_meta in metadata:
        image_path = TEST_IMAGES_DIR / image_meta["filename"]
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image file: {image_path}")

        print(f"\nImage {image_meta['id']}: {image_meta['filename']}")

        for model_name, runner in MODEL_RUNNERS:
            completed += 1
            print(f"  [{completed}/{total}] {model_name}...", end="", flush=True)

            try:
                row = score_one_pair(image_meta, image_path, model_name, runner)
                rows.append(row)

                # Save after every pair so partial progress is not lost.
                write_json(OUTPUT_DIR / "judgments_detailed.json", rows)
                write_csv(OUTPUT_DIR / "judgments_detailed.csv", rows)

                print(
                    f" sem={row['semantic_fit']} emo={row['emotional_fit']} "
                    f"overall={row['overall_resonance']}"
                )

            except Exception as e:
                print(f" ERROR: {e}")
                rows.append({
                    "image_id": image_meta.get("id"),
                    "filename": image_meta.get("filename"),
                    "image_description": image_meta.get("description"),
                    "expected_moods": ", ".join(image_meta.get("expected_moods", [])),
                    "scene_type": image_meta.get("scene_type"),
                    "model_name": model_name,
                    "poem_id": None,
                    "poem_title": None,
                    "poem_author": None,
                    "retrieval_score": None,
                    "retrieval_extra": "",
                    "semantic_fit": None,
                    "emotional_fit": None,
                    "overall_resonance": None,
                    "rationale": f"ERROR: {e}",
                })
                write_json(OUTPUT_DIR / "judgments_detailed.json", rows)
                write_csv(OUTPUT_DIR / "judgments_detailed.csv", rows)

            time.sleep(REQUEST_SLEEP_SECONDS)

    valid_rows = [
        r for r in rows
        if r.get("semantic_fit") is not None
        and r.get("emotional_fit") is not None
        and r.get("overall_resonance") is not None
    ]

    if valid_rows:
        summary_df = summarize(valid_rows)
        summary_df.to_csv(OUTPUT_DIR / "model_summary.csv", index=False)
        write_json(OUTPUT_DIR / "model_summary.json", summary_df.to_dict(orient="records"))

        print("\nModel summary:")
        print(summary_df.to_string(index=False))
    else:
        print("\nNo valid judgments were produced.")


if __name__ == "__main__":
    main()
