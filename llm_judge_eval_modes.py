"""
llm_judge_eval.py

Runs an Anthropic/Claude LLM judge over the top-1 retrieved poem from each
integration-level retrieval system in integrate.py:

  1. content mode
  2. mood mode
  3. balanced mode
  4. creative mode
  5. rerank retrieve

Important:
This evaluates the MODES from integrate.py, not the individual Track 1-4 models.

For each image-poem pair, the judge scores:
  - semantic_fit: 1-5
  - emotional_fit: 1-5
  - overall_resonance: 1-5

Outputs:
  - llm_judge_results_modes/evaluation_log.txt
  - llm_judge_results_modes/judgments_detailed.csv
  - llm_judge_results_modes/judgments_detailed.json
  - llm_judge_results_modes/model_summary.csv
  - llm_judge_results_modes/model_summary.json
  - llm_judge_results_modes/model_summary.txt

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
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image

from integrate import MODES, parallel_retrieve, rerank_retrieve


# =========================
# Config
# =========================

ANTHROPIC_API_KEY = "FILL IN LATER"  # or set environment variable ANTHROPIC_API_KEY
ANTHROPIC_MODEL = "claude-haiku-4-5"

BASE_DIR = Path(__file__).resolve().parent
TEST_IMAGES_DIR = BASE_DIR / "test_images"
METADATA_FILE = BASE_DIR / "test_images_metadata.json"
OUTPUT_DIR = BASE_DIR / "llm_judge_results_modes"
LOG_FILE = OUTPUT_DIR / "evaluation_log.txt"

TOP_K_PER_MODEL = 1
MAX_POEM_CHARS = 3500
REQUEST_SLEEP_SECONDS = 0.5
MAX_RETRIES = 3

# Anthropic rejects the BASE64 image payload when it is over 5 MB.
# Base64 is larger than raw JPEG bytes, so we validate len(base64_bytes).
ANTHROPIC_BASE64_MAX_BYTES = 5 * 1024 * 1024
IMAGE_TARGET_BASE64_BYTES = ANTHROPIC_BASE64_MAX_BYTES - 150_000

IMAGE_INITIAL_QUALITY = 95
IMAGE_MIN_QUALITY = 70
IMAGE_MIN_LONG_EDGE = 1200

# Set to None to evaluate every image in test_images_metadata.json.
# Or use a subset for debugging, e.g. IMAGE_IDS = [1, 2, 5]
IMAGE_IDS: Optional[List[int]] = None


# =========================
# File logging
# =========================

def reset_log() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("LLM Judge Evaluation Log\n")
        f.write("=" * 80 + "\n\n")


def log(message: str = "") -> None:
    """Write run progress to evaluation_log.txt instead of printing to terminal."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(str(message) + "\n")


# =========================
# Integration-mode retrieval wrappers
# =========================

def get_first_result(results: List[Dict[str, Any]], model_name: str, image_path: Path) -> Dict[str, Any]:
    if not results:
        raise RuntimeError(f"{model_name} returned no results for {image_path}")
    return results[0]


def run_parallel_mode(image_path: Path, mode: str) -> Dict[str, Any]:
    results = parallel_retrieve(str(image_path), mode=mode, top_k=TOP_K_PER_MODEL)
    poem = get_first_result(results, f"mode_{mode}", image_path)

    weights = MODES[mode]
    weights_str = (
        f"T1={weights[1]:.2f}, "
        f"T2={weights[2]:.2f}, "
        f"T3={weights[3]:.2f}, "
        f"T4={weights[4]:.2f}"
    )

    return {
        "model_name": f"mode_{mode}",
        "retrieval_score": poem.get("combined_score"),
        "retrieval_extra": f"parallel_retrieve mode='{mode}' with weights: {weights_str}",
        "poem": poem,
    }


def run_content_mode(image_path: Path) -> Dict[str, Any]:
    return run_parallel_mode(image_path, "content")


def run_mood_mode(image_path: Path) -> Dict[str, Any]:
    return run_parallel_mode(image_path, "mood")


def run_balanced_mode(image_path: Path) -> Dict[str, Any]:
    return run_parallel_mode(image_path, "balanced")


def run_creative_mode(image_path: Path) -> Dict[str, Any]:
    return run_parallel_mode(image_path, "creative")


def run_rerank(image_path: Path) -> Dict[str, Any]:
    results = rerank_retrieve(str(image_path), top_k=TOP_K_PER_MODEL)
    poem = get_first_result(results, "rerank_retrieve", image_path)
    return {
        "model_name": "rerank_retrieve",
        "retrieval_score": poem.get("combined_score"),
        "retrieval_extra": (
            "Two-stage retrieve-then-rerank: Track 1 and Track 2 generate content candidates; "
            "Track 3 and Track 4 rescore those candidates for mood/emotional fit."
        ),
        "poem": poem,
    }


# These are integration-level systems, not individual tracks.
MODEL_RUNNERS: List[Tuple[str, Callable[[Path], Dict[str, Any]]]] = [
    ("mode_content", run_content_mode),
    ("mode_mood", run_mood_mode),
    ("mode_balanced", run_balanced_mode),
    ("mode_creative", run_creative_mode),
    ("rerank_retrieve", run_rerank),
]


# =========================
# Anthropic image compression
# =========================

def base64_size(data: bytes) -> int:
    return len(base64.b64encode(data))


def make_anthropic_image_block(data: bytes, media_type: str) -> Dict[str, Any]:
    encoded = base64.b64encode(data).decode("utf-8")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": encoded,
        },
    }


def image_to_anthropic_block(image_path: Path) -> Dict[str, Any]:
    """
    Compress image only as much as needed so the actual base64 payload is under 5 MB.

    This is the same working compression logic from test_image2_eval_once.py.
    It fixes the bug where raw JPEG bytes were below 5 MB but the base64 string
    sent to Anthropic was still above 5 MB.
    """
    original_data = image_path.read_bytes()
    original_raw_size = len(original_data)
    original_b64_size = base64_size(original_data)

    if original_b64_size <= IMAGE_TARGET_BASE64_BYTES:
        media_type, _ = mimetypes.guess_type(str(image_path))
        if media_type is None:
            media_type = "image/jpeg"

        log(
            f"Using original image for Anthropic: {image_path.name} | "
            f"raw={original_raw_size:,} bytes | base64={original_b64_size:,} bytes"
        )
        return make_anthropic_image_block(original_data, media_type)

    image = Image.open(image_path).convert("RGB")
    original_width, original_height = image.size
    original_long_edge = max(original_width, original_height)

    best_data = None
    best_raw_size = 0
    best_b64_size = 0
    best_quality = None
    best_long_edge = None

    current_long_edge = original_long_edge

    while current_long_edge >= IMAGE_MIN_LONG_EDGE:
        working = image.copy()

        if current_long_edge < original_long_edge:
            scale = current_long_edge / original_long_edge
            new_size = (
                max(1, int(original_width * scale)),
                max(1, int(original_height * scale)),
            )
            working = working.resize(new_size, Image.LANCZOS)

        for quality in range(IMAGE_INITIAL_QUALITY, IMAGE_MIN_QUALITY - 1, -5):
            buffer = BytesIO()
            working.save(buffer, format="JPEG", quality=quality, optimize=True)
            data = buffer.getvalue()

            raw = len(data)
            b64 = base64_size(data)

            if b64 <= IMAGE_TARGET_BASE64_BYTES and b64 > best_b64_size:
                best_data = data
                best_raw_size = raw
                best_b64_size = b64
                best_quality = quality
                best_long_edge = current_long_edge

            if b64 <= IMAGE_TARGET_BASE64_BYTES:
                break

        if best_data is not None:
            break

        current_long_edge = int(current_long_edge * 0.9)

    if best_data is None:
        current_long_edge = IMAGE_MIN_LONG_EDGE

        while best_data is None and current_long_edge >= 400:
            scale = current_long_edge / original_long_edge
            new_size = (
                max(1, int(original_width * scale)),
                max(1, int(original_height * scale)),
            )
            working = image.resize(new_size, Image.LANCZOS)

            buffer = BytesIO()
            working.save(buffer, format="JPEG", quality=IMAGE_MIN_QUALITY, optimize=True)
            data = buffer.getvalue()

            raw = len(data)
            b64 = base64_size(data)

            if b64 <= IMAGE_TARGET_BASE64_BYTES:
                best_data = data
                best_raw_size = raw
                best_b64_size = b64
                best_quality = IMAGE_MIN_QUALITY
                best_long_edge = current_long_edge
                break

            current_long_edge = int(current_long_edge * 0.85)

    if best_data is None:
        raise RuntimeError(f"Could not compress image below Anthropic base64 limit: {image_path}")

    log(
        f"Compressed image for Anthropic: {image_path.name} | "
        f"original_raw={original_raw_size:,} bytes | "
        f"original_base64={original_b64_size:,} bytes | "
        f"sent_raw={best_raw_size:,} bytes | "
        f"sent_base64={best_b64_size:,} bytes | "
        f"quality={best_quality} | long_edge={best_long_edge}"
    )

    return make_anthropic_image_block(best_data, "image/jpeg")


# =========================
# Anthropic judge
# =========================

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

Retrieval system:
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
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
    reset_log()

    metadata = load_metadata()
    rows: List[Dict[str, Any]] = []

    total = len(metadata) * len(MODEL_RUNNERS)
    completed = 0

    log(f"Evaluating {len(metadata)} images x {len(MODEL_RUNNERS)} integration systems = {total} image-poem pairs")
    log("Systems: content mode, mood mode, balanced mode, creative mode, rerank retrieve")
    log(f"Output directory: {OUTPUT_DIR}")
    log(f"Detailed JSON: {OUTPUT_DIR / 'judgments_detailed.json'}")
    log(f"Detailed CSV:  {OUTPUT_DIR / 'judgments_detailed.csv'}")
    log("")

    for image_meta in metadata:
        image_path = TEST_IMAGES_DIR / image_meta["filename"]
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image file: {image_path}")

        log("=" * 80)
        log(f"Image {image_meta['id']}: {image_meta['filename']}")
        log(f"Description: {image_meta.get('description')}")
        log(f"Expected moods: {', '.join(image_meta.get('expected_moods', []))}")
        log("=" * 80)

        for model_name, runner in MODEL_RUNNERS:
            completed += 1

            try:
                row = score_one_pair(image_meta, image_path, model_name, runner)
                rows.append(row)

                # Save after every pair so partial progress is not lost.
                write_json(OUTPUT_DIR / "judgments_detailed.json", rows)
                write_csv(OUTPUT_DIR / "judgments_detailed.csv", rows)

                log(
                    f"[{completed}/{total}] {model_name} | "
                    f"poem=\"{row['poem_title']}\" by {row['poem_author']} | "
                    f"score={row['retrieval_score']} | "
                    f"sem={row['semantic_fit']} "
                    f"emo={row['emotional_fit']} "
                    f"overall={row['overall_resonance']}"
                )
                log(f"Rationale: {row['rationale']}")
                log("")

            except Exception as e:
                error_row = {
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
                }
                rows.append(error_row)
                write_json(OUTPUT_DIR / "judgments_detailed.json", rows)
                write_csv(OUTPUT_DIR / "judgments_detailed.csv", rows)

                log(f"[{completed}/{total}] {model_name} | ERROR: {e}")
                log("")

            time.sleep(REQUEST_SLEEP_SECONDS)

    valid_rows = [
        r for r in rows
        if r.get("semantic_fit") is not None
        and r.get("emotional_fit") is not None
        and r.get("overall_resonance") is not None
    ]

    log("=" * 80)
    log("Final Summary")
    log("=" * 80)

    if valid_rows:
        summary_df = summarize(valid_rows)

        summary_csv_path = OUTPUT_DIR / "model_summary.csv"
        summary_json_path = OUTPUT_DIR / "model_summary.json"
        summary_txt_path = OUTPUT_DIR / "model_summary.txt"

        summary_df.to_csv(summary_csv_path, index=False)
        write_json(summary_json_path, summary_df.to_dict(orient="records"))

        summary_text = summary_df.to_string(index=False)
        with open(summary_txt_path, "w", encoding="utf-8") as f:
            f.write(summary_text + "\n")

        log(summary_text)
        log("")
        log(f"Summary CSV:  {summary_csv_path}")
        log(f"Summary JSON: {summary_json_path}")
        log(f"Summary TXT:  {summary_txt_path}")
    else:
        log("No valid judgments were produced.")

    log("")
    log("Done.")


if __name__ == "__main__":
    main()
