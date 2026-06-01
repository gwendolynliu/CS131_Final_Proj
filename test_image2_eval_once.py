"""
test_image2_eval_once.py

Minimal test script for one Anthropic LLM judge call.

What it does:
  1. Loads only Image 2: test_images/2_beach_empty.jpg
  2. Runs only Track 1: CLIP semantic retrieval
  3. Compresses the image until the ACTUAL base64 payload is under Anthropic's 5 MB limit
  4. Calls Anthropic once
  5. Writes the result to test_image2_eval_result.json and test_image2_eval_log.txt

Run from repo root:
  python test_image2_eval_once.py

Before running:
  - Fill in ANTHROPIC_API_KEY below, or set env var ANTHROPIC_API_KEY.
"""

import base64
import json
import mimetypes
import os
import re
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

from PIL import Image

from clip_semantic import retrieve_poems as clip_retrieve


# =========================
# Config
# =========================

ANTHROPIC_API_KEY = "FILL IN LATER"  # or set environment variable ANTHROPIC_API_KEY
ANTHROPIC_MODEL = "claude-haiku-4-5"

BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "test_images" / "2_beach_empty.jpg"
OUTPUT_JSON = BASE_DIR / "test_image2_eval_result.json"
OUTPUT_LOG = BASE_DIR / "test_image2_eval_log.txt"

# Anthropic rejects the BASE64 image payload when it is over 5 MB.
# Base64 is larger than raw JPEG bytes, so we validate len(base64_bytes).
ANTHROPIC_BASE64_MAX_BYTES = 5 * 1024 * 1024
IMAGE_TARGET_BASE64_BYTES = ANTHROPIC_BASE64_MAX_BYTES - 150_000

IMAGE_INITIAL_QUALITY = 95
IMAGE_MIN_QUALITY = 70
IMAGE_MIN_LONG_EDGE = 1200
MAX_POEM_CHARS = 3500


# =========================
# Logging
# =========================

def reset_log() -> None:
    with open(OUTPUT_LOG, "w", encoding="utf-8") as f:
        f.write("Image 2 Single-Eval Test Log\n")
        f.write("=" * 80 + "\n\n")


def log(message: str = "") -> None:
    with open(OUTPUT_LOG, "a", encoding="utf-8") as f:
        f.write(str(message) + "\n")


# =========================
# Image compression
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
    This fixes the common bug where raw JPEG bytes are under 5 MB but base64 is not.
    """
    original_data = image_path.read_bytes()
    original_raw_size = len(original_data)
    original_b64_size = base64_size(original_data)

    log(f"Original image: {image_path}")
    log(f"Original raw bytes:    {original_raw_size:,}")
    log(f"Original base64 bytes: {original_b64_size:,}")
    log(f"Anthropic base64 cap:  {ANTHROPIC_BASE64_MAX_BYTES:,}")
    log(f"Target base64 bytes:   {IMAGE_TARGET_BASE64_BYTES:,}")
    log("")

    if original_b64_size <= IMAGE_TARGET_BASE64_BYTES:
        media_type, _ = mimetypes.guess_type(str(image_path))
        if media_type is None:
            media_type = "image/jpeg"

        log("Using original image because base64 payload already fits.")
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

            log(
                f"Try long_edge={current_long_edge}, quality={quality}: "
                f"raw={raw:,}, base64={b64:,}"
            )

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

            log(
                f"Fallback try long_edge={current_long_edge}, quality={IMAGE_MIN_QUALITY}: "
                f"raw={raw:,}, base64={b64:,}"
            )

            if b64 <= IMAGE_TARGET_BASE64_BYTES:
                best_data = data
                best_raw_size = raw
                best_b64_size = b64
                best_quality = IMAGE_MIN_QUALITY
                best_long_edge = current_long_edge
                break

            current_long_edge = int(current_long_edge * 0.85)

    if best_data is None:
        raise RuntimeError("Could not compress image below Anthropic base64 limit.")

    log("")
    log("Selected compressed image:")
    log(f"Sent raw bytes:    {best_raw_size:,}")
    log(f"Sent base64 bytes: {best_b64_size:,}")
    log(f"Quality:           {best_quality}")
    log(f"Long edge:         {best_long_edge}")
    log("")

    return make_anthropic_image_block(best_data, "image/jpeg")


# =========================
# Judge call
# =========================

def truncate_text(text: Any, max_chars: int = MAX_POEM_CHARS) -> str:
    text = "" if text is None else str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED]"


def build_prompt(poem: Dict[str, Any]) -> str:
    poem_text = truncate_text(poem.get("text", ""))

    return f"""
You are evaluating an image-to-poem retrieval system.

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

Image metadata:
- image_id: 2
- filename: 2_beach_empty.jpg
- category: Landscape
- description: Empty beach at dusk
- expected_moods: lonely, reflective, melancholic
- scene_type: dominant_subject

Retrieval model:
- model_name: track_1_clip_semantic

Retrieved poem:
Title: {poem.get("title", "")}
Author: {poem.get("author", "")}

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
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse judge response as JSON:\n{text}")
        parsed = json.loads(match.group(0))

    for key in ["semantic_fit", "emotional_fit", "overall_resonance"]:
        parsed[key] = int(parsed[key])
        if not 1 <= parsed[key] <= 5:
            raise ValueError(f"{key} out of range: {parsed[key]}")

    parsed["rationale"] = str(parsed.get("rationale", "")).strip()
    return parsed


def call_anthropic_once(poem: Dict[str, Any]) -> Dict[str, Any]:
    api_key = os.environ.get("ANTHROPIC_API_KEY") or ANTHROPIC_API_KEY
    if not api_key or api_key == "FILL IN LATER":
        raise RuntimeError(
            "Missing Anthropic API key. Fill in ANTHROPIC_API_KEY or set env var ANTHROPIC_API_KEY."
        )

    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 600,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    image_to_anthropic_block(IMAGE_PATH),
                    {"type": "text", "text": build_prompt(poem)},
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

    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            raw = response.read().decode("utf-8")
            data = json.loads(raw)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Anthropic HTTP error {e.code}: {body}") from e

    text = extract_text_from_anthropic_response(data)
    return parse_judge_json(text)


# =========================
# Main
# =========================

def main() -> None:
    reset_log()

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Missing image file: {IMAGE_PATH}")

    log("Running one-image, one-model Anthropic eval test.")
    log(f"Image: {IMAGE_PATH.name}")
    log("Model: track_1_clip_semantic")
    log("")

    results = clip_retrieve(str(IMAGE_PATH), top_k=1)
    if not results:
        raise RuntimeError("Track 1 returned no poems.")

    poem = results[0]
    log(f"Retrieved poem: \"{poem.get('title')}\" by {poem.get('author')}")
    log(f"Retrieval score: {poem.get('score')}")
    log("")

    judgment = call_anthropic_once(poem)

    output = {
        "image_id": 2,
        "filename": "2_beach_empty.jpg",
        "image_description": "Empty beach at dusk",
        "expected_moods": ["lonely", "reflective", "melancholic"],
        "model_name": "track_1_clip_semantic",
        "poem_id": poem.get("poem_id"),
        "poem_title": poem.get("title"),
        "poem_author": poem.get("author"),
        "retrieval_score": poem.get("score"),
        "semantic_fit": judgment["semantic_fit"],
        "emotional_fit": judgment["emotional_fit"],
        "overall_resonance": judgment["overall_resonance"],
        "rationale": judgment["rationale"],
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    log("Judge result:")
    log(json.dumps(output, indent=2))
    log("")
    log(f"Wrote result JSON to: {OUTPUT_JSON}")
    log("Done.")


if __name__ == "__main__":
    main()
