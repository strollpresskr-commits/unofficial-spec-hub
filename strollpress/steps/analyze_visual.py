"""Step 02 — Gemini Flash visual analysis per clip."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from strollpress.config import PipelineConfig, PROJECT_CONTEXT
from strollpress.utils.logging import get_logger
from strollpress.utils.sentinel import is_done, mark_done, clear

STEP_NAME = "analyze_visual"
STEP_NUM = 2

VISUAL_SCHEMA = """{
  "shot_size": "<wide|medium|cu|xcu|aerial>",
  "camera_movement": "<static|pan|tilt|handheld|dolly|rack>",
  "subjects": ["<list of subjects>"],
  "location_type": "<string>",
  "color_mood_keywords": ["<keyword>"],
  "dominant_objects": ["<object>"],
  "people_present": <true|false>,
  "speech_present": <true|false>,
  "usability_score": <0-10>,
  "stability_notes": "<string>",
  "exposure_notes": "<string>",
  "focus_notes": "<string>",
  "classification": "<interview|b_roll_landscape|b_roll_ritual|b_roll_people|b_roll_object|b_roll_motion|discard_candidate>"
}"""

VISUAL_PROMPT = f"""
{PROJECT_CONTEXT}

Analyze this video clip and return a JSON object matching this schema exactly:
{VISUAL_SCHEMA}

Guidelines:
- shot_size: dominant framing across the clip
- camera_movement: dominant movement; "handheld" if shaky/organic; "rack" if rack focus
- usability_score: composite 0–10 (10 = broadcast-ready, 0 = unusable)
- classification: choose the single best fit
- speech_present: true if any human speech is audible
- Only return valid JSON, no markdown fences.
"""


def _upload_video(clip_path: Path, client: Any) -> Any:
    """Upload video file to Gemini Files API and wait for processing."""
    import google.generativeai as genai  # lazy import
    uploaded = genai.upload_file(str(clip_path))
    # Poll until processing is complete (max 120s)
    for _ in range(60):
        file_info = genai.get_file(uploaded.name)
        if file_info.state.name == "ACTIVE":
            return file_info
        if file_info.state.name == "FAILED":
            raise RuntimeError(f"Gemini file processing failed for {clip_path.name}")
        time.sleep(2)
    raise TimeoutError(f"Gemini file processing timeout for {clip_path.name}")


def analyze_clip(clip_path: Path, config: PipelineConfig, model: Any) -> dict:
    import google.generativeai as genai  # lazy import
    video_file = _upload_video(clip_path, model)
    try:
        response = model.generate_content(
            [video_file, VISUAL_PROMPT],
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=1024,
            ),
        )
        text = response.text.strip()
        # Strip any accidental markdown fences
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    finally:
        # Clean up uploaded file to avoid storage costs
        try:
            genai.delete_file(video_file.name)
        except Exception:
            pass


def run(config: PipelineConfig) -> list[dict]:
    logger = get_logger("analyze_visual", config.log_path)

    if not config.force and is_done(config.output_dir, STEP_NAME):
        logger.info("[step 02] analyze_visual: already done, skipping")
        return _load_existing(config)

    if config.force:
        clear(config.output_dir, STEP_NAME)

    import google.generativeai as genai  # lazy import
    genai.configure(api_key=config.gemini_api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    clip_jsons = sorted(config.clips_dir.glob("*.json"))
    if not clip_jsons:
        logger.warning("[step 02] No clip JSONs found — run step 01 first")
        mark_done(config.output_dir, STEP_NAME)
        return []

    logger.info("[step 02] Visual analysis of %d clip(s) via Gemini Flash", len(clip_jsons))
    results = []

    for json_path in clip_jsons:
        with open(json_path) as f:
            clip_data: dict = json.load(f)

        # Skip if already analyzed
        if not config.force and clip_data.get("visual_analysis"):
            results.append(clip_data)
            continue

        source_path = Path(clip_data.get("source_path", ""))
        if not source_path.exists():
            logger.warning("  ✗ %s — source file not found: %s", json_path.stem, source_path)
            continue

        try:
            visual = analyze_clip(source_path, config, model)
            clip_data["visual_analysis"] = visual
            # Hoist key fields to top level for easy access by later steps
            clip_data["classification"] = visual.get("classification", "b_roll_landscape")
            clip_data["usability_score"] = visual.get("usability_score", 5)
            clip_data["speech_present"] = visual.get("speech_present", False)

            with open(json_path, "w") as f:
                json.dump(clip_data, f, ensure_ascii=False, indent=2)

            results.append(clip_data)
            logger.info(
                "  ✓ %s  class=%s  usability=%.1f  speech=%s",
                json_path.stem,
                visual.get("classification"),
                visual.get("usability_score", 0),
                visual.get("speech_present"),
            )
        except Exception as exc:
            logger.error("  ✗ %s — visual analysis failed: %s", json_path.stem, exc)
            results.append(clip_data)

    mark_done(config.output_dir, STEP_NAME)
    return results


def _load_existing(config: PipelineConfig) -> list[dict]:
    results = []
    for p in sorted(config.clips_dir.glob("*.json")):
        with open(p) as f:
            results.append(json.load(f))
    return results
