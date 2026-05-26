"""Step 07 — Claude Sonnet chapter / topic-shift detection."""

from __future__ import annotations

import json
import re
from pathlib import Path

import anthropic

from strollpress.config import PipelineConfig, PROJECT_CONTEXT
from strollpress.utils.logging import get_logger
from strollpress.utils.sentinel import is_done, mark_done, clear

STEP_NAME = "chapters"
STEP_NUM = 7

MODEL = "claude-sonnet-4-6"
SONNET_PRICE_IN = 3.0
SONNET_PRICE_OUT = 15.0


def _fmt_timecode_youtube(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _load_all_transcripts(config: PipelineConfig) -> tuple[str, list[dict]]:
    """Build a combined transcript ordered by interview clip order."""
    order = _load_order(config)
    all_clips = []
    for p in sorted(config.clips_dir.glob("*.json")):
        with open(p) as f:
            d = json.load(f)
        if d.get("classification") == "interview" and d.get("transcription"):
            all_clips.append(d)

    if order:
        order_map = {name: i for i, name in enumerate(order)}
        all_clips.sort(
            key=lambda c: order_map.get(Path(c["source_path"]).name, 9999)
        )

    segments_flat = []
    offset = 0.0
    lines = []

    for clip in all_clips:
        for seg in clip["transcription"].get("segments", []):
            abs_start = seg["start"] + offset
            abs_end = seg["end"] + offset
            segments_flat.append(
                {"start": abs_start, "end": abs_end, "text": seg["text"], "clip_id": clip["clip_id"]}
            )
            lines.append(f"[{abs_start:.1f}s] {seg['text']}")
        # Accumulate duration as offset for next clip
        offset += clip.get("duration", 0.0)

    return "\n".join(lines), segments_flat


def _load_order(config: PipelineConfig) -> list[str]:
    if config.order_path.exists():
        return [
            line.strip()
            for line in config.order_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return []


def detect_chapters(
    transcript_text: str,
    segments: list[dict],
    client: anthropic.Anthropic,
    config: PipelineConfig,
) -> list[dict]:
    prompt = f"""{PROJECT_CONTEXT}

You are analyzing a Jeju documentary interview transcript to detect topic-shift chapter boundaries.

Rules:
- Detect 4–8 chapters for a 10–30 minute edit (fewer if the material is shorter).
- A chapter boundary is where the conversation meaningfully shifts to a new subject.
- The first chapter always starts at 00:00.
- Return a JSON array, each item: {{"start_seconds": float, "title_ko": "...", "title_en": "..."}}
- title_ko: Korean chapter title (2–5 words), title_en: English chapter title (2–5 words)
- Only return valid JSON — no markdown, no commentary.

TRANSCRIPT:
{transcript_text}"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    config.track_cost(
        step="chapters",
        model=MODEL,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        unit_price_in=SONNET_PRICE_IN,
        unit_price_out=SONNET_PRICE_OUT,
    )
    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = re.sub(r"```\w*\n?", "", text).strip().rstrip("`").strip()
    return json.loads(text)


def run(config: PipelineConfig) -> list[dict]:
    logger = get_logger("chapters", config.log_path)

    if not config.force and is_done(config.output_dir, STEP_NAME):
        logger.info("[step 07] chapters: already done, skipping")
        return _load_existing(config)

    if config.force:
        clear(config.output_dir, STEP_NAME)

    transcript_text, segments = _load_all_transcripts(config)
    if not transcript_text.strip():
        logger.info("[step 07] No transcript found — skipping chapter detection")
        mark_done(config.output_dir, STEP_NAME)
        return []

    if config.dry_run:
        logger.info("[step 07] [dry-run] would detect chapters")
        mark_done(config.output_dir, STEP_NAME)
        return []

    if not config.anthropic_api_key:
        logger.warning("[step 07] ANTHROPIC_API_KEY not set — skipping chapter detection")
        mark_done(config.output_dir, STEP_NAME)
        return []
    client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    try:
        chapters = detect_chapters(transcript_text, segments, client, config)

        # Write chapters.txt (YouTube format)
        chapters_txt = config.output_dir / "chapters.txt"
        lines = []
        for ch in chapters:
            tc = _fmt_timecode_youtube(ch.get("start_seconds", 0))
            lines.append(f"{tc} {ch.get('title_en', 'Untitled')}")
        chapters_txt.write_text("\n".join(lines), encoding="utf-8")

        # Write chapters.json for use by FCPXML step
        chapters_json = config.output_dir / "chapters.json"
        with open(chapters_json, "w") as f:
            json.dump(chapters, f, ensure_ascii=False, indent=2)

        logger.info("[step 07] Detected %d chapters", len(chapters))
        for ch in chapters:
            logger.info("  %.1fs  %s", ch.get("start_seconds", 0), ch.get("title_en", ""))

        mark_done(config.output_dir, STEP_NAME)
        return chapters
    except Exception as exc:
        logger.error("[step 07] Chapter detection failed: %s", exc)
        mark_done(config.output_dir, STEP_NAME)
        return []


def _load_existing(config: PipelineConfig) -> list[dict]:
    p = config.output_dir / "chapters.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return []
