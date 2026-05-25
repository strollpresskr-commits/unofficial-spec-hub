"""Step 06 — Claude Opus utterance ↔ b-roll semantic matching."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import anthropic

from strollpress.config import PipelineConfig, PROJECT_CONTEXT
from strollpress.utils.logging import get_logger
from strollpress.utils.sentinel import is_done, mark_done, clear

STEP_NAME = "match"
STEP_NUM = 6

OPUS_PRICE_IN = 15.0
OPUS_PRICE_OUT = 75.0
SONNET_PRICE_IN = 3.0
SONNET_PRICE_OUT = 15.0

OPUS_MODEL = "claude-opus-4-7"
SONNET_MODEL = "claude-sonnet-4-6"


def _parse_utterances(segments: list[dict], word_timestamps: list[dict]) -> list[dict]:
    """Group Whisper segments into ~5–15s utterance units at pause boundaries."""
    utterances = []
    current: list[dict] = []
    current_start: Optional[float] = None

    for seg in segments:
        if current_start is None:
            current_start = seg["start"]
        current.append(seg)
        duration = seg["end"] - current_start
        # Natural boundary: end of segment + >0.4s gap, or duration ≥ 12s
        is_long = duration >= 12.0
        next_idx = segments.index(seg) + 1 if seg in segments else -1
        # Check for pause gap to next segment
        has_pause = True
        if next_idx < len(segments):
            gap = segments[next_idx]["start"] - seg["end"]
            has_pause = gap > 0.35

        if (is_long or has_pause) and duration >= 3.0:
            utterances.append(
                {
                    "start": current_start,
                    "end": seg["end"],
                    "text_ko": " ".join(s["text"] for s in current),
                    "segments": [s["id"] for s in current],
                }
            )
            current = []
            current_start = None

    if current and current_start is not None:
        last = current[-1]
        utterances.append(
            {
                "start": current_start,
                "end": last["end"],
                "text_ko": " ".join(s["text"] for s in current),
                "segments": [s["id"] for s in current],
            }
        )

    return utterances


def _enrich_utterance_keywords(
    utterances: list[dict],
    clip_id: str,
    client: anthropic.Anthropic,
    config: PipelineConfig,
    model: str,
) -> list[dict]:
    """Extract topical keywords + emotional tone per utterance via Claude."""
    utterances_text = "\n".join(
        f"[{i}] ({u['start']:.1f}s–{u['end']:.1f}s) {u['text_ko']}"
        for i, u in enumerate(utterances)
    )
    prompt = f"""{PROJECT_CONTEXT}

For interview clip: {clip_id}

Below are utterance units from a Korean documentary interview.
For EACH utterance, extract:
- keywords: 3–5 topical keywords in English (concepts, not words)
- tone: one of: reflective / nostalgic / instructional / emotional / contemplative / explanatory / urgent

Return JSON array, same order as input:
[{{"keywords": [...], "tone": "..."}}, ...]

Utterances:
{utterances_text}"""

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    config.track_cost(
        step="match_keywords",
        model=model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        unit_price_in=SONNET_PRICE_IN,
        unit_price_out=SONNET_PRICE_OUT,
    )
    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = re.sub(r"```\w*\n?", "", text).strip().rstrip("`").strip()

    enriched_raw = json.loads(text)
    for i, u in enumerate(utterances):
        if i < len(enriched_raw):
            u["keywords"] = enriched_raw[i].get("keywords", [])
            u["tone"] = enriched_raw[i].get("tone", "reflective")
    return utterances


def _score_broll(
    utterances: list[dict],
    broll_clips: list[dict],
    interview_clip_id: str,
    client: anthropic.Anthropic,
    config: PipelineConfig,
    model: str,
) -> list[dict]:
    """Score every b-roll against every utterance, return top 3 per utterance."""
    broll_summary = "\n".join(
        f"[{clip['clip_id']}] class={clip.get('classification','')} "
        f"shot={clip.get('visual_analysis',{}).get('shot_size','')} "
        f"movement={clip.get('visual_analysis',{}).get('camera_movement','')} "
        f"subjects={clip.get('visual_analysis',{}).get('subjects',[])} "
        f"mood={clip.get('visual_analysis',{}).get('color_mood_keywords',[])} "
        f"objects={clip.get('visual_analysis',{}).get('dominant_objects',[])} "
        f"usability={clip.get('usability_score',5)}"
        for clip in broll_clips
    )

    utterances_summary = "\n".join(
        f"[{i}] {u['start']:.1f}–{u['end']:.1f}s | "
        f"keywords={u.get('keywords',[])} tone={u.get('tone','')} | {u['text_ko'][:80]}"
        for i, u in enumerate(utterances)
    )

    prompt = f"""{PROJECT_CONTEXT}

Interview clip: {interview_clip_id}

You are a documentary editor. Match b-roll clips to interview utterances.

UTTERANCES (indexed):
{utterances_summary}

B-ROLL CLIPS (indexed by clip_id):
{broll_summary}

For EACH utterance, pick the top 3 b-roll candidates.
Score 0.0–1.0 based on: thematic match, mood match, visual rhythm fit.
Penalize reusing the same clip in back-to-back utterances.
Give a one-sentence rationale per candidate.

Return JSON:
{{
  "matches": [
    {{
      "utterance_idx": 0,
      "candidates": [
        {{"clip_id": "...", "score": 0.87, "rationale": "..."}},
        {{"clip_id": "...", "score": 0.71, "rationale": "..."}},
        {{"clip_id": "...", "score": 0.64, "rationale": "..."}}
      ]
    }},
    ...
  ]
}}

Only return valid JSON."""

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    config.track_cost(
        step="match_broll",
        model=model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        unit_price_in=OPUS_PRICE_IN if "opus" in model else SONNET_PRICE_IN,
        unit_price_out=OPUS_PRICE_OUT if "opus" in model else SONNET_PRICE_OUT,
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = re.sub(r"```\w*\n?", "", text).strip().rstrip("`").strip()
    raw = json.loads(text)

    # Inject candidates back into utterances
    for match in raw.get("matches", []):
        idx = match["utterance_idx"]
        if idx < len(utterances):
            utterances[idx]["b_roll_candidates"] = match["candidates"]

    return utterances


def run(config: PipelineConfig) -> list[dict]:
    logger = get_logger("match", config.log_path)

    if not config.force and is_done(config.output_dir, STEP_NAME):
        logger.info("[step 06] match: already done, skipping")
        return []

    if config.force:
        clear(config.output_dir, STEP_NAME)

    model = OPUS_MODEL if config.use_opus_matcher else SONNET_MODEL
    client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    # Load all clip data
    all_clips = []
    for p in sorted(config.clips_dir.glob("*.json")):
        with open(p) as f:
            all_clips.append(json.load(f))

    interview_clips = [c for c in all_clips if c.get("classification") == "interview"]
    broll_clips = [
        c for c in all_clips
        if c.get("classification", "").startswith("b_roll")
        and c.get("usability_score", 0) >= 4
    ]

    if not interview_clips:
        logger.info("[step 06] No interview clips found — skipping matching")
        mark_done(config.output_dir, STEP_NAME)
        return []

    if not broll_clips:
        logger.warning("[step 06] No usable b-roll clips found")
        mark_done(config.output_dir, STEP_NAME)
        return []

    logger.info(
        "[step 06] Matching %d interview(s) × %d b-roll clips via %s",
        len(interview_clips),
        len(broll_clips),
        model,
    )

    results = []
    for interview in interview_clips:
        clip_id = interview["clip_id"]
        transcription = interview.get("transcription", {})
        segments = transcription.get("segments", [])
        word_timestamps = transcription.get("word_timestamps", [])

        if not segments:
            logger.warning("  ✗ %s — no transcript segments, skipping", clip_id)
            continue

        if config.dry_run:
            logger.info("  [dry-run] would match %s", clip_id)
            continue

        try:
            utterances = _parse_utterances(segments, word_timestamps)
            logger.info("  %s: %d utterances", clip_id, len(utterances))

            # Enrich with keywords (use Sonnet for this — cheaper)
            utterances = _enrich_utterance_keywords(
                utterances, clip_id, client, config, SONNET_MODEL
            )

            # Get translation text for utterances if available
            en_srt_path = config.subtitles_dir / f"{clip_id}.en.srt"
            ja_srt_path = config.subtitles_dir / f"{clip_id}.ja.srt"
            # Basic text injection (timing lookup would be complex, skip for now)

            # Score b-roll (use selected model)
            utterances = _score_broll(
                utterances, broll_clips, clip_id, client, config, model
            )

            match_result = {
                "interview_clip_id": clip_id,
                "utterances": utterances,
            }
            out_path = config.matching_dir / f"{clip_id}.json"
            with open(out_path, "w") as f:
                json.dump(match_result, f, ensure_ascii=False, indent=2)

            results.append(match_result)
            logger.info("  ✓ %s — matching complete", clip_id)
        except Exception as exc:
            logger.error("  ✗ %s — matching failed: %s", clip_id, exc)

    mark_done(config.output_dir, STEP_NAME)
    return results
