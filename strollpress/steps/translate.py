"""Step 04 — Claude Sonnet SRT translation: Korean → English + Japanese."""

from __future__ import annotations

import json
import re
from pathlib import Path

import anthropic

from strollpress.config import PipelineConfig, PROJECT_CONTEXT
from strollpress.utils.logging import get_logger
from strollpress.utils.sentinel import is_done, mark_done, clear

STEP_NAME = "translate"
STEP_NUM = 4

# Anthropic pricing (per million tokens, as of mid-2025 — update as needed)
SONNET_PRICE_IN = 3.0
SONNET_PRICE_OUT = 15.0
MODEL = "claude-sonnet-4-6"


def _load_glossary(config: PipelineConfig) -> str:
    if config.glossary_path.exists():
        with open(config.glossary_path) as f:
            g = json.load(f)
        lines = [f"  - {k} → {v}" for k, v in g.items()]
        return "Locked glossary (never modify these terms):\n" + "\n".join(lines)
    return ""


def _build_translation_prompt(
    srt_ko: str,
    target_lang: str,
    glossary_note: str,
    clip_id: str,
) -> str:
    lang_instruction = {
        "en": (
            "Translate into natural, clear English. "
            "Preserve documentary register — avoid colloquialisms. "
            "Keep all timing lines exactly as-is."
        ),
        "ja": (
            "Translate into natural spoken Japanese. "
            "Use casual-formal register appropriate for documentary subtitles. "
            "Do NOT transliterate Korean particles. "
            "Preserve Jeju place names and ritual terms in Korean with furigana if helpful. "
            "Keep all timing lines exactly as-is."
        ),
    }.get(target_lang, f"Translate into {target_lang}.")

    return f"""{PROJECT_CONTEXT}

You are translating subtitles for clip: {clip_id}

{glossary_note}

Task: {lang_instruction}

Rules:
1. Output ONLY the translated SRT — no explanations, no markdown.
2. Preserve every subtitle index number and timing line unchanged.
3. Only translate the text lines.
4. Flag uncertain translations with a trailing [?] in that subtitle line.

--- SOURCE SRT (Korean) ---
{srt_ko}
--- END SOURCE ---"""


def translate_srt(
    srt_ko: str,
    target_lang: str,
    clip_id: str,
    client: anthropic.Anthropic,
    config: PipelineConfig,
    glossary_note: str,
) -> str:
    prompt = _build_translation_prompt(srt_ko, target_lang, glossary_note, clip_id)
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    config.track_cost(
        step=f"translate_{target_lang}",
        model=MODEL,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        unit_price_in=SONNET_PRICE_IN,
        unit_price_out=SONNET_PRICE_OUT,
    )
    return response.content[0].text.strip()


def run(config: PipelineConfig) -> list[dict]:
    logger = get_logger("translate", config.log_path)

    if not config.force and is_done(config.output_dir, STEP_NAME):
        logger.info("[step 04] translate: already done, skipping")
        return []

    if config.force:
        clear(config.output_dir, STEP_NAME)

    if not config.anthropic_api_key:
        logger.warning("[step 04] ANTHROPIC_API_KEY not set — skipping translation")
        mark_done(config.output_dir, STEP_NAME)
        return []
    client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    glossary_note = _load_glossary(config)
    target_langs = [lang for lang in config.languages if lang != "ko"]

    srt_files = sorted(config.subtitles_dir.glob("*.ko.srt"))
    if not srt_files:
        logger.info("[step 04] No Korean SRT files to translate")
        mark_done(config.output_dir, STEP_NAME)
        return []

    logger.info(
        "[step 04] Translating %d SRT(s) into %s via Claude Sonnet",
        len(srt_files),
        ", ".join(target_langs),
    )

    for srt_path in srt_files:
        clip_id = srt_path.name.replace(".ko.srt", "")
        srt_ko = srt_path.read_text(encoding="utf-8")

        for lang in target_langs:
            out_path = config.subtitles_dir / f"{clip_id}.{lang}.srt"
            if not config.force and out_path.exists():
                logger.info("  ↷ %s.%s.srt already exists, skipping", clip_id, lang)
                continue

            if config.dry_run:
                logger.info("  [dry-run] would translate %s → %s", clip_id, lang)
                continue

            try:
                translated = translate_srt(srt_ko, lang, clip_id, client, config, glossary_note)
                out_path.write_text(translated, encoding="utf-8")
                logger.info("  ✓ %s.%s.srt", clip_id, lang)
            except Exception as exc:
                logger.error("  ✗ %s → %s failed: %s", clip_id, lang, exc)

        # Update clip JSON with translation flags
        json_path = config.clips_dir / f"{clip_id}.json"
        if json_path.exists():
            with open(json_path) as f:
                clip_data = json.load(f)
            clip_data["translations_done"] = target_langs
            with open(json_path, "w") as f:
                json.dump(clip_data, f, ensure_ascii=False, indent=2)

    mark_done(config.output_dir, STEP_NAME)
    return []
