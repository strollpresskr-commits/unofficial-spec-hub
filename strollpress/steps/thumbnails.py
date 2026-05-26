"""Step 09 — Thumbnail candidate extraction + Pillow text-overlay variants."""

from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import anthropic
from PIL import Image, ImageDraw, ImageFont

from strollpress.config import PipelineConfig, PROJECT_CONTEXT
from strollpress.utils.logging import get_logger
from strollpress.utils.sentinel import is_done, mark_done, clear

STEP_NAME = "thumbnails"
STEP_NUM = 9

MODEL = "claude-sonnet-4-6"
SONNET_PRICE_IN = 3.0
SONNET_PRICE_OUT = 15.0

SAMPLE_INTERVAL_SECS = 2.0
TOP_N_CANDIDATES = 5
TOP_N_WITH_OVERLAY = 3


def _extract_frame(clip_path: Path, timestamp: float, out_path: Path) -> bool:
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(timestamp),
                "-i", str(clip_path),
                "-vframes", "1",
                "-q:v", "2",
                str(out_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return out_path.exists()
    except subprocess.CalledProcessError:
        return False


def _score_frame(img_path: Path) -> float:
    """Heuristic frame scoring: saturation, brightness balance, sharpness estimate."""
    try:
        img = Image.open(img_path).convert("RGB")
        img_small = img.resize((160, 90))
        pixels = list(img_small.getdata())
        n = len(pixels)

        # Saturation score
        from colorsys import rgb_to_hsv
        saturations = []
        brightnesses = []
        for r, g, b in pixels:
            h, s, v = rgb_to_hsv(r / 255, g / 255, b / 255)
            saturations.append(s)
            brightnesses.append(v)

        avg_sat = sum(saturations) / n
        avg_bright = sum(brightnesses) / n
        # Prefer 0.3–0.7 brightness range
        bright_score = 1.0 - abs(avg_bright - 0.5) * 2

        # Rule-of-thirds: check if bright region is off-center (simple proxy)
        # Split into 3x3 grid and check variance
        w, h = img_small.size
        cell_w, cell_h = w // 3, h // 3
        cell_means = []
        for row in range(3):
            for col in range(3):
                cell = img_small.crop((col*cell_w, row*cell_h, (col+1)*cell_w, (row+1)*cell_h))
                cell_pixels = list(cell.getdata())
                mean = sum(sum(p) for p in cell_pixels) / (len(cell_pixels) * 3 * 255)
                cell_means.append(mean)
        center = cell_means[4]
        off_center = sum(cell_means) - center
        thirds_score = min(off_center / (8 * 0.5), 1.0)

        total = (avg_sat * 0.4 + bright_score * 0.4 + thirds_score * 0.2)
        return round(total * 10, 2)
    except Exception:
        return 5.0


def _get_title_suggestions(
    chapters: list[dict],
    full_transcript: str,
    client: anthropic.Anthropic,
    config: PipelineConfig,
) -> list[str]:
    if chapters:
        titles = [ch.get("title_en", "") for ch in chapters[:3]]
        return titles if titles else ["Documentary Memory", "Place and Time", "Living Archive"]

    prompt = f"""{PROJECT_CONTEXT}

Based on this documentary transcript, suggest 3 short YouTube-style titles in English (max 60 chars each).
Return only a JSON array of 3 strings.

TRANSCRIPT (excerpt):
{full_transcript[:2000]}"""

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        config.track_cost(
            step="thumbnail_titles",
            model=MODEL,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            unit_price_in=SONNET_PRICE_IN,
            unit_price_out=SONNET_PRICE_OUT,
        )
        text = response.content[0].text.strip()
        if text.startswith("["):
            return json.loads(text)
    except Exception:
        pass
    return ["제주 다큐멘터리", "Jeju Documentary", "Archive Film"]


def _draw_overlay(
    src_path: Path,
    out_path: Path,
    title_text: Optional[str],
    variant: str,
) -> None:
    """Draw text overlay onto a frame image (variant: ko | en | minimal)."""
    img = Image.open(src_path).convert("RGB")
    if title_text and variant != "minimal":
        draw = ImageDraw.Draw(img)
        w, h = img.size

        # Try to load a bundled font, fall back to default
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/NanumGothic.ttf", size=max(36, h // 18))
        except Exception:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", size=max(36, h // 18))
            except Exception:
                font = ImageFont.load_default()

        # Semi-transparent bottom bar
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        bar = ImageDraw.Draw(overlay)
        bar_h = h // 5
        bar.rectangle([(0, h - bar_h), (w, h)], fill=(0, 0, 0, 160))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

        draw = ImageDraw.Draw(img)
        draw.text(
            (w // 2, h - bar_h // 2),
            title_text,
            font=font,
            fill=(255, 255, 255),
            anchor="mm",
            stroke_fill=(0, 0, 0),
            stroke_width=2,
        )
    img.save(str(out_path), "PNG")


def run(config: PipelineConfig) -> list[Path]:
    logger = get_logger("thumbnails", config.log_path)

    if not config.force and is_done(config.output_dir, STEP_NAME):
        logger.info("[step 09] thumbnails: already done, skipping")
        return list(config.thumbnails_dir.glob("candidate_*.jpg"))

    if config.force:
        clear(config.output_dir, STEP_NAME)

    clips = []
    for p in sorted(config.clips_dir.glob("*.json")):
        with open(p) as f:
            clips.append(json.load(f))

    eligible = [c for c in clips if c.get("usability_score", 0) >= 7]
    if not eligible:
        eligible = [c for c in clips if c.get("usability_score", 0) >= 5]

    if not eligible:
        logger.warning("[step 09] No eligible clips for thumbnails")
        mark_done(config.output_dir, STEP_NAME)
        return []

    logger.info("[step 09] Extracting thumbnail candidates from %d eligible clip(s)", len(eligible))

    candidates: list[tuple[float, Path, dict]] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for clip in eligible:
            sp = Path(clip.get("renamed_path") or clip.get("source_path", ""))
            if not sp.exists():
                continue
            dur = clip.get("duration", 0.0)
            ts = SAMPLE_INTERVAL_SECS
            frame_idx = 0
            while ts < dur:
                frame_path = tmp_dir / f"{clip['clip_id']}_{frame_idx:04d}.jpg"
                if _extract_frame(sp, ts, frame_path):
                    score = _score_frame(frame_path)
                    usability_boost = clip.get("usability_score", 5) * 0.5
                    combined = score + usability_boost
                    candidates.append((combined, frame_path, {"clip_id": clip["clip_id"], "ts": ts}))
                ts += SAMPLE_INTERVAL_SECS
                frame_idx += 1

        # Sort by score descending, take top 5
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = candidates[:TOP_N_CANDIDATES]

        # Copy best frames to output
        output_paths = []
        for n, (score, frame_path, meta) in enumerate(top_candidates, 1):
            out_jpg = config.thumbnails_dir / f"candidate_{n}.jpg"
            img = Image.open(frame_path)
            img.save(str(out_jpg), "JPEG", quality=95)
            output_paths.append(out_jpg)
            logger.info(
                "  candidate_%d  clip=%s  ts=%.1fs  score=%.2f",
                n, meta["clip_id"], meta["ts"], score,
            )

        # Generate title overlays for top 3
        if not config.dry_run and config.anthropic_api_key:
            client = anthropic.Anthropic(api_key=config.anthropic_api_key)
            chapters_path = config.output_dir / "chapters.json"
            chapters = []
            if chapters_path.exists():
                with open(chapters_path) as f:
                    chapters = json.load(f)

            # Collect full transcript
            full_transcript = ""
            for clip in clips:
                if clip.get("transcription"):
                    full_transcript += clip["transcription"].get("full_text_ko", "") + " "

            titles = _get_title_suggestions(chapters, full_transcript.strip(), client, config)
            # Ensure 3 titles
            while len(titles) < 3:
                titles.append("")

            ko_title = next((ch.get("title_ko", titles[0]) for ch in chapters[:1]), titles[0])
            en_title = titles[1] if len(titles) > 1 else titles[0]

            overlay_variants = [
                ("ko", ko_title),
                ("en", en_title),
                ("minimal", None),
            ]

            for n, (score, frame_path, meta) in enumerate(top_candidates[:TOP_N_WITH_OVERLAY], 1):
                src_jpg = config.thumbnails_dir / f"candidate_{n}.jpg"
                for v_name, v_text in overlay_variants:
                    out_png = config.thumbnails_dir / f"candidate_{n}_overlay_{v_name}.png"
                    try:
                        _draw_overlay(src_jpg, out_png, v_text, v_name)
                        logger.info("  ✓ candidate_%d_overlay_%s.png", n, v_name)
                    except Exception as exc:
                        logger.error("  ✗ overlay %d/%s failed: %s", n, v_name, exc)

    mark_done(config.output_dir, STEP_NAME)
    return output_paths
