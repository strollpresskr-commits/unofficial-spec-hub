"""Step 01 — ffprobe technical metadata extraction."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from strollpress.config import PipelineConfig, VIDEO_EXTENSIONS
from strollpress.utils.logging import get_logger
from strollpress.utils.sentinel import is_done, mark_done, clear

STEP_NAME = "probe"
STEP_NUM = 1


@dataclass
class ProbeResult:
    clip_id: str
    source_path: Path
    codec_video: str
    codec_audio: str
    resolution: str
    width: int
    height: int
    fps: float
    duration: float
    timecode: str
    audio_channels: int
    audio_sample_rate: int
    bit_rate: int
    gps: Optional[dict]
    raw: dict

    def to_dict(self) -> dict:
        return {
            "clip_id": self.clip_id,
            "source_path": str(self.source_path),
            "codec_video": self.codec_video,
            "codec_audio": self.codec_audio,
            "resolution": self.resolution,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "duration": self.duration,
            "timecode": self.timecode,
            "audio_channels": self.audio_channels,
            "audio_sample_rate": self.audio_sample_rate,
            "bit_rate": self.bit_rate,
            "gps": self.gps,
        }


def discover_clips(config: PipelineConfig) -> list[Path]:
    raw_dir = config.raw_dir
    if not raw_dir.exists():
        raw_dir = config.input_dir
    clips = sorted(
        p for p in raw_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if config.max_clips:
        clips = clips[: config.max_clips]
    return clips


def clip_id_from_path(p: Path) -> str:
    return re.sub(r"[^\w]", "_", p.stem)


def probe_clip(clip_path: Path) -> ProbeResult:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(clip_path),
    ]
    raw = json.loads(subprocess.check_output(cmd, stderr=subprocess.DEVNULL))

    video_stream = next(
        (s for s in raw.get("streams", []) if s.get("codec_type") == "video"), {}
    )
    audio_stream = next(
        (s for s in raw.get("streams", []) if s.get("codec_type") == "audio"), {}
    )
    fmt = raw.get("format", {})

    def fps_from_str(s: str) -> float:
        if "/" in s:
            a, b = s.split("/")
            return float(a) / float(b) if float(b) else 0.0
        return float(s) if s else 0.0

    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    fps = fps_from_str(video_stream.get("r_frame_rate", "0/1"))
    duration = float(fmt.get("duration") or video_stream.get("duration", 0))
    bit_rate = int(fmt.get("bit_rate", 0))

    # Extract GPS from format tags (common in drone footage)
    tags = fmt.get("tags", {})
    gps = _extract_gps(tags)

    return ProbeResult(
        clip_id=clip_id_from_path(clip_path),
        source_path=clip_path,
        codec_video=video_stream.get("codec_name", "unknown"),
        codec_audio=audio_stream.get("codec_name", "unknown"),
        resolution=f"{width}x{height}",
        width=width,
        height=height,
        fps=round(fps, 3),
        duration=round(duration, 3),
        timecode=video_stream.get("tags", {}).get("timecode", ""),
        audio_channels=int(audio_stream.get("channels", 0)),
        audio_sample_rate=int(audio_stream.get("sample_rate", 0)),
        bit_rate=bit_rate,
        gps=gps,
        raw=raw,
    )


def _extract_gps(tags: dict) -> Optional[dict]:
    lat_str = tags.get("location") or tags.get("com.apple.quicktime.location.ISO6709")
    if not lat_str:
        return None
    # ISO 6709 format: +33.4500+126.5700/
    m = re.match(r"([+-]\d+\.\d+)([+-]\d+\.\d+)", lat_str)
    if m:
        return {"lat": float(m.group(1)), "lon": float(m.group(2))}
    return {"raw": lat_str}


def run(config: PipelineConfig) -> list[dict]:
    logger = get_logger("probe", config.log_path)

    step_sentinel = STEP_NAME
    if not config.force and is_done(config.output_dir, step_sentinel):
        logger.info("[step 01] probe: already done, skipping (use --force to re-run)")
        return _load_existing(config)

    if config.force:
        clear(config.output_dir, step_sentinel)

    clips = discover_clips(config)
    if not clips:
        logger.warning("[step 01] No video files found in %s", config.raw_dir)
        mark_done(config.output_dir, step_sentinel)
        return []

    logger.info("[step 01] Probing %d clip(s)", len(clips))
    results = []

    for clip_path in clips:
        try:
            result = probe_clip(clip_path)
            out_path = config.clips_dir / f"{result.clip_id}.json"
            # Merge into existing JSON if present (other steps may have written there)
            existing: dict[str, Any] = {}
            if out_path.exists():
                with open(out_path) as f:
                    existing = json.load(f)
            existing.update(result.to_dict())
            with open(out_path, "w") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
            results.append(existing)
            logger.info(
                "  ✓ %s  %s  %.1fs  %.1ffps",
                result.clip_id,
                result.resolution,
                result.duration,
                result.fps,
            )
        except Exception as exc:
            logger.error("  ✗ %s — probe failed: %s", clip_path.name, exc)

    mark_done(config.output_dir, step_sentinel)
    return results


def _load_existing(config: PipelineConfig) -> list[dict]:
    results = []
    for p in sorted(config.clips_dir.glob("*.json")):
        with open(p) as f:
            results.append(json.load(f))
    return results
