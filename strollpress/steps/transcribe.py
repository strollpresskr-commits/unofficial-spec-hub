"""Step 03 — Whisper transcription → Korean SRT with word-level timestamps."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from strollpress.config import PipelineConfig
from strollpress.utils.logging import get_logger
from strollpress.utils.sentinel import is_done, mark_done, clear

STEP_NAME = "transcribe"
STEP_NUM = 3


def _extract_audio(clip_path: Path, wav_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(clip_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            str(wav_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _segments_to_srt(segments: list) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        start = _fmt_time(seg.start)
        end = _fmt_time(seg.end)
        text = seg.text.strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _segments_to_word_timestamps(segments: list) -> list[dict]:
    words = []
    for seg in segments:
        for w in getattr(seg, "words", []):
            words.append({"word": w.word, "start": w.start, "end": w.end})
    return words


def transcribe_clip(clip_path: Path, config: PipelineConfig) -> dict:
    from faster_whisper import WhisperModel  # lazy import

    model = WhisperModel("large-v3", device="auto", compute_type="auto")

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "audio.wav"
        _extract_audio(clip_path, wav_path)

        segments, info = model.transcribe(
            str(wav_path),
            language="ko",
            word_timestamps=True,
            beam_size=5,
        )
        segments = list(segments)

    srt_content = _segments_to_srt(segments)
    word_timestamps = _segments_to_word_timestamps(segments)
    full_text = " ".join(seg.text.strip() for seg in segments)

    return {
        "srt_ko": srt_content,
        "word_timestamps": word_timestamps,
        "full_text_ko": full_text,
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "segments": [
            {
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "avg_logprob": getattr(seg, "avg_logprob", None),
                "no_speech_prob": getattr(seg, "no_speech_prob", None),
            }
            for seg in segments
        ],
    }


def run(config: PipelineConfig) -> list[dict]:
    logger = get_logger("transcribe", config.log_path)

    if not config.force and is_done(config.output_dir, STEP_NAME):
        logger.info("[step 03] transcribe: already done, skipping")
        return _load_existing(config)

    if config.force:
        clear(config.output_dir, STEP_NAME)

    clip_jsons = sorted(config.clips_dir.glob("*.json"))
    speech_clips = [
        p for p in clip_jsons
        if _load_json(p).get("speech_present") is True
    ]

    if not speech_clips:
        logger.info("[step 03] No speech clips to transcribe")
        mark_done(config.output_dir, STEP_NAME)
        return []

    logger.info("[step 03] Transcribing %d speech clip(s) with Whisper large-v3", len(speech_clips))
    results = []

    for json_path in speech_clips:
        clip_data = _load_json(json_path)

        if not config.force and clip_data.get("transcription"):
            results.append(clip_data)
            continue

        source_path = Path(clip_data.get("source_path", ""))
        if not source_path.exists():
            logger.warning("  ✗ %s — source not found", json_path.stem)
            continue

        try:
            transcript = transcribe_clip(source_path, config)
            clip_data["transcription"] = transcript

            # Write SRT file
            srt_path = config.subtitles_dir / f"{json_path.stem}.ko.srt"
            srt_path.write_text(transcript["srt_ko"], encoding="utf-8")

            with open(json_path, "w") as f:
                json.dump(clip_data, f, ensure_ascii=False, indent=2)

            results.append(clip_data)
            words = len(transcript["word_timestamps"])
            logger.info("  ✓ %s  words=%d", json_path.stem, words)
        except Exception as exc:
            logger.error("  ✗ %s — transcription failed: %s", json_path.stem, exc)
            results.append(clip_data)

    mark_done(config.output_dir, STEP_NAME)
    return results


def _load_json(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def _load_existing(config: PipelineConfig) -> list[dict]:
    results = []
    for p in sorted(config.clips_dir.glob("*.json")):
        d = _load_json(p)
        if d.get("transcription"):
            results.append(d)
    return results
