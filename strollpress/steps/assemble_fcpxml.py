"""Step 08 — FCPXML 1.10 timeline assembly."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional
from lxml import etree

from strollpress.config import PipelineConfig
from strollpress.utils.logging import get_logger
from strollpress.utils.sentinel import is_done, mark_done, clear

STEP_NAME = "assemble_fcpxml"
STEP_NUM = 8

# FCPXML time is expressed as rational numbers: "numerator/denominator s"
# We use a 90000 timebase (standard for video)
TIMEBASE = 90000


def _secs_to_tc(seconds: float) -> str:
    """Convert seconds to FCPXML rational time string."""
    frames = round(seconds * TIMEBASE)
    return f"{frames}/{TIMEBASE}s"


def _dur_to_tc(duration: float) -> str:
    return _secs_to_tc(duration)


def _load_clips(config: PipelineConfig) -> list[dict]:
    clips = []
    for p in sorted(config.clips_dir.glob("*.json")):
        with open(p) as f:
            clips.append(json.load(f))
    return clips


def _load_chapters(config: PipelineConfig) -> list[dict]:
    p = config.output_dir / "chapters.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return []


def _load_matching(config: PipelineConfig, interview_clip_id: str) -> Optional[dict]:
    p = config.matching_dir / f"{interview_clip_id}.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def _load_order(config: PipelineConfig) -> list[str]:
    if config.order_path.exists():
        return [
            l.strip()
            for l in config.order_path.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
    return []


def _parse_srt_for_captions(srt_path: Path) -> list[dict]:
    if not srt_path.exists():
        return []
    blocks = []
    current: dict = {}
    for line in srt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.isdigit():
            current = {"index": int(line)}
        elif "-->" in line:
            parts = line.split("-->")
            current["start"] = _parse_srt_time(parts[0].strip())
            current["end"] = _parse_srt_time(parts[1].strip())
            current["text"] = ""
        elif line == "":
            if current.get("text"):
                blocks.append(current)
            current = {}
        elif "start" in current:
            current["text"] = (current.get("text", "") + " " + line).strip()
    if current.get("text"):
        blocks.append(current)
    return blocks


def _parse_srt_time(s: str) -> float:
    s = s.replace(",", ".")
    parts = s.split(":")
    h, m, rest = int(parts[0]), int(parts[1]), float(parts[2])
    return h * 3600 + m * 60 + rest


class FCPXMLBuilder:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = get_logger("assemble_fcpxml", config.log_path)
        self._format_id_counter = 1
        self._asset_id_counter = 1
        self._formats: dict[str, str] = {}  # key → format_id
        self._assets: dict[str, str] = {}   # source_path → asset_id

    def _make_format_key(self, clip: dict) -> str:
        return f"{clip.get('width',1920)}x{clip.get('height',1080)}@{clip.get('fps',25)}"

    def _get_or_create_format(self, resources: etree._Element, clip: dict) -> str:
        key = self._make_format_key(clip)
        if key in self._formats:
            return self._formats[key]
        fid = f"r{self._format_id_counter}"
        self._format_id_counter += 1
        w = clip.get("width", 1920)
        h = clip.get("height", 1080)
        fps = clip.get("fps", 25.0)
        # Determine frame duration
        common_fps = {23.976: "1001/24000s", 24.0: "1/24s", 25.0: "1/25s",
                      29.97: "1001/30000s", 30.0: "1/30s", 50.0: "1/50s",
                      59.94: "1001/60000s", 60.0: "1/60s"}
        frame_dur = common_fps.get(round(fps, 3), f"1/{int(fps)}s")
        etree.SubElement(
            resources, "format",
            id=fid, name=f"FFVideoFormat{h}p{int(fps)}",
            frameDuration=frame_dur,
            width=str(w), height=str(h),
        )
        self._formats[key] = fid
        return fid

    def _get_or_create_asset(self, resources: etree._Element, clip: dict) -> str:
        sp = clip.get("renamed_path") or clip.get("source_path", "")
        if sp in self._assets:
            return self._assets[sp]
        aid = f"a{self._asset_id_counter}"
        self._asset_id_counter += 1
        p = Path(sp)
        fid = self._get_or_create_format(resources, clip)
        has_audio = clip.get("audio_channels", 0) > 0
        duration_tc = _dur_to_tc(clip.get("duration", 0))
        asset = etree.SubElement(
            resources, "asset",
            id=aid, name=p.stem,
            uid=aid,
            src=f"file://{sp}",
            start="0s",
            duration=duration_tc,
            hasVideo="1",
            hasAudio="1" if has_audio else "0",
            format=fid,
        )
        if has_audio:
            etree.SubElement(asset, "media-rep", kind="original-media", src=f"file://{sp}")
        self._assets[sp] = aid
        return aid

    def build(self) -> etree._Element:
        clips = _load_clips(self.config)
        chapters = _load_chapters(self.config)
        order = _load_order(self.config)

        interview_clips = [c for c in clips if c.get("classification") == "interview"]
        broll_clips = {
            c["clip_id"]: c for c in clips
            if c.get("classification", "").startswith("b_roll")
        }

        if order:
            order_map = {name: i for i, name in enumerate(order)}
            interview_clips.sort(
                key=lambda c: order_map.get(Path(c.get("source_path", "")).name, 9999)
            )

        # ---- Root ----
        fcpxml = etree.Element("fcpxml", version="1.10")

        # ---- Resources ----
        resources = etree.SubElement(fcpxml, "resources")
        # Will be populated as we encounter clips

        # ---- Library ----
        library = etree.SubElement(fcpxml, "library")
        event = etree.SubElement(library, "event", name="Strollpress Rough Cut")
        project = etree.SubElement(event, "project", name="rough_cut")

        # Sequence
        sequence = etree.SubElement(
            project, "sequence",
            duration="0s",  # will update
            tcStart="0/1s",
            tcFormat="NDF",
            audioLayout="stereo",
            audioRate="48000",
        )
        spine = etree.SubElement(sequence, "spine")

        # ---- Build V1: interview clips on spine ----
        timeline_offset = 0.0
        interview_placements = []  # (clip, start_on_timeline, duration)

        for iv_clip in interview_clips:
            sp = iv_clip.get("renamed_path") or iv_clip.get("source_path", "")
            if not sp or not Path(sp).exists():
                self.logger.warning("  skip %s — file not found", iv_clip.get("clip_id"))
                continue

            aid = self._get_or_create_asset(resources, iv_clip)
            dur = iv_clip.get("duration", 0.0)

            clip_el = etree.SubElement(
                spine, "clip",
                name=iv_clip["clip_id"],
                offset=_secs_to_tc(timeline_offset),
                duration=_dur_to_tc(dur),
                start="0s",
                tcFormat="NDF",
            )
            etree.SubElement(clip_el, "asset-clip",
                             ref=aid,
                             offset="0s",
                             duration=_dur_to_tc(dur),
                             start="0s",
                             role="dialogue")

            # ---- Chapter markers on V1 ----
            for ch in chapters:
                ch_abs = ch.get("start_seconds", 0.0)
                ch_rel = ch_abs - timeline_offset
                if 0.0 <= ch_rel < dur:
                    m = etree.SubElement(
                        clip_el, "marker",
                        start=_secs_to_tc(ch_rel),
                        duration="1/25s",
                        value=ch.get("title_en", "Chapter"),
                        note=ch.get("title_ko", ""),
                    )

            # ---- Korean SRT captions ----
            srt_ko_path = self.config.subtitles_dir / f"{iv_clip['clip_id']}.ko.srt"
            captions_ko = _parse_srt_for_captions(srt_ko_path)
            if captions_ko:
                caption_lane = etree.SubElement(clip_el, "caption", lane="-1", role="captions.ITT:ko")
                for cap in captions_ko:
                    etree.SubElement(
                        caption_lane, "caption-annotation",
                        ref=aid,
                        offset=_secs_to_tc(cap["start"]),
                        duration=_secs_to_tc(cap["end"] - cap["start"]),
                    ).text = cap["text"]

            interview_placements.append((iv_clip, timeline_offset, dur))
            timeline_offset += dur

        total_duration = timeline_offset
        sequence.attrib["duration"] = _dur_to_tc(total_duration)

        # ---- Build V2/V3/V4: b-roll over interview ----
        for iv_clip, iv_offset, iv_dur in interview_placements:
            matching = _load_matching(self.config, iv_clip["clip_id"])
            if not matching:
                continue

            for utt in matching.get("utterances", []):
                utt_start = utt["start"] + iv_offset
                utt_end = utt["end"] + iv_offset
                utt_dur = utt_end - utt_start
                candidates = utt.get("b_roll_candidates", [])

                for lane_idx, cand in enumerate(candidates[:3]):
                    br_clip_id = cand["clip_id"]
                    br_clip = broll_clips.get(br_clip_id)
                    if not br_clip:
                        continue

                    br_sp = br_clip.get("renamed_path") or br_clip.get("source_path", "")
                    if not br_sp or not Path(br_sp).exists():
                        continue

                    br_aid = self._get_or_create_asset(resources, br_clip)
                    br_dur = min(utt_dur, br_clip.get("duration", utt_dur))
                    lane = str(lane_idx + 2)  # V2, V3, V4

                    # V2 is enabled, V3 and V4 are disabled (alt candidates)
                    enabled = "1" if lane_idx == 0 else "0"

                    br_el = etree.SubElement(
                        spine, "clip",
                        name=f"{br_clip_id}_lane{lane}",
                        offset=_secs_to_tc(utt_start),
                        duration=_dur_to_tc(br_dur),
                        start="0s",
                        lane=lane,
                        enabled=enabled,
                        tcFormat="NDF",
                    )
                    etree.SubElement(
                        br_el, "asset-clip",
                        ref=br_aid,
                        offset="0s",
                        duration=_dur_to_tc(br_dur),
                        start="0s",
                        role="B-Roll",
                    )
                    # Marker with rationale
                    etree.SubElement(
                        br_el, "marker",
                        start="0s",
                        duration="1/25s",
                        value=f"B-roll alt {lane_idx+1}: score={cand.get('score',0):.2f}",
                        note=cand.get("rationale", ""),
                    )

        # ---- A2: ambient room tone for b-roll-only sections ----
        ambient_clips = [
            c for c in clips
            if c.get("classification") in ("b_roll_landscape", "b_roll_ritual")
            and not c.get("speech_present")
            and c.get("audio_channels", 0) > 0
        ]
        if ambient_clips:
            amb = ambient_clips[0]
            amb_sp = amb.get("renamed_path") or amb.get("source_path", "")
            if amb_sp and Path(amb_sp).exists():
                amb_aid = self._get_or_create_asset(resources, amb)
                etree.SubElement(
                    spine, "clip",
                    name="ambient_room_tone",
                    offset="0s",
                    duration=_dur_to_tc(total_duration),
                    start="0s",
                    lane="-2",
                    enabled="1",
                    audioRole="dialogue",
                )

        return fcpxml


def run(config: PipelineConfig) -> None:
    logger = get_logger("assemble_fcpxml", config.log_path)

    if not config.force and is_done(config.output_dir, STEP_NAME):
        logger.info("[step 08] assemble_fcpxml: already done, skipping")
        return

    if config.force:
        clear(config.output_dir, STEP_NAME)

    if config.dry_run:
        logger.info("[step 08] [dry-run] would generate rough_cut.fcpxml")
        mark_done(config.output_dir, STEP_NAME)
        return

    builder = FCPXMLBuilder(config)
    try:
        root = builder.build()
        out_path = config.output_dir / "rough_cut.fcpxml"
        tree = etree.ElementTree(root)
        tree.write(
            str(out_path),
            xml_declaration=True,
            encoding="UTF-8",
            pretty_print=True,
            doctype='<!DOCTYPE fcpxml>',
        )
        logger.info("[step 08] FCPXML written: %s", out_path)
        mark_done(config.output_dir, STEP_NAME)
    except Exception as exc:
        logger.error("[step 08] FCPXML assembly failed: %s", exc)
        raise
