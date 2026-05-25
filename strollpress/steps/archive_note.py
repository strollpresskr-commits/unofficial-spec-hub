"""Step 10 — Obsidian-compatible archive note generation."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import anthropic

from strollpress.config import PipelineConfig, PROJECT_CONTEXT
from strollpress.utils.logging import get_logger
from strollpress.utils.sentinel import is_done, mark_done, clear

STEP_NAME = "archive_note"
STEP_NUM = 10

MODEL = "claude-sonnet-4-6"
SONNET_PRICE_IN = 3.0
SONNET_PRICE_OUT = 15.0


def _fmt_duration(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


def _generate_summary(
    full_transcript: str,
    project_meta: dict,
    chapters: list[dict],
    client: anthropic.Anthropic,
    config: PipelineConfig,
) -> str:
    prompt = f"""{PROJECT_CONTEXT}

Project: {project_meta.get('title', 'Untitled')}
Theme: {project_meta.get('theme', '')}

Write a 2–3 sentence archival summary of this documentary project based on:
- The transcript excerpts below
- The chapter structure
- Strollpress's documentary aesthetic (slow, observational, Jeju-focused)

Chapters: {', '.join(ch.get('title_en', '') for ch in chapters)}

Transcript excerpt:
{full_transcript[:3000]}

Return plain text only, no markdown."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    config.track_cost(
        step="archive_summary",
        model=MODEL,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        unit_price_in=SONNET_PRICE_IN,
        unit_price_out=SONNET_PRICE_OUT,
    )
    return response.content[0].text.strip()


def _load_srt_pairs(config: PipelineConfig, clip_id: str) -> list[dict]:
    """Load ko/en/ja SRT blocks into aligned list."""
    def parse_srt(path: Path) -> dict[int, dict]:
        blocks: dict[int, dict] = {}
        if not path.exists():
            return blocks
        current: dict = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.isdigit():
                current = {"index": int(line)}
            elif "-->" in line:
                parts = line.split("-->")
                def t(s: str) -> float:
                    s = s.strip().replace(",", ".")
                    p = s.split(":")
                    return int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])
                current["start"] = t(parts[0])
                current["end"] = t(parts[1])
                current["text"] = ""
            elif line == "":
                if current.get("text"):
                    blocks[current["index"]] = dict(current)
                current = {}
            elif "start" in current:
                current["text"] = (current.get("text", "") + " " + line).strip()
        if current.get("text"):
            blocks[current["index"]] = dict(current)
        return blocks

    ko = parse_srt(config.subtitles_dir / f"{clip_id}.ko.srt")
    en = parse_srt(config.subtitles_dir / f"{clip_id}.en.srt")
    ja = parse_srt(config.subtitles_dir / f"{clip_id}.ja.srt")

    pairs = []
    for idx in sorted(ko.keys()):
        row = ko[idx]
        tc = f"{int(row['start']//60):02d}:{int(row['start']%60):02d}"
        pairs.append({
            "tc": tc,
            "ko": row["text"],
            "en": en.get(idx, {}).get("text", ""),
            "ja": ja.get(idx, {}).get("text", ""),
            "flagged": "[?]" in row["text"],
        })
    return pairs


def _collect_editor_notes(clips: list[dict], config: PipelineConfig) -> list[str]:
    notes = []
    for clip in clips:
        cid = clip["clip_id"]
        score = clip.get("usability_score", 10)
        if score < 6:
            notes.append(
                f"- **{cid}**: usability score {score}/10 — consider replacing or excluding"
            )
        # Check matching for clips with multiple b-roll candidates
        match_path = config.matching_dir / f"{cid}.json"
        if match_path.exists():
            with open(match_path) as f:
                matching = json.load(f)
            for utt in matching.get("utterances", []):
                cands = utt.get("b_roll_candidates", [])
                if len(cands) > 1:
                    tc = f"{int(utt['start']//60):02d}:{int(utt['start']%60):02d}"
                    top = cands[0]
                    notes.append(
                        f"- **{cid}** @ {tc}: top b-roll is `{top['clip_id']}` "
                        f"(score {top['score']:.2f}) — 2 alt candidates available on V3/V4"
                    )

    # Flag low-confidence translations
    for srt_path in config.subtitles_dir.glob("*.en.srt"):
        text = srt_path.read_text(encoding="utf-8")
        if "[?]" in text:
            notes.append(f"- **{srt_path.stem}**: contains uncertain translations flagged with [?]")

    return notes or ["- No specific issues flagged — review the b-roll choices before export."]


def run(config: PipelineConfig) -> Path:
    logger = get_logger("archive_note", config.log_path)

    if not config.force and is_done(config.output_dir, STEP_NAME):
        logger.info("[step 10] archive_note: already done, skipping")
        return _find_note(config)

    if config.force:
        clear(config.output_dir, STEP_NAME)

    # Load everything
    clips = []
    for p in sorted(config.clips_dir.glob("*.json")):
        with open(p) as f:
            clips.append(json.load(f))

    chapters: list[dict] = []
    chapters_path = config.output_dir / "chapters.json"
    if chapters_path.exists():
        with open(chapters_path) as f:
            chapters = json.load(f)

    project_meta = config.project_meta
    slug = project_meta.get("slug", config.input_dir.name or "project")
    today = date.today().strftime("%Y%m%d")
    title = project_meta.get("title", "Untitled Documentary")
    theme = project_meta.get("theme", "")

    full_transcript_ko = ""
    for clip in clips:
        if clip.get("transcription"):
            full_transcript_ko += clip["transcription"].get("full_text_ko", "") + " "

    if config.dry_run:
        logger.info("[step 10] [dry-run] would write archive note")
        mark_done(config.output_dir, STEP_NAME)
        return config.output_dir

    client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    # Generate project summary
    summary = ""
    if full_transcript_ko.strip():
        try:
            summary = _generate_summary(full_transcript_ko, project_meta, chapters, client, config)
        except Exception as exc:
            logger.warning("[step 10] Summary generation failed: %s", exc)
            summary = theme or "Jeju documentary archive project."

    # ---- Build Obsidian Markdown ----
    lines = []

    # YAML frontmatter
    lines += [
        "---",
        f"title: \"{title}\"",
        f"slug: {slug}",
        f"date: {date.today().isoformat()}",
        f"project: strollpress",
        "tags: [strollpress, jeju, archive]",
        f"target_length: {project_meta.get('target_length_min', 15)} min",
        f"audience: {project_meta.get('audience', 'local + international archive viewers')}",
        "---",
        "",
    ]

    # Title
    lines += [
        f"# {title}",
        "",
        "> [!info] Project Summary",
        f"> {summary}",
        "",
    ]

    # Clip inventory table
    lines += [
        "## Clip Inventory",
        "",
        "| Clip ID | Classification | Duration | Usability | Location |",
        "|---------|---------------|----------|-----------|----------|",
    ]
    for clip in clips:
        cid = clip["clip_id"]
        cls = clip.get("classification", "—")
        dur = _fmt_duration(clip.get("duration", 0))
        usability = clip.get("usability_score", "—")
        gps = clip.get("gps")
        loc = (
            clip.get("visual_analysis", {}).get("location_type", "")
            or (f"{gps['lat']:.4f},{gps['lon']:.4f}" if gps else "—")
        )
        lines.append(f"| [[{cid}]] | {cls} | {dur} | {usability}/10 | {loc} |")
    lines.append("")

    # Full transcript with translations
    interview_clips = [c for c in clips if c.get("classification") == "interview" and c.get("transcription")]
    if interview_clips:
        lines += [
            "## Full Transcript",
            "",
        ]
        for clip in interview_clips:
            cid = clip["clip_id"]
            lines += [
                f"### {cid}",
                "",
                "> [!quote] Korean / English / Japanese",
            ]
            pairs = _load_srt_pairs(config, cid)
            if pairs:
                for row in pairs:
                    flag = " ⚠️" if row.get("flagged") else ""
                    lines.append(f"> **[{row['tc']}]** {row['ko']}{flag}")
                    if row["en"]:
                        lines.append(f"> _EN: {row['en']}_")
                    if row["ja"]:
                        lines.append(f"> _JA: {row['ja']}_")
                    lines.append(">")
            else:
                full_ko = clip["transcription"].get("full_text_ko", "")
                lines.append(f"> {full_ko}")
                lines.append(">")
            lines.append("")

    # Chapter list
    if chapters:
        lines += [
            "## Chapters",
            "",
        ]
        for ch in chapters:
            tc_secs = ch.get("start_seconds", 0)
            tc_str = f"{int(tc_secs // 60):02d}:{int(tc_secs % 60):02d}"
            lines.append(f"- `{tc_str}` **{ch.get('title_en', '')}** / {ch.get('title_ko', '')}")
        lines.append("")

    # Editor notes
    editor_notes = _collect_editor_notes(clips, config)
    lines += [
        "## Editor Notes for Finalization",
        "",
        "> [!warning] Decisions remaining before export",
    ]
    for note in editor_notes:
        lines.append(f"> {note}")
    lines.append("")

    # Cost log
    if config.cost_log:
        total_cost = sum(e["cost_usd"] for e in config.cost_log)
        lines += [
            "## Pipeline Cost",
            "",
            f"Total estimated API cost: **${total_cost:.4f} USD**",
            "",
            "| Step | Model | In Tokens | Out Tokens | Cost |",
            "|------|-------|-----------|------------|------|",
        ]
        for entry in config.cost_log:
            lines.append(
                f"| {entry['step']} | {entry['model']} | "
                f"{entry['input_tokens']:,} | {entry['output_tokens']:,} | "
                f"${entry['cost_usd']:.4f} |"
            )
        lines.append("")

    note_path = config.output_dir / f"STROLL_{slug.upper()}_{today}.md"
    note_path.write_text("\n".join(lines), encoding="utf-8")

    logger.info("[step 10] Archive note written: %s", note_path.name)
    mark_done(config.output_dir, STEP_NAME)
    return note_path


def _find_note(config: PipelineConfig) -> Path:
    notes = list(config.output_dir.glob("STROLL_*.md"))
    return notes[0] if notes else config.output_dir
