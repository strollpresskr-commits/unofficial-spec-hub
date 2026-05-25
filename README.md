# Strollpress Rough-Cut Pipeline

Automated rough-cut pipeline for the Jeju documentary archive project. Drop raw footage → get a reviewable FCPXML timeline, Korean/English/Japanese subtitles, chapter markers, thumbnail candidates, and an Obsidian archive note.

## Requirements

- macOS (Apple Silicon recommended), Python 3.11+
- `ffmpeg` and `ffprobe` installed via Homebrew: `brew install ffmpeg`
- API keys: `GEMINI_API_KEY` (Google AI Studio) and `ANTHROPIC_API_KEY`

## Install

```bash
# Clone the repo
git clone https://github.com/strollpresskr-commits/unofficial-spec-hub
cd unofficial-spec-hub

# Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install the package and dependencies
pip install -e .
```

> **Note on faster-whisper:** On Apple Silicon, `faster-whisper` will use Core ML / Metal via `ctranslate2`. The first run downloads the `large-v3` model (~3 GB) to `~/.cache/huggingface`. Subsequent runs skip the download.

## Setup

Set your API keys in your shell profile or `.env`:

```bash
export GEMINI_API_KEY="your-gemini-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

## Folder Structure

```
/Volumes/T7 Shield/footage/         ← default input root
  raw/                              ← drop footage here
  project.yaml                      ← project context (optional but recommended)
  glossary.json                     ← Jeju proper-noun lock list (optional)
  order.txt                         ← narrative clip order override (optional)

/Volumes/T7 Shield/rough_cuts/<slug>_<date>/   ← output
  clips/<clip_id>.json              ← per-clip metadata + analysis
  subtitles/<clip_id>.{ko,en,ja}.srt
  matching/<clip_id>.json           ← utterance ↔ b-roll candidates
  thumbnails/candidate_*.{jpg,png}
  chapters.txt                      ← YouTube-format chapter list
  chapters.json                     ← structured chapter data
  rough_cut.fcpxml                  ← timeline for FCP / Resolve / Premiere
  STROLL_<SLUG>_<DATE>.md           ← Obsidian archive note
  pipeline.log
```

## Usage

```bash
# Default run (uses /Volumes/T7 Shield/footage as input)
python -m strollpress.run

# Explicit paths (quote paths with spaces)
python -m strollpress.run \
  --input "/Volumes/T7 Shield/footage/jeju_spring_2025" \
  --output "/Volumes/T7 Shield/rough_cuts/jeju_spring_2025"

# Watch mode — reprocesses when new clips are dropped in raw/
python -m strollpress.run --watch

# Re-run only specific steps (useful after editing project.yaml)
python -m strollpress.run --steps 06,08

# Force re-run all steps (ignores completion sentinels)
python -m strollpress.run --force

# Dry run — shows what would happen without calling APIs
python -m strollpress.run --dry-run

# Cost-controlled test with 2 clips
python -m strollpress.run --input ./test_clips --output ./test_out --max-clips 2

# Use Claude Sonnet instead of Opus for b-roll matching (cheaper)
python -m strollpress.run --no-use-opus-matcher
```

## Pipeline Steps

| # | Module | Description | API |
|---|--------|-------------|-----|
| 01 | `probe` | ffprobe technical metadata | — |
| 02 | `analyze_visual` | Shot size, movement, classification, usability score | Gemini 2.0 Flash |
| 03 | `transcribe` | Korean Whisper large-v3 with word timestamps | local |
| 04 | `translate` | Korean → English + Japanese SRT | Claude Sonnet |
| 05 | `classify` | Rename/symlink clips with shot+class labels | — |
| 06 | `match` | Utterance ↔ b-roll semantic scoring | Claude Opus / Sonnet |
| 07 | `chapters` | Topic-shift boundary detection | Claude Sonnet |
| 08 | `assemble_fcpxml` | Build FCPXML 1.10 timeline | — |
| 09 | `thumbnails` | Frame extraction + Pillow text-overlay variants | Claude Sonnet |
| 10 | `archive_note` | Obsidian-compatible Markdown archive note | Claude Sonnet |

Each step writes a `.done` sentinel in the output folder. Re-running skips completed steps unless `--force` is passed. You can re-run any single step with `--steps <n>` without invalidating the rest.

## Opening the Timeline

- **Final Cut Pro**: File → Import → `rough_cut.fcpxml`
- **DaVinci Resolve**: File → Import → Timeline → `rough_cut.fcpxml`
- **Adobe Premiere**: File → Import → `rough_cut.fcpxml`

V2 contains the top-1 b-roll per utterance. V3 and V4 contain alternative b-roll candidates (disabled by default) — toggle visibility in the timeline inspector to compare options.

## Cost Estimates (approximate)

For a 15-minute documentary (≈10 clips, 3 interview + 7 b-roll):

| Step | Typical cost |
|------|-------------|
| Gemini Flash (visual analysis) | ~$0.10 |
| Claude Sonnet (translation, chapters, thumbnails) | ~$0.30 |
| Claude Opus (b-roll matching) | ~$0.80 |
| **Total** | **~$1.20** |

Costs vary with clip count, transcript length, and language count. Use `--max-clips 2` and `--no-use-opus-matcher` for low-cost testing.

## project.yaml Reference

See `examples/project.yaml` for a filled-in example. Key fields:

```yaml
slug: my_project          # used in output folder and archive note filename
title: "My Documentary"
theme: "Project description for AI context"
target_length_min: 15
audience: "archive viewers"
```
