"""Central configuration: environment loading, path resolution, project context."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

# ---------------------------------------------------------------------------
# Project context injected into every AI prompt
# ---------------------------------------------------------------------------
PROJECT_CONTEXT = (
    "This footage is from Strollpress, a Jeju (Korea) regional documentary archive. "
    "Subjects include Jeju cultural rituals, local figures, place memory, groundwater "
    "and natural landscapes, and traditional craft. Aesthetic: observational documentary, "
    "slow, ritual-aware, archive-oriented. The editor prefers narrow loyal audiences over "
    "mass appeal, emotionally resonant pacing over optimized retention, and source materials "
    "that fold into multiple media forms. Treat all proper nouns and Jeju dialect terms as "
    "sacred — never translate place names or ritual terms; transliterate or leave in Korean "
    "if unsure."
)

DEFAULT_SSD = Path("/Volumes/T7 Shield")
DEFAULT_INPUT = DEFAULT_SSD / "footage"
DEFAULT_CACHE = DEFAULT_SSD / ".strollpress_cache"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mxf", ".avi", ".mkv", ".m4v", ".r3d", ".braw"}


@dataclass
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    cache_dir: Path
    project_yaml: Optional[Path] = None
    project_meta: dict = field(default_factory=dict)

    # Feature flags
    use_opus_matcher: bool = True
    dry_run: bool = False
    force: bool = False
    watch_mode: bool = False
    max_clips: Optional[int] = None
    steps: Optional[list[int]] = None
    languages: list[str] = field(default_factory=lambda: ["ko", "en", "ja"])
    rename_symlink: bool = False  # True = symlink instead of in-place rename

    # API keys (populated at build time)
    gemini_api_key: str = ""
    anthropic_api_key: str = ""

    # Cost tracking accumulator
    cost_log: list[dict] = field(default_factory=list)

    @property
    def clips_dir(self) -> Path:
        return self.output_dir / "clips"

    @property
    def subtitles_dir(self) -> Path:
        return self.output_dir / "subtitles"

    @property
    def matching_dir(self) -> Path:
        return self.output_dir / "matching"

    @property
    def thumbnails_dir(self) -> Path:
        return self.output_dir / "thumbnails"

    @property
    def raw_dir(self) -> Path:
        return self.input_dir / "raw"

    @property
    def glossary_path(self) -> Path:
        return self.input_dir / "glossary.json"

    @property
    def order_path(self) -> Path:
        return self.input_dir / "order.txt"

    @property
    def log_path(self) -> Path:
        return self.output_dir / "pipeline.log"

    def ensure_dirs(self) -> None:
        for d in [
            self.output_dir,
            self.clips_dir,
            self.subtitles_dir,
            self.matching_dir,
            self.thumbnails_dir,
            self.cache_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def track_cost(
        self,
        step: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        unit_price_in: float,
        unit_price_out: float,
    ) -> None:
        cost = (input_tokens * unit_price_in + output_tokens * unit_price_out) / 1_000_000
        self.cost_log.append(
            {
                "step": step,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": round(cost, 6),
            }
        )


def build_config(
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
    use_opus_matcher: bool = True,
    dry_run: bool = False,
    force: bool = False,
    watch_mode: bool = False,
    max_clips: Optional[int] = None,
    steps_str: Optional[str] = None,
    languages_str: str = "ko,en,ja",
    rename_symlink: bool = False,
) -> PipelineConfig:
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not gemini_key:
        print("[ERROR] GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    if not anthropic_key:
        print(
            "[WARNING] ANTHROPIC_API_KEY not set — steps 04, 06, 07, 09 (titles), 10 (summary) will be skipped.",
            file=sys.stderr,
        )

    # Resolve input directory
    if input_path:
        resolved_input = Path(input_path)
    elif os.environ.get("STROLL_INPUT"):
        resolved_input = Path(os.environ["STROLL_INPUT"])
    else:
        _check_ssd_mounted()
        resolved_input = DEFAULT_INPUT

    # Try to load project.yaml from input dir
    project_meta: dict = {}
    project_yaml_path: Optional[Path] = None
    yaml_candidate = resolved_input / "project.yaml"
    if yaml_candidate.exists():
        with open(yaml_candidate) as f:
            project_meta = yaml.safe_load(f) or {}
        project_yaml_path = yaml_candidate

    slug = project_meta.get("slug", resolved_input.name or "project")
    from datetime import date
    today = date.today().strftime("%Y%m%d")

    # Resolve output directory
    if output_path:
        resolved_output = Path(output_path)
    elif os.environ.get("STROLL_OUTPUT"):
        resolved_output = Path(os.environ["STROLL_OUTPUT"])
    elif project_meta.get("paths", {}).get("output"):
        resolved_output = Path(project_meta["paths"]["output"])
    else:
        if not input_path and not os.environ.get("STROLL_INPUT"):
            _check_ssd_mounted()
        rough_cuts_base = (
            Path(os.environ["STROLL_OUTPUT"]).parent
            if os.environ.get("STROLL_OUTPUT")
            else DEFAULT_SSD / "rough_cuts"
        )
        resolved_output = rough_cuts_base / f"{slug}_{today}"

    # Resolve cache directory
    if os.environ.get("STROLL_CACHE"):
        resolved_cache = Path(os.environ["STROLL_CACHE"])
    elif project_meta.get("paths", {}).get("cache"):
        resolved_cache = Path(project_meta["paths"]["cache"])
    else:
        resolved_cache = DEFAULT_CACHE

    # Parse steps
    steps: Optional[list[int]] = None
    if steps_str:
        steps = [int(s.strip()) for s in steps_str.split(",")]

    languages = [lang.strip() for lang in languages_str.split(",")]

    return PipelineConfig(
        input_dir=resolved_input,
        output_dir=resolved_output,
        cache_dir=resolved_cache,
        project_yaml=project_yaml_path,
        project_meta=project_meta,
        use_opus_matcher=use_opus_matcher,
        dry_run=dry_run,
        force=force,
        watch_mode=watch_mode,
        max_clips=max_clips,
        steps=steps,
        languages=languages,
        rename_symlink=rename_symlink,
        gemini_api_key=gemini_key,
        anthropic_api_key=anthropic_key,
    )


def _check_ssd_mounted() -> None:
    if not DEFAULT_SSD.exists():
        print(
            f"[ERROR] External SSD not mounted at {DEFAULT_SSD}.\n"
            "Mount the drive or pass --input / --output to override.",
            file=sys.stderr,
        )
        sys.exit(2)
