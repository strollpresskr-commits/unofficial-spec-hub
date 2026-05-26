"""Step 05 — File renaming / symlinking based on visual classification."""

from __future__ import annotations

import json
import re
from pathlib import Path

from strollpress.config import PipelineConfig
from strollpress.utils.logging import get_logger
from strollpress.utils.sentinel import is_done, mark_done, clear

STEP_NAME = "classify"
STEP_NUM = 5

VALID_CLASSIFICATIONS = {
    "interview",
    "b_roll_landscape",
    "b_roll_ritual",
    "b_roll_people",
    "b_roll_object",
    "b_roll_motion",
    "discard_candidate",
}

VALID_SHOT_SIZES = {"wide", "medium", "cu", "xcu", "aerial"}


def _safe_label(val: str, valid: set[str], default: str) -> str:
    return val if val in valid else default


def _build_new_name(clip_data: dict, source_path: Path) -> str:
    visual = clip_data.get("visual_analysis", {})
    shot = _safe_label(visual.get("shot_size", ""), VALID_SHOT_SIZES, "unknown")
    cls = _safe_label(
        clip_data.get("classification", ""),
        VALID_CLASSIFICATIONS,
        "b_roll_landscape",
    )
    return f"{source_path.stem}_{shot}_{cls}{source_path.suffix}"


def run(config: PipelineConfig) -> list[dict]:
    logger = get_logger("classify", config.log_path)

    if not config.force and is_done(config.output_dir, STEP_NAME):
        logger.info("[step 05] classify: already done, skipping")
        return []

    if config.force:
        clear(config.output_dir, STEP_NAME)

    clip_jsons = sorted(config.clips_dir.glob("*.json"))
    if not clip_jsons:
        logger.info("[step 05] No clip JSONs — nothing to rename")
        mark_done(config.output_dir, STEP_NAME)
        return []

    logger.info(
        "[step 05] Renaming/symlinking %d clip(s) (%s mode)",
        len(clip_jsons),
        "symlink" if config.rename_symlink else "in-place rename",
    )

    for json_path in clip_jsons:
        with open(json_path) as f:
            clip_data = json.load(f)

        source_path = Path(clip_data.get("source_path", ""))
        if not source_path.exists():
            continue

        new_name = _build_new_name(clip_data, source_path)
        new_path = source_path.parent / new_name

        if new_path == source_path:
            continue

        if config.dry_run:
            logger.info("  [dry-run] %s → %s", source_path.name, new_name)
            continue

        try:
            if config.rename_symlink:
                if not new_path.exists():
                    new_path.symlink_to(source_path)
                clip_data["renamed_path"] = str(new_path)
                clip_data["rename_mode"] = "symlink"
            else:
                source_path.rename(new_path)
                clip_data["source_path"] = str(new_path)
                clip_data["rename_mode"] = "in-place"

            with open(json_path, "w") as f:
                json.dump(clip_data, f, ensure_ascii=False, indent=2)

            logger.info("  ✓ %s → %s", source_path.name, new_name)
        except Exception as exc:
            logger.error("  ✗ %s — rename failed: %s", source_path.name, exc)

    mark_done(config.output_dir, STEP_NAME)
    return []
