"""Sentinel file helpers for pipeline step caching."""

from __future__ import annotations

from pathlib import Path


def sentinel_path(output_dir: Path, step_name: str) -> Path:
    return output_dir / f".{step_name}.done"


def is_done(output_dir: Path, step_name: str) -> bool:
    return sentinel_path(output_dir, step_name).exists()


def mark_done(output_dir: Path, step_name: str) -> None:
    sentinel_path(output_dir, step_name).touch()


def clear(output_dir: Path, step_name: str) -> None:
    p = sentinel_path(output_dir, step_name)
    if p.exists():
        p.unlink()
