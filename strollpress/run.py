"""Main pipeline orchestrator and CLI entry point."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from strollpress.config import build_config, PipelineConfig

console = Console()

STEPS = [
    (1, "probe", "strollpress.steps.probe"),
    (2, "analyze_visual", "strollpress.steps.analyze_visual"),
    (3, "transcribe", "strollpress.steps.transcribe"),
    (4, "translate", "strollpress.steps.translate"),
    (5, "classify", "strollpress.steps.classify"),
    (6, "match", "strollpress.steps.match"),
    (7, "chapters", "strollpress.steps.chapters"),
    (8, "assemble_fcpxml", "strollpress.steps.assemble_fcpxml"),
    (9, "thumbnails", "strollpress.steps.thumbnails"),
    (10, "archive_note", "strollpress.steps.archive_note"),
]


def _should_run(step_num: int, config: PipelineConfig) -> bool:
    if config.steps is None:
        return True
    return step_num in config.steps


def run_pipeline(config: PipelineConfig) -> None:
    from strollpress.utils.logging import get_logger
    logger = get_logger("pipeline", config.log_path)

    config.ensure_dirs()

    console.print(Rule("[bold cyan]Strollpress Rough-Cut Pipeline[/bold cyan]"))
    console.print(f"  Input:  {config.input_dir}")
    console.print(f"  Output: {config.output_dir}")
    console.print(f"  Steps:  {config.steps or 'all'}")
    if config.dry_run:
        console.print("[yellow]  DRY RUN — no files will be written[/yellow]")
    console.print()

    t_start = time.monotonic()
    step_times: list[tuple[str, float]] = []
    errors: list[str] = []

    for step_num, step_name, module_path in STEPS:
        if not _should_run(step_num, config):
            continue

        import importlib
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            console.print(f"[red]Step {step_num:02d} {step_name}: import error — {exc}[/red]")
            errors.append(f"step {step_num} import: {exc}")
            continue

        console.print(Rule(f"[bold]Step {step_num:02d}  {step_name}[/bold]", style="dim"))
        t0 = time.monotonic()
        try:
            module.run(config)
        except BaseException as exc:  # noqa: BLE001 — catch Rust panics too
            elapsed = time.monotonic() - t0
            console.print(f"[red]  ✗ Step {step_num:02d} FAILED: {exc}[/red]")
            logger.exception("Step %02d %s failed", step_num, step_name)
            errors.append(f"step {step_num} {step_name}: {exc}")
            step_times.append((step_name, elapsed))
            continue

        elapsed = time.monotonic() - t0
        step_times.append((step_name, elapsed))
        console.print(f"[green]  ✓ {step_name}[/green]  ({elapsed:.1f}s)")

    total_elapsed = time.monotonic() - t_start
    console.print()
    console.print(Rule("[bold cyan]Run complete[/bold cyan]"))
    _print_summary(step_times, errors, config, total_elapsed)


def _print_summary(
    step_times: list[tuple[str, float]],
    errors: list[str],
    config: PipelineConfig,
    total_elapsed: float,
) -> None:
    table = Table(title="Step Timing", show_header=True)
    table.add_column("Step", style="cyan")
    table.add_column("Elapsed", justify="right")
    for name, elapsed in step_times:
        table.add_row(name, f"{elapsed:.1f}s")
    console.print(table)

    if config.cost_log:
        total_cost = sum(e["cost_usd"] for e in config.cost_log)
        cost_table = Table(title=f"API Cost (est. ${total_cost:.4f} USD)", show_header=True)
        cost_table.add_column("Step")
        cost_table.add_column("Model")
        cost_table.add_column("In Tokens", justify="right")
        cost_table.add_column("Out Tokens", justify="right")
        cost_table.add_column("Cost", justify="right")
        for entry in config.cost_log:
            cost_table.add_row(
                entry["step"],
                entry["model"],
                f"{entry['input_tokens']:,}",
                f"{entry['output_tokens']:,}",
                f"${entry['cost_usd']:.4f}",
            )
        console.print(cost_table)

    if errors:
        console.print(f"\n[red]Errors ({len(errors)}):[/red]")
        for e in errors:
            console.print(f"  • {e}")
    else:
        console.print("\n[green]All steps completed without errors.[/green]")

    console.print(f"\nTotal runtime: [bold]{total_elapsed:.1f}s[/bold]")
    console.print(f"Output: [bold]{config.output_dir}[/bold]")


def _watch_loop(config: PipelineConfig) -> None:
    """Watch mode: poll input directory for new video files and re-run pipeline."""
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    from strollpress.config import VIDEO_EXTENSIONS

    class Handler(FileSystemEventHandler):
        def __init__(self):
            self.pending = False

        def on_created(self, event):
            if not event.is_directory:
                if Path(event.src_path).suffix.lower() in VIDEO_EXTENSIONS:
                    self.pending = True

    handler = Handler()
    observer = Observer()
    observer.schedule(handler, str(config.raw_dir), recursive=False)
    observer.start()
    console.print(f"[cyan]Watch mode active — monitoring {config.raw_dir}[/cyan]")
    console.print("Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(5)
            if handler.pending:
                handler.pending = False
                console.print("[yellow]New files detected — re-running pipeline...[/yellow]")
                run_pipeline(config)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="strollpress",
        description="Strollpress rough-cut pipeline for Jeju documentary archive",
    )
    p.add_argument("--input", "-i", help="Input folder path (overrides defaults)")
    p.add_argument("--output", "-o", help="Output folder path (overrides defaults)")
    p.add_argument("--watch", action="store_true", help="Watch input folder for new clips")
    p.add_argument(
        "--use-opus-matcher",
        dest="use_opus_matcher",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Claude Opus for b-roll matching (default: true)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    p.add_argument("--force", action="store_true", help="Re-run all steps ignoring sentinels")
    p.add_argument("--max-clips", type=int, default=None, metavar="N", help="Limit clip count")
    p.add_argument(
        "--language",
        dest="languages_str",
        default="ko,en,ja",
        help="Subtitle languages, comma-separated (default: ko,en,ja)",
    )
    p.add_argument(
        "--steps",
        dest="steps_str",
        default=None,
        metavar="NUMS",
        help="Comma-separated step numbers to run, e.g. 06,08",
    )
    p.add_argument(
        "--rename-symlink",
        action="store_true",
        help="Symlink renamed files instead of moving them in-place",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    config = build_config(
        input_path=args.input,
        output_path=args.output,
        use_opus_matcher=args.use_opus_matcher,
        dry_run=args.dry_run,
        force=args.force,
        watch_mode=args.watch,
        max_clips=args.max_clips,
        steps_str=args.steps_str,
        languages_str=args.languages_str,
        rename_symlink=args.rename_symlink,
    )

    if args.watch:
        _watch_loop(config)
    else:
        run_pipeline(config)


if __name__ == "__main__":
    main()
