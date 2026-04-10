"""
Rich-powered terminal UI for interactive token counting runs.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional

from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from token_counter.reporting import (
    TokenCountStats,
    compute_iqr,
    format_decimal,
    format_integer,
)


def _compact_number(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    abs_value = abs(float(value))
    thresholds = (
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "K"),
    )
    for threshold, suffix in thresholds:
        if abs_value >= threshold:
            scaled = value / threshold
            decimals = 2 if abs(scaled) < 10 else 1
            return f"{scaled:.{decimals}f}{suffix}"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.1f}"


class CountingUI:
    """
    Small terminal UI wrapper used by both CLIs.

    The live display is rendered on stderr so stdout can still be used for
    machine-readable payloads when necessary.
    """

    def __init__(
        self,
        *,
        title: str,
        source_label: str,
        source_value: str,
        model: str,
        total_docs: Optional[int] = None,
        progress_total: Optional[int] = None,
        progress_unit: str = "docs",
        progress_metric: str = "documents",
        dataset_total_rows: Optional[int] = None,
        checkpoint_every: Optional[int] = None,
        report_path: Optional[str] = None,
        json_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        refresh_interval: float = 0.2,
    ) -> None:
        self.title = title
        self.source_label = source_label
        self.source_value = source_value
        self.model = model
        self.total_docs = total_docs
        self.progress_total = progress_total if progress_total is not None else total_docs
        self.progress_unit = progress_unit
        self.progress_metric = progress_metric
        self.dataset_total_rows = dataset_total_rows
        self.checkpoint_every = checkpoint_every
        self.report_path = report_path
        self.json_path = json_path
        self.checkpoint_path = checkpoint_path
        self.refresh_interval = refresh_interval
        self.last_checkpoint_docs: int = 0

        self.console = Console(stderr=True, log_path=False, log_time=False, soft_wrap=True)
        self.progress = self._build_progress(self.progress_total)
        self.task_id = self.progress.add_task("Preparing", total=self.progress_total)
        self.checkpoint_progress = self._build_checkpoint_progress(checkpoint_every)
        self.checkpoint_task_id = (
            self.checkpoint_progress.add_task("Next checkpoint", total=checkpoint_every)
            if self.checkpoint_progress is not None
            else None
        )
        self.phase = "Preparing"
        self.message: Optional[str] = None
        self.last_error: Optional[str] = None
        self.restart_attempts = 0
        self.events: Deque[tuple[str, str]] = deque(maxlen=4)
        self.stats = TokenCountStats()
        self._last_refresh = 0.0
        self._last_progress_value = 0
        self._started = False
        self.live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=max(4, int(1 / max(refresh_interval, 0.1))),
            transient=False,
        )

    def start(self) -> None:
        if self._started:
            return
        self.live.start()
        self._started = True
        self.update(self.stats, force=True)

    def stop(self) -> None:
        if not self._started:
            return
        self.refresh(force=True)
        self.live.stop()
        self._started = False

    def set_phase(self, phase: str, message: Optional[str] = None, *, force: bool = True) -> None:
        self.phase = phase
        self.message = message
        self.progress.update(self.task_id, description=phase)
        self.refresh(force=force)

    def set_message(self, message: Optional[str], *, force: bool = True) -> None:
        self.message = message
        self.refresh(force=force)

    def set_restart_attempts(self, attempts: int) -> None:
        self.restart_attempts = attempts
        self.refresh(force=True)

    def set_checkpoint(self, docs: int) -> None:
        self.last_checkpoint_docs = docs
        self.message = f"Checkpoint saved at {_compact_number(docs)} docs"
        if self.checkpoint_progress is not None and self.checkpoint_task_id is not None:
            self.checkpoint_progress.update(self.checkpoint_task_id, completed=0)
        self.refresh(force=True)

    def set_checkpoint_anchor(self, docs: int) -> None:
        self.last_checkpoint_docs = docs
        if self.checkpoint_progress is not None and self.checkpoint_task_id is not None:
            self.checkpoint_progress.update(self.checkpoint_task_id, completed=0)
        self.refresh(force=True)

    def set_error(self, message: Optional[str]) -> None:
        self.last_error = message
        self.refresh(force=True)

    def clear_error(self) -> None:
        if self.last_error is None:
            return
        self.last_error = None
        self.refresh(force=True)

    def update(self, stats: TokenCountStats, *, force: bool = False) -> None:
        self.stats = stats
        progress_value = self._progress_value(stats)
        completed = progress_value
        if self.progress_total is not None:
            completed = min(completed, self.progress_total)
        self.progress.update(self.task_id, completed=completed, total=self.progress_total)
        self._last_progress_value = progress_value
        if self.checkpoint_progress is not None and self.checkpoint_task_id is not None:
            completed = max(0, stats.documents_processed - self.last_checkpoint_docs)
            total = self.checkpoint_every or max(completed, 1)
            completed = min(completed, total)
            self.checkpoint_progress.update(
                self.checkpoint_task_id,
                completed=completed,
                total=total,
            )
        self.refresh(force=force)

    def log(self, message: str, *, style: str = "cyan") -> None:
        self.events.append((message, style))
        self.message = message
        self.refresh(force=True)

    def print_final_summary(
        self,
        payload: dict[str, object],
        *,
        markdown_path: Optional[Path],
        json_path: Optional[Path],
        plot_path: Optional[Path],
        pdf_path: Optional[Path],
    ) -> None:
        summary = self._build_summary_table(payload)
        outputs = self._build_outputs_table(
            markdown_path=markdown_path,
            json_path=json_path,
            plot_path=plot_path,
            pdf_path=pdf_path,
        )
        self.console.print(
            Panel(
                summary,
                title=f"[bold green]Run {str(payload.get('status', 'completed')).replace('_', ' ').title()}[/bold green]",
                border_style="green",
            )
        )
        self.console.print(outputs)

    def refresh(self, *, force: bool = False) -> None:
        if not self._started:
            return
        now = time.monotonic()
        if not force and now - self._last_refresh < self.refresh_interval:
            return
        self.live.update(self._render(), refresh=True)
        self._last_refresh = now

    def _progress_value(self, stats: TokenCountStats) -> int:
        if self.progress_metric == "rows":
            return stats.rows_seen
        return stats.documents_processed

    def _build_progress(self, total_docs: Optional[int]) -> Progress:
        columns = [
            SpinnerColumn(style="cyan"),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=None, pulse_style="cyan", complete_style="green"),
        ]
        if total_docs is not None:
            columns.append(TaskProgressColumn())
        columns.extend(
            [
                TextColumn(f"[magenta]{{task.completed:,.0f}} {self.progress_unit}"),
                TimeElapsedColumn(),
            ]
        )
        if total_docs is not None:
            columns.append(TimeRemainingColumn())
        return Progress(*columns, console=self.console, expand=True)

    def _build_checkpoint_progress(self, checkpoint_every: Optional[int]) -> Optional[Progress]:
        if not checkpoint_every or checkpoint_every <= 0:
            return None
        return Progress(
            TextColumn("[bold yellow]{task.description}"),
            BarColumn(bar_width=None, complete_style="yellow", finished_style="green"),
            TaskProgressColumn(),
            TextColumn(
                "[yellow]{task.completed:,.0f}[/yellow]/[yellow]{task.total:,.0f} docs[/yellow]"
            ),
            console=self.console,
            expand=True,
        )

    def _render(self) -> Group:
        parts = [self._build_context_panel(), self.progress]
        if self.checkpoint_progress is not None:
            parts.append(self.checkpoint_progress)
        parts.append(
            Columns(
                [self._build_metrics_panel(), self._build_activity_panel()],
                expand=True,
                equal=True,
            )
        )
        if self.last_error:
            parts.append(
                Panel(
                    Text(self.last_error, style="bold red"),
                    title="[bold red]Last Error[/bold red]",
                    border_style="red",
                )
            )
        return Group(*parts)

    def _build_context_panel(self) -> Panel:
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(style="bold cyan", ratio=1)
        table.add_column(ratio=3)
        table.add_column(style="bold cyan", ratio=1)
        table.add_column(ratio=3)

        table.add_row(
            self.source_label,
            Text(self.source_value, overflow="ellipsis"),
            "Model",
            Text(self.model, overflow="ellipsis"),
        )
        table.add_row(
            "Phase",
            Text(self.phase, style="bold white"),
            "Run target",
            Text(
                f"{format_integer(self.total_docs)} docs"
                if self.total_docs is not None
                else "All docs"
            ),
        )
        table.add_row(
            "Markdown",
            Text(self.report_path or "disabled", overflow="ellipsis"),
            "JSON",
            Text(self.json_path or "disabled", overflow="ellipsis"),
        )
        if self.dataset_total_rows is not None:
            completion_percent = (
                (self.stats.rows_seen / self.dataset_total_rows) * 100
                if self.dataset_total_rows > 0
                else 0.0
            )
            table.add_row(
                "Dataset size",
                Text(f"{_compact_number(self.dataset_total_rows)} rows"),
                "Completed",
                Text(f"{completion_percent:.1f}% of rows"),
            )
        if self.checkpoint_path and self.last_checkpoint_docs == 0:
            checkpoint_display = self.checkpoint_path
        elif self.last_checkpoint_docs > 0:
            checkpoint_display = f"{_compact_number(self.last_checkpoint_docs)} docs"
        else:
            checkpoint_display = "n/a"
        if self.checkpoint_path:
            table.add_row(
                "Checkpoint",
                Text(checkpoint_display, overflow="ellipsis"),
                "Message",
                Text(self.message or "Streaming"),
            )
        else:
            table.add_row(
                "Message",
                Text(self.message or "Streaming"),
                "Restarts",
                Text(str(self.restart_attempts)),
            )

        return Panel(table, title=f"[bold]{self.title}[/bold]", border_style="cyan")

    def _build_metrics_panel(self) -> Panel:
        percentiles = self.stats.quantiles
        iqr = compute_iqr(percentiles)
        metrics = Table.grid(expand=True, padding=(0, 1))
        for _ in range(4):
            metrics.add_column(ratio=1)

        metrics.add_row(
            "[bold cyan]Docs[/bold cyan]",
            f"[bold white]{_compact_number(self.stats.documents_processed)}[/bold white]",
            "[bold cyan]Tokens[/bold cyan]",
            f"[bold white]{_compact_number(self.stats.total_tokens)}[/bold white]",
        )
        metrics.add_row(
            "[bold cyan]Avg/doc[/bold cyan]",
            format_decimal(self.stats.average_tokens),
            "[bold cyan]Tok/s[/bold cyan]",
            f"[bold white]{_compact_number(self.stats.tokens_per_second)}[/bold white]",
        )
        metrics.add_row(
            "[bold cyan]P50[/bold cyan]",
            format_decimal(percentiles["p50"]),
            "[bold cyan]P95[/bold cyan]",
            format_decimal(percentiles["p95"]),
        )
        metrics.add_row(
            "[bold cyan]IQR[/bold cyan]",
            format_decimal(iqr),
            "[bold cyan]Doc/s[/bold cyan]",
            f"[bold white]{_compact_number(self.stats.docs_per_second)}[/bold white]",
        )
        return Panel(metrics, title="[bold]Snapshot[/bold]", border_style="blue")

    def _build_activity_panel(self) -> Panel:
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(style="bold cyan", ratio=1)
        table.add_column(ratio=1)
        table.add_row("Rows seen", format_integer(self.stats.rows_seen))
        table.add_row("Empty rows", format_integer(self.stats.empty_text_rows))
        table.add_row("Min / max", f"{format_integer(self.stats.min_tokens)} / {format_integer(self.stats.max_tokens)}")
        table.add_row("Retries", format_integer(self.restart_attempts))
        if self.dataset_total_rows is not None:
            remaining_rows = max(0, self.dataset_total_rows - self.stats.rows_seen)
            completion_percent = (
                (self.stats.rows_seen / self.dataset_total_rows) * 100
                if self.dataset_total_rows > 0
                else 0.0
            )
            table.add_row("Rows left", f"{format_integer(remaining_rows)}")
            table.add_row("Dataset done", f"{completion_percent:.2f}%")
        if self.checkpoint_every:
            remaining = max(0, self.checkpoint_every - (self.stats.documents_processed - self.last_checkpoint_docs))
            table.add_row("Next checkpoint", f"in {_compact_number(remaining)} docs")

        if self.events:
            table.add_row("", "")
            recent = Text()
            for index, (message, style) in enumerate(self.events):
                if index:
                    recent.append("\n")
                recent.append("• ", style="dim")
                recent.append(message, style=style)
            table.add_row("Recent", recent)

        return Panel(table, title="[bold]Activity[/bold]", border_style="magenta")

    def _build_summary_table(self, payload: dict[str, object]) -> Table:
        summary_stats = payload["summary_stats"]  # type: ignore[index]
        distribution_stats = payload["distribution_stats"]  # type: ignore[index]
        performance_stats = payload["performance_stats"]  # type: ignore[index]

        table = Table(box=box.SIMPLE_HEAVY, expand=True, show_header=False)
        table.add_column(style="bold cyan", width=20)
        table.add_column()
        table.add_column(style="bold cyan", width=20)
        table.add_column()

        percentiles = distribution_stats["percentiles"]  # type: ignore[index]
        table.add_row(
            "Documents",
            format_integer(summary_stats["documents_processed"]),  # type: ignore[index]
            "Rows seen",
            format_integer(summary_stats["rows_seen"]),  # type: ignore[index]
        )
        table.add_row(
            "Total tokens",
            format_integer(summary_stats["total_tokens"]),  # type: ignore[index]
            "Mean / doc",
            format_decimal(distribution_stats["mean_tokens_per_doc"]),  # type: ignore[index]
        )
        table.add_row(
            "Median / doc",
            format_decimal(distribution_stats["median_tokens_per_doc"]),  # type: ignore[index]
            "IQR / doc",
            format_decimal(distribution_stats["iqr_tokens_per_doc"]),  # type: ignore[index]
        )
        table.add_row(
            "P95 / doc",
            format_decimal(percentiles["p95"]),  # type: ignore[index]
            "P99 / doc",
            format_decimal(percentiles["p99"]),  # type: ignore[index]
        )
        table.add_row(
            "Tokens / sec",
            format_decimal(performance_stats["tokens_per_second"]),  # type: ignore[index]
            "Docs / sec",
            format_decimal(performance_stats["documents_per_second"]),  # type: ignore[index]
        )
        return table

    def _build_outputs_table(
        self,
        *,
        markdown_path: Optional[Path],
        json_path: Optional[Path],
        plot_path: Optional[Path],
        pdf_path: Optional[Path],
    ) -> Table:
        table = Table(title="Artifacts", box=box.SIMPLE_HEAVY, expand=True)
        table.add_column("Type", style="bold cyan", width=14)
        table.add_column("Path", style="white")

        rows = [
            ("Markdown", markdown_path),
            ("JSON", json_path),
            ("Histogram", plot_path),
            ("PDF", pdf_path),
        ]
        for label, path in rows:
            table.add_row(label, str(path) if path else "n/a")
        return table
