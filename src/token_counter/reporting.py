"""
Shared rich reporting utilities for token counting runs.
"""

from __future__ import annotations

import bisect
import json
import math
import os
import platform
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import datasets
import transformers

from token_counter import __version__
from token_counter.pdf_export import export_markdown_report_to_pdf

SCHEMA_VERSION = 1
HISTOGRAM_BUCKETS = (
    ("0-32", 0, 32),
    ("33-64", 33, 64),
    ("65-128", 65, 128),
    ("129-256", 129, 256),
    ("257-512", 257, 512),
    ("513-1024", 513, 1024),
    ("1025-2048", 1025, 2048),
    ("2049-4096", 2049, 4096),
    (">4096", 4097, None),
)
PERCENTILE_TARGETS = (
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p95", 0.95),
    ("p99", 0.99),
)


def now_local() -> datetime:
    return datetime.now().astimezone()


def format_timestamp(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.isoformat(timespec="seconds")


def datetime_from_epoch(value: Optional[float]) -> Optional[datetime]:
    if value is None:
        return None
    return datetime.fromtimestamp(value).astimezone()


def format_integer(value: Optional[int]) -> str:
    if value is None:
        return "n/a"
    return f"{value:,}".replace(",", ".")


def format_decimal(value: Optional[float], decimals: int = 2) -> str:
    if value is None:
        return "n/a"
    formatted = f"{value:,.{decimals}f}"
    return formatted.replace(",", "_").replace(".", ",").replace("_", ".")


def distribution_plot_relative_path(report_path: Path) -> str:
    return f"{report_path.stem}_distribution.png"


def pdf_report_path(report_path: Path) -> Path:
    return report_path.with_suffix(".pdf")


def _interpolate_quantile(sorted_values: list[float], quantile: float) -> Optional[float]:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])

    weight = position - lower
    return float(sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * weight)


def compute_iqr(percentiles: dict[str, Optional[float]]) -> Optional[float]:
    p25 = percentiles.get("p25")
    p75 = percentiles.get("p75")
    if p25 is None or p75 is None:
        return None
    return float(p75 - p25)


class P2QuantileEstimator:
    """
    Streaming quantile estimator using the P2 algorithm.
    """

    def __init__(self, quantile: float) -> None:
        if not 0 < quantile < 1:
            raise ValueError("quantile must be between 0 and 1")

        self.quantile = quantile
        self.count = 0
        self._initial_values: list[float] = []
        self._markers: list[float] = []
        self._positions: list[int] = []
        self._desired_positions: list[float] = []
        self._desired_position_increments = [
            0.0,
            quantile / 2.0,
            quantile,
            (1.0 + quantile) / 2.0,
            1.0,
        ]

    def add(self, value: float) -> None:
        value = float(value)
        self.count += 1

        if self.count <= 5:
            bisect.insort(self._initial_values, value)
            if self.count == 5:
                self._markers = list(self._initial_values)
                self._positions = [1, 2, 3, 4, 5]
                q = self.quantile
                self._desired_positions = [
                    1.0,
                    1.0 + 2.0 * q,
                    1.0 + 4.0 * q,
                    3.0 + 2.0 * q,
                    5.0,
                ]
            return

        markers = self._markers
        positions = self._positions

        if value < markers[0]:
            markers[0] = value
            bucket = 0
        elif value < markers[1]:
            bucket = 0
        elif value < markers[2]:
            bucket = 1
        elif value < markers[3]:
            bucket = 2
        elif value <= markers[4]:
            bucket = 3
        else:
            markers[4] = value
            bucket = 3

        for index in range(bucket + 1, 5):
            positions[index] += 1

        for index, increment in enumerate(self._desired_position_increments):
            self._desired_positions[index] += increment

        for index in range(1, 4):
            delta = self._desired_positions[index] - positions[index]
            if (delta >= 1 and positions[index + 1] - positions[index] > 1) or (
                delta <= -1 and positions[index - 1] - positions[index] < -1
            ):
                direction = 1 if delta > 0 else -1
                candidate = self._parabolic(index, direction)
                if markers[index - 1] < candidate < markers[index + 1]:
                    markers[index] = candidate
                else:
                    markers[index] = self._linear(index, direction)
                positions[index] += direction

    def result(self) -> Optional[float]:
        if self.count == 0:
            return None
        if self.count <= 5:
            return _interpolate_quantile(self._initial_values, self.quantile)
        return float(self._markers[2])

    def to_state(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "initial_values": list(self._initial_values),
            "markers": list(self._markers),
            "positions": list(self._positions),
            "desired_positions": list(self._desired_positions),
        }

    @classmethod
    def from_state(cls, quantile: float, state: Optional[dict[str, Any]]) -> "P2QuantileEstimator":
        estimator = cls(quantile)
        if not state:
            return estimator
        estimator.count = int(state.get("count", 0))
        estimator._initial_values = [float(value) for value in state.get("initial_values", [])]
        estimator._markers = [float(value) for value in state.get("markers", [])]
        estimator._positions = [int(value) for value in state.get("positions", [])]
        estimator._desired_positions = [
            float(value) for value in state.get("desired_positions", [])
        ]
        return estimator

    def _linear(self, index: int, direction: int) -> float:
        next_index = index + direction
        return self._markers[index] + direction * (
            (self._markers[next_index] - self._markers[index])
            / (self._positions[next_index] - self._positions[index])
        )

    def _parabolic(self, index: int, direction: int) -> float:
        left = index - 1
        right = index + 1
        positions = self._positions
        markers = self._markers

        return markers[index] + direction / (positions[right] - positions[left]) * (
            (positions[index] - positions[left] + direction)
            * (markers[right] - markers[index])
            / (positions[right] - positions[index])
            + (positions[right] - positions[index] - direction)
            * (markers[index] - markers[left])
            / (positions[index] - positions[left])
        )


def default_histogram() -> dict[str, int]:
    return {label: 0 for label, _, _ in HISTOGRAM_BUCKETS}


def default_percentile_estimators() -> dict[str, P2QuantileEstimator]:
    return {name: P2QuantileEstimator(target) for name, target in PERCENTILE_TARGETS}


def histogram_bucket_label(token_length: int) -> str:
    for label, lower_bound, upper_bound in HISTOGRAM_BUCKETS:
        if token_length < lower_bound:
            continue
        if upper_bound is None or token_length <= upper_bound:
            return label
    return HISTOGRAM_BUCKETS[-1][0]


@dataclass
class TokenCountStats:
    rows_seen: int = 0
    documents_processed: int = 0
    null_field_rows: int = 0
    empty_text_rows: int = 0
    non_string_rows_coerced: int = 0
    total_tokens: int = 0
    total_characters: int = 0
    min_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    wall_time: float = 0.0
    started_at_epoch: Optional[float] = None
    completed_at_epoch: Optional[float] = None
    histogram: dict[str, int] = field(default_factory=default_histogram)
    percentile_estimators: dict[str, P2QuantileEstimator] = field(
        default_factory=default_percentile_estimators,
        repr=False,
    )
    _token_mean: float = field(default=0.0, init=False, repr=False)
    _token_m2: float = field(default=0.0, init=False, repr=False)

    @property
    def documents(self) -> int:
        return self.documents_processed

    @property
    def rows_skipped(self) -> int:
        return self.rows_seen - self.documents_processed

    @property
    def average_tokens(self) -> float:
        return self.total_tokens / self.documents_processed if self.documents_processed else 0.0

    @property
    def average_characters(self) -> float:
        return (
            self.total_characters / self.documents_processed if self.documents_processed else 0.0
        )

    @property
    def average_tokens_per_char(self) -> float:
        return self.total_tokens / self.total_characters if self.total_characters else 0.0

    @property
    def token_stddev(self) -> float:
        if not self.documents_processed:
            return 0.0
        return math.sqrt(self._token_m2 / self.documents_processed)

    @property
    def tokens_per_second(self) -> float:
        return self.total_tokens / self.wall_time if self.wall_time > 0 else 0.0

    @property
    def docs_per_second(self) -> float:
        return self.documents_processed / self.wall_time if self.wall_time > 0 else 0.0

    @property
    def characters_per_second(self) -> float:
        return self.total_characters / self.wall_time if self.wall_time > 0 else 0.0

    @property
    def quantiles(self) -> dict[str, Optional[float]]:
        return {name: estimator.result() for name, estimator in self.percentile_estimators.items()}

    def observe_document(self, text: str, token_length: int) -> None:
        self.documents_processed += 1
        self.total_tokens += token_length
        self.total_characters += len(text)
        self.min_tokens = token_length if self.min_tokens is None else min(self.min_tokens, token_length)
        self.max_tokens = token_length if self.max_tokens is None else max(self.max_tokens, token_length)

        delta = token_length - self._token_mean
        self._token_mean += delta / self.documents_processed
        delta2 = token_length - self._token_mean
        self._token_m2 += delta * delta2

        for estimator in self.percentile_estimators.values():
            estimator.add(token_length)

        self.histogram[histogram_bucket_label(token_length)] += 1

    def to_checkpoint_state(self) -> dict[str, Any]:
        return {
            "rows_seen": self.rows_seen,
            "documents_processed": self.documents_processed,
            "null_field_rows": self.null_field_rows,
            "empty_text_rows": self.empty_text_rows,
            "non_string_rows_coerced": self.non_string_rows_coerced,
            "total_tokens": self.total_tokens,
            "total_characters": self.total_characters,
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
            "started_at_epoch": self.started_at_epoch,
            "completed_at_epoch": self.completed_at_epoch,
            "wall_time": self.wall_time,
            "histogram": dict(self.histogram),
            "percentile_estimators": {
                name: estimator.to_state()
                for name, estimator in self.percentile_estimators.items()
            },
            "token_mean": self._token_mean,
            "token_m2": self._token_m2,
        }

    @classmethod
    def from_checkpoint_state(cls, state: dict[str, Any]) -> "TokenCountStats":
        stats = cls()
        stats.rows_seen = int(state.get("rows_seen", 0))
        stats.documents_processed = int(state.get("documents_processed", 0))
        stats.null_field_rows = int(state.get("null_field_rows", 0))
        stats.empty_text_rows = int(state.get("empty_text_rows", 0))
        stats.non_string_rows_coerced = int(state.get("non_string_rows_coerced", 0))
        stats.total_tokens = int(state.get("total_tokens", 0))
        stats.total_characters = int(state.get("total_characters", 0))
        stats.min_tokens = state.get("min_tokens")
        stats.max_tokens = state.get("max_tokens")
        stats.started_at_epoch = state.get("started_at_epoch")
        stats.completed_at_epoch = state.get("completed_at_epoch")
        stats.wall_time = float(state.get("wall_time", 0.0))
        histogram = default_histogram()
        histogram.update(
            {
                str(label): int(value)
                for label, value in (state.get("histogram") or {}).items()
                if str(label) in histogram
            }
        )
        stats.histogram = histogram
        estimator_state = state.get("percentile_estimators") or {}
        stats.percentile_estimators = {
            name: P2QuantileEstimator.from_state(target, estimator_state.get(name))
            for name, target in PERCENTILE_TARGETS
        }
        stats._token_mean = float(state.get("token_mean", 0.0))
        stats._token_m2 = float(state.get("token_m2", 0.0))
        return stats


def build_run_metadata(
    *,
    input_value: str,
    data_format: str,
    field: str,
    model: str,
    add_special_tokens: bool,
    max_docs: Optional[int],
    trust_remote_code: bool,
    report_path: Optional[str],
    report_json_path: Optional[str],
    report_pdf_path: Optional[str],
    started_at_epoch: Optional[float],
    completed_at_epoch: Optional[float],
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    metadata = {
        "generated_at": format_timestamp(now_local()),
        "started_at": format_timestamp(datetime_from_epoch(started_at_epoch)),
        "completed_at": format_timestamp(datetime_from_epoch(completed_at_epoch)),
        "input": input_value,
        "format": data_format,
        "field": field,
        "model": model,
        "add_special_tokens": bool(add_special_tokens),
        "max_docs": max_docs,
        "trust_remote_code": bool(trust_remote_code),
        "report_path": report_path,
        "report_json_path": report_json_path,
        "report_pdf_path": report_pdf_path,
        "package_versions": {
            "token_counter": __version__,
            "datasets": datasets.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
        },
    }
    if extra:
        for key, value in extra.items():
            if value is not None:
                metadata[key] = value
    return metadata


def build_report_payload(
    run_metadata: dict[str, Any],
    stats: TokenCountStats,
    status: str = "completed",
) -> dict[str, Any]:
    percentiles = stats.quantiles
    iqr = compute_iqr(percentiles)
    plot_relative_path = None
    if run_metadata.get("report_path"):
        plot_relative_path = distribution_plot_relative_path(Path(run_metadata["report_path"]))

    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "run_metadata": run_metadata,
        "summary_stats": {
            "rows_seen": stats.rows_seen,
            "documents_processed": stats.documents_processed,
            "total_tokens": stats.total_tokens,
            "total_characters": stats.total_characters,
            "avg_tokens_per_doc": stats.average_tokens,
            "avg_characters_per_doc": stats.average_characters,
            "avg_tokens_per_char": stats.average_tokens_per_char,
            "min_tokens_per_doc": stats.min_tokens,
            "max_tokens_per_doc": stats.max_tokens,
        },
        "distribution_stats": {
            "mean_tokens_per_doc": stats.average_tokens,
            "median_tokens_per_doc": percentiles["p50"],
            "iqr_tokens_per_doc": iqr,
            "min_tokens_per_doc": stats.min_tokens,
            "max_tokens_per_doc": stats.max_tokens,
            "token_stddev": stats.token_stddev,
            "percentiles": percentiles,
            "plot": {
                "relative_path": plot_relative_path,
                "format": "png",
            }
            if plot_relative_path
            else None,
            "histogram": [
                {
                    "label": label,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "documents": stats.histogram[label],
                    "share_percent": (
                        (stats.histogram[label] / stats.documents_processed) * 100
                        if stats.documents_processed
                        else 0.0
                    ),
                }
                for label, lower_bound, upper_bound in HISTOGRAM_BUCKETS
            ],
        },
        "data_quality_stats": {
            "rows_seen": stats.rows_seen,
            "documents_processed": stats.documents_processed,
            "rows_skipped": stats.rows_skipped,
            "null_field_rows": stats.null_field_rows,
            "empty_text_rows": stats.empty_text_rows,
            "non_string_rows_coerced": stats.non_string_rows_coerced,
        },
        "performance_stats": {
            "wall_time_seconds": stats.wall_time,
            "tokens_per_second": stats.tokens_per_second,
            "documents_per_second": stats.docs_per_second,
            "characters_per_second": stats.characters_per_second,
        },
    }


def render_markdown_report(payload: dict[str, Any]) -> str:
    run_metadata = payload["run_metadata"]
    distribution_stats = payload["distribution_stats"]
    data_quality_stats = payload["data_quality_stats"]
    performance_stats = payload["performance_stats"]
    plot_info = distribution_stats.get("plot") or {}

    lines = [
        "# Token Count Report",
        "",
        f"- Status: {payload['status']}",
        f"- Schema version: {payload['schema_version']}",
        "",
        "## Run Context",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Generated at | {run_metadata['generated_at'] or 'n/a'} |",
        f"| Started at | {run_metadata['started_at'] or 'n/a'} |",
        f"| Completed at | {run_metadata['completed_at'] or 'n/a'} |",
        f"| Input | {run_metadata['input']} |",
        f"| Format | {run_metadata['format']} |",
    ]

    if run_metadata.get("split") is not None:
        lines.append(f"| Split | {run_metadata['split']} |")

    lines.extend(
        [
            f"| Field | {run_metadata['field']} |",
            f"| Model | {run_metadata['model']} |",
            f"| Add special tokens | {bool(run_metadata['add_special_tokens'])} |",
            f"| Max docs | {run_metadata['max_docs'] if run_metadata['max_docs'] is not None else 'All'} |",
            f"| Trust remote code | {bool(run_metadata['trust_remote_code'])} |",
        ]
    )

    if run_metadata.get("checkpoint_path") is not None:
        lines.append(f"| Checkpoint JSON path | {run_metadata['checkpoint_path']} |")

    if run_metadata.get("runner") is not None:
        lines.append(f"| Runner | {run_metadata['runner']} |")

    lines.extend(
        [
            f"| Markdown report path | {run_metadata['report_path'] or 'n/a'} |",
            f"| JSON report path | {run_metadata['report_json_path'] or 'n/a'} |",
            f"| PDF report path | {run_metadata['report_pdf_path'] or 'n/a'} |",
            f"| token_counter version | {run_metadata['package_versions']['token_counter']} |",
            f"| datasets version | {run_metadata['package_versions']['datasets']} |",
            f"| transformers version | {run_metadata['package_versions']['transformers']} |",
            f"| Python version | {run_metadata['package_versions']['python']} |",
            "",
            "## Distribution Snapshot",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Documents processed | {format_integer(data_quality_stats['documents_processed'])} |",
            f"| Mean tokens / doc | {format_decimal(distribution_stats['mean_tokens_per_doc'])} |",
            f"| Median tokens / doc | {format_decimal(distribution_stats['median_tokens_per_doc'])} |",
            f"| IQR tokens / doc | {format_decimal(distribution_stats['iqr_tokens_per_doc'])} |",
            f"| P95 tokens / doc | {format_decimal(distribution_stats['percentiles']['p95'])} |",
            f"| P99 tokens / doc | {format_decimal(distribution_stats['percentiles']['p99'])} |",
            f"| Std. dev. tokens / doc | {format_decimal(distribution_stats['token_stddev'])} |",
            "",
            "## Distribution Histogram",
            "",
        ]
    )

    if plot_info.get("relative_path"):
        lines.extend(
            [
                f"![Distribution histogram]({plot_info['relative_path']})",
                "",
            ]
        )

    lines.extend(
        [
            "| Token Range | Docs | Share |",
            "| --- | --- | --- |",
        ]
    )

    for bucket in distribution_stats["histogram"]:
        lines.append(
            f"| {bucket['label']} | {format_integer(bucket['documents'])} | {format_decimal(bucket['share_percent'])}% |"
        )

    lines.extend(
        [
            "",
            "## Data Quality",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Rows seen | {format_integer(data_quality_stats['rows_seen'])} |",
            f"| Documents processed | {format_integer(data_quality_stats['documents_processed'])} |",
            f"| Rows skipped | {format_integer(data_quality_stats['rows_skipped'])} |",
            f"| Null field rows | {format_integer(data_quality_stats['null_field_rows'])} |",
            f"| Empty text rows | {format_integer(data_quality_stats['empty_text_rows'])} |",
            f"| Non-string rows coerced | {format_integer(data_quality_stats['non_string_rows_coerced'])} |",
            "",
            "## Performance",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Wall time (s) | {format_decimal(performance_stats['wall_time_seconds'])} |",
            f"| Tokens per second | {format_decimal(performance_stats['tokens_per_second'])} |",
            f"| Documents per second | {format_decimal(performance_stats['documents_per_second'])} |",
            f"| Characters per second | {format_decimal(performance_stats['characters_per_second'])} |",
        ]
    )

    return "\n".join(lines)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        dir=path.parent,
        prefix=f"{path.stem}_",
        suffix=f"{path.suffix}.tmp",
    ) as handle:
        handle.write(content)
        temp_path = Path(handle.name)

    temp_path.replace(path)


def write_markdown_report(path: Path, payload: dict[str, Any]) -> Path:
    markdown = render_markdown_report(payload)
    _atomic_write_text(path, markdown)
    return path


def write_json_report(path: Path, payload: dict[str, Any]) -> Path:
    json_content = json.dumps(payload, ensure_ascii=False, indent=2)
    _atomic_write_text(path, json_content + "\n")
    return path


def write_distribution_plot(path: Path, payload: dict[str, Any]) -> Path:
    try:
        mpl_config_dir = Path(tempfile.gettempdir()) / "token_counter_mpl"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to generate the distribution plot. "
            "Install project dependencies and rerun the command."
        ) from exc

    histogram = payload["distribution_stats"]["histogram"]
    labels = [bucket["label"] for bucket in histogram]
    shares = [bucket["share_percent"] for bucket in histogram]
    documents = [bucket["documents"] for bucket in histogram]

    figure_height = max(4.0, 0.6 * len(labels) + 1.5)
    figure, axis = plt.subplots(figsize=(10, figure_height))
    bars = axis.barh(labels, shares, color="#2F6B8A")

    max_share = max(shares, default=0.0)
    axis.set_xlim(0, max(100.0, max_share * 1.1 if max_share > 0 else 1.0))
    axis.set_xlabel("Share of documents (%)")
    axis.set_title("Token length distribution")
    axis.grid(axis="x", linestyle="--", alpha=0.3)
    axis.invert_yaxis()

    x_limit = axis.get_xlim()[1]
    annotation_offset = x_limit * 0.01
    for bar, share, documents_count in zip(bars, shares, documents):
        text_x = min(bar.get_width() + annotation_offset, x_limit * 0.98)
        axis.text(
            text_x,
            bar.get_y() + bar.get_height() / 2,
            f"{share:.2f}% ({documents_count:,})",
            va="center",
            ha="left",
            fontsize=9,
        )

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        delete=False,
        dir=path.parent,
        prefix=f"{path.stem}_",
        suffix=path.suffix,
    ) as handle:
        temp_path = Path(handle.name)

    try:
        figure.savefig(temp_path, dpi=160, bbox_inches="tight", format="png")
        temp_path.replace(path)
    finally:
        plt.close(figure)
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

    return path


def write_outputs(
    payload: dict[str, Any],
) -> tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path]]:
    run_metadata = payload["run_metadata"]
    markdown_path = None
    json_path = None
    plot_path = None
    pdf_path = None

    report_path_value = run_metadata.get("report_path")
    if report_path_value:
        markdown_path = Path(report_path_value)
        plot_info = payload["distribution_stats"].get("plot") or {}
        if plot_info.get("relative_path"):
            plot_path = markdown_path.parent / plot_info["relative_path"]
            write_distribution_plot(plot_path, payload)
        markdown_path = write_markdown_report(markdown_path, payload)

        report_pdf_value = run_metadata.get("report_pdf_path")
        if report_pdf_value:
            pdf_path = export_markdown_report_to_pdf(markdown_path, Path(report_pdf_value))

    report_json_value = run_metadata.get("report_json_path")
    if report_json_value:
        json_path = write_json_report(Path(report_json_value), payload)

    return markdown_path, json_path, plot_path, pdf_path


def render_console_summary(payload: dict[str, Any]) -> list[str]:
    summary_stats = payload["summary_stats"]
    distribution_stats = payload["distribution_stats"]
    performance_stats = payload["performance_stats"]
    median = distribution_stats["median_tokens_per_doc"]
    iqr = distribution_stats["iqr_tokens_per_doc"]
    p95 = distribution_stats["percentiles"]["p95"]
    p99 = distribution_stats["percentiles"]["p99"]

    return [
        f"Status: {payload['status']}",
        f"Rows seen: {summary_stats['rows_seen']}",
        f"Documents processed: {summary_stats['documents_processed']}",
        f"Total tokens: {summary_stats['total_tokens']}",
        f"Mean tokens/doc: {distribution_stats['mean_tokens_per_doc']:.2f}",
        f"Median tokens/doc: {median:.2f}" if median is not None else "Median tokens/doc: n/a",
        f"IQR tokens/doc: {iqr:.2f}" if iqr is not None else "IQR tokens/doc: n/a",
        f"P95 tokens/doc: {p95:.2f}" if p95 is not None else "P95 tokens/doc: n/a",
        f"P99 tokens/doc: {p99:.2f}" if p99 is not None else "P99 tokens/doc: n/a",
        f"Tokens/sec: {performance_stats['tokens_per_second']:.2f}",
        f"Docs/sec: {performance_stats['documents_per_second']:.2f}",
    ]
