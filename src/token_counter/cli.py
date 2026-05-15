"""
Universal CLI entrypoint for counting tokenizer tokens in datasets.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional

from datasets import load_dataset

from token_counter.hf_auth import load_hf_token
from token_counter.hf_dataset_meta import resolve_dataset_split_size
from token_counter.reporting import (
    TokenCountStats,
    build_report_payload,
    build_run_metadata,
    pdf_report_path,
    write_json_report,
    write_outputs,
)
from token_counter.terminal_ui import CountingUI

DEFAULT_MODEL = "Qwen/Qwen3-1.7B-Base"
DEFAULT_FORMAT = "parquet"
DEFAULT_FIELD = "text"
DEFAULT_BATCH_SIZE = 256
DEFAULT_REPORT_PATH = "reports/token_count_report.md"
DEFAULT_REPORT_JSON_PATH = "reports/token_count_report.json"
DEFAULT_CHECKPOINT_EVERY = 10_000
TERMINAL_STATUSES = {"completed", "max_docs_reached", "interrupted", "error"}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count tokenizer tokens in local, remote, or Hugging Face datasets."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input",
        help="Input JSONL/Parquet path, URL, or hf:// glob.",
    )
    source.add_argument(
        "--dataset",
        help="Hugging Face dataset id, for example org/name.",
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "parquet"],
        default=DEFAULT_FORMAT,
        help="Input format for --input. Default: parquet.",
    )
    parser.add_argument("--config", default=None, help="Hugging Face dataset config/name.")
    parser.add_argument("--revision", default=None, help="Hugging Face dataset revision.")
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        help="Hugging Face split to process. Can be repeated. Default: train.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Hugging Face splits to process. Default: train.",
    )
    parser.add_argument(
        "--field",
        default=DEFAULT_FIELD,
        help=f"Field in each record containing text to tokenize. Default: {DEFAULT_FIELD}.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Tokenizer model to use. Default: {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Documents to tokenize per batch. Default: {DEFAULT_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--add-special-tokens",
        action="store_true",
        help="Include tokenizer special tokens when encoding.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum number of processed documents. Defaults to all.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow loading custom tokenizer or dataset code from remote repositories.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from --report-json if it contains a token-counter checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help=f"Write checkpoint JSON every N processed docs. Use 0 to disable. Default: {DEFAULT_CHECKPOINT_EVERY}.",
    )
    parser.add_argument(
        "--report",
        default=DEFAULT_REPORT_PATH,
        help=f"Path for Markdown report. Use empty string to skip. Default: {DEFAULT_REPORT_PATH}.",
    )
    parser.add_argument(
        "--report-json",
        default=DEFAULT_REPORT_JSON_PATH,
        help=f"Path for JSON report/checkpoint. Use empty string to skip. Default: {DEFAULT_REPORT_JSON_PATH}.",
    )
    parser.add_argument(
        "--report-pdf",
        action="store_true",
        help="Generate a PDF report next to the Markdown report.",
    )
    args = parser.parse_args(argv)
    if args.split and args.splits:
        parser.error("Use either --split or --splits, not both.")
    if args.input and (args.split or args.splits):
        parser.error("--split/--splits only apply with --dataset.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero.")
    if args.max_docs is not None and args.max_docs <= 0:
        parser.error("--max-docs must be greater than zero.")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every cannot be negative.")
    if args.report_pdf and not args.report:
        parser.error("--report-pdf requires --report to be set to a Markdown path.")
    if args.resume and not args.report_json:
        parser.error("--resume requires --report-json.")
    return args


def _resolve_splits(args: argparse.Namespace) -> list[str]:
    if not args.dataset:
        return []
    if args.splits:
        return list(args.splits)
    if args.split:
        return list(args.split)
    return ["train"]


def _extract_field(row: Any, field: str) -> Any:
    if not isinstance(row, dict):
        raise TypeError(f"Expected dataset rows to be dictionaries, got {type(row).__name__}.")
    if field not in row:
        raise KeyError(f"Field '{field}' not found in record: keys={list(row.keys())}")
    return row[field]


def _iter_input_values(
    input_path: str,
    data_format: str,
    field: str,
    hf_token: Optional[str] = None,
) -> Iterator[Any]:
    data_format = data_format.lower()
    path = Path(input_path)
    if data_format not in {"jsonl", "parquet"}:
        raise ValueError(f"Unsupported format '{data_format}'. Use jsonl or parquet.")
    if not input_path.startswith(("http://", "https://", "hf://")) and not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    loader = "json" if data_format == "jsonl" else "parquet"
    storage_options = {"token": hf_token} if hf_token and input_path.startswith("hf://") else None
    dataset = load_dataset(
        loader,
        data_files=input_path,
        streaming=True,
        token=hf_token,
        storage_options=storage_options,
    )
    for row in dataset["train"]:
        yield _extract_field(row, field)


def _iter_dataset_split_values(
    dataset_id: str,
    split: str,
    field: str,
    *,
    config: Optional[str] = None,
    revision: Optional[str] = None,
    hf_token: Optional[str] = None,
    trust_remote_code: bool = False,
) -> Iterator[Any]:
    kwargs: dict[str, Any] = {
        "split": split,
        "streaming": True,
        "token": hf_token,
        "trust_remote_code": trust_remote_code,
    }
    if revision:
        kwargs["revision"] = revision
    if config:
        stream = load_dataset(dataset_id, config, **kwargs)
    else:
        stream = load_dataset(dataset_id, **kwargs)
    for row in stream:
        yield _extract_field(row, field)


def _load_stream(
    input_path: str,
    data_format: str,
    field: str,
    hf_token: Optional[str] = None,
) -> Iterator[Any]:
    """Backward-compatible private helper for tests and one-off callers."""

    yield from _iter_input_values(input_path, data_format, field, hf_token=hf_token)


def _token_lengths(tokenizer: Any, texts: list[str], *, add_special_tokens: bool) -> list[int]:
    if not texts:
        return []
    try:
        encoded = tokenizer(
            texts,
            add_special_tokens=add_special_tokens,
            return_attention_mask=False,
        )
        input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        return [len(ids) for ids in input_ids]
    except (AttributeError, KeyError, TypeError):
        return [
            len(tokenizer.encode(text, add_special_tokens=add_special_tokens))
            for text in texts
        ]


def _load_tokenizer(model: str, *, trust_remote_code: bool, token: Optional[str]) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model,
        trust_remote_code=trust_remote_code,
        token=token,
    )


def _refresh_timing(stats: TokenCountStats, *, terminal: bool = False) -> None:
    now_epoch = time.time()
    if stats.started_at_epoch is None:
        stats.started_at_epoch = now_epoch
    stats.wall_time = max(0.0, now_epoch - stats.started_at_epoch)
    if terminal:
        stats.completed_at_epoch = now_epoch


def _observe_rows(
    values: Iterable[Any],
    *,
    tokenizer: Any,
    stats: TokenCountStats,
    split_stats: Optional[TokenCountStats],
    add_special_tokens: bool,
    max_docs: Optional[int],
    batch_size: int,
    rows_to_skip: int = 0,
    ui: Optional[CountingUI] = None,
    checkpoint_callback: Optional[Callable[[], None]] = None,
    stop_requested: Optional[Callable[[], bool]] = None,
) -> bool:
    pending_texts: list[str] = []
    split_target = split_stats

    def targets() -> list[TokenCountStats]:
        if split_target is None:
            return [stats]
        return [stats, split_target]

    def flush() -> None:
        if not pending_texts:
            return
        lengths = _token_lengths(
            tokenizer,
            pending_texts,
            add_special_tokens=add_special_tokens,
        )
        for text, token_length in zip(pending_texts, lengths):
            stats.observe_document(text=text, token_length=token_length)
            if split_target is not None:
                split_target.observe_document(text=text, token_length=token_length)
        pending_texts.clear()
        _refresh_timing(stats)
        if split_target is not None:
            _refresh_timing(split_target)
        if ui is not None:
            ui.update(stats)
        if checkpoint_callback is not None:
            checkpoint_callback()

    completed_stream = True
    for row_number, raw_value in enumerate(values, start=1):
        if row_number <= rows_to_skip:
            continue
        if stop_requested is not None and stop_requested():
            completed_stream = False
            break
        if max_docs is not None and stats.documents_processed + len(pending_texts) >= max_docs:
            flush()
            completed_stream = False
            break

        for target in targets():
            target.rows_seen += 1

        if raw_value is None:
            for target in targets():
                target.null_field_rows += 1
            if ui is not None:
                ui.update(stats)
            continue

        if isinstance(raw_value, str):
            text = raw_value
        else:
            for target in targets():
                target.non_string_rows_coerced += 1
            text = str(raw_value)

        if text == "":
            for target in targets():
                target.empty_text_rows += 1

        pending_texts.append(text)
        if max_docs is not None and stats.documents_processed + len(pending_texts) >= max_docs:
            flush()
            completed_stream = False
            break
        if len(pending_texts) >= batch_size:
            flush()
            if max_docs is not None and stats.documents_processed >= max_docs:
                completed_stream = False
                break

    flush()
    if max_docs is not None and stats.documents_processed >= max_docs:
        completed_stream = False
    return completed_stream


def _count_tokens(
    text_stream: Iterable[Any],
    tokenizer: Any,
    add_special_tokens: bool = False,
    max_docs: Optional[int] = None,
    ui: Optional[CountingUI] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> TokenCountStats:
    stats = TokenCountStats()
    stats.started_at_epoch = time.time()
    _observe_rows(
        text_stream,
        tokenizer=tokenizer,
        stats=stats,
        split_stats=None,
        add_special_tokens=add_special_tokens,
        max_docs=max_docs,
        batch_size=batch_size,
        ui=ui,
    )
    _refresh_timing(stats, terminal=True)
    return stats


def _checkpoint_state(
    stats: TokenCountStats,
    by_split: dict[str, TokenCountStats],
    *,
    completed_splits: set[str],
    last_checkpoint_docs: int,
) -> dict[str, Any]:
    return {
        "overall": stats.to_checkpoint_state(),
        "by_split": {
            split: split_stats.to_checkpoint_state()
            for split, split_stats in by_split.items()
        },
        "completed_splits": sorted(completed_splits),
        "last_checkpoint_docs": last_checkpoint_docs,
    }


def _load_checkpoint(path: Path) -> tuple[TokenCountStats, dict[str, TokenCountStats], set[str], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    state = payload.get("checkpoint_state")
    if not state:
        raise ValueError(f"Checkpoint JSON does not contain checkpoint_state: {path}")

    if "overall" in state:
        stats = TokenCountStats.from_checkpoint_state(state["overall"])
        by_split = {
            split: TokenCountStats.from_checkpoint_state(split_state)
            for split, split_state in (state.get("by_split") or {}).items()
        }
        completed_splits = set(state.get("completed_splits") or [])
        last_checkpoint_docs = int(state.get("last_checkpoint_docs", stats.documents_processed))
        return stats, by_split, completed_splits, last_checkpoint_docs

    stats = TokenCountStats.from_checkpoint_state(state)
    return stats, {}, set(), stats.documents_processed


def _build_payload(
    args: argparse.Namespace,
    *,
    stats: TokenCountStats,
    by_split: dict[str, TokenCountStats],
    completed_splits: set[str],
    last_checkpoint_docs: int,
    report_path: Optional[str],
    report_json_path: Optional[str],
    report_pdf_path_value: Optional[str],
    status: str,
    dataset_total_rows: Optional[int],
    dataset_split_rows: dict[str, int],
) -> dict[str, Any]:
    _refresh_timing(stats, terminal=status in TERMINAL_STATUSES)
    if status in TERMINAL_STATUSES:
        for split_stats in by_split.values():
            _refresh_timing(split_stats, terminal=True)

    splits = _resolve_splits(args)
    dataset_rows_remaining = (
        max(0, dataset_total_rows - stats.rows_seen) if dataset_total_rows is not None else None
    )
    dataset_completion_percent = (
        (stats.rows_seen / dataset_total_rows) * 100 if dataset_total_rows else None
    )
    extra = {
        "source_kind": "huggingface-dataset" if args.dataset else "input",
        "dataset": args.dataset,
        "config": args.config,
        "revision": args.revision,
        "split": splits[0] if len(splits) == 1 else None,
        "splits": splits if len(splits) > 1 else None,
        "batch_size": args.batch_size,
        "checkpoint_path": report_json_path,
        "dataset_total_rows": dataset_total_rows,
        "dataset_split_rows": dataset_split_rows or None,
        "dataset_rows_remaining": dataset_rows_remaining,
        "dataset_completion_percent": dataset_completion_percent,
        "runner": "token_counter.cli",
    }
    run_metadata = build_run_metadata(
        input_value=args.dataset or args.input,
        data_format="huggingface-dataset" if args.dataset else args.format,
        field=args.field,
        model=args.model,
        add_special_tokens=args.add_special_tokens,
        max_docs=args.max_docs,
        trust_remote_code=args.trust_remote_code,
        report_path=report_path,
        report_json_path=report_json_path,
        report_pdf_path=report_pdf_path_value,
        started_at_epoch=stats.started_at_epoch,
        completed_at_epoch=stats.completed_at_epoch,
        extra=extra,
    )
    payload = build_report_payload(
        run_metadata,
        stats,
        status=status,
        by_split=by_split if by_split else None,
    )
    payload["checkpoint_state"] = _checkpoint_state(
        stats,
        by_split,
        completed_splits=completed_splits,
        last_checkpoint_docs=last_checkpoint_docs,
    )
    return payload


def _resolve_dataset_sizes(
    args: argparse.Namespace,
    splits: list[str],
    *,
    token: Optional[str],
) -> tuple[Optional[int], dict[str, int], list[str]]:
    if not args.dataset:
        return None, {}, []

    split_rows: dict[str, int] = {}
    notes: list[str] = []
    for split in splits:
        size_info = resolve_dataset_split_size(
            args.dataset,
            split,
            config=args.config,
            revision=args.revision,
            token=token,
            trust_remote_code=args.trust_remote_code,
        )
        if size_info.num_examples is not None:
            split_rows[split] = size_info.num_examples
        if size_info.note:
            notes.append(size_info.note)

    total_rows = sum(split_rows.values()) if len(split_rows) == len(splits) else None
    return total_rows, split_rows, notes


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    splits = _resolve_splits(args)
    report_path = args.report or None
    report_json_path = args.report_json or None
    report_pdf_path_value = (
        str(pdf_report_path(Path(args.report))) if args.report_pdf and args.report else None
    )

    auth = load_hf_token()
    stats = TokenCountStats()
    by_split: dict[str, TokenCountStats] = {}
    completed_splits: set[str] = set()
    last_checkpoint_docs = 0
    if args.resume:
        checkpoint_path = Path(report_json_path)
        if checkpoint_path.exists():
            stats, by_split, completed_splits, last_checkpoint_docs = _load_checkpoint(
                checkpoint_path
            )
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if stats.started_at_epoch is None:
        stats.started_at_epoch = time.time()

    if args.dataset and len(splits) == 1 and stats.rows_seen and not by_split:
        by_split[splits[0]] = TokenCountStats.from_checkpoint_state(stats.to_checkpoint_state())

    dataset_total_rows, dataset_split_rows, dataset_notes = _resolve_dataset_sizes(
        args,
        splits,
        token=auth.token,
    )
    progress_total = args.max_docs if args.max_docs is not None else dataset_total_rows
    progress_metric = "documents" if args.max_docs is not None else "rows"
    progress_unit = "docs" if progress_metric == "documents" else "rows"
    source_label = "Dataset" if args.dataset else "Input"
    source_value = args.dataset if args.dataset else args.input

    ui = CountingUI(
        title="Token Counter",
        source_label=source_label,
        source_value=source_value,
        model=args.model,
        total_docs=args.max_docs,
        progress_total=progress_total,
        progress_unit=progress_unit,
        progress_metric=progress_metric,
        dataset_total_rows=dataset_total_rows,
        checkpoint_every=args.checkpoint_every if report_json_path else None,
        report_path=report_path,
        json_path=report_json_path,
        checkpoint_path=report_json_path,
    )

    stop_requested = False

    def handle_signal(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        ui.set_phase("Stopping", f"Signal {signum} received; saving checkpoint", force=True)

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    payload: Optional[dict[str, Any]] = None
    markdown_path = json_path = plot_path = pdf_path = None

    def build_current_payload(status: str) -> dict[str, Any]:
        return _build_payload(
            args,
            stats=stats,
            by_split=by_split,
            completed_splits=completed_splits,
            last_checkpoint_docs=last_checkpoint_docs,
            report_path=report_path,
            report_json_path=report_json_path,
            report_pdf_path_value=report_pdf_path_value,
            status=status,
            dataset_total_rows=dataset_total_rows,
            dataset_split_rows=dataset_split_rows,
        )

    def maybe_checkpoint(force: bool = False) -> None:
        nonlocal last_checkpoint_docs
        if not report_json_path:
            return
        if not force:
            if args.checkpoint_every <= 0:
                return
            if stats.documents_processed - last_checkpoint_docs < args.checkpoint_every:
                return
        last_checkpoint_docs = stats.documents_processed
        checkpoint_payload = build_current_payload("in_progress")
        write_json_report(Path(report_json_path), checkpoint_payload)
        ui.set_checkpoint(last_checkpoint_docs)

    try:
        ui.start()
        ui.set_phase("Authenticating", "Checking HF_TOKEN from environment")
        if auth.token:
            ui.log(
                f"Using HF_TOKEN from {auth.source or 'environment'}",
                style="green",
            )
        for note in dataset_notes:
            ui.log(note, style="yellow")
        if args.resume:
            ui.set_checkpoint_anchor(last_checkpoint_docs)
            ui.log(
                "Resuming from checkpoint: "
                f"docs={stats.documents_processed}, rows_seen={stats.rows_seen}, "
                f"total_tokens={stats.total_tokens}",
                style="cyan",
            )

        ui.set_phase("Loading tokenizer", "Preparing tokenizer")
        tokenizer = _load_tokenizer(
            args.model,
            trust_remote_code=args.trust_remote_code,
            token=auth.token,
        )

        finished_all_sources = True
        if args.dataset:
            for split in splits:
                if stop_requested:
                    finished_all_sources = False
                    break
                if args.max_docs is not None and stats.documents_processed >= args.max_docs:
                    finished_all_sources = False
                    break
                if split in completed_splits:
                    continue

                split_stats = by_split.setdefault(split, TokenCountStats())
                if split_stats.started_at_epoch is None:
                    split_stats.started_at_epoch = time.time()
                ui.set_phase("Streaming dataset", f"Reading split {split}")
                values = _iter_dataset_split_values(
                    args.dataset,
                    split,
                    args.field,
                    config=args.config,
                    revision=args.revision,
                    hf_token=auth.token,
                    trust_remote_code=args.trust_remote_code,
                )
                split_completed = _observe_rows(
                    values,
                    tokenizer=tokenizer,
                    stats=stats,
                    split_stats=split_stats,
                    add_special_tokens=args.add_special_tokens,
                    max_docs=args.max_docs,
                    batch_size=args.batch_size,
                    rows_to_skip=split_stats.rows_seen,
                    ui=ui,
                    checkpoint_callback=maybe_checkpoint,
                    stop_requested=lambda: stop_requested,
                )
                if split_completed:
                    completed_splits.add(split)
                else:
                    finished_all_sources = False
                    break
        else:
            ui.set_phase("Streaming dataset", f"Reading {args.format} stream")
            values = _iter_input_values(
                args.input,
                args.format,
                args.field,
                hf_token=auth.token,
            )
            finished_all_sources = _observe_rows(
                values,
                tokenizer=tokenizer,
                stats=stats,
                split_stats=None,
                add_special_tokens=args.add_special_tokens,
                max_docs=args.max_docs,
                batch_size=args.batch_size,
                rows_to_skip=stats.rows_seen if args.resume else 0,
                ui=ui,
                checkpoint_callback=maybe_checkpoint,
                stop_requested=lambda: stop_requested,
            )

        if stop_requested:
            final_status = "interrupted"
        elif args.max_docs is not None and stats.documents_processed >= args.max_docs:
            final_status = "max_docs_reached"
        elif finished_all_sources:
            final_status = "completed"
        else:
            final_status = "in_progress"

        maybe_checkpoint(force=True)
        ui.set_phase("Writing reports", "Persisting Markdown, JSON, and plots")
        payload = build_current_payload(final_status)
        markdown_path, json_path, plot_path, pdf_path = write_outputs(payload)
    except Exception:
        if report_json_path:
            try:
                error_payload = build_current_payload("error")
                write_json_report(Path(report_json_path), error_payload)
            except Exception:
                pass
        raise
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        ui.stop()

    if payload is not None:
        ui.print_final_summary(
            payload,
            markdown_path=markdown_path,
            json_path=json_path,
            plot_path=plot_path,
            pdf_path=pdf_path,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
