#!/usr/bin/env python3
"""
Standalone token counting runner for Hugging Face datasets.

Features:
- Streams dataset rows without full in-memory loading
- Supports checkpoint/resume
- Persists rich statistics in the checkpoint JSON
- Generates Markdown reports and optional PDF exports
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import load_dataset
from huggingface_hub.utils import close_session
from transformers import AutoTokenizer

from token_counter.hf_auth import ensure_hf_auth
from token_counter.hf_dataset_meta import resolve_dataset_split_size
from token_counter.reporting import (
    TokenCountStats,
    build_report_payload,
    build_run_metadata,
    pdf_report_path,
    write_outputs,
)
from token_counter.terminal_ui import CountingUI


TERMINAL_STATUSES = {"completed", "max_docs_reached", "interrupted", "error"}
COMPLETE_STATUSES = {"completed", "max_docs_reached"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Count tokens from a Hugging Face dataset with checkpoint/resume."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Hugging Face dataset id (e.g. ggg-llms-team/aroeira-bertimbau-filter).",
    )
    parser.add_argument("--split", default="train", help="Dataset split. Default: train.")
    parser.add_argument("--field", default="text", help="Text field name. Default: text.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-1.7B-Base",
        help="Tokenizer model id. Default: Qwen/Qwen3-1.7B-Base.",
    )
    parser.add_argument(
        "--add-special-tokens",
        action="store_true",
        help="Count with tokenizer special tokens.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom code from tokenizer repo.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Stop after N processed documents.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10_000,
        help="Print progress every N processed documents. Default: 10000.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10_000,
        help="Save checkpoint every N processed documents. Default: 10000.",
    )
    parser.add_argument(
        "--output",
        default="reports/hf_token_count_result.json",
        help="Output JSON checkpoint path. Default: reports/hf_token_count_result.json",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Markdown report path. Defaults to the --output path with a .md suffix. Use empty string to skip.",
    )
    parser.add_argument(
        "--report-pdf",
        action="store_true",
        help="Generate a PDF report next to the Markdown report.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing --output if present.",
    )
    parser.add_argument(
        "--max-stream-restarts",
        type=int,
        default=10,
        help="How many times to reopen the dataset stream after a retryable read error. Default: 10.",
    )
    parser.add_argument(
        "--retry-wait-seconds",
        type=float,
        default=5.0,
        help="Seconds to wait before reopening the dataset stream after a retryable error. Default: 5.",
    )
    return parser


def resolve_report_path(report_arg: Optional[str], output_path: str) -> Optional[str]:
    if report_arg is None:
        return str(Path(output_path).with_suffix(".md"))
    return report_arg or None


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def is_retryable_stream_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return isinstance(exc, RuntimeError) and "client has been closed" in message


def load_stats_from_checkpoint(payload: dict[str, Any]) -> TokenCountStats:
    checkpoint_state = payload.get("checkpoint_state")
    if checkpoint_state:
        return TokenCountStats.from_checkpoint_state(checkpoint_state)

    if payload:
        raise ValueError(
            "Existing checkpoint is from the legacy lightweight format and cannot resume rich distribution metrics accurately. "
            "Start with a new --output path or remove the old checkpoint file."
        )

    stats = TokenCountStats()
    stats.started_at_epoch = time.time()
    return stats


def build_hf_payload(
    args: argparse.Namespace,
    *,
    stats: TokenCountStats,
    out_path: Path,
    report_path: Optional[str],
    status: str,
    restart_attempts: int,
    last_checkpoint_docs: int,
    dataset_total_rows: Optional[int],
    last_error: Optional[str] = None,
) -> dict[str, Any]:
    now_epoch = time.time()
    if stats.started_at_epoch is None:
        stats.started_at_epoch = now_epoch
    stats.wall_time = max(0.0, now_epoch - stats.started_at_epoch)
    stats.completed_at_epoch = now_epoch if status in TERMINAL_STATUSES else None

    report_pdf_path_value = (
        str(pdf_report_path(Path(report_path))) if args.report_pdf and report_path else None
    )
    dataset_rows_remaining = (
        max(0, dataset_total_rows - stats.rows_seen) if dataset_total_rows is not None else None
    )
    dataset_completion_percent = (
        ((stats.rows_seen / dataset_total_rows) * 100) if dataset_total_rows else None
    )
    run_metadata = build_run_metadata(
        input_value=args.dataset,
        data_format="huggingface-dataset",
        field=args.field,
        model=args.model,
        add_special_tokens=args.add_special_tokens,
        max_docs=args.max_docs,
        trust_remote_code=args.trust_remote_code,
        report_path=report_path,
        report_json_path=str(out_path),
        report_pdf_path=report_pdf_path_value,
        started_at_epoch=stats.started_at_epoch,
        completed_at_epoch=stats.completed_at_epoch,
        extra={
            "split": args.split,
            "checkpoint_path": str(out_path),
            "dataset_total_rows": dataset_total_rows,
            "dataset_rows_remaining": dataset_rows_remaining,
            "dataset_completion_percent": dataset_completion_percent,
            "runner": "hf_token_count_run",
        },
    )
    payload = build_report_payload(run_metadata, stats, status=status)
    payload.update(
        {
            "dataset": args.dataset,
            "split": args.split,
            "field": args.field,
            "model": args.model,
            "add_special_tokens": bool(args.add_special_tokens),
            "trust_remote_code": bool(args.trust_remote_code),
            "documents": stats.documents_processed,
            "rows_seen": stats.rows_seen,
            "total_tokens": stats.total_tokens,
            "avg_tokens_per_doc": stats.average_tokens,
            "min_tokens_per_doc": stats.min_tokens,
            "max_tokens_per_doc": stats.max_tokens,
            "started_at_epoch": stats.started_at_epoch,
            "wall_time_seconds": stats.wall_time,
            "tokens_per_second": stats.tokens_per_second,
            "docs_per_second": stats.docs_per_second,
            "incomplete": status not in COMPLETE_STATUSES,
            "restart_attempts": restart_attempts,
            "last_checkpoint_docs": last_checkpoint_docs,
            "dataset_total_rows": dataset_total_rows,
            "dataset_rows_remaining": dataset_rows_remaining,
            "dataset_completion_percent": dataset_completion_percent,
            "updated_at_epoch": now_epoch,
            "checkpoint_state": stats.to_checkpoint_state(),
        }
    )
    if last_error:
        payload["last_error"] = last_error
    return payload


def main() -> int:
    args = build_parser().parse_args()
    out_path = Path(args.output)
    report_path = resolve_report_path(args.report, args.output)
    if args.report_pdf and not report_path:
        raise ValueError("--report-pdf requires a Markdown report path.")

    checkpoint = load_checkpoint(out_path) if args.resume else {}
    dataset_total_rows = checkpoint.get("dataset_total_rows")
    auth = ensure_hf_auth()
    dataset_size_info = None
    if dataset_total_rows is None:
        dataset_size_info = resolve_dataset_split_size(
            args.dataset,
            args.split,
            token=auth.token,
            trust_remote_code=args.trust_remote_code,
        )
        dataset_total_rows = dataset_size_info.num_examples

    progress_total = dataset_total_rows if args.max_docs is None else args.max_docs
    progress_metric = "rows" if dataset_total_rows is not None and args.max_docs is None else "documents"
    progress_unit = "rows" if progress_metric == "rows" else "docs"
    ui = CountingUI(
        title="HF Token Counter",
        source_label="Dataset",
        source_value=f"{args.dataset} [{args.split}]",
        model=args.model,
        total_docs=args.max_docs,
        progress_total=progress_total,
        progress_unit=progress_unit,
        progress_metric=progress_metric,
        dataset_total_rows=dataset_total_rows,
        checkpoint_every=args.checkpoint_every,
        report_path=report_path,
        json_path=str(out_path),
        checkpoint_path=str(out_path),
    )
    ui.start()

    stop_requested = False
    limit_reached = False
    completed = False
    terminal_error: Optional[Exception] = None

    def _handle_signal(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        ui.set_phase("Stopping", f"Signal {signum} received; saving checkpoint", force=True)
        ui.log(f"Signal {signum} received. Saving checkpoint and exiting...", style="yellow")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    if checkpoint:
        stats = load_stats_from_checkpoint(checkpoint)
        restart_attempts = int(checkpoint.get("restart_attempts", 0))
        last_checkpoint_docs = int(checkpoint.get("last_checkpoint_docs", stats.documents_processed))
        last_error = checkpoint.get("last_error")
        ui.set_restart_attempts(restart_attempts)
        ui.set_checkpoint_anchor(last_checkpoint_docs)
        ui.update(stats, force=True)
        ui.log(
            "Resuming from checkpoint: "
            f"docs={stats.documents_processed}, rows_seen={stats.rows_seen}, total_tokens={stats.total_tokens}",
            style="cyan",
        )
    else:
        stats = TokenCountStats()
        stats.started_at_epoch = time.time()
        restart_attempts = 0
        last_checkpoint_docs = 0
        last_error = None
        ui.update(stats, force=True)
        ui.log("Starting a fresh run", style="cyan")

    try:
        ui.set_phase("Authenticating", "Checking HF_TOKEN from environment")
        if auth.token:
            ui.log(
                f"Authenticated to Hugging Face with HF_TOKEN from {auth.source or 'environment'}",
                style="green",
            )
        if dataset_total_rows is not None:
            size_message = (
                f"Split size detected: {dataset_total_rows:,} rows".replace(",", ".")
            )
            if dataset_size_info and dataset_size_info.source:
                size_message += f" ({dataset_size_info.source})"
            ui.log(size_message, style="green")
        elif dataset_size_info and dataset_size_info.note:
            ui.log(dataset_size_info.note, style="yellow")

        ui.set_phase("Loading tokenizer", "Preparing tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
            token=auth.token,
        )

        while not stop_requested and not limit_reached and terminal_error is None:
            ui.set_phase("Streaming dataset", "Reading streaming split")
            ui.set_restart_attempts(restart_attempts)
            if restart_attempts:
                ui.log(
                    f"Reopening dataset stream after retryable error (attempt {restart_attempts}/{args.max_stream_restarts})",
                    style="yellow",
                )

            ui.log(f"Loading dataset stream: {args.dataset} [{args.split}]", style="cyan")

            try:
                stream = load_dataset(
                    args.dataset,
                    split=args.split,
                    streaming=True,
                    token=auth.token,
                )
                seen_in_stream = 0

                for row in stream:
                    seen_in_stream += 1

                    if seen_in_stream <= stats.rows_seen:
                        continue

                    stats.rows_seen += 1

                    raw_value = row.get(args.field)
                    if raw_value is None:
                        stats.null_field_rows += 1
                        ui.update(stats)
                        continue

                    if isinstance(raw_value, str):
                        text = raw_value
                    else:
                        stats.non_string_rows_coerced += 1
                        text = str(raw_value)

                    if text == "":
                        stats.empty_text_rows += 1

                    token_length = len(
                        tokenizer.encode(text, add_special_tokens=bool(args.add_special_tokens))
                    )
                    stats.observe_document(text=text, token_length=token_length)
                    last_error = None
                    ui.clear_error()
                    ui.update(stats)

                    if args.max_docs is not None and stats.documents_processed >= args.max_docs:
                        ui.log(f"Reached --max-docs={args.max_docs}", style="green")
                        limit_reached = True

                    if (
                        stats.documents_processed > 0
                        and args.progress_every > 0
                        and stats.documents_processed % args.progress_every == 0
                    ):
                        ui.set_message(
                            f"Processed {stats.documents_processed:,} docs".replace(",", ".")
                        )

                    if stats.documents_processed > 0 and (
                        (
                            args.checkpoint_every > 0
                            and (stats.documents_processed - last_checkpoint_docs) >= args.checkpoint_every
                        )
                        or stop_requested
                        or limit_reached
                    ):
                        checkpoint_status = (
                            "max_docs_reached"
                            if limit_reached
                            else "interrupted"
                            if stop_requested
                            else "in_progress"
                        )
                        save_checkpoint(
                            out_path,
                            build_hf_payload(
                                args,
                                stats=stats,
                                out_path=out_path,
                                report_path=report_path,
                                status=checkpoint_status,
                                restart_attempts=restart_attempts,
                                last_checkpoint_docs=stats.documents_processed,
                                dataset_total_rows=dataset_total_rows,
                                last_error=last_error,
                            ),
                        )
                        last_checkpoint_docs = stats.documents_processed
                        ui.set_checkpoint(stats.documents_processed)

                    if stop_requested or limit_reached:
                        break

                if not stop_requested and not limit_reached:
                    completed = True
                break

            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                ui.set_error(last_error)
                retryable = is_retryable_stream_error(exc)
                can_retry = retryable and restart_attempts < args.max_stream_restarts
                save_checkpoint(
                    out_path,
                    build_hf_payload(
                        args,
                        stats=stats,
                        out_path=out_path,
                        report_path=report_path,
                        status="retrying" if can_retry else "error",
                        restart_attempts=restart_attempts,
                        last_checkpoint_docs=stats.documents_processed,
                        dataset_total_rows=dataset_total_rows,
                        last_error=last_error,
                    ),
                )
                last_checkpoint_docs = stats.documents_processed
                ui.set_checkpoint(stats.documents_processed)

                if not can_retry:
                    ui.log(f"Unrecoverable stream error: {last_error}", style="bold red")
                    terminal_error = exc
                    break

                restart_attempts += 1
                ui.set_restart_attempts(restart_attempts)
                ui.log(
                    (
                        f"Retryable stream error: {last_error}. "
                        f"Waiting {args.retry_wait_seconds:.1f}s before retry."
                    ),
                    style="yellow",
                )
                close_session()
                time.sleep(args.retry_wait_seconds)
    finally:
        ui.update(stats, force=True)

    if completed:
        final_status = "completed"
    elif limit_reached:
        final_status = "max_docs_reached"
    elif stop_requested:
        final_status = "interrupted"
    elif terminal_error is not None:
        final_status = "error"
    else:
        final_status = "in_progress"

    try:
        result = build_hf_payload(
            args,
            stats=stats,
            out_path=out_path,
            report_path=report_path,
            status=final_status,
            restart_attempts=restart_attempts,
            last_checkpoint_docs=last_checkpoint_docs,
            dataset_total_rows=dataset_total_rows,
            last_error=last_error,
        )
        ui.set_phase("Writing reports", "Persisting checkpoint and artifacts", force=True)
        save_checkpoint(out_path, result)
        markdown_path, json_path, plot_path, pdf_path = write_outputs(result)
    finally:
        ui.stop()

    ui.print_final_summary(
        result,
        markdown_path=markdown_path,
        json_path=json_path,
        plot_path=plot_path,
        pdf_path=pdf_path,
    )

    print("FINAL_RESULT_START", flush=True)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    print("FINAL_RESULT_END", flush=True)
    print(f"Wrote {out_path}", flush=True)

    if terminal_error is not None:
        raise terminal_error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
