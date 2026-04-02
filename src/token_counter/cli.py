"""
CLI entrypoint for counting tokenizer tokens in JSONL or Parquet datasets.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from token_counter.reporting import (
    TokenCountStats,
    build_report_payload,
    build_run_metadata,
    pdf_report_path,
    render_console_summary,
    write_outputs,
)

DEFAULT_MODEL = "Qwen/Qwen3-1.7B-Base"
DEFAULT_FORMAT = "parquet"
DEFAULT_FIELD = "text"
DEFAULT_REPORT_PATH = "reports/token_count_report.md"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count tokenizer tokens in JSONL or Parquet datasets."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input dataset path (local file or remote URL).",
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "parquet"],
        default=DEFAULT_FORMAT,
        help="Dataset format. Supports newline-delimited JSON (.jsonl) or Parquet.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Tokenizer model to use. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--field",
        default=DEFAULT_FIELD,
        help=f"Field in each record containing text to tokenize. Default: {DEFAULT_FIELD}",
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
        help="Maximum number of documents to process (streaming). Defaults to all.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow loading custom tokenizer code from remote repositories.",
    )
    parser.add_argument(
        "--report",
        default=DEFAULT_REPORT_PATH,
        help=f"Path for Markdown report. Use empty string to skip. Default: {DEFAULT_REPORT_PATH}",
    )
    parser.add_argument(
        "--report-json",
        default="",
        help="Path for JSON report. Disabled by default.",
    )
    parser.add_argument(
        "--report-pdf",
        action="store_true",
        help="Generate a PDF report next to the Markdown report, reusing the same base filename.",
    )
    return parser.parse_args(argv)


def _load_stream(input_path: str, data_format: str, field: str) -> Iterator[Any]:
    data_format = data_format.lower()
    path = Path(input_path)
    if data_format not in {"jsonl", "parquet"}:
        raise ValueError(f"Unsupported format '{data_format}'. Use jsonl or parquet.")

    if not input_path.startswith(("http://", "https://", "hf://")) and not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    loader = "json" if data_format == "jsonl" else "parquet"
    dataset = load_dataset(loader, data_files=input_path, streaming=True)
    iterable = dataset["train"]

    for example in iterable:
        if field not in example:
            raise KeyError(f"Field '{field}' not found in record: keys={list(example.keys())}")
        yield example[field]


def _count_tokens(
    text_stream: Iterable[Any],
    tokenizer,
    add_special_tokens: bool = False,
    max_docs: Optional[int] = None,
) -> TokenCountStats:
    stats = TokenCountStats()
    stats.started_at_epoch = time.time()
    start = time.perf_counter()

    progress = tqdm(desc="Counting tokens", unit="docs", total=max_docs)
    try:
        for raw_value in text_stream:
            stats.rows_seen += 1

            if raw_value is None:
                stats.null_field_rows += 1
                continue

            if isinstance(raw_value, str):
                text = raw_value
            else:
                stats.non_string_rows_coerced += 1
                text = str(raw_value)

            if text == "":
                stats.empty_text_rows += 1

            token_length = len(tokenizer.encode(text, add_special_tokens=add_special_tokens))
            stats.observe_document(text=text, token_length=token_length)
            progress.update(1)

            if max_docs is not None and stats.documents_processed >= max_docs:
                break
    finally:
        progress.close()
        stats.wall_time = time.perf_counter() - start
        stats.completed_at_epoch = time.time()

    return stats


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    if args.report_pdf and not args.report:
        raise ValueError("--report-pdf requires --report to be set to a Markdown path.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )

    text_stream = _load_stream(args.input, args.format, args.field)
    stats = _count_tokens(
        text_stream,
        tokenizer=tokenizer,
        add_special_tokens=args.add_special_tokens,
        max_docs=args.max_docs,
    )

    report_path = args.report or None
    report_json_path = args.report_json or None
    report_pdf_path_value = (
        str(pdf_report_path(Path(args.report))) if args.report_pdf and args.report else None
    )
    run_metadata = build_run_metadata(
        input_value=args.input,
        data_format=args.format,
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
    )
    payload = build_report_payload(run_metadata, stats)
    markdown_path, json_path, plot_path, pdf_path = write_outputs(payload)

    for line in render_console_summary(payload):
        print(line)
    if markdown_path:
        print(f"Report written to: {markdown_path}")
    if plot_path:
        print(f"Distribution plot written to: {plot_path}")
    if pdf_path:
        print(f"PDF report written to: {pdf_path}")
    if json_path:
        print(f"JSON report written to: {json_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
