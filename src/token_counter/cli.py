"""
CLI entrypoint for counting tokenizer tokens in JSONL or Parquet datasets.

The tool streams datasets via `datasets.load_dataset(..., streaming=True)` so it
can handle large files without loading everything into memory. Results are
summarized into a Markdown report for later inspection.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# Defaults to mirror the original script behavior.
DEFAULT_MODEL = "Qwen/Qwen3-1.7B-Base"
DEFAULT_INPUT = "data/ptwiki_articles1.parquet"
DEFAULT_FORMAT = "parquet"
DEFAULT_FIELD = "text"
DEFAULT_REPORT_PATH = "reports/token_count_report.md"


@dataclass
class TokenCountStats:
    documents: int = 0
    total_tokens: int = 0
    min_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    wall_time: float = 0.0

    @property
    def average_tokens(self) -> float:
        return self.total_tokens / self.documents if self.documents else 0.0

    @property
    def tokens_per_second(self) -> float:
        return self.total_tokens / self.wall_time if self.wall_time > 0 else 0.0

    @property
    def docs_per_second(self) -> float:
        return self.documents / self.wall_time if self.wall_time > 0 else 0.0


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count tokenizer tokens in JSONL or Parquet datasets."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Input dataset path (local file or remote URL). Default: {DEFAULT_INPUT}",
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
    return parser.parse_args(argv)


def _load_stream(input_path: str, data_format: str, field: str) -> Iterator[str]:
    """
    Stream records from the dataset, yielding the target field as text.
    """
    data_format = data_format.lower()
    path = Path(input_path)
    if data_format not in {"jsonl", "parquet"}:
        raise ValueError(f"Unsupported format '{data_format}'. Use jsonl or parquet.")

    if not input_path.startswith(("http://", "https://")) and not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    loader = "json" if data_format == "jsonl" else "parquet"
    dataset = load_dataset(loader, data_files=input_path, streaming=True)
    iterable = dataset["train"]

    for example in iterable:
        if field not in example:
            raise KeyError(f"Field '{field}' not found in record: keys={list(example.keys())}")
        value = example[field]
        if value is None:
            continue
        yield str(value)


def _count_tokens(
    text_stream: Iterable[str],
    tokenizer,
    add_special_tokens: bool = False,
    max_docs: Optional[int] = None,
) -> TokenCountStats:
    stats = TokenCountStats()
    start = time.time()

    progress = tqdm(text_stream, desc="Counting tokens", unit="docs", total=max_docs)
    for idx, text in enumerate(progress, start=1):
        encoded = tokenizer.encode(text, add_special_tokens=add_special_tokens)
        length = len(encoded)

        stats.documents += 1
        stats.total_tokens += length
        stats.min_tokens = length if stats.min_tokens is None else min(stats.min_tokens, length)
        stats.max_tokens = length if stats.max_tokens is None else max(stats.max_tokens, length)

        if max_docs is not None and idx >= max_docs:
            break

    stats.wall_time = time.time() - start
    return stats


def _write_report(args: argparse.Namespace, stats: TokenCountStats) -> Optional[Path]:
    if not args.report:
        return None

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Token Count Report",
        "",
        f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Input: {args.input}",
        f"- Format: {args.format}",
        f"- Field: {args.field}",
        f"- Model: {args.model}",
        f"- Add special tokens: {bool(args.add_special_tokens)}",
        f"- Max docs: {args.max_docs if args.max_docs is not None else 'All'}",
        f"- Trust remote code: {bool(args.trust_remote_code)}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Documents processed | {stats.documents:,} |",
        f"| Total tokens | {stats.total_tokens:,} |",
        f"| Avg tokens / doc | {stats.average_tokens:,.2f} |",
        f"| Min tokens / doc | {stats.min_tokens if stats.min_tokens is not None else 'n/a'} |",
        f"| Max tokens / doc | {stats.max_tokens if stats.max_tokens is not None else 'n/a'} |",
        "",
        "## Performance",
        "",
        f"- Wall time (s): {stats.wall_time:.2f}",
        f"- Tokens per second: {stats.tokens_per_second:,.2f}",
        f"- Docs per second: {stats.docs_per_second:,.2f}",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

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

    report_path = _write_report(args, stats)

    # Console summary
    print(f"Documents processed: {stats.documents}")
    print(f"Total tokens: {stats.total_tokens}")
    print(f"Avg tokens/doc: {stats.average_tokens:.2f}")
    print(f"Min tokens/doc: {stats.min_tokens if stats.min_tokens is not None else 'n/a'}")
    print(f"Max tokens/doc: {stats.max_tokens if stats.max_tokens is not None else 'n/a'}")
    print(f"Tokens/sec: {stats.tokens_per_second:.2f}")
    print(f"Docs/sec: {stats.docs_per_second:.2f}")
    if report_path:
        print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
