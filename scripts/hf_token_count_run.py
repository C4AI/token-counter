#!/usr/bin/env python3
"""
Standalone token counting runner for Hugging Face datasets.

Features:
- Streams dataset rows (no full in-memory load)
- Supports checkpoint/resume
- Writes incremental stats to JSON
- Optional hard stop with --max-docs for quick validation
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer


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
        help="Output JSON path. Default: reports/hf_token_count_result.json",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing --output if present.",
    )
    return parser


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    out_path = Path(args.output)

    stop_requested = False

    def _handle_signal(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"\nSignal {signum} received. Saving checkpoint and exiting...", flush=True)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    checkpoint = load_checkpoint(out_path) if args.resume else {}

    if checkpoint:
        docs = int(checkpoint.get("documents", 0))
        total_tokens = int(checkpoint.get("total_tokens", 0))
        min_tokens = checkpoint.get("min_tokens_per_doc")
        max_tokens = checkpoint.get("max_tokens_per_doc")
        started_at = float(checkpoint.get("started_at_epoch", time.time()))
        skipped = docs
        print(f"Resuming from checkpoint: docs={docs}, total_tokens={total_tokens}", flush=True)
    else:
        docs = 0
        total_tokens = 0
        min_tokens = None
        max_tokens = None
        started_at = time.time()
        skipped = 0
        print("Starting a fresh run", flush=True)

    print(f"Loading tokenizer: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    print(f"Loading dataset stream: {args.dataset} [{args.split}]", flush=True)
    stream = load_dataset(args.dataset, split=args.split, streaming=True)

    seen = 0
    for row in stream:
        seen += 1

        # Resume by skipping already-counted rows.
        if seen <= skipped:
            continue

        text = row.get(args.field)
        if text is None:
            continue

        n_tokens = len(
            tokenizer.encode(str(text), add_special_tokens=bool(args.add_special_tokens))
        )
        docs += 1
        total_tokens += n_tokens
        min_tokens = n_tokens if min_tokens is None else min(min_tokens, n_tokens)
        max_tokens = n_tokens if max_tokens is None else max(max_tokens, n_tokens)

        if args.max_docs is not None and docs >= args.max_docs:
            print(f"Reached --max-docs={args.max_docs}", flush=True)
            stop_requested = True

        if docs % args.progress_every == 0:
            elapsed = time.time() - started_at
            tps = total_tokens / elapsed if elapsed > 0 else 0.0
            dps = docs / elapsed if elapsed > 0 else 0.0
            print(
                (
                    f"progress docs={docs} total_tokens={total_tokens} "
                    f"avg={total_tokens / docs:.2f} min={min_tokens} max={max_tokens} "
                    f"tok_s={tps:.2f} doc_s={dps:.2f}"
                ),
                flush=True,
            )

        if docs % args.checkpoint_every == 0 or stop_requested:
            elapsed = time.time() - started_at
            payload = {
                "dataset": args.dataset,
                "split": args.split,
                "field": args.field,
                "model": args.model,
                "add_special_tokens": bool(args.add_special_tokens),
                "trust_remote_code": bool(args.trust_remote_code),
                "documents": docs,
                "total_tokens": total_tokens,
                "avg_tokens_per_doc": (total_tokens / docs) if docs else 0.0,
                "min_tokens_per_doc": min_tokens,
                "max_tokens_per_doc": max_tokens,
                "started_at_epoch": started_at,
                "wall_time_seconds": elapsed,
                "tokens_per_second": (total_tokens / elapsed) if elapsed > 0 else 0.0,
                "docs_per_second": (docs / elapsed) if elapsed > 0 else 0.0,
                "incomplete": bool(stop_requested),
                "updated_at_epoch": time.time(),
            }
            save_checkpoint(out_path, payload)

        if stop_requested:
            break

    elapsed = time.time() - started_at
    result = {
        "dataset": args.dataset,
        "split": args.split,
        "field": args.field,
        "model": args.model,
        "add_special_tokens": bool(args.add_special_tokens),
        "trust_remote_code": bool(args.trust_remote_code),
        "documents": docs,
        "total_tokens": total_tokens,
        "avg_tokens_per_doc": (total_tokens / docs) if docs else 0.0,
        "min_tokens_per_doc": min_tokens,
        "max_tokens_per_doc": max_tokens,
        "started_at_epoch": started_at,
        "wall_time_seconds": elapsed,
        "tokens_per_second": (total_tokens / elapsed) if elapsed > 0 else 0.0,
        "docs_per_second": (docs / elapsed) if elapsed > 0 else 0.0,
        "incomplete": False,
        "updated_at_epoch": time.time(),
    }
    save_checkpoint(out_path, result)

    print("FINAL_RESULT_START", flush=True)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    print("FINAL_RESULT_END", flush=True)
    print(f"Wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
