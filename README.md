# Token Counter

CLI to stream JSONL or Parquet datasets, count tokenizer tokens with a base model tokenizer (default: `Qwen/Qwen3-1.7B-Base`), and write a Markdown summary report.

## Installation
```
pip install -r requirements.txt
pip install -e .
```

## Quick start
- Parquet:  
  `python -m token_counter.cli --input data/ptwiki_articles1.parquet --format parquet`

- JSONL (limit to first 500 rows):  
  `python -m token_counter.cli --input data/ptwiki_articles1.jsonl --format jsonl --max-docs 500`

The console script alias is also available after installation: `token-counter --input ...`

## Flags
- `--input` (default `data/ptwiki_articles1.parquet`): dataset path or URL.
- `--format` (`jsonl` | `parquet`, default `parquet`): dataset format.
- `--model` (default `Qwen/Qwen3-1.7B-Base`): tokenizer to load via `transformers.AutoTokenizer`.
- `--field` (default `text`): record field containing the text to tokenize.
- `--add-special-tokens` (default `False`): include special tokens in the length.
- `--max-docs` (int, optional): stop after N documents.
- `--trust-remote-code` (flag): allow custom tokenizer code from the model repo.
- `--report` (default `reports/token_count_report.md`): Markdown report path. Use empty string to skip.

## Report
A Markdown file is generated with summary and performance metrics:
- Documents processed, total tokens, min/avg/max tokens per document.
- Wall-clock time, tokens per second, documents per second.

## Entrypoints
- Module: `python -m token_counter.cli ...`
- Console script: `token-counter ...`
- Wrapper script: `python scripts/count_tokens.py ...`

## Notes
- Uses streaming `datasets.load_dataset(..., streaming=True)` to avoid loading full datasets into memory.
- Parquet and JSONL files can be local or remote URLs.
