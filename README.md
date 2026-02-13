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

- Hugging Face sharded Parquet (all matching files):  
  `python -m token_counter.cli --input "hf://datasets/g4me/corpus-carolina-v2@main/data/corpus/part-*.parquet" --format parquet`

The console script alias is also available after installation: `token-counter --input ...`

## Hugging Face datasets
You can stream files directly from the Hub by passing a remote path or glob pattern to `--input`.

- Public dataset shards (glob):  
  `python -m token_counter.cli --input "hf://datasets/<org>/<dataset>@main/<folder>/part-*.parquet" --format parquet`

- Private dataset shards (PowerShell):  
  `$env:HF_TOKEN="hf_xxx"`  
  `python -m token_counter.cli --input "hf://datasets/<org>/<dataset>@main/<folder>/part-*.parquet" --format parquet`

If the text field is not named `text`, pass `--field <column_name>`.

## Flags
- `--input` (required): dataset path, URL, or glob pattern.
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
- Parquet and JSONL files can be local files, remote URLs, or remote glob patterns.
