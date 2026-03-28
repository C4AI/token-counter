# Token Counter

CLI to stream JSONL or Parquet datasets, count tokenizer tokens with a base model tokenizer (default: `Qwen/Qwen3-1.7B-Base`), and write a distribution-focused Markdown report with optional JSON output.

## Installation
```
pip install -r requirements.txt
pip install -e .
```

For PDF export support only, you can install the optional extra:
```
pip install -e ".[pdf]"
```

## Quick start
- Parquet:  
  `python -m token_counter.cli --input data/your_file_dataset.parquet --format parquet`

- JSONL (limit to first 500 rows):  
  `python -m token_counter.cli --input data/your_file_dataset.jsonl --format jsonl --max-docs 500`

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
- `--report-json` (optional): structured JSON report path. Disabled by default.
- `--report-pdf` (flag): generate a PDF next to the Markdown report using the same base filename.

## Report
The CLI builds one canonical report payload and can render it to Markdown and JSON.

Markdown sections:
- Run context with tokenizer settings, timestamps, report paths, and package versions.
- Distribution snapshot with documents processed, mean, median, IQR, P95, and P99.
- Distribution histogram rendered as a single PNG chart embedded in Markdown with fixed token buckets.
- Data quality with skipped/null/empty/coerced row counts.
- Performance with wall-clock time and throughput metrics.

JSON report:
- Includes `schema_version`, `status`, `run_metadata`, `summary_stats`, `distribution_stats`, `data_quality_stats`, and `performance_stats`.
- Keeps richer distribution details such as percentiles, IQR, histogram buckets, min/max, and standard deviation for programmatic comparison.

## PDF export
You can convert a generated Markdown report to PDF with the cross-platform Python exporter. It renders the Markdown to HTML and writes the PDF directly from Python, preserving tables and the distribution plot image.

During counting, you can ask the CLI to emit the PDF automatically from the Markdown report path:

```bash
python -m token_counter.cli --input data/your_file_dataset.parquet --format parquet --report reports/token_count_report.md --report-pdf
```

That generates `reports/token_count_report.pdf` automatically.

```bash
python -m token_counter.pdf_export --input reports/gutenberg_distribution_report.md
```

To choose the output file explicitly:

```bash
python -m token_counter.pdf_export --input reports/gutenberg_distribution_report.md --output reports/gutenberg_distribution_report.pdf
```

The package also exposes:
- Module: `python -m token_counter.pdf_export ...`
- Console script: `token-counter-report-pdf --input ...`
- Wrapper script: `python scripts/export_report_pdf.py --input ...`

## Entrypoints
- Module: `python -m token_counter.cli ...`
- Console script: `token-counter ...`
- Wrapper script: `python scripts/count_tokens.py ...`

## Notes
- Uses streaming `datasets.load_dataset(..., streaming=True)` to avoid loading full datasets into memory.
- Parquet and JSONL files can be local files, remote URLs, or remote glob patterns.
