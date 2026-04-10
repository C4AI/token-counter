# Token Counter

CLI and runner utilities to stream datasets, count tokenizer tokens with a base model tokenizer (default: `Qwen/Qwen3-1.7B-Base`), and generate rich reports with distribution metrics.

The terminal experience is powered by `rich`, with live progress, continuously updated counters, and a more legible final summary.

## Installation
```bash
pip install -r requirements.txt
pip install -e .
```

For gated or private Hugging Face assets, place your token in a local `.env` file:
```bash
HF_TOKEN=hf_xxx
```

The CLI and the Hugging Face runner automatically load `HF_TOKEN` from `.env`, log in to the Hub, and reuse that token for `datasets` and `transformers`.

For PDF export support only, you can install the optional extra:
```bash
pip install -e ".[pdf]"
```

## Standard CLI
- Parquet:  
  `python -m token_counter.cli --input data/your_file_dataset.parquet --format parquet`

- JSONL (limit to first 500 docs):  
  `python -m token_counter.cli --input data/your_file_dataset.jsonl --format jsonl --max-docs 500`

- Hugging Face sharded Parquet:  
  `python -m token_counter.cli --input "hf://datasets/<org>/<dataset>@main/<folder>/part-*.parquet" --format parquet`

The console script alias is also available after installation: `token-counter --input ...`

## Hugging Face dataset runner with checkpoint
Use the standalone runner when you want resume/checkpoint behavior for a Hub dataset id such as `costadev00/wikipedia-pt-br-extract-cpt-2048`.

```bash
.venv/bin/python scripts/hf_token_count_run.py \
  --dataset costadev00/wikipedia-pt-br-extract-cpt-2048 \
  --split train \
  --field text \
  --output reports/wikipedia_pt_br_extract_cpt_2048_token_count.json \
  --resume
```

By default this runner now:
- writes a rich JSON checkpoint/report to `--output`
- writes a Markdown report next to it using the same basename and `.md`
- can generate a PDF with `--report-pdf`

To skip Markdown generation:
```bash
.venv/bin/python scripts/hf_token_count_run.py --dataset <org/dataset> --report ""
```

## Report outputs
The project builds one canonical payload and can render it to Markdown, JSON, PNG, and PDF.

Markdown sections:
- Run context with timestamps, tokenizer settings, report paths, and package versions
- Distribution snapshot with mean, median, IQR, P95, P99, and standard deviation
- Distribution histogram rendered as a PNG and a Markdown table by token bucket
- Data quality metrics such as rows seen, skipped, null, empty, and coerced values
- Performance metrics such as wall time, docs/sec, chars/sec, and tokens/sec

JSON payload:
- `schema_version`, `status`, `run_metadata`
- `summary_stats`, `distribution_stats`, `data_quality_stats`, `performance_stats`
- for the Hugging Face runner, `checkpoint_state` is also stored so `--resume` can continue with full rich metrics

## Main CLI flags
- `--input` (required): dataset path, URL, or glob pattern
- `--format` (`jsonl` | `parquet`, default `parquet`)
- `--model` (default `Qwen/Qwen3-1.7B-Base`)
- `--field` (default `text`)
- `--add-special-tokens`
- `--max-docs`
- `--trust-remote-code`
- `--report` (default `reports/token_count_report.md`)
- `--report-json` (optional structured JSON report path)
- `--report-pdf` (generate PDF next to the Markdown report)

## Hugging Face runner flags
- `--dataset` (required): Hub dataset id
- `--split` (default `train`)
- `--field` (default `text`)
- `--output`: JSON checkpoint/report path
- `--report`: Markdown path. Defaults to the same basename as `--output`
- `--report-pdf`: generate PDF next to the Markdown report
- `--resume`: resume from the checkpoint JSON
- `--checkpoint-every`: save checkpoint every N processed documents
- `--progress-every`: print progress every N processed documents

## PDF export only
You can convert an existing Markdown report to PDF directly:

```bash
python -m token_counter.pdf_export --input reports/token_count_report.md
```

Or via the wrapper script:

```bash
python scripts/export_report_pdf.py --input reports/token_count_report.md
```

If `xhtml2pdf` pulls native `cairo` dependencies on your machine, you may need the system `cairo` toolchain installed before `pip install -r requirements.txt` succeeds.

## Entrypoints
- Module: `python -m token_counter.cli ...`
- Console script: `token-counter ...`
- Wrapper script: `python scripts/count_tokens.py ...`
- PDF module: `python -m token_counter.pdf_export ...`
- PDF console script: `token-counter-report-pdf ...`
- PDF wrapper script: `python scripts/export_report_pdf.py ...`

## Notes
- Uses streaming `datasets.load_dataset(..., streaming=True)` to avoid loading full datasets into memory
- Parquet and JSONL files can be local files, remote URLs, or `hf://` glob patterns
- PDF export requires `markdown` and `xhtml2pdf`
