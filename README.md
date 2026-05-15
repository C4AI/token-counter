# Token Counter

Universal CLI for counting tokenizer tokens in datasets and generating reports.

The main goal is to answer one question clearly: how many tokens are in this
dataset for this tokenizer? The same run also produces useful distribution,
quality, and performance insights.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

For gated or private Hugging Face assets, place your token in the environment or
in a local `.env` file:

```bash
HF_TOKEN=hf_xxx
```

PDF export support is optional:

```bash
pip install -e ".[pdf]"
```

## Usage

Local or remote Parquet:

```bash
token-counter --input data/dataset.parquet --format parquet --field text
```

JSONL:

```bash
token-counter --input data/dataset.jsonl --format jsonl --field text
```

Hugging Face `hf://` Parquet glob:

```bash
token-counter \
  --input "hf://datasets/<org>/<dataset>@main/path/part-*.parquet" \
  --format parquet \
  --field text
```

Hugging Face dataset streaming:

```bash
token-counter \
  --dataset <org>/<dataset> \
  --split train \
  --field text
```

Multiple Hugging Face splits:

```bash
token-counter \
  --dataset felipeoes/br_legislation \
  --revision refs/convert/parquet \
  --config default \
  --splits valid invalid \
  --field text_markdown \
  --model Qwen/Qwen3-1.7B-Base
```

The module entrypoint is equivalent:

```bash
python -m token_counter.cli --dataset <org>/<dataset> --split train --field text
```

## Reports

By default, the CLI writes:

- Markdown: `reports/token_count_report.md`
- JSON/checkpoint: `reports/token_count_report.json`
- Distribution plot next to the Markdown report

The Markdown report starts with a summary table that includes `Total tokens`.
The JSON report stores the same value at `summary_stats.total_tokens`.

Useful report flags:

```bash
token-counter --dataset <org>/<dataset> --report reports/count.md
token-counter --dataset <org>/<dataset> --report-json reports/count.json
token-counter --dataset <org>/<dataset> --report "" --report-json reports/count.json
token-counter --dataset <org>/<dataset> --report-pdf
```

For long runs, the JSON report is also the checkpoint file:

```bash
token-counter --dataset <org>/<dataset> --report-json reports/count.json
token-counter --dataset <org>/<dataset> --report-json reports/count.json --resume
```

## Main Flags

- `--input`: JSONL/Parquet path, URL, or `hf://` glob
- `--dataset`: Hugging Face dataset id
- `--format`: `jsonl` or `parquet` for `--input`
- `--config`: Hugging Face dataset config/name
- `--revision`: Hugging Face dataset revision
- `--split`: one Hugging Face split, repeatable
- `--splits`: multiple Hugging Face splits in one flag
- `--field`: text field to tokenize, default `text`
- `--model`: tokenizer model, default `Qwen/Qwen3-1.7B-Base`
- `--batch-size`: documents per tokenizer batch, default `256`
- `--max-docs`: stop after N processed documents
- `--add-special-tokens`: include tokenizer special tokens
- `--trust-remote-code`: allow custom tokenizer or dataset code
- `--resume`: resume from `--report-json`
- `--checkpoint-every`: checkpoint every N processed documents, default `10000`
- `--report`: Markdown report path, empty string disables Markdown
- `--report-json`: JSON report/checkpoint path, empty string disables JSON
- `--report-pdf`: generate a PDF next to the Markdown report

## Output Shape

JSON reports include:

- `schema_version`, `status`, `run_metadata`
- `summary_stats.total_tokens`
- `distribution_stats`
- `data_quality_stats`
- `performance_stats`
- `by_split` when multiple Hugging Face splits are processed
- `checkpoint_state` for `--resume`

Markdown reports include:

- Summary with total tokens
- Run context
- Optional by-split table
- Distribution snapshot and histogram
- Data quality metrics
- Performance metrics
