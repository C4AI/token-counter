# Token Counter

Count tokenizer tokens in local files or Hugging Face datasets, then generate a
Markdown/JSON report with the total token count and distribution insights.

The primary output is always:

- Markdown summary: `Total tokens`
- JSON field: `summary_stats.total_tokens`
- Terminal summary: `Total tokens`

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

For private or gated Hugging Face datasets, set `HF_TOKEN` in the environment or
in a local `.env` file:

```bash
HF_TOKEN=hf_xxx
```

Optional PDF support:

```bash
pip install -e ".[pdf]"
```

## Quick Start

Run a small sample from a Hugging Face dataset:

```bash
token-counter \
  --dataset costadev00/gutenberg-project-tokenweaver-cpt-2048 \
  --config default \
  --split train \
  --field text \
  --model Qwen/Qwen3-1.7B-Base \
  --max-docs 100 \
  --report reports/gutenberg_sample.md \
  --report-json reports/gutenberg_sample.json
```

## Common Commands

Single local Parquet:

```bash
token-counter \
  --input data/dataset.parquet \
  --format parquet \
  --field text
```

Multiple local Parquet shards:

```bash
token-counter \
  --input "data/shards/*.parquet" \
  --format parquet \
  --field text
```

Use quotes around globs so the CLI receives the pattern.

Remote or Hugging Face Parquet glob:

```bash
token-counter \
  --input "hf://datasets/<org>/<dataset>@main/path/*.parquet" \
  --format parquet \
  --field text
```

Hugging Face dataset:

```bash
token-counter \
  --dataset <org>/<dataset> \
  --split train \
  --field text
```

Hugging Face dataset with config/revision:

```bash
token-counter \
  --dataset costadev00/gutenberg-project-tokenweaver-cpt-2048 \
  --config default \
  --split train \
  --field text \
  --model Qwen/Qwen3-1.7B-Base \
  --report reports/gutenberg_project_tokenweaver_cpt_2048.md \
  --report-json reports/gutenberg_project_tokenweaver_cpt_2048.json
```

Multiple Hugging Face splits:

```bash
token-counter \
  --dataset felipeoes/br_legislation \
  --revision refs/convert/parquet \
  --config default \
  --splits valid invalid \
  --field text_markdown \
  --model Qwen/Qwen3-1.7B-Base \
  --report reports/br_legislation.md \
  --report-json reports/br_legislation.json
```

The module form is equivalent:

```bash
python -m token_counter.cli --dataset <org>/<dataset> --split train --field text
```

## Reports And Resume

By default, `token-counter` writes:

- `reports/token_count_report.md`
- `reports/token_count_report.json`
- `reports/token_count_report_distribution.png`

Choose output paths:

```bash
token-counter \
  --dataset <org>/<dataset> \
  --field text \
  --report reports/count.md \
  --report-json reports/count.json
```

Disable Markdown and write only JSON:

```bash
token-counter \
  --dataset <org>/<dataset> \
  --field text \
  --report "" \
  --report-json reports/count.json
```

Resume from the JSON checkpoint:

```bash
token-counter \
  --dataset <org>/<dataset> \
  --field text \
  --report-json reports/count.json \
  --resume
```

Generate PDF:

```bash
token-counter \
  --dataset <org>/<dataset> \
  --field text \
  --report reports/count.md \
  --report-pdf
```

## Important Flags

| Flag | Use |
| --- | --- |
| `--input` | Local path, URL, local glob, or `hf://` Parquet/JSONL input |
| `--dataset` | Hugging Face dataset id |
| `--format` | `jsonl` or `parquet` for `--input` |
| `--field` | Text column to tokenize, default `text` |
| `--model` | Tokenizer model, default `Qwen/Qwen3-1.7B-Base` |
| `--batch-size` | Documents per tokenizer batch, default `256` |
| `--max-docs` | Stop after N processed documents |
| `--config` | Hugging Face dataset config |
| `--revision` | Hugging Face dataset revision |
| `--split` | One Hugging Face split, repeatable |
| `--splits` | Multiple Hugging Face splits in one flag |
| `--resume` | Resume from `--report-json` checkpoint |
| `--report` | Markdown report path; `""` disables Markdown |
| `--report-json` | JSON report/checkpoint path; `""` disables JSON |
| `--report-pdf` | Generate a PDF next to the Markdown report |

## Report Contents

Markdown reports include:

- Summary with `Total tokens`
- Run context
- Optional by-split table
- Distribution snapshot: mean, median, IQR, P95, P99, stddev
- Histogram plot and table
- Data quality metrics
- Performance metrics

JSON reports include:

- `summary_stats.total_tokens`
- `distribution_stats`
- `data_quality_stats`
- `performance_stats`
- `by_split` for multi-split runs
- `checkpoint_state` for `--resume`
