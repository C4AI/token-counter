import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from token_counter import cli
from token_counter.hf_dataset_meta import HFSplitSize
from token_counter.reporting import (
    TokenCountStats,
    build_report_payload,
    build_run_metadata,
    render_console_summary,
    render_markdown_report,
)


class FakeTokenizer:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def __call__(
        self,
        texts: list[str],
        *,
        add_special_tokens: bool = False,
        return_attention_mask: bool = False,
    ) -> dict[str, list[list[int]]]:
        _ = return_attention_mask
        self.batches.append(list(texts))
        extra = 1 if add_special_tokens else 0
        return {"input_ids": [[0] * (len(text) + extra) for text in texts]}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        extra = 1 if add_special_tokens else 0
        return [0] * (len(text) + extra)


class FakeUI:
    instances: list["FakeUI"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.final_payload = None
        FakeUI.instances.append(self)

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def set_phase(self, *args, **kwargs) -> None:
        return None

    def set_message(self, *args, **kwargs) -> None:
        return None

    def set_checkpoint(self, *args, **kwargs) -> None:
        return None

    def set_checkpoint_anchor(self, *args, **kwargs) -> None:
        return None

    def set_restart_attempts(self, *args, **kwargs) -> None:
        return None

    def log(self, *args, **kwargs) -> None:
        return None

    def update(self, *args, **kwargs) -> None:
        return None

    def print_final_summary(self, payload, **kwargs) -> None:
        _ = kwargs
        self.final_payload = payload
        for line in render_console_summary(payload):
            print(line)


def make_stats(lengths: list[int], *, rows_seen: int | None = None) -> TokenCountStats:
    stats = TokenCountStats()
    for length in lengths:
        stats.rows_seen += 1
        stats.observe_document("x" * length, length)
    if rows_seen is not None:
        stats.rows_seen = rows_seen
    stats.wall_time = 1.0
    return stats


class TokenCounterReportTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeUI.instances = []

    def test_count_tokens_batches_and_tracks_quality_metrics(self) -> None:
        tokenizer = FakeTokenizer()
        stats = cli._count_tokens(
            [None, "", "xx", 7],
            tokenizer,
            add_special_tokens=True,
            batch_size=2,
        )

        self.assertEqual(tokenizer.batches, [["", "xx"], ["7"]])
        self.assertEqual(stats.rows_seen, 4)
        self.assertEqual(stats.documents_processed, 3)
        self.assertEqual(stats.null_field_rows, 1)
        self.assertEqual(stats.empty_text_rows, 1)
        self.assertEqual(stats.non_string_rows_coerced, 1)
        self.assertEqual(stats.total_tokens, 6)

    def test_markdown_report_highlights_total_tokens_and_by_split(self) -> None:
        valid = make_stats([1, 2])
        invalid = make_stats([3])
        aggregate = make_stats([1, 2, 3])
        metadata = build_run_metadata(
            input_value="org/dataset",
            data_format="huggingface-dataset",
            field="text",
            model="fake-model",
            add_special_tokens=False,
            max_docs=None,
            trust_remote_code=False,
            report_path="reports/out.md",
            report_json_path="reports/out.json",
            report_pdf_path=None,
            started_at_epoch=1.0,
            completed_at_epoch=2.0,
            extra={"splits": ["valid", "invalid"], "batch_size": 256},
        )
        payload = build_report_payload(
            metadata,
            aggregate,
            by_split={"valid": valid, "invalid": invalid},
        )

        markdown = render_markdown_report(payload)

        self.assertIn("## Summary", markdown)
        self.assertIn("| Total tokens | 6 |", markdown)
        self.assertLess(markdown.index("## Summary"), markdown.index("## Distribution Snapshot"))
        self.assertIn("## By Split", markdown)
        self.assertIn("| valid | 2 | 2 | 3 |", markdown)
        self.assertIn("| invalid | 1 | 1 | 3 |", markdown)
        self.assertEqual(payload["summary_stats"]["total_tokens"], 6)

    def test_hf_glob_input_does_not_require_local_file(self) -> None:
        with patch(
            "token_counter.cli.load_dataset",
            return_value={"train": [{"text": "ok"}]},
        ) as loader:
            values = list(
                cli._iter_input_values(
                    "hf://datasets/org/name@main/data/*.parquet",
                    "parquet",
                    "text",
                )
            )

        self.assertEqual(values, ["ok"])
        loader.assert_called_once()

    def test_local_glob_input_is_allowed_when_it_matches_files(self) -> None:
        with TemporaryDirectory() as temp_dir:
            shard = Path(temp_dir) / "part-0000.parquet"
            shard.write_bytes(b"placeholder")
            pattern = str(Path(temp_dir) / "*.parquet")

            with patch(
                "token_counter.cli.load_dataset",
                return_value={"train": [{"text": "ok"}]},
            ) as loader:
                values = list(cli._iter_input_values(pattern, "parquet", "text"))

        self.assertEqual(values, ["ok"])
        loader.assert_called_once()

    def test_main_counts_input_and_prints_total_tokens(self) -> None:
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "sample.jsonl"
            input_path.write_text('{"text":"a"}\n', encoding="utf-8")
            json_path = Path(temp_dir) / "report.json"
            stdout = io.StringIO()

            with (
                patch("token_counter.cli.load_hf_token", return_value=SimpleNamespace(token=None, source=None)),
                patch("token_counter.cli._load_tokenizer", return_value=FakeTokenizer()),
                patch(
                    "token_counter.cli.load_dataset",
                    return_value={"train": [{"text": None}, {"text": "a"}, {"text": "aa"}, {"text": "aaa"}]},
                ),
                patch("token_counter.cli.CountingUI", FakeUI),
                redirect_stdout(stdout),
            ):
                cli.main(
                    [
                        "--input",
                        str(input_path),
                        "--format",
                        "jsonl",
                        "--report",
                        "",
                        "--report-json",
                        str(json_path),
                        "--max-docs",
                        "2",
                        "--add-special-tokens",
                    ]
                )

            payload = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "max_docs_reached")
        self.assertEqual(payload["summary_stats"]["rows_seen"], 3)
        self.assertEqual(payload["summary_stats"]["documents_processed"], 2)
        self.assertEqual(payload["summary_stats"]["total_tokens"], 5)
        self.assertIn("Total tokens: 5", stdout.getvalue())

    def test_main_counts_multiple_hugging_face_splits(self) -> None:
        rows_by_split = {
            "valid": [{"text": "a"}, {"text": "aa"}],
            "invalid": [{"text": None}, {"text": "bbb"}],
        }

        def fake_load_dataset(path, *args, **kwargs):
            _ = args
            if path == "org/name":
                return rows_by_split[kwargs["split"]]
            raise AssertionError(f"Unexpected load_dataset path: {path}")

        with TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "report.json"
            stdout = io.StringIO()
            with (
                patch("token_counter.cli.load_hf_token", return_value=SimpleNamespace(token=None, source=None)),
                patch("token_counter.cli._load_tokenizer", return_value=FakeTokenizer()),
                patch("token_counter.cli.load_dataset", side_effect=fake_load_dataset),
                patch(
                    "token_counter.cli.resolve_dataset_split_size",
                    side_effect=[
                        HFSplitSize(num_examples=2, source="test"),
                        HFSplitSize(num_examples=2, source="test"),
                    ],
                ),
                patch("token_counter.cli.CountingUI", FakeUI),
                redirect_stdout(stdout),
            ):
                cli.main(
                    [
                        "--dataset",
                        "org/name",
                        "--splits",
                        "valid",
                        "invalid",
                        "--field",
                        "text",
                        "--report",
                        "",
                        "--report-json",
                        str(json_path),
                        "--batch-size",
                        "2",
                    ]
                )

            payload = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["run_metadata"]["splits"], ["valid", "invalid"])
        self.assertEqual(payload["summary_stats"]["rows_seen"], 4)
        self.assertEqual(payload["summary_stats"]["documents_processed"], 3)
        self.assertEqual(payload["summary_stats"]["total_tokens"], 6)
        self.assertEqual(payload["by_split"]["valid"]["total_tokens"], 3)
        self.assertEqual(payload["by_split"]["invalid"]["total_tokens"], 3)
        self.assertIn("checkpoint_state", payload)

    def test_resume_uses_json_checkpoint_and_skips_seen_rows(self) -> None:
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "sample.jsonl"
            input_path.write_text('{"text":"a"}\n{"text":"bb"}\n', encoding="utf-8")
            json_path = Path(temp_dir) / "report.json"
            stdout = io.StringIO()
            checkpoint_stats = make_stats([1])
            checkpoint_payload = {
                "checkpoint_state": cli._checkpoint_state(
                    checkpoint_stats,
                    {},
                    completed_splits=set(),
                    last_checkpoint_docs=1,
                )
            }
            json_path.write_text(json.dumps(checkpoint_payload), encoding="utf-8")

            with (
                patch("token_counter.cli.load_hf_token", return_value=SimpleNamespace(token=None, source=None)),
                patch("token_counter.cli._load_tokenizer", return_value=FakeTokenizer()),
                patch(
                    "token_counter.cli.load_dataset",
                    return_value={"train": [{"text": "a"}, {"text": "bb"}]},
                ),
                patch("token_counter.cli.CountingUI", FakeUI),
                redirect_stdout(stdout),
            ):
                cli.main(
                    [
                        "--input",
                        str(input_path),
                        "--format",
                        "jsonl",
                        "--report",
                        "",
                        "--report-json",
                        str(json_path),
                        "--resume",
                    ]
                )

            payload = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["summary_stats"]["rows_seen"], 2)
        self.assertEqual(payload["summary_stats"]["documents_processed"], 2)
        self.assertEqual(payload["summary_stats"]["total_tokens"], 3)

    def test_report_pdf_requires_markdown_report(self) -> None:
        stderr = io.StringIO()
        with self.assertRaises(SystemExit):
            with redirect_stderr(stderr):
                cli.main(["--input", "data/sample.parquet", "--report", "", "--report-pdf"])


if __name__ == "__main__":
    unittest.main()
