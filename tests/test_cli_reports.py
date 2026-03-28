import io
import json
import sys
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from token_counter import cli


class DummyProgress:
    def __init__(self, *args, **kwargs) -> None:
        self.total = kwargs.get("total")

    def update(self, value: int) -> None:
        _ = value

    def close(self) -> None:
        return None


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        token_count = len(text) + (1 if add_special_tokens else 0)
        return [0] * token_count


def make_args(**overrides) -> Namespace:
    values = {
        "input": "data/sample.parquet",
        "format": "parquet",
        "field": "text",
        "model": "Qwen/Qwen3-1.7B-Base",
        "add_special_tokens": False,
        "max_docs": None,
        "trust_remote_code": False,
        "report": "reports/token_count_report.md",
        "report_json": "reports/token_count_report.json",
        "report_pdf": False,
    }
    values.update(overrides)
    return Namespace(**values)


def make_stats_from_lengths(lengths: list[int], **overrides) -> cli.TokenCountStats:
    stats = cli.TokenCountStats(**overrides)
    for length in lengths:
        stats.observe_document("x" * length, length)
    return stats


class TokenCounterReportTests(unittest.TestCase):
    @patch("token_counter.cli.tqdm", DummyProgress)
    def test_count_tokens_tracks_bucket_boundaries_and_quality(self) -> None:
        stream = [
            None,
            "",
            "x" * 32,
            "x" * 33,
            "x" * 64,
            "x" * 65,
            "x" * 128,
            "x" * 129,
            "x" * 256,
            "x" * 257,
            "x" * 512,
            "x" * 513,
            "x" * 1024,
            "x" * 1025,
            "x" * 2048,
            "x" * 2049,
            "x" * 4096,
            "x" * 4097,
            7,
        ]
        stats = cli._count_tokens(stream, FakeTokenizer())

        self.assertEqual(stats.rows_seen, 19)
        self.assertEqual(stats.documents_processed, 18)
        self.assertEqual(stats.null_field_rows, 1)
        self.assertEqual(stats.empty_text_rows, 1)
        self.assertEqual(stats.non_string_rows_coerced, 1)
        self.assertEqual(stats.histogram["0-32"], 3)
        self.assertEqual(stats.histogram["33-64"], 2)
        self.assertEqual(stats.histogram["65-128"], 2)
        self.assertEqual(stats.histogram["129-256"], 2)
        self.assertEqual(stats.histogram["257-512"], 2)
        self.assertEqual(stats.histogram["513-1024"], 2)
        self.assertEqual(stats.histogram["1025-2048"], 2)
        self.assertEqual(stats.histogram["2049-4096"], 2)
        self.assertEqual(stats.histogram[">4096"], 1)

    @patch("token_counter.cli.tqdm", DummyProgress)
    def test_distribution_payload_captures_median_iqr_and_outlier_gap(self) -> None:
        stats = cli._count_tokens(["a", "aa", "aaa", "aaaa", "x" * 100], FakeTokenizer())

        payload = cli.build_report_payload(make_args(), stats)
        distribution = payload["distribution_stats"]

        self.assertAlmostEqual(distribution["mean_tokens_per_doc"], 22.0, places=6)
        self.assertAlmostEqual(distribution["median_tokens_per_doc"], 3.0, places=6)
        self.assertAlmostEqual(distribution["percentiles"]["p25"], 2.0, places=6)
        self.assertAlmostEqual(distribution["percentiles"]["p75"], 4.0, places=6)
        self.assertAlmostEqual(distribution["iqr_tokens_per_doc"], 2.0, places=6)
        self.assertAlmostEqual(distribution["percentiles"]["p95"], 80.8, places=6)
        self.assertAlmostEqual(distribution["percentiles"]["p99"], 96.16, places=6)
        self.assertGreater(distribution["mean_tokens_per_doc"], distribution["median_tokens_per_doc"])

    def test_render_markdown_report_uses_single_histogram_layout(self) -> None:
        lengths = [1, 2, 3, 4, 100]
        stats = make_stats_from_lengths(
            lengths,
            rows_seen=5,
            wall_time=12.34,
            started_at=datetime.fromisoformat("2026-03-28T09:00:00-03:00"),
            completed_at=datetime.fromisoformat("2026-03-28T09:00:12-03:00"),
        )
        args = make_args(report="reports/out.md", report_json="reports/out.json", max_docs=500)

        with patch("token_counter.cli._now_local", return_value=datetime.fromisoformat("2026-03-28T09:00:13-03:00")):
            payload = cli.build_report_payload(args, stats)

        markdown = cli.render_markdown_report(payload)

        self.assertIn("## Run Context", markdown)
        self.assertIn("## Distribution Snapshot", markdown)
        self.assertIn("## Distribution Histogram", markdown)
        self.assertIn("## Data Quality", markdown)
        self.assertIn("## Performance", markdown)
        self.assertNotIn("## Summary", markdown)
        self.assertNotIn("## Distribution\n", markdown)
        self.assertIn("| Documents processed | 5 |", markdown)
        self.assertIn("| PDF report path | n/a |", markdown)
        self.assertIn("| Mean tokens / doc | 22,00 |", markdown)
        self.assertIn("| Median tokens / doc | 3,00 |", markdown)
        self.assertIn("| IQR tokens / doc | 2,00 |", markdown)
        self.assertIn("| P95 tokens / doc | 80,80 |", markdown)
        self.assertIn("| P99 tokens / doc | 96,16 |", markdown)
        self.assertIn("![Distribution histogram](out_distribution.png)", markdown)
        self.assertIn("| Token Range | Docs | Share |", markdown)
        self.assertIn("| 0-32 | 4 | 80,00% |", markdown)
        self.assertIn("| 65-128 | 1 | 20,00% |", markdown)
        self.assertNotIn("| Bar |", markdown)
        self.assertNotIn("100,00", markdown)

    def test_write_json_report_persists_complete_distribution_shape(self) -> None:
        lengths = [1, 2, 3, 4, 100]
        stats = make_stats_from_lengths(
            lengths,
            rows_seen=5,
            wall_time=2.0,
            started_at=datetime.fromisoformat("2026-03-28T10:00:00-03:00"),
            completed_at=datetime.fromisoformat("2026-03-28T10:00:02-03:00"),
        )
        args = make_args(report="reports/out.md", report_json="reports/out.json")

        with patch("token_counter.cli._now_local", return_value=datetime.fromisoformat("2026-03-28T10:00:03-03:00")):
            payload = cli.build_report_payload(args, stats)

        with TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "report.json"
            cli.write_json_report(json_path, payload)
            written_payload = json.loads(json_path.read_text(encoding="utf-8"))

        distribution = written_payload["distribution_stats"]
        self.assertEqual(written_payload["schema_version"], 1)
        self.assertEqual(written_payload["status"], "completed")
        self.assertEqual(len(distribution["histogram"]), 9)
        self.assertEqual(distribution["histogram"][0]["label"], "0-32")
        self.assertEqual(distribution["histogram"][-1]["label"], ">4096")
        self.assertEqual(distribution["min_tokens_per_doc"], 1)
        self.assertEqual(distribution["max_tokens_per_doc"], 100)
        self.assertIn("token_stddev", distribution)
        self.assertIn("p25", distribution["percentiles"])
        self.assertIn("p75", distribution["percentiles"])
        self.assertIn("iqr_tokens_per_doc", distribution)
        self.assertEqual(distribution["plot"]["relative_path"], "out_distribution.png")
        self.assertIsNone(written_payload["run_metadata"]["report_pdf_path"])

    def test_build_report_payload_includes_pdf_path_when_enabled(self) -> None:
        payload = cli.build_report_payload(
            make_args(report="reports/out.md", report_json="", report_pdf=True),
            make_stats_from_lengths([1, 2, 3], rows_seen=3),
        )

        self.assertEqual(payload["run_metadata"]["report_pdf_path"], "reports/out.pdf")

    @patch("token_counter.cli.tqdm", DummyProgress)
    def test_main_supports_json_output_max_docs_and_special_tokens(self) -> None:
        with TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "report.json"
            markdown_path = Path(temp_dir) / "report.md"
            stdout = io.StringIO()

            with (
                patch("token_counter.cli.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()),
                patch("token_counter.cli._load_stream", return_value=iter([None, "a", "aa", "aaa"])),
                patch("token_counter.cli._now_local", return_value=datetime.fromisoformat("2026-03-28T11:00:00-03:00")),
                patch("token_counter.cli._write_distribution_plot") as plot_writer,
                redirect_stdout(stdout),
            ):
                cli.main(
                    [
                        "--input",
                        "data/sample.parquet",
                        "--report",
                        "",
                        "--report-json",
                        str(json_path),
                        "--max-docs",
                        "2",
                        "--add-special-tokens",
                    ]
                )

            written_payload = json.loads(json_path.read_text(encoding="utf-8"))
            output = stdout.getvalue()
            self.assertFalse(markdown_path.exists())
            plot_writer.assert_not_called()

        distribution = written_payload["distribution_stats"]
        self.assertEqual(written_payload["summary_stats"]["rows_seen"], 3)
        self.assertEqual(written_payload["summary_stats"]["documents_processed"], 2)
        self.assertEqual(written_payload["summary_stats"]["total_tokens"], 5)
        self.assertTrue(written_payload["run_metadata"]["add_special_tokens"])
        self.assertEqual(written_payload["run_metadata"]["max_docs"], 2)
        self.assertAlmostEqual(distribution["mean_tokens_per_doc"], 2.5, places=6)
        self.assertAlmostEqual(distribution["median_tokens_per_doc"], 2.5, places=6)
        self.assertIn("Status: completed", output)
        self.assertIn("Median tokens/doc: 2.50", output)
        self.assertIn("JSON report written to:", output)
        self.assertNotIn("Report written to:", output)

    def test_write_outputs_generates_plot_for_markdown_report(self) -> None:
        payload = cli.build_report_payload(
            make_args(report="reports/out.md", report_json="reports/out.json"),
            make_stats_from_lengths([1, 2, 3], rows_seen=3),
        )

        with TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "out.md"
            json_path = Path(temp_dir) / "out.json"
            args = make_args(report=str(report_path), report_json=str(json_path))
            payload["distribution_stats"]["plot"]["relative_path"] = "out_distribution.png"

            with (
                patch("token_counter.cli.write_markdown_report", return_value=report_path) as markdown_writer,
                patch("token_counter.cli.write_json_report", return_value=json_path) as json_writer,
                patch("token_counter.cli._write_distribution_plot") as plot_writer,
                patch("token_counter.cli.export_markdown_report_to_pdf") as pdf_writer,
            ):
                written_markdown, written_json, written_plot, written_pdf = cli._write_outputs(args, payload)

            self.assertEqual(written_markdown, report_path)
            self.assertEqual(written_json, json_path)
            self.assertEqual(written_plot, report_path.parent / "out_distribution.png")
            self.assertIsNone(written_pdf)
            plot_writer.assert_called_once_with(report_path.parent / "out_distribution.png", payload)
            pdf_writer.assert_not_called()
            markdown_writer.assert_called_once()
            json_writer.assert_called_once()

    def test_write_outputs_generates_pdf_when_flag_is_enabled(self) -> None:
        payload = cli.build_report_payload(
            make_args(report="reports/out.md", report_json="", report_pdf=True),
            make_stats_from_lengths([1, 2, 3], rows_seen=3),
        )

        with TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "out.md"
            pdf_path = report_path.with_suffix(".pdf")
            args = make_args(report=str(report_path), report_json="", report_pdf=True)
            payload["distribution_stats"]["plot"]["relative_path"] = "out_distribution.png"

            with (
                patch("token_counter.cli.write_markdown_report", return_value=report_path),
                patch("token_counter.cli._write_distribution_plot"),
                patch(
                    "token_counter.cli.export_markdown_report_to_pdf",
                    return_value=pdf_path,
                ) as pdf_writer,
            ):
                written_markdown, written_json, written_plot, written_pdf = cli._write_outputs(args, payload)

            self.assertEqual(written_markdown, report_path)
            self.assertIsNone(written_json)
            self.assertEqual(written_plot, report_path.parent / "out_distribution.png")
            self.assertEqual(written_pdf, pdf_path)
            pdf_writer.assert_called_once_with(report_path, pdf_path)

    @patch("token_counter.cli.tqdm", DummyProgress)
    def test_main_supports_pdf_output(self) -> None:
        with TemporaryDirectory() as temp_dir:
            markdown_path = Path(temp_dir) / "report.md"
            pdf_path = markdown_path.with_suffix(".pdf")
            stdout = io.StringIO()

            with (
                patch("token_counter.cli.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()),
                patch("token_counter.cli._load_stream", return_value=iter(["a", "aa"])),
                patch("token_counter.cli._now_local", return_value=datetime.fromisoformat("2026-03-28T11:00:00-03:00")),
                patch("token_counter.cli._write_distribution_plot"),
                patch("token_counter.cli.write_markdown_report", return_value=markdown_path),
                patch("token_counter.cli.export_markdown_report_to_pdf", return_value=pdf_path) as pdf_writer,
                redirect_stdout(stdout),
            ):
                cli.main(
                    [
                        "--input",
                        "data/sample.parquet",
                        "--report",
                        str(markdown_path),
                        "--report-pdf",
                    ]
                )

            output = stdout.getvalue()
            pdf_writer.assert_called_once_with(markdown_path, pdf_path)
            self.assertIn("PDF report written to:", output)

    def test_main_rejects_pdf_without_markdown_report(self) -> None:
        with self.assertRaisesRegex(ValueError, "--report-pdf requires --report"):
            cli.main(
                [
                    "--input",
                    "data/sample.parquet",
                    "--report",
                    "",
                    "--report-pdf",
                ]
            )


if __name__ == "__main__":
    unittest.main()
