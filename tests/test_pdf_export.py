import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from token_counter import pdf_export


class PdfExportTests(unittest.TestCase):
    def test_extract_title_prefers_first_h1(self) -> None:
        markdown_text = "Intro\n\n# Token Count Report\n\nBody"
        title = pdf_export._extract_title(markdown_text, "fallback")
        self.assertEqual(title, "Token Count Report")

    def test_resolve_asset_uri_handles_relative_and_absolute_links(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            image_path = base_dir / "chart.png"
            image_path.write_bytes(b"png")

            self.assertEqual(
                pdf_export._resolve_asset_uri("chart.png", base_dir),
                str(image_path.resolve()),
            )
            self.assertEqual(
                pdf_export._resolve_asset_uri("https://example.com/chart.png", base_dir),
                "https://example.com/chart.png",
            )
            self.assertEqual(pdf_export._resolve_asset_uri("#section", base_dir), "#section")

    def test_export_markdown_report_to_pdf_renders_via_xhtml2pdf(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            markdown_path = temp_path / "report.md"
            image_path = temp_path / "chart.png"
            image_path.write_bytes(b"png")
            markdown_path.write_text(
                "# Token Count Report\n\n![Distribution](chart.png)\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n",
                encoding="utf-8",
            )

            fake_markdown = SimpleNamespace(
                markdown=lambda text, extensions, output_format: (
                    "<h1>Token Count Report</h1>"
                    '<p><img alt="Distribution" src="chart.png" /></p>'
                    "<table><tr><td>1</td><td>2</td></tr></table>"
                )
            )

            def fake_create_pdf(html: str, dest, encoding: str, link_callback):
                self.assertEqual(encoding, "utf-8")
                self.assertIn("Token Count Report", html)
                self.assertIn("<table>", html)
                self.assertEqual(link_callback("chart.png", None), str(image_path.resolve()))
                dest.write(b"%PDF-1.4 fake\n")
                return SimpleNamespace(err=0)

            with (
                patch.object(pdf_export, "markdown_lib", fake_markdown),
                patch.object(
                    pdf_export,
                    "pisa",
                    SimpleNamespace(CreatePDF=fake_create_pdf),
                ),
            ):
                output_path = pdf_export.export_markdown_report_to_pdf(markdown_path)

            self.assertEqual(output_path, markdown_path.with_suffix(".pdf"))
            self.assertEqual(output_path.read_bytes(), b"%PDF-1.4 fake\n")

    def test_export_markdown_report_to_pdf_raises_for_missing_source(self) -> None:
        with TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "missing.md"
            with self.assertRaises(FileNotFoundError):
                pdf_export.export_markdown_report_to_pdf(missing_path)


if __name__ == "__main__":
    unittest.main()
