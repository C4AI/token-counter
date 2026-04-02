"""
Compatibility wrapper for Markdown report PDF export.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from token_counter.pdf_export import main


if __name__ == "__main__":
    main()
