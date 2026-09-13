"""Small CLI for local, non-HTTP parser runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .docling_service import DoclingParser
from .models import ParseOptions, ParserSettings


def main() -> None:
    """Parse a paper that has first been placed in ``paper_folder``."""
    parser = argparse.ArgumentParser(description="Parse a PDF with SARAL's Docling module")
    parser.add_argument("input", type=Path, help="PDF inside this workspace's paper_folder")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--options-json", default="{}", help="JSON object matching ParseOptions")
    arguments = parser.parse_args()
    options = ParseOptions.model_validate(json.loads(arguments.options_json))
    manifest = DoclingParser(ParserSettings.from_workspace(arguments.workspace)).parse_path(
        arguments.input, options
    )
    print(json.dumps(manifest.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
