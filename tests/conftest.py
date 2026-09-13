"""Shared test fixtures; no test invokes a Docling model run."""

from pathlib import Path

import pytest

from saral_parser.models import ParserSettings
from saral_parser.validation import ensure_directories


@pytest.fixture
def settings(tmp_path: Path) -> ParserSettings:
    active = ParserSettings.from_workspace(tmp_path)
    ensure_directories(active)
    return active


@pytest.fixture
def valid_pdf(settings: ParserSettings) -> Path:
    path = settings.paper_folder / "sample.pdf"
    path.write_bytes(b"%PDF-1.7\nminimal test fixture\n")
    return path
