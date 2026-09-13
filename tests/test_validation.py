from io import BytesIO
from pathlib import Path

import pytest

from saral_parser.exceptions import InputValidationError
from saral_parser.validation import (
    identify_pdf,
    save_upload,
    validate_document_id,
    validate_local_pdf,
)


def test_identify_pdf_assigns_repeatable_content_identity(settings, valid_pdf):
    result = identify_pdf(valid_pdf, settings)
    assert result.document_id == f"doc_{result.content_sha256[:16]}"
    assert result.size_bytes == valid_pdf.stat().st_size


def test_rejects_non_pdf_signature(settings):
    bad = settings.paper_folder / "misleading.pdf"
    bad.write_bytes(b"not a PDF")
    with pytest.raises(InputValidationError, match="signature"):
        validate_local_pdf(bad, settings)


def test_rejects_input_outside_owned_source_directories(settings, tmp_path):
    external = tmp_path.parent / "external.pdf"
    external.write_bytes(b"%PDF-1.7\nexternal")
    with pytest.raises(InputValidationError, match="inside"):
        validate_local_pdf(external, settings)


def test_upload_is_streamed_and_filename_is_sanitized(settings):
    result = save_upload(BytesIO(b"%PDF-1.7\nupload"), "../../unsafe name.pdf", settings)
    assert result.original_filename == "unsafe_name.pdf"
    assert result.stored_filename.startswith(result.document_id)
    assert ".." not in result.stored_filename
    assert Path(result.source_path).is_file()


@pytest.mark.parametrize(
    "value", ["../doc_deadbeefdeadbeef", "doc_ABC", "doc_deadbeefdeadbeef/extra"]
)
def test_document_id_cannot_be_a_path(value):
    with pytest.raises(InputValidationError):
        validate_document_id(value)
