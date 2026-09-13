"""Input validation, file identity, and safe path handling."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import BinaryIO, Iterable

from werkzeug.utils import secure_filename

from .exceptions import InputValidationError
from .models import InputMetadata, ParserSettings

_PDF_HEADER = b"%PDF-"
_SAFE_ID = re.compile(r"^doc_(?:[a-f0-9]{8}_)?[a-f0-9]{16}$")


def ensure_directories(settings: ParserSettings) -> None:
    """Create only parser-owned directories beneath the workspace root."""
    for folder in (settings.paper_folder, settings.upload_folder, settings.output_folder):
        folder.mkdir(parents=True, exist_ok=True)


def is_within(path: Path, allowed_roots: Iterable[Path]) -> bool:
    """Return whether ``path`` resolves under one of ``allowed_roots``."""
    resolved = path.resolve(strict=False)
    for root in allowed_roots:
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def validate_local_pdf(path: Path, settings: ParserSettings) -> InputMetadata:
    """Validate a non-symlink PDF located in a configured source directory."""
    if not path.exists() or not path.is_file():
        raise InputValidationError("Input file does not exist or is not a regular file")
    if path.is_symlink():
        raise InputValidationError("Symbolic links are not accepted as document input")
    if not is_within(path, (settings.paper_folder, settings.upload_folder)):
        raise InputValidationError("Input file must be inside paper_folder or uploads")
    return identify_pdf(path, settings)


def identify_pdf(path: Path, settings: ParserSettings) -> InputMetadata:
    """Validate PDF extension, limit, signature, and calculate SHA-256 in one pass."""
    if path.suffix.lower() not in settings.allowed_extensions:
        raise InputValidationError(f"Unsupported file type: {path.suffix or '<none>'}")
    size = path.stat().st_size
    if size <= 0:
        raise InputValidationError("Empty files are not accepted")
    if size > settings.max_upload_bytes:
        raise InputValidationError(f"File exceeds {settings.max_upload_bytes} byte limit")
    with path.open("rb") as stream:
        header = stream.read(len(_PDF_HEADER))
        if header != _PDF_HEADER:
            raise InputValidationError("File does not have a PDF signature")
        digest = _hash_stream(stream, prefix=header)
    document_id = f"doc_{digest[:16]}"
    return InputMetadata(
        document_id=document_id,
        content_sha256=digest,
        original_filename=path.name,
        stored_filename=path.name,
        media_type="application/pdf",
        size_bytes=size,
        source_path=str(path.resolve()),
    )


def save_upload(stream: BinaryIO, filename: str, settings: ParserSettings) -> InputMetadata:
    """Stream a request upload to an owned, safe path, then validate its bytes."""
    safe_name = secure_filename(filename)
    if not safe_name:
        raise InputValidationError("Filename is empty or unsafe")
    if Path(safe_name).suffix.lower() not in settings.allowed_extensions:
        raise InputValidationError("Only PDF uploads are currently accepted")
    temporary = settings.upload_folder / f".incoming-{os.urandom(12).hex()}.pdf"
    total = 0
    try:
        with temporary.open("xb") as destination:
            while chunk := stream.read(1024 * 1024):
                total += len(chunk)
                if total > settings.max_upload_bytes:
                    raise InputValidationError(
                        f"File exceeds {settings.max_upload_bytes} byte limit"
                    )
                destination.write(chunk)
        metadata = identify_pdf(temporary, settings)
        final_path = settings.upload_folder / f"{metadata.document_id}-{safe_name}"
        temporary.replace(final_path)
        return metadata.model_copy(
            update={
                "original_filename": safe_name,
                "stored_filename": final_path.name,
                "source_path": str(final_path.resolve()),
            }
        )
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def validate_document_id(document_id: str) -> str:
    """Validate externally provided IDs before using them in a filesystem path."""
    if not _SAFE_ID.fullmatch(document_id):
        raise InputValidationError("Invalid document ID")
    return document_id


def _hash_stream(stream: BinaryIO, prefix: bytes = b"") -> str:
    digest = hashlib.sha256()
    digest.update(prefix)
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()
