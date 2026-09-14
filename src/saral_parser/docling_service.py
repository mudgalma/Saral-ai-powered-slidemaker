"""Docling-backed parsing service; it has no dependency on FastAPI."""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .exceptions import ArtifactError, ConversionError
from .models import (
    ArtifactManifest,
    InputMetadata,
    OcrEngine,
    ParseOptions,
    ParserSettings,
    PictureDescriptionMode,
)
from .observability import JobLogger
from .validation import ensure_directories, validate_local_pdf


class DoclingParser:
    """Convert validated PDFs and save immutable-style, inspectable output bundles."""

    def __init__(self, settings: ParserSettings) -> None:
        self.settings = settings
        ensure_directories(settings)

    def parse_path(self, path: Path, options: ParseOptions | None = None) -> ArtifactManifest:
        """Parse a PDF inside a configured input root and return its artifact manifest."""
        metadata = validate_local_pdf(path, self.settings)
        return self.parse_metadata(metadata, options or ParseOptions())

    def parse_metadata(self, metadata: InputMetadata, options: ParseOptions) -> ArtifactManifest:
        """Run one document job. Partial diagnostics remain available after a failure."""
        job_dir = self.settings.output_folder / metadata.document_id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "assets" / "figures").mkdir(parents=True, exist_ok=True)
        (job_dir / "assets" / "tables").mkdir(parents=True, exist_ok=True)
        (job_dir / "assets" / "pages").mkdir(parents=True, exist_ok=True)
        (job_dir / "intermediate").mkdir(parents=True, exist_ok=True)
        logger = JobLogger(metadata.document_id, job_dir / "debug.jsonl")
        started = time.perf_counter()
        now = _utc_now()
        manifest = ArtifactManifest(
            document_id=metadata.document_id,
            content_sha256=metadata.content_sha256,
            input=metadata,
            options=options,
            status="running",
            created_at=now,
            artifacts={
                "debug_log": "debug.jsonl",
                "input_metadata": "intermediate/input_metadata.json",
            },
        )
        try:
            with logger.stage("persist_intermediate", input_bytes=metadata.size_bytes):
                _write_json(job_dir / "intermediate" / "input_metadata.json", metadata.model_dump())
                _write_json(
                    job_dir / "intermediate" / "pipeline_options.json",
                    options.model_dump(mode="json"),
                )
                manifest.artifacts["pipeline_options"] = "intermediate/pipeline_options.json"
            with logger.stage("configure_docling", configuration=options.model_dump(mode="json")):
                converter = self._create_converter(options)
            with logger.stage(
                "convert", source=metadata.stored_filename, timeout_seconds=options.timeout_seconds
            ):
                result = converter.convert(Path(metadata.source_path))
                document = result.document
            with logger.stage("export", output_directory=str(job_dir)):
                counts, warnings, artifacts = self._export(document, job_dir, logger, options)
                manifest.counts = counts
                manifest.warnings.extend(warnings)
                manifest.artifacts.update(artifacts)
            with logger.stage("validate", counts=manifest.counts):
                report = self._validation_report(manifest, job_dir)
                _write_json(job_dir / "validation_report.json", report)
                manifest.artifacts["validation_report"] = "validation_report.json"
                manifest.warnings.extend(report["warnings"])
            manifest.status = "completed"
            manifest.completed_at = _utc_now()
            manifest.duration_seconds = round(time.perf_counter() - started, 3)
            _write_json(job_dir / "manifest.json", manifest.model_dump(mode="json"))
            logger.event(
                logging.INFO,
                "job",
                "completed",
                counts=manifest.counts,
                artifacts=manifest.artifacts,
                warnings=manifest.warnings,
            )
            return manifest
        except Exception as exc:
            manifest.status = "failed"
            manifest.completed_at = _utc_now()
            manifest.duration_seconds = round(time.perf_counter() - started, 3)
            manifest.errors.append(f"{type(exc).__name__}: {exc}")
            _write_json(
                job_dir / "validation_report.json", self._validation_report(manifest, job_dir)
            )
            manifest.artifacts["validation_report"] = "validation_report.json"
            _write_json(job_dir / "manifest.json", manifest.model_dump(mode="json"))
            logger.event(
                logging.ERROR, "job", "failed", error=repr(exc), artifacts=manifest.artifacts
            )
            if isinstance(exc, (ArtifactError, ConversionError)):
                raise
            raise ConversionError("Docling conversion failed; inspect the job debug log") from exc

    def _create_converter(self, options: ParseOptions) -> Any:
        """Build documented PDF options without enabling remote model services."""
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ImportError as exc:  # pragma: no cover - depends on runtime install
            raise ConversionError(
                "Docling is not installed. Install the locked dependencies first."
            ) from exc

        pipeline = PdfPipelineOptions()
        pipeline.do_ocr = options.enable_ocr
        pipeline.do_table_structure = options.extract_table_structure
        pipeline.table_structure_options = TableStructureOptions(do_cell_matching=True)
        pipeline.do_formula_enrichment = options.enable_formula_enrichment
        pipeline.do_picture_classification = options.enable_picture_classification
        pipeline.generate_page_images = options.generate_page_images
        pipeline.generate_picture_images = options.generate_picture_images
        pipeline.images_scale = options.image_scale
        pipeline.document_timeout = options.timeout_seconds

        if options.enable_ocr:
            self._configure_ocr(pipeline, options)
        if options.picture_description_mode != PictureDescriptionMode.DISABLED:
            self._configure_local_picture_description(pipeline, options.picture_description_mode)

        return DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)}
        )

    @staticmethod
    def _configure_ocr(pipeline: Any, options: ParseOptions) -> None:
        """Select a documented local OCR backend; no OCR service is contacted remotely."""
        if options.ocr_engine == OcrEngine.AUTO:
            pipeline.ocr_options.lang = options.ocr_languages
            return
        if options.ocr_engine == OcrEngine.OCRMAC:
            from docling.datamodel.pipeline_options import OcrMacOptions

            pipeline.ocr_options = OcrMacOptions()
            return
        if options.ocr_engine == OcrEngine.TESSERACT_CLI:
            from docling.datamodel.pipeline_options import TesseractCliOcrOptions

            pipeline.ocr_options = TesseractCliOcrOptions()
            return
        raise ConversionError(f"Unsupported OCR engine: {options.ocr_engine}")

    @staticmethod
    def _configure_local_picture_description(pipeline: Any, mode: PictureDescriptionMode) -> None:
        """Configure a local documented VLM preset, explicitly avoiding remote APIs."""
        if mode == PictureDescriptionMode.SMOLVLM_LOCAL:
            from docling.datamodel.pipeline_options import smolvlm_picture_description

            pipeline.picture_description_options = smolvlm_picture_description
        elif mode == PictureDescriptionMode.GRANITE_LOCAL:
            from docling.datamodel.pipeline_options import granite_picture_description

            pipeline.picture_description_options = granite_picture_description
        else:
            raise ConversionError(f"Unsupported picture description mode: {mode}")
        pipeline.do_picture_description = True

    def _export(
        self, document: Any, job_dir: Path, logger: JobLogger, options: ParseOptions
    ) -> Tuple[Dict[str, int], List[str], Dict[str, str]]:
        """Export lossless JSON, readable text, assets, and inspection-friendly derivatives."""
        try:
            from docling_core.types.doc import FormulaItem, ImageRefMode, PictureItem, TableItem
        except ImportError as exc:  # pragma: no cover
            raise ArtifactError("Docling core types are unavailable") from exc

        warnings: List[str] = []
        raw = document.export_to_dict()
        _write_json(job_dir / "document.json", raw)
        # Embed images in readable exports: Docling's referenced paths are converter-internal
        # and would not be stable after this job bundle is moved. Stable PNG assets are also
        # emitted below for consumers that want files rather than data URIs.
        document.save_as_markdown(job_dir / "document.md", image_mode=ImageRefMode.EMBEDDED)
        document.save_as_html(job_dir / "document.html", image_mode=ImageRefMode.EMBEDDED)
        (job_dir / "document.txt").write_text(
            document.export_to_markdown(strict_text=True), encoding="utf-8"
        )
        artifacts = {
            "docling_json": "document.json",
            "readable_markdown": "document.md",
            "readable_html": "document.html",
            "plain_text": "document.txt",
        }

        figures: List[Dict[str, Any]] = []
        tables: List[Dict[str, Any]] = []
        formulas: List[Dict[str, Any]] = []
        counts = {
            "pages": len(getattr(document, "pages", {})),
            "pictures": 0,
            "tables": 0,
            "formulas": 0,
        }

        for index, (element, _level) in enumerate(document.iterate_items(), start=1):
            if isinstance(element, TableItem):
                counts["tables"] += 1
                table_id = f"table_{counts['tables']:04d}"
                item = {
                    "id": table_id,
                    "source_element": _self_ref(element),
                    "provenance": _provenance(element),
                }
                try:
                    dataframe = element.export_to_dataframe(doc=document)
                    csv_path = job_dir / "assets" / "tables" / f"{table_id}.csv"
                    html_path = job_dir / "assets" / "tables" / f"{table_id}.html"
                    dataframe.to_csv(csv_path, index=False)
                    html_path.write_text(element.export_to_html(doc=document), encoding="utf-8")
                    item.update(
                        {
                            "csv": str(csv_path.relative_to(job_dir)),
                            "html": str(html_path.relative_to(job_dir)),
                            "shape": [int(dataframe.shape[0]), int(dataframe.shape[1])],
                        }
                    )
                except Exception as exc:
                    warnings.append(f"Could not export {table_id}: {type(exc).__name__}: {exc}")
                self._save_element_image(
                    element,
                    document,
                    job_dir,
                    job_dir / "assets" / "tables" / f"{table_id}.png",
                    item,
                    warnings,
                )
                tables.append(item)
            elif isinstance(element, PictureItem):
                counts["pictures"] += 1
                figure_id = f"figure_{counts['pictures']:04d}"
                item = {
                    "id": figure_id,
                    "source_element": _self_ref(element),
                    "provenance": _provenance(element),
                    "original_caption": _caption_text(element, document),
                    "generated_description": _generated_description(element),
                }
                self._save_element_image(
                    element,
                    document,
                    job_dir,
                    job_dir / "assets" / "figures" / f"{figure_id}.png",
                    item,
                    warnings,
                )
                figures.append(item)
            elif isinstance(element, FormulaItem):
                counts["formulas"] += 1
                formula_id = f"formula_{counts['formulas']:04d}"
                latex_text = getattr(element, "text", None)
                if _is_formula_text_garbled(latex_text):
                    source_ref = _self_ref(element) or "unknown"
                    warnings.append(
                        f"{formula_id} ({source_ref}): formula enrichment model produced "
                        "space-separated character artifacts — review this formula's "
                        "LaTeX against the source PDF before any scientific reuse."
                    )
                formulas.append(
                    {
                        "id": formula_id,
                        "source_element": _self_ref(element),
                        "provenance": _provenance(element),
                        "latex": latex_text,
                    }
                )

        self._save_page_images(document, job_dir, warnings)
        _write_json(job_dir / "figures.json", figures)
        _write_json(job_dir / "tables.json", tables)
        _write_json(job_dir / "formulas.json", formulas)
        _write_json(
            job_dir / "intermediate" / "structure_preview.json",
            _bounded_structure_preview(raw, self.settings.preview_characters),
        )
        if options.debug_full_document_logging:
            (job_dir / "intermediate" / "full_document_diagnostic.txt").write_text(
                document.export_to_markdown(), encoding="utf-8"
            )
            artifacts["full_document_diagnostic"] = "intermediate/full_document_diagnostic.txt"
        artifacts.update(
            {
                "figures": "figures.json",
                "tables": "tables.json",
                "formulas": "formulas.json",
                "structure_preview": "intermediate/structure_preview.json",
                "asset_root": "assets",
            }
        )
        logger.event(logging.DEBUG, "export", "item_counts", **counts, warnings=warnings)
        return counts, warnings, artifacts

    @staticmethod
    def _save_element_image(
        element: Any,
        document: Any,
        job_dir: Path,
        path: Path,
        item: Dict[str, Any],
        warnings: List[str],
    ) -> None:
        try:
            image = element.get_image(document)
            if image is None:
                warnings.append(f"No rendered image available for {item['id']}")
                return
            image.save(path, "PNG")
            item["image"] = str(path.relative_to(job_dir))
        except Exception as exc:
            warnings.append(f"Could not export image for {item['id']}: {type(exc).__name__}: {exc}")

    @staticmethod
    def _save_page_images(document: Any, job_dir: Path, warnings: List[str]) -> None:
        for page_number, page in getattr(document, "pages", {}).items():
            try:
                image = page.image.pil_image
                path = job_dir / "assets" / "pages" / f"page_{int(page_number):04d}.png"
                image.save(path, "PNG")
            except Exception as exc:
                warnings.append(
                    f"Could not export page {page_number} image: {type(exc).__name__}: {exc}"
                )

    @staticmethod
    def _validation_report(manifest: ArtifactManifest, job_dir: Path) -> Dict[str, Any]:
        """Validate files and surface review items without claiming semantic correctness."""
        failures = list(manifest.errors)
        warnings = list(manifest.warnings)
        required = (
            ("document.json", "document.md", "document.txt") if manifest.status != "failed" else ()
        )
        for relative in required:
            if not (job_dir / relative).is_file():
                failures.append(f"Required artifact is missing: {relative}")
        if manifest.status != "failed":
            try:
                json.loads((job_dir / "document.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                failures.append(f"Primary Docling JSON is unreadable: {exc}")
        review_items: List[str] = []
        if manifest.status != "failed" and manifest.counts.get("tables", 0) == 0:
            review_items.append(
                "No tables detected; confirm this is expected for the source paper."
            )
        if manifest.status != "failed" and manifest.counts.get("pictures", 0) == 0:
            review_items.append("No pictures detected; inspect pages if figures are expected.")
        if manifest.options.enable_formula_enrichment and manifest.counts.get("formulas", 0) == 0:
            review_items.append(
                "Formula enrichment was enabled but no formula items were detected."
            )
        if manifest.options.enable_formula_enrichment and manifest.counts.get("formulas", 0) > 0:
            warnings.append(
                "Formula LaTeX text is model output from the enrichment pipeline. "
                "Review each formula against the source PDF; space-separated character "
                "errors are possible in complex or multi-symbol notation."
            )
        if not manifest.options.enable_ocr:
            warnings.append(
                "OCR was disabled; scanned or raster-only text requires a separate OCR-enabled run."
            )
        if manifest.options.picture_description_mode == PictureDescriptionMode.DISABLED:
            warnings.append(
                "Picture descriptions were disabled; figures retain source captions and provenance only."
            )
        return {
            "document_id": manifest.document_id,
            "status": "failed"
            if failures
            else "passed_with_review_items"
            if review_items or warnings
            else "passed",
            "failures": sorted(set(failures)),
            "warnings": sorted(set(warnings)),
            "review_items": review_items,
            "validated_artifacts": sorted(manifest.artifacts.values()),
        }


def _caption_text(element: Any, document: Any) -> List[str]:
    """Resolve only explicit Docling caption references; never infer a caption."""
    captions: List[str] = []
    for reference in getattr(element, "captions", []) or []:
        source_ref = getattr(reference, "cref", None) or str(reference)
        resolved = next(
            (item for item in getattr(document, "texts", []) if _self_ref(item) == source_ref),
            None,
        )
        text = getattr(resolved, "text", None)
        if text:
            captions.append(str(text))
    return captions


def _generated_description(element: Any) -> Any:
    """Return Docling's model-generated description separately from original source captions."""
    description = getattr(getattr(element, "meta", None), "description", None)
    if description is None:
        return None
    return (
        description.model_dump(mode="json")
        if hasattr(description, "model_dump")
        else str(description)
    )


def _provenance(element: Any) -> List[Dict[str, Any]]:
    """Serialize page and bounding-box information exposed by Docling."""
    result: List[Dict[str, Any]] = []
    for source in getattr(element, "prov", []) or []:
        bbox = getattr(source, "bbox", None)
        result.append(
            {
                "page_number": getattr(source, "page_no", None),
                "bbox": bbox.model_dump(mode="json")
                if hasattr(bbox, "model_dump")
                else str(bbox)
                if bbox
                else None,
                "charspan": getattr(source, "charspan", None),
            }
        )
    return result


def _self_ref(element: Any) -> str | None:
    reference = getattr(element, "self_ref", None)
    return str(reference) if reference else None


def _bounded_structure_preview(raw: Dict[str, Any], limit: int) -> Dict[str, Any]:
    """Provide a bounded debug preview, never a full-document log duplicate."""
    return {
        "top_level_keys": sorted(raw.keys()),
        "counts": {
            key: len(raw.get(key, [])) if isinstance(raw.get(key), list) else None
            for key in ("texts", "tables", "pictures", "groups")
        },
        "text_preview": str(raw.get("texts", []))[:limit],
    }


def _write_json(path: Path, data: Any) -> None:
    try:
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        raise ArtifactError(f"Could not write {path.name}: {exc}") from exc


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Three or more consecutive space-separated single word-characters, e.g. "M u l t i H e a d".
# Threshold of 3 avoids false positives on legitimate two-symbol notation like "A B".
_GARBLED_FORMULA_RE = re.compile(r"(?<![\w])(\w ){3,}\w(?![\w])")


def _is_formula_text_garbled(text: str | None) -> bool:
    """Detect the space-separated single-character artifact common in formula enrichment output.

    The formula enrichment model occasionally renders multi-character tokens as
    individual characters separated by spaces (e.g. ``M u l t i H e a d`` instead
    of ``\\mathrm{MultiHead}``).  Three or more consecutive space-separated
    single word-characters is treated as a positive signal; this threshold avoids
    false positives on two-symbol notation like ``A B`` or ``p \\eta``.
    """
    return bool(text and _GARBLED_FORMULA_RE.search(text))
