"""Docling integration contract tests; configuration only, no model conversion."""

from saral_parser.docling_service import DoclingParser
from saral_parser.models import ParseOptions


def test_builds_docling_converter_with_formula_and_local_asset_options(settings):
    converter = DoclingParser(settings)._create_converter(
        ParseOptions(
            enable_formula_enrichment=True, generate_page_images=True, generate_picture_images=True
        )
    )
    assert type(converter).__name__ == "DocumentConverter"
