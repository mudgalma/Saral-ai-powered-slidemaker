from uuid import UUID

from saral_parser.models import DocumentChunk
from saral_parser.persistence import SupabasePersistence


def test_private_object_path_is_tenant_and_document_scoped():
    owner = UUID("11111111-1111-4111-8111-111111111111")
    service = object.__new__(SupabasePersistence)
    assert service.original_object_path(owner, "doc_11111111_0123456789abcdef") == (
        "11111111-1111-4111-8111-111111111111/doc_11111111_0123456789abcdef/original/paper.pdf"
    )


def test_chunk_row_serializes_grounding_and_asset_relationships():
    owner = UUID("11111111-1111-4111-8111-111111111111")
    job = UUID("22222222-2222-4222-8222-222222222222")
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        document_id="doc_11111111_0123456789abcdef",
        chunk_index=0,
        text="equation",
        contextualized_text="Methods\nequation",
        headings=["Methods"],
        source_refs=["#/texts/4"],
        page_numbers=[2],
        provenance=[{"source_ref": "#/texts/4", "page_number": 2}],
        asset_ids=["asset-1"],
        content_types=["formula"],
        token_count=4,
        chunker_version="docling-hybrid-v1",
    )
    row = SupabasePersistence._chunk_row(owner, job, chunk)
    assert row["source_refs"] == ["#/texts/4"]
    assert row["asset_ids"] == ["asset-1"]
    assert row["owner_id"] == str(owner)
