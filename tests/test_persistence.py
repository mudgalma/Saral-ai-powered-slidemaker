from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

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


@pytest.mark.parametrize("rpc_data", [{"version_number": 1}, [{"version_number": 1}]])
def test_create_artifact_version_accepts_object_or_list_rpc_response(rpc_data):
    class FakeClient:
        def rpc(self, name, params):
            assert name == "create_artifact_version"
            assert params["p_document_id"] == "doc_11111111_0123456789abcdef"
            return SimpleNamespace(execute=lambda: SimpleNamespace(data=rpc_data))

    owner = UUID("11111111-1111-4111-8111-111111111111")
    service = object.__new__(SupabasePersistence)
    service._client = FakeClient()
    service._execute = lambda operation: operation()

    result = service.create_artifact_version(
        uuid4(),
        "doc_11111111_0123456789abcdef",
        owner,
        {"title": "Version one"},
        None,
        None,
    )

    assert result == {"version_number": 1}
