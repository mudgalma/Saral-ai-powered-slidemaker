from saral_parser.retrieval import fuse_rankings


def _row(chunk_id: str, index: int, **extra):
    return {
        "chunk_id": chunk_id,
        "document_id": "doc_11111111_0123456789abcdef",
        "chunk_index": index,
        "text": f"text-{chunk_id}",
        "contextualized_text": f"Methods\\ntext-{chunk_id}",
        "headings": ["Methods"],
        "source_refs": [f"#/texts/{index}"],
        "page_numbers": [index + 1],
        "provenance": [{"page_number": index + 1}],
        "asset_ids": [f"asset-{chunk_id}"],
        "content_types": ["text"],
        **extra,
    }


def test_rrf_prioritizes_a_chunk_found_in_both_indexes_and_preserves_provenance():
    chunks = fuse_rankings(
        [_row("dense-only", 0), _row("both", 1)],
        [_row("both", 1), _row("sparse-only", 2)],
        top_k=3,
    )

    assert [chunk.chunk_id for chunk in chunks] == ["both", "dense-only", "sparse-only"]
    assert chunks[0].dense_rank == 2
    assert chunks[0].sparse_rank == 1
    assert chunks[0].asset_ids == ["asset-both"]
    assert chunks[0].page_numbers == [2]


def test_rrf_ties_are_deterministic_by_chunk_order():
    chunks = fuse_rankings([_row("later", 4)], [_row("earlier", 2)], top_k=2)

    assert [chunk.chunk_id for chunk in chunks] == ["earlier", "later"]
