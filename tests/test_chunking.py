from types import SimpleNamespace

import pytest

from saral_parser.chunking import HybridDocumentChunker
from saral_parser.exceptions import ChunkingError
from saral_parser.models import ChunkOptions


class Prov:
    page_no = 2

    def model_dump(self, mode):
        return {"page_no": self.page_no, "bbox": {"l": 1, "t": 2, "r": 3, "b": 4}}


class Item:
    def __init__(self, ref, label="text"):
        self.self_ref = ref
        self.label = label
        self.prov = [Prov()]


class FakeDocument:
    def __init__(self):
        self.items = [Item("#/texts/0"), Item("#/texts/1", "formula")]

    def iterate_items(self):
        return [(item, 0) for item in self.items]

    def get_ref(self, ref):
        item = next(item for item in self.items if item.self_ref == ref)
        return SimpleNamespace(resolve=lambda _doc: item)


class FakeTokenizer:
    def count_tokens(self, text):
        return len(text.split())


class FakeChunker:
    tokenizer = FakeTokenizer()

    def __init__(self, reverse=False, duplicate_formula=False):
        items = FakeDocument().items
        if reverse:
            items = list(reversed(items))
        self.items = items + ([items[-1]] if duplicate_formula else [])

    def chunk(self, _document):
        for item in self.items:
            yield SimpleNamespace(
                text="equation" if item.label == "formula" else "body text",
                meta=SimpleNamespace(doc_items=[item], headings=["Methods"], captions=[]),
            )

    def contextualize(self, chunk):
        return "Methods\n" + chunk.text


def test_chunks_keep_order_provenance_and_formula_atomicity():
    result = HybridDocumentChunker(ChunkOptions(max_tokens=64), FakeChunker()).chunk_document(
        "doc_0123456789abcdef", FakeDocument()
    )
    assert [chunk.chunk_index for chunk in result.chunks] == [0, 1]
    assert result.chunks[0].source_refs == ["#/texts/0"]
    assert result.chunks[0].page_numbers == [2]
    assert result.chunks[1].content_types == ["formula"]


def test_chunk_order_violation_fails():
    with pytest.raises(ChunkingError, match="document order"):
        HybridDocumentChunker(
            ChunkOptions(max_tokens=64), FakeChunker(reverse=True)
        ).chunk_document("doc_0123456789abcdef", FakeDocument())


def test_formula_duplication_fails():
    with pytest.raises(ChunkingError, match="exactly one"):
        HybridDocumentChunker(
            ChunkOptions(max_tokens=64), FakeChunker(duplicate_formula=True)
        ).chunk_document("doc_0123456789abcdef", FakeDocument())
