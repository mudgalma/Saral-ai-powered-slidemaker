import json

from saral_parser.docling_service import (
    _bounded_structure_preview,
    _caption_text,
    _provenance,
    _write_json,
)


class FakeBox:
    def model_dump(self, mode):
        assert mode == "json"
        return {"l": 1, "t": 2, "r": 3, "b": 4}


class FakeSource:
    page_no = 2
    bbox = FakeBox()
    charspan = (4, 9)


class FakeElement:
    prov = [FakeSource()]


class FakeRef:
    cref = "#/texts/0"


class FakeCaptionedPicture:
    captions = [FakeRef()]


class FakeText:
    text = "Figure 1: Explicit source caption."

    @property
    def self_ref(self):
        return "#/texts/0"


class FakeDocument:
    texts = [FakeText()]


def test_provenance_preserves_page_bbox_and_source_span():
    assert _provenance(FakeElement()) == [
        {"page_number": 2, "bbox": {"l": 1, "t": 2, "r": 3, "b": 4}, "charspan": (4, 9)}
    ]


def test_caption_uses_only_explicit_docling_caption_reference():
    assert _caption_text(FakeCaptionedPicture(), FakeDocument()) == [
        "Figure 1: Explicit source caption."
    ]


def test_json_artifact_and_bounded_preview_are_valid(tmp_path):
    path = tmp_path / "artifact.json"
    _write_json(path, {"ok": True})
    assert json.loads(path.read_text()) == {"ok": True}
    preview = _bounded_structure_preview(
        {"texts": ["a" * 100], "tables": [], "pictures": [], "groups": []}, 20
    )
    assert len(preview["text_preview"]) <= 20
