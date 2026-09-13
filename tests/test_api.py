from io import BytesIO
from uuid import UUID

from fastapi.testclient import TestClient

from saral_parser.app import create_app
from saral_parser.models import RetrievalResponse, RetrievedChunk

OWNER_ID = UUID("11111111-1111-4111-8111-111111111111")


class FakePersistence:
    def __init__(self):
        self.created = None

    def create_queued_document(self, **kwargs):
        self.created = kwargs
        return "private/original/paper.pdf"

    def mark_failed(self, *args):
        raise AssertionError(f"unexpected failure: {args}")

    def get_document(self, document_id, owner_id):
        if owner_id != OWNER_ID:
            return None
        return {
            "id": document_id,
            "status": "ready",
            "original_filename": "paper.pdf",
            "page_count": 2,
            "figure_count": 1,
            "table_count": 1,
            "formula_count": 3,
            "chunk_count": 7,
            "warnings": [],
            "errors": [],
            "embedding_status": "ready",
        }


class NotReadyPersistence(FakePersistence):
    def get_document(self, document_id, owner_id):
        row = super().get_document(document_id, owner_id)
        row["embedding_status"] = "processing"
        return row


class FakeRetriever:
    def retrieve(self, document_id, owner_id, question, top_k):
        return RetrievalResponse(
            document_id=document_id,
            chunks=[
                RetrievedChunk(
                    chunk_id="chunk-1",
                    document_id=document_id,
                    chunk_index=0,
                    text="Method text",
                    contextualized_text="Methods\\nMethod text",
                    source_refs=["#/texts/0"],
                    page_numbers=[1],
                    provenance=[{"page_number": 1}],
                    asset_ids=["figure-1"],
                    content_types=["text"],
                    rrf_score=0.03,
                    dense_rank=1,
                )
            ],
        )


class FakeDispatcher:
    def __init__(self):
        self.dispatched = None

    def dispatch(self, *args):
        self.dispatched = args


def client(settings):
    return TestClient(
        create_app(settings, FakePersistence(), FakeDispatcher(), lambda _token: OWNER_ID)
    )


def test_health_endpoint(settings):
    response = client(settings).get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_upload_returns_202_and_dispatches(settings):
    persistence = FakePersistence()
    dispatcher = FakeDispatcher()
    api = TestClient(create_app(settings, persistence, dispatcher, lambda _token: OWNER_ID))
    response = api.post(
        "/v1/documents",
        files={"file": ("paper.pdf", BytesIO(b"%PDF-1.7\napi test"), "application/pdf")},
    )
    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    assert persistence.created["owner_id"] == OWNER_ID
    assert dispatcher.dispatched[0] == response.json()["document_id"]


def test_upload_requires_file(settings):
    response = client(settings).post("/v1/documents")
    assert response.status_code == 422


def test_status_is_owner_scoped(settings):
    response = client(settings).get("/v1/documents/doc_0123456789abcdef")
    assert response.status_code == 200
    assert response.json()["chunk_count"] == 7


def test_retrieve_returns_fused_chunks_with_provenance(settings):
    api = TestClient(
        create_app(settings, FakePersistence(), FakeDispatcher(), lambda _token: OWNER_ID, FakeRetriever())
    )
    response = api.post(
        "/v1/documents/doc_0123456789abcdef/retrieve", json={"question": "Which methods?"}
    )

    assert response.status_code == 200
    assert response.json()["chunks"][0]["asset_ids"] == ["figure-1"]
    assert response.json()["chunks"][0]["page_numbers"] == [1]


def test_retrieve_returns_409_until_embedding_is_ready(settings):
    api = TestClient(
        create_app(settings, NotReadyPersistence(), FakeDispatcher(), lambda _token: OWNER_ID)
    )
    response = api.post(
        "/v1/documents/doc_0123456789abcdef/retrieve", json={"question": "Which methods?"}
    )

    assert response.status_code == 409




def test_api_allows_only_configured_frontend_origin(settings):
    api = client(settings)
    allowed = api.get("/healthz", headers={"Origin": "http://localhost:5173"})
    blocked = api.get("/healthz", headers={"Origin": "https://untrusted.example"})
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "access-control-allow-origin" not in blocked.headers


def test_default_settings_allow_actual_vite_development_origin(tmp_path):
    from saral_parser.models import ParserSettings

    settings = ParserSettings.from_workspace(tmp_path)
    assert "http://localhost:8080" in settings.allowed_origins
    assert "http://127.0.0.1:8080" in settings.allowed_origins
