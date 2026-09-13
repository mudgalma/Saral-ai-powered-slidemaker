from types import SimpleNamespace

import pytest

from saral_parser.embeddings import OpenAIEmbedder, vector_literal
from saral_parser.exceptions import EmbeddingError
from saral_parser.models import EmbeddingSettings


def embedding_settings() -> EmbeddingSettings:
    return EmbeddingSettings(api_key="test-key", dimensions=3, batch_size=2)


class FakeEmbeddingsAPI:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            data=[SimpleNamespace(index=index, embedding=vector) for index, vector in self.vectors]
        )


def test_embedder_preserves_provider_order_and_uses_configured_dimensions():
    api = FakeEmbeddingsAPI([(1, [4.0, 5.0, 6.0]), (0, [1.0, 2.0, 3.0])])
    client = SimpleNamespace(embeddings=api)
    embedder = OpenAIEmbedder(embedding_settings(), client=client)

    assert embedder.embed_texts(["first", "second"]) == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert api.calls[0]["dimensions"] == 3


def test_embedder_rejects_empty_text_without_provider_call():
    api = FakeEmbeddingsAPI([])
    embedder = OpenAIEmbedder(embedding_settings(), client=SimpleNamespace(embeddings=api))

    with pytest.raises(EmbeddingError, match="must not be empty"):
        embedder.embed_texts(["   "])
    assert api.calls == []


def test_embedder_rejects_wrong_vector_shape():
    api = FakeEmbeddingsAPI([(0, [1.0, 2.0])])
    embedder = OpenAIEmbedder(embedding_settings(), client=SimpleNamespace(embeddings=api))

    with pytest.raises(EmbeddingError, match="invalid vector shape"):
        embedder.embed_texts(["chunk"])


def test_vector_literal_is_pgvector_compatible_json():
    assert vector_literal([1.0, 2.5]) == "[1.0,2.5]"
