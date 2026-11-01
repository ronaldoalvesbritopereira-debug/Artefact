from __future__ import annotations

from artefact_rag.documents import load_policy_documents
from artefact_rag.retrieval import _load_compatible_page_embeddings
from artefact_rag.settings import Settings


def test_policy_pages_keep_source_metadata_and_reuse_compatible_vectors() -> None:
    settings = Settings.from_env()
    documents = load_policy_documents(
        settings.data_dir,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    assert len(documents) == 8
    assert [document.metadata["page"] for document in documents] == list(range(1, 9))
    assert all(
        document.metadata["source"] == "políticas_da_loja.pdf"
        for document in documents
    )

    vectors = _load_compatible_page_embeddings(documents)
    assert vectors is not None
    assert len(vectors) == 8
    assert all(len(vector) == 1024 for vector in vectors)
