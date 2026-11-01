from __future__ import annotations

import hashlib
import json
import logging
import gc
from pathlib import Path

import numpy as np
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from artefact_rag.settings import Settings

logger = logging.getLogger(__name__)
REFERENCE_PAGES = Path(__file__).resolve().parents[2] / "artifacts"


def _index_fingerprint(documents: list[Document], settings: Settings) -> str:
    digest = hashlib.sha256()
    digest.update(settings.embedding_model.encode("utf-8"))
    digest.update(f"{settings.chunk_size}:{settings.chunk_overlap}".encode("ascii"))
    for document in documents:
        digest.update(str(document.metadata["source_hash"]).encode("ascii"))
        digest.update(str(document.metadata["page"]).encode("ascii"))
        digest.update(document.page_content.encode("utf-8"))
    return digest.hexdigest()


def _load_compatible_page_embeddings(
    documents: list[Document],
) -> list[list[float]] | None:
    manifest_path = REFERENCE_PAGES / "policy_embeddings.json"
    chunks_path = REFERENCE_PAGES / "policy_pages.jsonl"
    embeddings_path = REFERENCE_PAGES / "policy_embeddings.npy"
    if not all(path.exists() for path in (manifest_path, chunks_path, embeddings_path)):
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(documents) != manifest["page_count"]:
        return None
    if len({document.metadata["source_hash"] for document in documents}) != 1:
        return None
    if documents[0].metadata["source_hash"] != manifest["source_sha256"].lower():
        return None
    if any(
        document.metadata["source"] != manifest["source_name"]
        for document in documents
    ):
        return None
    if any(
        document.metadata["page"] != page
        for page, document in enumerate(documents, start=1)
    ):
        return None

    chunks = [
        json.loads(line)
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [chunk.get("page") for chunk in chunks] != list(
        range(1, manifest["page_count"] + 1)
    ):
        return None

    embeddings = np.load(embeddings_path, allow_pickle=False)
    if embeddings.shape != (manifest["page_count"], manifest["embedding_dimension"]):
        return None
    if manifest["model"] != "BAAI/bge-m3":
        return None
    return embeddings.tolist()


def _has_current_index(
    store: InMemoryVectorStore,
    fingerprint: str,
    embedding_model: str,
    expected_count: int,
) -> bool:
    entries = list(store.store.values())
    return len(entries) == expected_count and all(
        entry.get("metadata", {}).get("_fingerprint") == fingerprint
        and entry.get("metadata", {}).get("_embedding_model") == embedding_model
        for entry in entries
    )


def _load_vector_store(
    documents: list[Document],
    settings: Settings,
    embeddings: HuggingFaceEmbeddings,
) -> InMemoryVectorStore:
    settings.vector_store_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _index_fingerprint(documents, settings)
    index_path = settings.vector_store_dir / "policy_vector_store.json"
    store = InMemoryVectorStore(embedding=embeddings)
    if index_path.exists():
        try:
            store = InMemoryVectorStore.load(str(index_path), embedding=embeddings)
        except (OSError, TypeError, ValueError, KeyError) as error:
            logger.warning("Índice vetorial inválido; será recriado: %s", error)

    if not _has_current_index(
        store,
        fingerprint,
        settings.embedding_model,
        len(documents),
    ):
        metadata = [
            {
                **document.metadata,
                "_fingerprint": fingerprint,
                "_embedding_model": settings.embedding_model,
            }
            for document in documents
        ]
        ids = [
            hashlib.sha256(
                (
                    str(document.metadata["source_hash"])
                    + ":"
                    + str(document.metadata["page"])
                    + ":"
                    + str(index)
                ).encode("utf-8")
            ).hexdigest()
            for index, document in enumerate(documents)
        ]
        cached_vectors = (
            _load_compatible_page_embeddings(documents)
            if settings.embedding_model == "BAAI/bge-m3"
            and settings.chunk_size == 4096
            and settings.chunk_overlap == 200
            else None
        )
        store = InMemoryVectorStore(embedding=embeddings)
        if cached_vectors is not None:
            store.store.update(
                {
                    document_id: {
                        "id": document_id,
                        "vector": vector,
                        "text": document.page_content,
                        "metadata": document_metadata,
                    }
                    for document_id, document, vector, document_metadata in zip(
                        ids,
                        documents,
                        cached_vectors,
                        metadata,
                        strict=True,
                    )
                }
            )
            logger.info("Embeddings BGE-M3 compatíveis foram reutilizados.")
        else:
            store.add_documents(
                [
                    Document(
                        page_content=document.page_content,
                        metadata=document_metadata,
                    )
                    for document, document_metadata in zip(
                        documents,
                        metadata,
                        strict=True,
                    )
                ],
                ids=ids,
            )
            logger.info("Embeddings do PDF foram gerados para o índice local.")
        store.dump(str(index_path))

    return store


def retrieve_documents(
    documents: list[Document],
    query: str,
    settings: Settings,
) -> list[Document]:
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )
    store = _load_vector_store(documents, settings, embeddings)
    lexical_retriever = BM25Retriever.from_documents(
        documents,
        k=settings.retrieval_k,
    )
    semantic_retriever = store.as_retriever(
        search_kwargs={"k": settings.retrieval_k}
    )
    hybrid_retriever = EnsembleRetriever(
        retrievers=[lexical_retriever, semantic_retriever],
        weights=[0.5, 0.5],
        c=60,
    )
    candidates = hybrid_retriever.invoke(query)
    del hybrid_retriever, lexical_retriever, semantic_retriever, store, embeddings
    gc.collect()

    reranker = CrossEncoderReranker(
        model=HuggingFaceCrossEncoder(
            model_name=settings.reranker_model,
            model_kwargs={"device": settings.reranker_device},
        ),
        top_n=settings.rerank_k,
    )
    reranked = reranker.compress_documents(candidates, query)
    del reranker
    gc.collect()
    return reranked[: settings.rerank_k]
