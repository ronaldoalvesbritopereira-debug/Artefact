from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_policy_documents(
    data_dir: Path,
    chunk_size: int = 4096,
    chunk_overlap: int = 200,
) -> list[Document]:
    pdf_paths = sorted(data_dir.rglob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"Nenhum PDF de políticas encontrado em {data_dir}.")

    documents: list[Document] = []
    for pdf_path in pdf_paths:
        source_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        for document in PyPDFLoader(str(pdf_path)).load():
            document.metadata["source"] = pdf_path.name
            document.metadata["page"] = int(document.metadata.get("page", 0)) + 1
            document.metadata["source_hash"] = source_hash
            documents.append(document)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(documents)
