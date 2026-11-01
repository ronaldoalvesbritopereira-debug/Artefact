from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "Desafio Técnico AI Engineer 2026" / "data"


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    database_path: Path
    vector_store_dir: Path
    llm_model: str = "qwen3:8b"
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    embedding_device: str = "cpu"
    reranker_device: str = "cpu"
    llm_num_gpu: int = 20
    ollama_base_url: str = "http://127.0.0.1:11434"
    chunk_size: int = 4096
    chunk_overlap: int = 200
    retrieval_k: int = 5
    rerank_k: int = 1

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv(PROJECT_ROOT / ".env")
        artifacts_dir = Path(
            os.getenv("ARTEFACT_ARTIFACTS_DIR", str(PROJECT_ROOT / ".artifacts"))
        ).expanduser()
        data_dir = Path(
            os.getenv("ARTEFACT_DATA_DIR", str(DEFAULT_DATA_DIR))
        ).expanduser()

        return cls(
            data_dir=data_dir.resolve(),
            database_path=Path(
                os.getenv("ARTEFACT_DATABASE_PATH", str(artifacts_dir / "artefact.sqlite3"))
            ).expanduser().resolve(),
            vector_store_dir=Path(
                os.getenv(
                    "ARTEFACT_VECTOR_STORE_DIR",
                    str(artifacts_dir / "vector_store"),
                )
            ).expanduser().resolve(),
            llm_model=os.getenv("ARTEFACT_LLM_MODEL", "qwen3:8b"),
            embedding_model=os.getenv("ARTEFACT_EMBEDDING_MODEL", "BAAI/bge-m3"),
            reranker_model=os.getenv(
                "ARTEFACT_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
            ),
            embedding_device=os.getenv("ARTEFACT_EMBEDDING_DEVICE", "cpu"),
            reranker_device=os.getenv("ARTEFACT_RERANKER_DEVICE", "cpu"),
            llm_num_gpu=int(os.getenv("ARTEFACT_LLM_NUM_GPU", "20")),
            ollama_base_url=os.getenv(
                "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
            ),
        )
