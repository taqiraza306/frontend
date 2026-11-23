"""Open-source embeddings (sentence-transformers) + FAISS vector index."""

from __future__ import annotations

import os
from pathlib import Path

import faiss
import numpy as np

from .config import settings


class Embedder:
    """Wraps a sentence-transformers model. Vectors are L2-normalised, so inner product == cosine."""

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

        self.model_name = model_name or settings.embedding_model
        self.model = SentenceTransformer(self.model_name)
        self.dim = int(self.model.encode(["dimension probe"], normalize_embeddings=True).shape[1])

        name = self.model_name.lower()
        # Retrieval models trained with instructions/prefixes perform noticeably better with them.
        if "bge" in name and "-en" in name:
            self.query_prefix, self.doc_prefix = "Represent this sentence for searching relevant passages: ", ""
        elif "e5" in name:
            self.query_prefix, self.doc_prefix = "query: ", "passage: "
        else:
            self.query_prefix, self.doc_prefix = "", ""
        self.query_prefix = os.getenv("EMBEDDING_QUERY_PREFIX", self.query_prefix)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(
            [self.doc_prefix + t for t in texts],
            normalize_embeddings=True, batch_size=32, show_progress_bar=False,
        )
        return np.asarray(vecs, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        vec = self.model.encode([self.query_prefix + text], normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vec, dtype="float32")


class VectorStore:
    """FAISS IndexIDMap2(IndexFlatIP): exact cosine search, vectors addressed by SQLite ids."""

    def __init__(self, dim: int, path: Path | None = None):
        self.path = path or settings.faiss_path
        self.dim = dim
        self.index = self._load_or_create()

    def _new(self) -> faiss.Index:
        return faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))

    def _load_or_create(self) -> faiss.Index:
        if self.path.exists():
            index = faiss.read_index(str(self.path))
            if index.d == self.dim:
                return index
        return self._new()

    def reset(self) -> None:
        self.index = self._new()

    @property
    def size(self) -> int:
        return int(self.index.ntotal)

    def ids(self) -> set[int]:
        if self.size == 0:
            return set()
        return set(faiss.vector_to_array(self.index.id_map).tolist())

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        if ids:
            self.index.add_with_ids(vectors, np.asarray(ids, dtype="int64"))

    def remove(self, ids: list[int]) -> None:
        if ids and self.size:
            self.index.remove_ids(np.asarray(ids, dtype="int64"))

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        if self.size == 0:
            return []
        k = min(k, self.size)
        scores, ids = self.index.search(query_vec, k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        faiss.write_index(self.index, str(tmp))
        os.replace(tmp, self.path)
