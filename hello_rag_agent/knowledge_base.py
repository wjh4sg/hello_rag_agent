from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

import httpx

try:
    import chromadb
except ImportError:  # pragma: no cover
    chromadb = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None

from hello_rag_agent.config import KnowledgeBaseSettings


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
SUPPORTED_RETRIEVAL_MODES = {"keyword", "vector", "hybrid"}
EMBED_BATCH_SIZE = 8
RRF_K = 60


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    source: str
    title: str
    content: str


@dataclass(frozen=True)
class SearchResult:
    chunk: KnowledgeChunk
    score: float


class KnowledgeBase:
    def __init__(
        self,
        settings: KnowledgeBaseSettings,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.settings = settings
        self.api_key = api_key or ""
        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self._chunks: list[KnowledgeChunk] = []
        self._chunks_by_id: dict[str, KnowledgeChunk] = {}
        self._skipped_files: list[str] = []
        self._vector_error: str | None = None
        self._embedding_client: OpenAI | None = None
        self._client = None
        self._collection = None
        self._collection_meta_path = self.settings.vector_store_dir / f"{self.settings.collection_name}_meta.json"
        self._load()
        self._initialize_vector_store()

    def stats(self) -> dict[str, object]:
        return {
            "data_dir": str(self.settings.data_dir),
            "document_count": len({chunk.source for chunk in self._chunks}),
            "chunk_count": len(self._chunks),
            "skipped_files": list(self._skipped_files),
            "retrieval_mode": self.settings.retrieval_mode,
            "vector_store_dir": str(self.settings.vector_store_dir),
            "collection_name": self.settings.collection_name,
            "embedding_model": self.settings.embedding_model,
            "embedding_dimensions": self.settings.embedding_dimensions,
            "vector_index_ready": self._collection is not None,
            "vector_error": self._vector_error,
        }

    def search(
        self,
        query: str,
        top_k: int | None = None,
        strategy: str | None = None,
    ) -> list[SearchResult]:
        normalized_query = query.strip()
        if not normalized_query:
            return []

        limit = top_k or self.settings.top_k
        mode = self._normalize_retrieval_mode(strategy or self.settings.retrieval_mode)
        if mode == "keyword":
            return self._keyword_search(normalized_query, limit)
        if mode == "vector":
            vector_results = self._vector_search(normalized_query, limit)
            return vector_results or self._keyword_search(normalized_query, limit)
        return self._hybrid_search(normalized_query, limit)

    def render_context(
        self,
        query: str,
        top_k: int | None = None,
        strategy: str | None = None,
    ) -> str:
        results = self.search(query, top_k=top_k, strategy=strategy)
        if not results:
            return "No relevant knowledge was found in the local knowledge base."

        lines = []
        for index, item in enumerate(results, start=1):
            lines.append(f"[Source {index}] file: {item.chunk.source}")
            lines.append(f"title: {item.chunk.title}")
            lines.append(f"score: {item.score:.4f}")
            lines.append(f"content: {item.chunk.content}")
            lines.append("")
        return "\n".join(lines).strip()

    def _initialize_vector_store(self) -> None:
        if not self._chunks:
            return

        if chromadb is None or OpenAI is None:
            self._vector_error = "chromadb or openai dependency is missing."
            return

        if not self.api_key:
            self._vector_error = "No API key was provided for embedding generation."
            return

        try:
            self.settings.vector_store_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.settings.vector_store_dir))
            signature = self._build_signature()
            current_meta = self._read_collection_meta()

            if (
                current_meta.get("signature") != signature
                or current_meta.get("chunk_count") != len(self._chunks)
                or not self._collection_exists()
            ):
                self._rebuild_collection(signature)
            else:
                self._collection = self._client.get_collection(self.settings.collection_name)

            self._vector_error = None
        except Exception as exc:  # pragma: no cover - depends on local services and credentials
            self._collection = None
            self._vector_error = str(exc)

    def _collection_exists(self) -> bool:
        if self._client is None:
            return False
        try:
            self._client.get_collection(self.settings.collection_name)
            return True
        except Exception:
            return False

    def _read_collection_meta(self) -> dict[str, object]:
        if not self._collection_meta_path.exists():
            return {}
        try:
            return json.loads(self._collection_meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _write_collection_meta(self, payload: dict[str, object]) -> None:
        self._collection_meta_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _rebuild_collection(self, signature: str) -> None:
        assert self._client is not None

        try:
            self._client.delete_collection(self.settings.collection_name)
        except Exception:
            pass

        self._collection = self._client.get_or_create_collection(
            name=self.settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        for batch in self._batched(self._chunks, EMBED_BATCH_SIZE):
            documents = [chunk.content for chunk in batch]
            embeddings = self._embed_texts(documents)
            self._collection.add(
                ids=[chunk.chunk_id for chunk in batch],
                documents=documents,
                embeddings=embeddings,
                metadatas=[
                    {
                        "source": chunk.source,
                        "title": chunk.title,
                    }
                    for chunk in batch
                ],
            )

        self._write_collection_meta(
            {
                "signature": signature,
                "chunk_count": len(self._chunks),
                "embedding_model": self.settings.embedding_model,
                "embedding_dimensions": self.settings.embedding_dimensions,
            }
        )

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self._embedding_client is None:
            http_client = httpx.Client(timeout=120, trust_env=False)
            self._embedding_client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=120,
                http_client=http_client,
            )

        request: dict[str, object] = {
            "model": self.settings.embedding_model,
            "input": texts,
        }
        if self.settings.embedding_dimensions is not None:
            request["dimensions"] = self.settings.embedding_dimensions

        response = self._embedding_client.embeddings.create(**request)
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]

    def _load(self) -> None:
        if not self.settings.data_dir.exists():
            self._skipped_files.append(f"missing:{self.settings.data_dir}")
            return

        for file_path in self._iter_files(self.settings.data_dir):
            try:
                text = self._read_text(file_path)
                if not text.strip():
                    self._skipped_files.append(f"empty:{file_path.name}")
                    continue

                relative_source = str(file_path.relative_to(self.settings.data_dir)).replace("\\", "/")
                chunk_prefix = relative_source.replace("/", "__").replace(".", "_")
                title = file_path.stem
                for index, chunk_text in enumerate(self._split_text(text), start=1):
                    chunk = KnowledgeChunk(
                        chunk_id=f"{chunk_prefix}-{index}",
                        source=relative_source,
                        title=title,
                        content=chunk_text,
                    )
                    self._chunks.append(chunk)
                    self._chunks_by_id[chunk.chunk_id] = chunk
            except Exception:
                self._skipped_files.append(f"failed:{file_path.name}")

    def _iter_files(self, root: Path) -> Iterable[Path]:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in self.settings.supported_extensions:
                yield path

    def _read_text(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            if PdfReader is None:
                raise RuntimeError("pypdf is required to read PDF files.")
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)

        for encoding in ("utf-8", "utf-8-sig", "gbk"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        return path.read_text(encoding="utf-8", errors="ignore")

    def _split_text(self, text: str) -> list[str]:
        normalized = re.sub(r"\r\n?", "\n", text)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
        if not normalized:
            return []

        chunk_size = self.settings.chunk_size
        overlap = self.settings.chunk_overlap
        if len(normalized) <= chunk_size:
            return [normalized]

        chunks = []
        start = 0
        min_boundary = max(chunk_size // 2, 1)

        while start < len(normalized):
            end = min(start + chunk_size, len(normalized))
            if end < len(normalized):
                boundary = normalized.rfind("\n\n", start + min_boundary, end)
                if boundary > start:
                    end = boundary

            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= len(normalized):
                break

            next_start = max(end - overlap, start + 1)
            if next_start <= start:
                next_start = end
            start = next_start

        return chunks

    def _keyword_search(self, query: str, limit: int) -> list[SearchResult]:
        scored = []
        for chunk in self._chunks:
            score = self._score(query, chunk)
            if score > 0:
                scored.append(SearchResult(chunk=chunk, score=score))

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]

    def _vector_search(self, query: str, limit: int) -> list[SearchResult]:
        if self._collection is None:
            return []

        try:
            query_embedding = self._embed_texts([query])[0]
            result = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(max(limit, 1), len(self._chunks)),
                include=["distances", "metadatas"],
            )
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            self._vector_error = str(exc)
            return []

        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        ranked: list[SearchResult] = []
        for chunk_id, distance in zip(ids, distances):
            chunk = self._chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            ranked.append(
                SearchResult(
                    chunk=chunk,
                    score=self._distance_to_score(distance),
                )
            )
        return ranked[:limit]

    def _hybrid_search(self, query: str, limit: int) -> list[SearchResult]:
        keyword_results = self._keyword_search(query, max(limit * 3, self.settings.top_k * 2))
        vector_results = self._vector_search(query, max(limit * 3, self.settings.top_k * 2))
        if not vector_results:
            return keyword_results[:limit]

        combined_scores: dict[str, float] = {}
        for ranking in (keyword_results, vector_results):
            for rank, item in enumerate(ranking, start=1):
                combined_scores[item.chunk.chunk_id] = combined_scores.get(item.chunk.chunk_id, 0.0) + (
                    1.0 / (RRF_K + rank)
                )

        ranked = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
        return [
            SearchResult(chunk=self._chunks_by_id[chunk_id], score=score)
            for chunk_id, score in ranked[:limit]
            if chunk_id in self._chunks_by_id
        ]

    def _build_signature(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.settings.embedding_model.encode("utf-8"))
        digest.update(str(self.settings.embedding_dimensions).encode("utf-8"))
        for chunk in self._chunks:
            digest.update(chunk.chunk_id.encode("utf-8"))
            digest.update(chunk.content.encode("utf-8"))
        return digest.hexdigest()

    @staticmethod
    def _batched(items: list[KnowledgeChunk], size: int) -> Iterable[list[KnowledgeChunk]]:
        for index in range(0, len(items), size):
            yield items[index : index + size]

    def _score(self, query: str, chunk: KnowledgeChunk) -> float:
        query_tokens = Counter(self._tokenize(query))
        if not query_tokens:
            return 0.0

        haystack = f"{chunk.title}\n{chunk.content}"
        chunk_tokens = Counter(self._tokenize(haystack))
        overlap = sum(min(count, chunk_tokens[token]) for token, count in query_tokens.items())
        if overlap == 0 and query not in haystack:
            return 0.0

        title_tokens = set(self._tokenize(chunk.title))
        title_bonus = sum(1 for token in query_tokens if token in title_tokens) * 1.5
        exact_bonus = 2.0 if query in haystack else 0.0
        density_bonus = overlap / max(len(query_tokens), 1)
        return overlap * 2.0 + title_bonus + exact_bonus + density_bonus

    @staticmethod
    def _distance_to_score(distance: float) -> float:
        return 1.0 / (1.0 + max(distance, 0.0))

    @staticmethod
    def _normalize_retrieval_mode(value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in SUPPORTED_RETRIEVAL_MODES:
            return "hybrid"
        return normalized

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]
