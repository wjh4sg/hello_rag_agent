from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
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
SECTION_PRIORITY = {
    "heading": 0.6,
    "bullet": 0.8,
    "qa": 1.1,
    "paragraph": 0.7,
}


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    source: str
    title: str
    content: str
    chunk_index: int
    heading_path: tuple[str, ...]
    section_type: str
    start_offset: int
    end_offset: int


@dataclass(frozen=True)
class SearchResult:
    chunk: KnowledgeChunk
    score: float
    strategy: str
    snippet: str
    citation: str
    match_terms: tuple[str, ...]


@dataclass(frozen=True)
class _ChunkSegment:
    text: str
    heading_path: tuple[str, ...]
    section_type: str
    start_offset: int
    end_offset: int


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
            "rerank_pool_size": self.settings.rerank_pool_size,
            "max_query_variants": self.settings.max_query_variants,
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
        if self._detect_query_intent(normalized_query) == "overview":
            overview_results = self._overview_search(normalized_query, limit)
            if overview_results:
                return overview_results

        retrieval_query = self._expand_query_for_retrieval(normalized_query)
        mode = self._normalize_retrieval_mode(strategy or self.settings.retrieval_mode)
        if mode == "keyword":
            results = self._keyword_search(retrieval_query, self._candidate_pool_size(limit))
            return self._rerank_results(normalized_query, results, limit, strategy="keyword")
        if mode == "vector":
            vector_results = self._vector_search(retrieval_query, self._candidate_pool_size(limit))
            if vector_results:
                return self._rerank_results(normalized_query, vector_results, limit, strategy="vector")
            keyword_results = self._keyword_search(retrieval_query, self._candidate_pool_size(limit))
            return self._rerank_results(normalized_query, keyword_results, limit, strategy="keyword")

        return self._hybrid_search(retrieval_query, limit, rerank_query=normalized_query)

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
            lines.append(f"citation: {item.citation}")
            lines.append(f"strategy: {item.strategy}")
            lines.append(f"score: {item.score:.4f}")
            if item.match_terms:
                lines.append(f"match_terms: {', '.join(item.match_terms)}")
            lines.append(f"content: {item.snippet or item.chunk.content}")
            lines.append("")
        return "\n".join(lines).strip()

    def format_citation(self, chunk: KnowledgeChunk) -> str:
        location = f"chunk {chunk.chunk_index}"
        if chunk.heading_path:
            return f"{chunk.source} > {' > '.join(chunk.heading_path)} ({location})"
        return f"{chunk.source} ({location})"

    def build_snippet(
        self,
        chunk: KnowledgeChunk,
        query: str,
        *,
        max_chars: int | None = None,
    ) -> tuple[str, tuple[str, ...]]:
        limit = max_chars or self.settings.snippet_max_chars
        query_terms = self._tokenize(query)
        match_terms = tuple(term for term in query_terms if term in chunk.content.lower() or term in chunk.title.lower())

        if not chunk.content:
            return "", match_terms

        best_line = chunk.content.strip()
        best_score = -1.0
        for line in chunk.content.splitlines():
            candidate = line.strip(" -*\t")
            if not candidate:
                continue
            score = self._score_text_overlap(query_terms, candidate)
            if score > best_score:
                best_score = score
                best_line = candidate

        snippet = best_line
        if len(snippet) > limit:
            snippet = f"{snippet[: max(limit - 3, 1)].rstrip()}..."
        return snippet, match_terms

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
                        "chunk_index": chunk.chunk_index,
                        "heading_path": " > ".join(chunk.heading_path),
                        "section_type": chunk.section_type,
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
                for index, chunk in enumerate(self._build_chunks(relative_source, title, text), start=1):
                    chunk_id = f"{chunk_prefix}-{index}"
                    final_chunk = KnowledgeChunk(
                        chunk_id=chunk_id,
                        source=relative_source,
                        title=title,
                        content=chunk.content,
                        chunk_index=index,
                        heading_path=chunk.heading_path,
                        section_type=chunk.section_type,
                        start_offset=chunk.start_offset,
                        end_offset=chunk.end_offset,
                    )
                    self._chunks.append(final_chunk)
                    self._chunks_by_id[chunk_id] = final_chunk
            except Exception:
                self._skipped_files.append(f"failed:{file_path.name}")

    def _build_chunks(self, source: str, title: str, text: str) -> list[KnowledgeChunk]:
        normalized = self._normalize_document_text(text)
        if not normalized:
            return []

        segments = self._segment_text(normalized, title)
        if not segments:
            return [
                KnowledgeChunk(
                    chunk_id=f"{source}-1",
                    source=source,
                    title=title,
                    content=normalized,
                    chunk_index=1,
                    heading_path=(title,),
                    section_type="paragraph",
                    start_offset=0,
                    end_offset=len(normalized),
                )
            ]

        chunk_size = max(self.settings.chunk_size, 120)
        overlap = max(self.settings.chunk_overlap, 0)
        chunks: list[KnowledgeChunk] = []
        buffer: list[_ChunkSegment] = []
        buffer_len = 0

        def flush_buffer() -> None:
            nonlocal buffer, buffer_len
            if not buffer:
                return

            content = "\n".join(segment.text for segment in buffer).strip()
            if not content:
                buffer = []
                buffer_len = 0
                return

            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"{source}-{len(chunks) + 1}",
                    source=source,
                    title=title,
                    content=content,
                    chunk_index=len(chunks) + 1,
                    heading_path=self._resolve_heading_path(buffer, title),
                    section_type=self._resolve_section_type(buffer),
                    start_offset=buffer[0].start_offset,
                    end_offset=buffer[-1].end_offset,
                )
            )

            buffer = self._build_overlap_segments(buffer, overlap)
            buffer_len = sum(len(segment.text) + 1 for segment in buffer)

        for segment in segments:
            if buffer:
                heading_changed = segment.heading_path != buffer[-1].heading_path
                if heading_changed and buffer_len >= chunk_size // 3:
                    flush_buffer()

            prospective_len = buffer_len + len(segment.text) + (1 if buffer else 0)
            if buffer and prospective_len > chunk_size:
                flush_buffer()

            if len(segment.text) > chunk_size:
                for part in self._split_large_segment(segment, chunk_size, overlap):
                    if buffer and buffer_len + len(part.text) + 1 > chunk_size:
                        flush_buffer()
                    buffer.append(part)
                    buffer_len += len(part.text) + 1
                    flush_buffer()
                continue

            buffer.append(segment)
            buffer_len += len(segment.text) + 1

        flush_buffer()
        return chunks

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

    @staticmethod
    def _normalize_document_text(text: str) -> str:
        normalized = re.sub(r"\r\n?", "\n", text)
        normalized = re.sub(r"(?<!\n)(#{1,6}\s+)", r"\n\1", normalized)
        normalized = re.sub(r"(?<!\n)(\d+\.\s*\*\*[^*]{2,80}\*\*)", r"\n\1", normalized)
        normalized = re.sub(r"(?<!\n)(\d+[、.]\s*[^\n]{2,40}[:：])", r"\n\1", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    def _segment_text(self, text: str, title: str) -> list[_ChunkSegment]:
        lines = text.splitlines()
        heading_stack: list[str] = [title]
        segments: list[_ChunkSegment] = []
        buffer: list[str] = []
        buffer_type = "paragraph"
        buffer_heading = tuple(heading_stack)
        offset = 0
        segment_start = 0

        def flush() -> None:
            nonlocal buffer, segment_start
            if not buffer:
                return
            segment_text = "\n".join(buffer).strip()
            if segment_text:
                segments.append(
                    _ChunkSegment(
                        text=segment_text,
                        heading_path=buffer_heading,
                        section_type=buffer_type,
                        start_offset=segment_start,
                        end_offset=segment_start + len(segment_text),
                    )
                )
            buffer = []

        for raw_line in lines:
            line = raw_line.strip()
            line_start = offset
            offset += len(raw_line) + 1

            if not line:
                flush()
                continue

            heading_info = self._parse_heading(line)
            if heading_info is not None:
                flush()
                level, heading = heading_info
                level = max(level, 1)
                while len(heading_stack) > level:
                    heading_stack.pop()
                if len(heading_stack) == level:
                    heading_stack[-1] = heading
                else:
                    heading_stack.append(heading)
                continue

            line_type = self._detect_section_type(line)
            current_heading = tuple(heading_stack)
            if not buffer:
                buffer_heading = current_heading
                buffer_type = line_type
                segment_start = line_start
            elif buffer_heading != current_heading or (line_type != buffer_type and line_type in {"bullet", "qa"}):
                flush()
                buffer_heading = current_heading
                buffer_type = line_type
                segment_start = line_start

            buffer.append(line)

        flush()
        return segments

    @staticmethod
    def _parse_heading(line: str) -> tuple[int, str] | None:
        markdown_match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if markdown_match:
            title = markdown_match.group(2).strip(" -*")
            return len(markdown_match.group(1)), title or "Untitled"

        numbered_match = re.match(r"^(\d+(?:\.\d+)*)[、.]\s*(.+)$", line)
        if numbered_match and len(line) <= 36:
            level = min(len(numbered_match.group(1).split(".")) + 1, 4)
            return level, numbered_match.group(2).strip(" :：")

        colon_heading = re.match(r"^([^:：]{2,30})[:：]$", line)
        if colon_heading:
            return 2, colon_heading.group(1).strip()

        return None

    @staticmethod
    def _detect_section_type(line: str) -> str:
        if re.match(r"^[-*•]\s+", line):
            return "bullet"
        if re.match(r"^\d+[.)、]\s+", line):
            return "bullet"
        if "**" in line and "?" in line or "？" in line:
            return "qa"
        if any(marker in line for marker in ("如何", "为什么", "怎么", "是否", "吗？", "吗?")) and len(line) <= 80:
            return "qa"
        return "paragraph"

    @staticmethod
    def _resolve_heading_path(segments: list[_ChunkSegment], title: str) -> tuple[str, ...]:
        if not segments:
            return (title,)
        common = list(segments[0].heading_path)
        for segment in segments[1:]:
            while common and tuple(common) != segment.heading_path[: len(common)]:
                common.pop()
            if not common:
                break
        if common:
            return tuple(common)
        for segment in reversed(segments):
            if segment.heading_path:
                return segment.heading_path
        return (title,)

    @staticmethod
    def _resolve_section_type(segments: list[_ChunkSegment]) -> str:
        scores = Counter(segment.section_type for segment in segments)
        return max(scores, key=lambda item: (scores[item], SECTION_PRIORITY.get(item, 0.0)))

    @staticmethod
    def _build_overlap_segments(segments: list[_ChunkSegment], overlap: int) -> list[_ChunkSegment]:
        if overlap <= 0:
            return []
        tail: list[_ChunkSegment] = []
        char_count = 0
        for segment in reversed(segments):
            tail.insert(0, segment)
            char_count += len(segment.text) + 1
            if char_count >= overlap:
                break
        return tail

    def _split_large_segment(self, segment: _ChunkSegment, chunk_size: int, overlap: int) -> list[_ChunkSegment]:
        parts: list[_ChunkSegment] = []
        start = 0
        text = segment.text
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                boundary = text.rfind("。", start + max(chunk_size // 2, 1), end)
                if boundary <= start:
                    boundary = text.rfind("\n", start + max(chunk_size // 2, 1), end)
                if boundary > start:
                    end = boundary + 1

            content = text[start:end].strip()
            if content:
                parts.append(
                    _ChunkSegment(
                        text=content,
                        heading_path=segment.heading_path,
                        section_type=segment.section_type,
                        start_offset=segment.start_offset + start,
                        end_offset=segment.start_offset + end,
                    )
                )

            if end >= len(text):
                break

            next_start = max(end - overlap, start + 1)
            if next_start <= start:
                next_start = end
            start = next_start
        return parts

    def _keyword_search(self, query: str, limit: int) -> list[tuple[KnowledgeChunk, float]]:
        scored: list[tuple[KnowledgeChunk, float]] = []
        for chunk in self._chunks:
            score = self._score(query, chunk)
            if score > 0:
                scored.append((chunk, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def _vector_search(self, query: str, limit: int) -> list[tuple[KnowledgeChunk, float]]:
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
        ranked: list[tuple[KnowledgeChunk, float]] = []
        for chunk_id, distance in zip(ids, distances):
            chunk = self._chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            ranked.append((chunk, self._distance_to_score(distance)))
        return ranked[:limit]

    def _hybrid_search(self, query: str, limit: int, *, rerank_query: str | None = None) -> list[SearchResult]:
        pool_size = self._candidate_pool_size(limit)
        keyword_results = self._keyword_search(query, pool_size)
        vector_results = self._vector_search(query, pool_size)
        if not vector_results:
            return self._rerank_results(rerank_query or query, keyword_results, limit, strategy="keyword")

        combined_scores: dict[str, dict[str, float | str]] = {}
        for strategy_name, ranking in (("keyword", keyword_results), ("vector", vector_results)):
            for rank, (chunk, score) in enumerate(ranking, start=1):
                payload = combined_scores.setdefault(
                    chunk.chunk_id,
                    {"rrf": 0.0, "keyword": 0.0, "vector": 0.0, "strategy": strategy_name},
                )
                payload["rrf"] = float(payload["rrf"]) + (1.0 / (RRF_K + rank))
                payload[strategy_name] = max(float(payload[strategy_name]), score)
                if strategy_name == "keyword" and float(payload["vector"]) > 0:
                    payload["strategy"] = "hybrid"
                elif strategy_name == "vector" and float(payload["keyword"]) > 0:
                    payload["strategy"] = "hybrid"

        merged = [
            (
                self._chunks_by_id[chunk_id],
                float(payload["rrf"]) * 10.0 + float(payload["keyword"]) * 0.35 + float(payload["vector"]) * 0.65,
            )
            for chunk_id, payload in combined_scores.items()
            if chunk_id in self._chunks_by_id
        ]
        merged.sort(key=lambda item: item[1], reverse=True)
        return self._rerank_results(rerank_query or query, merged[:pool_size], limit, strategy="hybrid")

    def _rerank_results(
        self,
        query: str,
        candidates: list[tuple[KnowledgeChunk, float]],
        limit: int,
        *,
        strategy: str,
    ) -> list[SearchResult]:
        query_terms = self._tokenize(query)
        same_doc_hits = Counter(chunk.source for chunk, _ in candidates)
        intent = self._detect_query_intent(query)
        ranked: list[tuple[float, KnowledgeChunk]] = []
        for rank, (chunk, base_score) in enumerate(candidates, start=1):
            text = f"{chunk.title}\n{chunk.content}"
            lexical = self._score_text_overlap(query_terms, text)
            phrase_bonus = 1.2 if query.strip().lower() in text.lower() else 0.0
            heading_bonus = 0.6 * self._score_text_overlap(query_terms, " ".join(chunk.heading_path))
            section_bonus = SECTION_PRIORITY.get(chunk.section_type, 0.0)
            density_bonus = min(same_doc_hits.get(chunk.source, 0), 3) * 0.2
            rank_bonus = max(limit + 2 - rank, 0) * 0.08
            source_bonus = self._source_preference_bonus(chunk.source)
            intent_bonus = self._intent_relevance_bonus(intent, chunk)
            final_score = (
                base_score * 100.0
                + lexical * 4.0
                + phrase_bonus
                + heading_bonus
                + section_bonus
                + density_bonus
                + rank_bonus
                + source_bonus
                + intent_bonus
            )
            ranked.append((final_score, chunk))

        ranked.sort(key=lambda item: item[0], reverse=True)
        results: list[SearchResult] = []
        for score, chunk in ranked:
            snippet, match_terms = self.build_snippet(chunk, query)
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    strategy=strategy,
                    snippet=snippet,
                    citation=self.format_citation(chunk),
                    match_terms=match_terms,
                )
            )
        return self._select_diverse_results(query, results, limit)

    @staticmethod
    def _source_preference_bonus(source: str) -> float:
        normalized = source.lower()
        if normalized.endswith(".txt"):
            return 4.0
        if normalized.endswith(".md"):
            return 2.0
        if normalized.endswith(".pdf"):
            return -4.0
        return 0.0

    @staticmethod
    def _detect_query_intent(query: str) -> str:
        normalized = query.strip()
        if not normalized:
            return "general"
        if any(marker in normalized for marker in ("知识库", "主要讲什么", "主要内容", "讲什么")):
            return "overview"
        if any(marker in normalized for marker in ("选购", "推荐", "适合", "参数", "大户型")):
            return "selection"
        if any(marker in normalized for marker in ("维护", "保养", "多久", "更换", "清理")):
            return "maintenance"
        if any(marker in normalized for marker in ("故障", "异常", "回充", "出水", "排查", "无法", "报警", "APP")):
            return "troubleshooting"
        return "general"

    @staticmethod
    def _intent_relevance_bonus(intent: str, chunk: KnowledgeChunk) -> float:
        title = chunk.title
        if intent == "selection":
            if title == "选购指南":
                return 180.0
            if "选购指南" in title:
                return 60.0
            if "故障排除" in title:
                return -6.0
        if intent == "maintenance":
            if title == "维护保养":
                return 180.0
            if "维护保养" in title:
                return 60.0
            if "选购指南" in title:
                return -4.0
        if intent == "troubleshooting":
            if title == "故障排除":
                return 180.0
            if "故障排除" in title:
                return 60.0
            if "选购指南" in title:
                return -6.0
            if chunk.section_type == "qa":
                return 6.0
        if intent == "overview" and title in {"选购指南", "维护保养", "故障排除"}:
            return 120.0
        return 0.0

    def _select_diverse_results(self, query: str, results: list[SearchResult], limit: int) -> list[SearchResult]:
        if len(results) <= limit:
            return results[:limit]

        intent = self._detect_query_intent(query)
        max_per_source = 1 if intent == "overview" else 2
        if "app" in query.lower():
            max_per_source = 1
        source_counts: dict[str, int] = {}
        selected: list[SearchResult] = []
        deferred: list[SearchResult] = []
        for item in results:
            current = source_counts.get(item.chunk.source, 0)
            if current >= max_per_source:
                deferred.append(item)
                continue
            source_counts[item.chunk.source] = current + 1
            selected.append(item)
            if len(selected) >= limit:
                return selected

        for item in deferred:
            selected.append(item)
            if len(selected) >= limit:
                break
        return selected

    def _candidate_pool_size(self, limit: int) -> int:
        return max(limit, self.settings.rerank_pool_size, self.settings.top_k * 2)

    def _expand_query_for_retrieval(self, query: str) -> str:
        expansions = list(self._intent_expansion_terms(query))
        if not expansions:
            return query
        return f"{query} {' '.join(expansions[:4])}"

    def _overview_search(self, query: str, limit: int) -> list[SearchResult]:
        preferred_titles = ("选购指南", "维护保养", "故障排除")
        selected_chunks: list[KnowledgeChunk] = []
        seen_sources: set[str] = set()
        for preferred_title in preferred_titles:
            for chunk in self._chunks:
                if chunk.title != preferred_title or chunk.source in seen_sources:
                    continue
                selected_chunks.append(replace(chunk, content=f"{chunk.title}\n{chunk.content}"))
                seen_sources.add(chunk.source)
                break

        if not selected_chunks:
            return []
        candidates = [(chunk, 5.0) for chunk in selected_chunks]
        return self._rerank_results(query, candidates, limit, strategy="overview")

    def _build_signature(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.settings.embedding_model.encode("utf-8"))
        digest.update(str(self.settings.embedding_dimensions).encode("utf-8"))
        for chunk in self._chunks:
            digest.update(chunk.chunk_id.encode("utf-8"))
            digest.update(chunk.content.encode("utf-8"))
            digest.update("|".join(chunk.heading_path).encode("utf-8"))
        return digest.hexdigest()

    @staticmethod
    def _batched(items: list[KnowledgeChunk], size: int) -> Iterable[list[KnowledgeChunk]]:
        for index in range(0, len(items), size):
            yield items[index : index + size]

    def _score(self, query: str, chunk: KnowledgeChunk) -> float:
        expanded_query = f"{query} {' '.join(self._intent_expansion_terms(query))}".strip()
        query_tokens = Counter(self._tokenize(expanded_query))
        if not query_tokens:
            return 0.0

        heading_text = " ".join(chunk.heading_path)
        haystack = f"{chunk.title}\n{heading_text}\n{chunk.content}"
        chunk_tokens = Counter(self._tokenize(haystack))
        overlap = sum(min(count, chunk_tokens[token]) for token, count in query_tokens.items())
        if overlap == 0 and query.lower() not in haystack.lower():
            return 0.0

        title_tokens = set(self._tokenize(f"{chunk.title} {heading_text}"))
        title_bonus = sum(1 for token in query_tokens if token in title_tokens) * 1.5
        exact_bonus = 2.0 if query.lower() in haystack.lower() else 0.0
        density_bonus = overlap / max(len(query_tokens), 1)
        section_bonus = SECTION_PRIORITY.get(chunk.section_type, 0.0)
        return overlap * 2.0 + title_bonus + exact_bonus + density_bonus + section_bonus

    @staticmethod
    def _intent_expansion_terms(query: str) -> tuple[str, ...]:
        intent = KnowledgeBase._detect_query_intent(query)
        if intent == "selection":
            return ("选购", "参数", "续航", "断点续扫", "水箱", "大户型")
        if intent == "maintenance":
            return ("维护", "保养", "滚刷", "边刷", "滤网", "更换")
        if intent == "troubleshooting":
            return ("故障", "排查", "回充", "APP", "WiFi", "出水")
        if intent == "overview":
            return ("选购", "维护", "故障")
        return ()

    @staticmethod
    def _score_text_overlap(query_terms: list[str], text: str) -> float:
        if not query_terms or not text.strip():
            return 0.0
        text_tokens = Counter(KnowledgeBase._tokenize(text))
        overlap = sum(1 for term in query_terms if term in text_tokens)
        return overlap / max(len(set(query_terms)), 1)

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
