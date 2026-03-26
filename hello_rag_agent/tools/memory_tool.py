from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math
import re
from typing import Any
from uuid import uuid4

from hello_agents.context import ContextPacket
from hello_agents.core.message import Message
from hello_agents.tools.base import Tool, ToolParameter
from hello_agents.tools.errors import ToolErrorCode
from hello_agents.tools.response import ToolResponse


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
SUPPORTED_MEMORY_TYPES = {"working", "episodic", "semantic", "perceptual"}


@dataclass(frozen=True)
class MemoryEntry:
    entry_id: str
    role: str
    content: str
    memory_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryTool(Tool):
    """Session-scoped memory manager with minimal lifecycle actions."""

    def __init__(
        self,
        session_id: str,
        default_top_k: int = 3,
        max_entries: int = 200,
    ):
        super().__init__(
            name="memory_tool",
            description=(
                "Manage session memory. Use action=search to retrieve prior context, "
                "action=add to store memory, action=summary for a compact recap, "
                "action=stats for diagnostics, and action=clear_all to reset memory."
            ),
        )
        self.session_id = session_id
        self.default_top_k = default_top_k
        self.max_entries = max_entries
        self._entries: list[MemoryEntry] = []

    def remember_message(self, message: Message) -> None:
        content = message.content.strip()
        if not content:
            return

        memory_type = "episodic" if message.role == "assistant" else "working"
        importance = 0.65 if message.role == "assistant" else 0.55
        self.add(
            content=content,
            role=message.role,
            memory_type=memory_type,
            importance=importance,
            metadata={"session_id": self.session_id, "source": "conversation"},
            timestamp=message.timestamp,
        )

    def add(
        self,
        *,
        content: str,
        role: str = "system",
        memory_type: str = "working",
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> MemoryEntry:
        normalized_content = content.strip()
        if not normalized_content:
            raise ValueError("Memory content cannot be empty.")

        normalized_type = self._normalize_memory_type(memory_type)
        entry = MemoryEntry(
            entry_id=uuid4().hex,
            role=role,
            content=normalized_content,
            memory_type=normalized_type,
            timestamp=timestamp or datetime.now(),
            importance=self._clamp_importance(importance),
            metadata=dict(metadata or {}),
        )
        self._entries.append(entry)
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries :]
        return entry

    def clear(self) -> None:
        self.clear_all()

    def clear_all(self) -> None:
        self._entries.clear()

    def update_entry(
        self,
        entry_id: str,
        *,
        content: str | None = None,
        importance: float | None = None,
        memory_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry | None:
        for index, entry in enumerate(self._entries):
            if entry.entry_id != entry_id:
                continue

            updated = MemoryEntry(
                entry_id=entry.entry_id,
                role=entry.role,
                content=content.strip() if isinstance(content, str) and content.strip() else entry.content,
                memory_type=self._normalize_memory_type(memory_type or entry.memory_type),
                timestamp=entry.timestamp,
                importance=self._clamp_importance(importance if importance is not None else entry.importance),
                metadata=dict(metadata) if metadata is not None else dict(entry.metadata),
            )
            self._entries[index] = updated
            return updated
        return None

    def remove_entry(self, entry_id: str) -> MemoryEntry | None:
        for index, entry in enumerate(self._entries):
            if entry.entry_id == entry_id:
                return self._entries.pop(index)
        return None

    def search(
        self,
        query: str,
        top_k: int | None = None,
        memory_types: list[str] | None = None,
    ) -> list[MemoryEntry]:
        normalized_query = query.strip()
        if not normalized_query:
            return []

        query_tokens = self._tokenize(normalized_query)
        if not query_tokens:
            return []

        normalized_types = {self._normalize_memory_type(item) for item in memory_types or []}
        scored: list[tuple[float, MemoryEntry]] = []
        now = datetime.now()

        for entry in self._entries:
            if normalized_types and entry.memory_type not in normalized_types:
                continue

            content_tokens = self._tokenize(entry.content)
            if not content_tokens:
                continue

            overlap = sum(
                min(query_tokens.count(token), content_tokens.count(token))
                for token in set(query_tokens)
            )
            if overlap == 0 and normalized_query not in entry.content:
                continue

            age_seconds = max((now - entry.timestamp).total_seconds(), 0.0)
            recency_bonus = math.exp(-age_seconds / 3600.0)
            exact_bonus = 1.5 if normalized_query in entry.content else 0.0
            role_bonus = 0.2 if entry.role == "assistant" else 0.0
            type_bonus = 0.15 if entry.memory_type == "working" else 0.05
            score = overlap * 2.0 + recency_bonus + exact_bonus + role_bonus + type_bonus + entry.importance
            scored.append((score, entry))

        scored.sort(key=lambda item: item[0], reverse=True)
        limit = top_k or self.default_top_k
        return [entry for _, entry in scored[:limit]]

    def render_context(
        self,
        query: str,
        top_k: int | None = None,
        memory_types: list[str] | None = None,
    ) -> str:
        results = self.search(query, top_k=top_k, memory_types=memory_types)
        if not results:
            return "No relevant session memory found."

        lines = []
        for index, entry in enumerate(results, start=1):
            preview = entry.content if len(entry.content) <= 320 else f"{entry.content[:320]}..."
            lines.append(f"[Memory {index}] role: {entry.role}")
            lines.append(f"type: {entry.memory_type}")
            lines.append(f"content: {preview}")
            lines.append("")
        return "\n".join(lines).strip()

    def build_context_packet(
        self,
        query: str,
        top_k: int | None = None,
        memory_types: list[str] | None = None,
    ) -> ContextPacket | None:
        context = self.render_context(query, top_k=top_k, memory_types=memory_types)
        if context == "No relevant session memory found.":
            return None
        return ContextPacket(
            content=context,
            metadata={"type": "related_memory", "session_id": self.session_id},
        )

    def summary(self, limit: int = 5) -> dict[str, Any]:
        recent_entries = self._entries[-limit:]
        counts_by_type: dict[str, int] = {}
        counts_by_role: dict[str, int] = {}
        for entry in self._entries:
            counts_by_type[entry.memory_type] = counts_by_type.get(entry.memory_type, 0) + 1
            counts_by_role[entry.role] = counts_by_role.get(entry.role, 0) + 1

        preview = [
            {
                "entry_id": entry.entry_id,
                "role": entry.role,
                "memory_type": entry.memory_type,
                "content": entry.content if len(entry.content) <= 160 else f"{entry.content[:160]}...",
                "timestamp": entry.timestamp.isoformat(),
            }
            for entry in recent_entries
        ]
        return {
            "session_id": self.session_id,
            "entry_count": len(self._entries),
            "counts_by_type": counts_by_type,
            "counts_by_role": counts_by_role,
            "recent_entries": preview,
        }

    def stats(self) -> dict[str, Any]:
        summary = self.summary(limit=min(3, len(self._entries)))
        summary.update(
            {
                "max_entries": self.max_entries,
                "default_top_k": self.default_top_k,
                "supported_memory_types": sorted(SUPPORTED_MEMORY_TYPES),
            }
        )
        return summary

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        action = self._extract_action(parameters)

        if action == "add":
            content = self._extract_content(parameters)
            if not content:
                return ToolResponse.error(
                    code=ToolErrorCode.INVALID_PARAM,
                    message="The memory_tool add action requires content.",
                )

            try:
                entry = self.add(
                    content=content,
                    role=self._extract_role(parameters),
                    memory_type=self._extract_memory_type(parameters),
                    importance=self._extract_importance(parameters),
                    metadata=self._extract_metadata(parameters),
                )
            except ValueError as exc:
                return ToolResponse.error(
                    code=ToolErrorCode.INVALID_PARAM,
                    message=str(exc),
                )

            return ToolResponse.success(
                text=f"Stored memory entry {entry.entry_id} as {entry.memory_type} memory.",
                data={
                    "entry": {
                        "entry_id": entry.entry_id,
                        "role": entry.role,
                        "content": entry.content,
                        "memory_type": entry.memory_type,
                        "timestamp": entry.timestamp.isoformat(),
                        "importance": entry.importance,
                        "metadata": entry.metadata,
                    }
                },
            )

        if action == "search":
            query = self._extract_query(parameters)
            if not query:
                return ToolResponse.error(
                    code=ToolErrorCode.INVALID_PARAM,
                    message="The memory_tool search action requires a query.",
                )

            top_k = self._extract_top_k(parameters)
            memory_types = self._extract_memory_types(parameters)
            results = self.search(query, top_k=top_k, memory_types=memory_types)
            if not results:
                return ToolResponse.success(
                    text="No relevant session memory found.",
                    data={"matches": []},
                )

            return ToolResponse.success(
                text=self.render_context(query, top_k=top_k, memory_types=memory_types),
                data={
                    "matches": [
                        {
                            "entry_id": entry.entry_id,
                            "role": entry.role,
                            "content": entry.content,
                            "memory_type": entry.memory_type,
                            "timestamp": entry.timestamp.isoformat(),
                            "importance": entry.importance,
                            "metadata": entry.metadata,
                        }
                        for entry in results
                    ]
                },
            )

        if action == "summary":
            summary = self.summary()
            return ToolResponse.success(
                text=(
                    f"Memory summary for session {self.session_id}: "
                    f"{summary['entry_count']} entries across {len(summary['counts_by_type'])} memory types."
                ),
                data=summary,
            )

        if action == "stats":
            stats = self.stats()
            return ToolResponse.success(
                text=(
                    f"Memory stats for session {self.session_id}: "
                    f"{stats['entry_count']} entries, max {stats['max_entries']}."
                ),
                data=stats,
            )

        if action == "update":
            entry_id = self._extract_entry_id(parameters)
            if not entry_id:
                return ToolResponse.error(
                    code=ToolErrorCode.INVALID_PARAM,
                    message="The memory_tool update action requires entry_id.",
                )

            entry = self.update_entry(
                entry_id,
                content=self._extract_optional_content(parameters),
                importance=self._extract_optional_importance(parameters),
                memory_type=self._extract_optional_memory_type(parameters),
                metadata=self._extract_optional_metadata(parameters),
            )
            if entry is None:
                return ToolResponse.error(
                    code=ToolErrorCode.NOT_FOUND,
                    message=f"Memory entry {entry_id} was not found.",
                )

            return ToolResponse.success(
                text=f"Updated memory entry {entry.entry_id}.",
                data={
                    "entry": {
                        "entry_id": entry.entry_id,
                        "role": entry.role,
                        "content": entry.content,
                        "memory_type": entry.memory_type,
                        "timestamp": entry.timestamp.isoformat(),
                        "importance": entry.importance,
                        "metadata": entry.metadata,
                    }
                },
            )

        if action in {"remove", "forget"}:
            entry_id = self._extract_entry_id(parameters)
            if not entry_id:
                return ToolResponse.error(
                    code=ToolErrorCode.INVALID_PARAM,
                    message="The memory_tool remove action requires entry_id.",
                )

            removed = self.remove_entry(entry_id)
            if removed is None:
                return ToolResponse.error(
                    code=ToolErrorCode.NOT_FOUND,
                    message=f"Memory entry {entry_id} was not found.",
                )

            return ToolResponse.success(
                text=f"Removed memory entry {removed.entry_id}.",
                data={"removed_entry_id": removed.entry_id},
            )

        if action == "clear_all":
            removed_count = len(self._entries)
            self.clear_all()
            return ToolResponse.success(
                text=f"Cleared {removed_count} memory entries for session {self.session_id}.",
                data={"cleared": removed_count},
            )

        return ToolResponse.error(
            code=ToolErrorCode.INVALID_PARAM,
            message=f"Unsupported memory_tool action: {action}",
        )

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description=(
                    "Memory action to execute. Supported values: add, search, summary, "
                    "stats, update, remove, forget, clear_all."
                ),
                required=True,
            ),
            ToolParameter(
                name="query",
                type="string",
                description="Query used by action=search.",
                required=False,
            ),
            ToolParameter(
                name="content",
                type="string",
                description="Memory content used by action=add or action=update.",
                required=False,
            ),
            ToolParameter(
                name="entry_id",
                type="string",
                description="Target memory entry id used by update/remove/forget.",
                required=False,
            ),
            ToolParameter(
                name="role",
                type="string",
                description="Role associated with the stored memory entry.",
                required=False,
                default="system",
            ),
            ToolParameter(
                name="memory_type",
                type="string",
                description="One of working, episodic, semantic, perceptual.",
                required=False,
                default="working",
            ),
            ToolParameter(
                name="memory_types",
                type="array",
                description="Optional list of memory types to filter during search.",
                required=False,
            ),
            ToolParameter(
                name="importance",
                type="number",
                description="Importance score between 0.0 and 1.0.",
                required=False,
                default=0.5,
            ),
            ToolParameter(
                name="top_k",
                type="integer",
                description="Maximum number of memory hits to return for search.",
                required=False,
                default=self.default_top_k,
            ),
            ToolParameter(
                name="metadata",
                type="object",
                description="Optional metadata object for add/update.",
                required=False,
            ),
        ]

    @staticmethod
    def _extract_action(parameters: dict[str, Any]) -> str:
        value = parameters.get("action")
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
        return "search"

    @staticmethod
    def _extract_query(parameters: dict[str, Any]) -> str:
        for key in ("query", "input", "question"):
            value = parameters.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _extract_content(parameters: dict[str, Any]) -> str:
        value = parameters.get("content")
        return value.strip() if isinstance(value, str) and value.strip() else ""

    @staticmethod
    def _extract_optional_content(parameters: dict[str, Any]) -> str | None:
        value = parameters.get("content")
        return value if isinstance(value, str) else None

    @staticmethod
    def _extract_role(parameters: dict[str, Any]) -> str:
        value = parameters.get("role")
        return value.strip() if isinstance(value, str) and value.strip() else "system"

    @staticmethod
    def _extract_entry_id(parameters: dict[str, Any]) -> str:
        value = parameters.get("entry_id")
        return value.strip() if isinstance(value, str) and value.strip() else ""

    def _extract_memory_type(self, parameters: dict[str, Any]) -> str:
        value = parameters.get("memory_type")
        return self._normalize_memory_type(value if isinstance(value, str) else "working")

    def _extract_optional_memory_type(self, parameters: dict[str, Any]) -> str | None:
        value = parameters.get("memory_type")
        if isinstance(value, str) and value.strip():
            return self._normalize_memory_type(value)
        return None

    def _extract_memory_types(self, parameters: dict[str, Any]) -> list[str] | None:
        value = parameters.get("memory_types")
        if isinstance(value, list):
            normalized = []
            for item in value:
                if isinstance(item, str) and item.strip():
                    normalized.append(self._normalize_memory_type(item))
            return normalized or None

        single = parameters.get("memory_type")
        if isinstance(single, str) and single.strip():
            return [self._normalize_memory_type(single)]
        return None

    @staticmethod
    def _extract_top_k(parameters: dict[str, Any]) -> int | None:
        value = parameters.get("top_k")
        if isinstance(value, int) and value > 0:
            return value
        return None

    @staticmethod
    def _extract_importance(parameters: dict[str, Any]) -> float:
        value = parameters.get("importance")
        if isinstance(value, (int, float)):
            return float(value)
        return 0.5

    @staticmethod
    def _extract_optional_importance(parameters: dict[str, Any]) -> float | None:
        value = parameters.get("importance")
        if isinstance(value, (int, float)):
            return float(value)
        return None

    @staticmethod
    def _extract_metadata(parameters: dict[str, Any]) -> dict[str, Any]:
        value = parameters.get("metadata")
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _extract_optional_metadata(parameters: dict[str, Any]) -> dict[str, Any] | None:
        value = parameters.get("metadata")
        return dict(value) if isinstance(value, dict) else None

    @staticmethod
    def _normalize_memory_type(value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in SUPPORTED_MEMORY_TYPES:
            return "working"
        return normalized

    @staticmethod
    def _clamp_importance(value: float) -> float:
        return max(0.0, min(float(value), 1.0))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]
