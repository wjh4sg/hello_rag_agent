from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from hello_agents.context import ContextPacket
from hello_agents.core.message import Message
from hello_agents.tools.base import Tool, ToolParameter
from hello_agents.tools.errors import ToolErrorCode
from hello_agents.tools.response import ToolResponse

from hello_rag_agent.config import load_settings
from hello_rag_agent.memory_manager import MemoryManager
from hello_rag_agent.memory_profile import format_profile_fact_line
from hello_rag_agent.memory_store import MemoryRecord, SQLiteMemoryStore


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
    """Session-scoped memory tool backed by MemoryManager + SQLite store."""

    def __init__(
        self,
        session_id: str,
        user_id: str | None = None,
        default_top_k: int = 3,
        max_entries: int | None = None,
        *,
        store: SQLiteMemoryStore | None = None,
        manager: MemoryManager | None = None,
        working_ttl_minutes: int | None = None,
        profile_enabled: bool | None = None,
        profile_max_facts: int | None = None,
        assistant_memory_min_chars: int | None = None,
    ):
        super().__init__(
            name="memory_tool",
            description=(
                "Manage session memory. Use action=search to retrieve prior context, "
                "action=add to store memory, action=summary for a compact recap, "
                "action=stats for diagnostics, and action=clear_all to reset memory."
            ),
        )
        settings = load_settings().memory
        self.session_id = session_id
        self.user_id = user_id or f"session:{session_id}"
        self.default_top_k = default_top_k or settings.default_top_k
        self.max_entries = max_entries or settings.working_max_entries
        self.working_ttl_minutes = working_ttl_minutes or settings.working_ttl_minutes
        self.profile_enabled = settings.profile_enabled if profile_enabled is None else profile_enabled
        self.profile_max_facts = profile_max_facts or settings.profile_max_facts
        self.assistant_memory_min_chars = assistant_memory_min_chars or settings.assistant_memory_min_chars
        self.store = store or SQLiteMemoryStore(settings.db_path)
        self.manager = manager or MemoryManager(
            store=self.store,
            default_top_k=self.default_top_k,
            working_max_entries=self.max_entries,
            working_ttl_minutes=self.working_ttl_minutes,
            profile_enabled=self.profile_enabled,
            profile_max_facts=self.profile_max_facts,
            assistant_memory_min_chars=self.assistant_memory_min_chars,
        )

    def remember_message(self, message: Message) -> None:
        self.manager.remember_message(user_id=self.user_id, session_id=self.session_id, message=message)

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

        record = self.manager.add_memory(
            user_id=self.user_id,
            session_id=self.session_id,
            content=normalized_content,
            role=role,
            memory_type=self._normalize_memory_type(memory_type),
            importance=self._clamp_importance(importance),
            metadata=self._build_metadata(metadata),
            timestamp=timestamp,
        )
        return self._record_to_entry(record)

    def clear(self) -> None:
        self.clear_all()

    def clear_all(self) -> None:
        self.manager.clear_all_for_user_session(user_id=self.user_id, session_id=self.session_id)

    def clear_session_context(self) -> None:
        self.manager.clear_session_context(self.session_id)

    def update_entry(
        self,
        entry_id: str,
        *,
        content: str | None = None,
        importance: float | None = None,
        memory_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry | None:
        updated = self.manager.update_memory(
            entry_id,
            content=content,
            importance=self._clamp_importance(importance) if importance is not None else None,
            memory_type=self._normalize_memory_type(memory_type) if memory_type else None,
            metadata=metadata,
        )
        if updated is None:
            return None
        return self._record_to_entry(updated)

    def remove_entry(self, entry_id: str) -> MemoryEntry | None:
        removed = self.manager.remove_memory(entry_id)
        if removed is None:
            return None
        return self._record_to_entry(removed)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        memory_types: list[str] | None = None,
    ) -> list[MemoryEntry]:
        normalized_query = query.strip()
        if not normalized_query:
            return []

        normalized_types = [self._normalize_memory_type(item) for item in memory_types or []]
        bundle = self.manager.search(
            user_id=self.user_id,
            session_id=self.session_id,
            query=normalized_query,
            top_k=top_k or self.default_top_k,
            memory_types=normalized_types or None,
        )

        combined: list[tuple[float, MemoryEntry]] = []
        seen_contents: set[str] = set()

        for score, fact in bundle.fact_matches:
            line = format_profile_fact_line(fact.fact_key, fact.fact_value)
            if line in seen_contents:
                continue
            seen_contents.add(line)
            combined.append(
                (
                    score,
                    MemoryEntry(
                        entry_id=fact.fact_id,
                        role="profile",
                        content=line,
                        memory_type="semantic",
                        timestamp=fact.updated_at,
                        importance=fact.confidence,
                        metadata={
                            "fact_key": fact.fact_key,
                            "fact_value": fact.fact_value,
                            "source_entry_id": fact.source_entry_id,
                            "source": "profile_fact",
                            "user_id": fact.user_id,
                            "session_id": fact.session_id,
                        },
                    ),
                )
            )

        for score, record in bundle.entry_matches:
            entry = self._record_to_entry(record)
            normalized_content = entry.content.strip()
            if not normalized_content or normalized_content in seen_contents:
                continue
            seen_contents.add(normalized_content)
            combined.append((score, entry))

        combined.sort(key=lambda item: (item[0], item[1].timestamp), reverse=True)
        return [entry for _, entry in combined[: top_k or self.default_top_k]]

    def get_profile_lines(self, *, query: str | None = None, limit: int | None = None) -> list[str]:
        return self.manager.get_profile_lines(
            user_id=self.user_id,
            query=query,
            limit=limit or self.profile_max_facts,
        )

    def get_profile_history_lines(self, *, fact_key: str | None = None, limit: int | None = None) -> list[str]:
        return self.manager.get_profile_history_lines(
            user_id=self.user_id,
            fact_key=fact_key,
            limit=limit or self.profile_max_facts,
        )

    def build_profile_packet(self, *, query: str, top_k: int | None = None) -> ContextPacket | None:
        lines = self.get_profile_lines(query=query, limit=top_k or min(self.default_top_k, self.profile_max_facts))
        if not lines:
            return None
        content = "[User Profile]\n" + "\n".join(f"- {line}" for line in lines)
        return ContextPacket(
            content=content,
            metadata={"type": "user_profile", "session_id": self.session_id, "user_id": self.user_id},
        )

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
            metadata={"type": "related_memory", "session_id": self.session_id, "user_id": self.user_id},
        )

    def summary(self, limit: int = 5) -> dict[str, Any]:
        return self.manager.summarize(user_id=self.user_id, session_id=self.session_id, limit=limit)

    def stats(self) -> dict[str, Any]:
        summary = self.summary(limit=min(3, max(self.store.count_entries(self.session_id), 1)))
        summary.update(
            {
                "max_entries": self.max_entries,
                "default_top_k": self.default_top_k,
                "supported_memory_types": sorted(SUPPORTED_MEMORY_TYPES),
                "working_ttl_minutes": self.working_ttl_minutes,
                "profile_enabled": self.profile_enabled,
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
                data={"entry": self._serialize_entry(entry)},
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
                data={"matches": [self._serialize_entry(entry) for entry in results]},
            )

        if action == "summary":
            summary = self.summary()
            return ToolResponse.success(
                text=(
                    f"Memory summary for session {self.session_id} / user {self.user_id}: "
                    f"{summary['entry_count']} entries and {summary['fact_count']} profile facts."
                ),
                data=summary,
            )

        if action == "stats":
            stats = self.stats()
            return ToolResponse.success(
                text=(
                    f"Memory stats for session {self.session_id} / user {self.user_id}: "
                    f"{stats['entry_count']} entries, {stats['fact_count']} facts, max {stats['max_entries']}."
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
                data={"entry": self._serialize_entry(entry)},
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
            counts = self.manager.clear_all_for_user_session(user_id=self.user_id, session_id=self.session_id)
            return ToolResponse.success(
                text=(
                    f"Cleared {counts['entries_cleared']} session entries and "
                    f"{counts['facts_cleared']} user facts for {self.user_id}."
                ),
                data=counts,
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
            ToolParameter(name="query", type="string", description="Query used by action=search.", required=False),
            ToolParameter(name="content", type="string", description="Memory content used by action=add or action=update.", required=False),
            ToolParameter(name="entry_id", type="string", description="Target memory entry id used by update/remove/forget.", required=False),
            ToolParameter(name="role", type="string", description="Role associated with the stored memory entry.", required=False, default="system"),
            ToolParameter(name="memory_type", type="string", description="One of working, episodic, semantic, perceptual.", required=False, default="working"),
            ToolParameter(name="memory_types", type="array", description="Optional list of memory types to filter during search.", required=False),
            ToolParameter(name="importance", type="number", description="Importance score between 0.0 and 1.0.", required=False, default=0.5),
            ToolParameter(name="top_k", type="integer", description="Maximum number of memory hits to return for search.", required=False, default=self.default_top_k),
            ToolParameter(name="metadata", type="object", description="Optional metadata object for add/update.", required=False),
        ]

    def _build_metadata(self, metadata: dict[str, Any] | None) -> dict[str, Any]:
        payload = dict(metadata or {})
        payload.setdefault("session_id", self.session_id)
        payload.setdefault("user_id", self.user_id)
        payload.setdefault("source", payload.get("source", "conversation"))
        return payload

    @staticmethod
    def _record_to_entry(record: MemoryRecord) -> MemoryEntry:
        return MemoryEntry(
            entry_id=record.entry_id,
            role=record.role,
            content=record.content,
            memory_type=record.memory_type,
            timestamp=record.timestamp,
            importance=record.importance,
            metadata=dict(record.metadata),
        )

    @staticmethod
    def _serialize_entry(entry: MemoryEntry) -> dict[str, Any]:
        return {
            "entry_id": entry.entry_id,
            "role": entry.role,
            "content": entry.content,
            "memory_type": entry.memory_type,
            "timestamp": entry.timestamp.isoformat(),
            "importance": entry.importance,
            "metadata": entry.metadata,
        }

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
