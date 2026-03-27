from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4

from hello_agents.core.message import Message

from hello_rag_agent.memory_profile import extract_profile_facts, format_profile_fact_line
from hello_rag_agent.memory_store import MemoryRecord, SQLiteMemoryStore, StoredProfileFact, StoredProfileFactHistory


@dataclass(frozen=True)
class MemorySearchBundle:
    fact_matches: list[tuple[float, StoredProfileFact]]
    entry_matches: list[tuple[float, MemoryRecord]]


class MemoryManager:
    def __init__(
        self,
        *,
        store: SQLiteMemoryStore,
        default_top_k: int,
        working_max_entries: int,
        working_ttl_minutes: int,
        profile_enabled: bool,
        profile_max_facts: int,
        assistant_memory_min_chars: int,
    ):
        self.store = store
        self.default_top_k = default_top_k
        self.working_max_entries = working_max_entries
        self.working_ttl_minutes = working_ttl_minutes
        self.profile_enabled = profile_enabled
        self.profile_max_facts = profile_max_facts
        self.assistant_memory_min_chars = assistant_memory_min_chars

    def remember_message(self, *, user_id: str, session_id: str, message: Message) -> MemoryRecord | None:
        content = message.content.strip()
        if not content:
            return None

        if message.role == "assistant" and len(content) < self.assistant_memory_min_chars:
            return None

        memory_type = "episodic" if message.role == "assistant" else "working"
        importance = 0.65 if message.role == "assistant" else 0.55
        return self.add_memory(
            user_id=user_id,
            session_id=session_id,
            content=content,
            role=message.role,
            memory_type=memory_type,
            importance=importance,
            metadata={"session_id": session_id, "user_id": user_id, "source": "conversation"},
            timestamp=message.timestamp,
        )

    def add_memory(
        self,
        *,
        user_id: str,
        session_id: str,
        content: str,
        role: str,
        memory_type: str,
        importance: float,
        metadata: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> MemoryRecord:
        record = MemoryRecord(
            entry_id=uuid4().hex,
            user_id=user_id,
            session_id=session_id,
            role=role,
            content=content.strip(),
            memory_type=memory_type,
            timestamp=timestamp or datetime.now(),
            importance=max(0.0, min(float(importance), 1.0)),
            source=str((metadata or {}).get("source", "conversation")),
            metadata=dict(metadata or {}),
        )
        self.store.add_entry(record)
        self._ingest_profile_facts(record)
        self.enforce_working_limits(session_id)
        return record

    def update_memory(
        self,
        entry_id: str,
        *,
        content: str | None = None,
        importance: float | None = None,
        memory_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord | None:
        updated = self.store.update_entry(
            entry_id,
            content=content,
            importance=importance,
            memory_type=memory_type,
            metadata=metadata,
        )
        if updated is None:
            return None
        self._ingest_profile_facts(updated)
        self.enforce_working_limits(updated.session_id)
        return updated

    def remove_memory(self, entry_id: str) -> MemoryRecord | None:
        current = self.store.get_entry(entry_id)
        if current is None:
            return None
        if not self.store.remove_entry(entry_id):
            return None
        return current

    def search(
        self,
        *,
        user_id: str,
        session_id: str,
        query: str,
        top_k: int | None = None,
        memory_types: list[str] | None = None,
    ) -> MemorySearchBundle:
        limit = top_k or self.default_top_k
        self.enforce_working_limits(session_id)
        fact_matches = self.store.search_facts(user_id=user_id, query=query, top_k=limit)
        entry_matches = self.store.search_entries(
            session_id=session_id,
            query=query,
            top_k=max(limit * 2, 6),
            memory_types=memory_types,
        )
        return MemorySearchBundle(fact_matches=fact_matches, entry_matches=entry_matches)

    def get_profile_lines(self, *, user_id: str, query: str | None = None, limit: int | None = None) -> list[str]:
        if not self.profile_enabled:
            return []

        fetch_limit = limit or self.profile_max_facts
        if query:
            facts = [fact for _, fact in self.store.search_facts(user_id=user_id, query=query, top_k=fetch_limit)]
        else:
            facts = self.store.get_facts(user_id=user_id, limit=fetch_limit)
        return [format_profile_fact_line(fact.fact_key, fact.fact_value) for fact in facts]

    def get_profile_history_lines(
        self,
        *,
        user_id: str,
        fact_key: str | None = None,
        limit: int = 10,
    ) -> list[str]:
        history = self.store.list_fact_history(user_id=user_id, fact_key=fact_key, limit=limit)
        return [self._format_history_line(item) for item in history]

    def clear_session_context(self, session_id: str) -> int:
        return self.store.clear_session(session_id)

    def clear_all_for_user_session(self, *, user_id: str, session_id: str) -> dict[str, int]:
        return {
            "entries_cleared": self.store.clear_session(session_id),
            "facts_cleared": self.store.clear_user_facts(user_id),
        }

    def summarize(self, *, user_id: str, session_id: str, limit: int = 5) -> dict[str, Any]:
        recent_entries = self.store.list_recent_entries(session_id, limit=limit)
        current_facts = self.get_profile_lines(user_id=user_id, limit=min(limit, self.profile_max_facts))
        fact_history = self.get_profile_history_lines(user_id=user_id, limit=min(limit, self.profile_max_facts))
        return {
            "user_id": user_id,
            "session_id": session_id,
            "entry_count": self.store.count_entries(session_id),
            "fact_count": self.store.fact_count(user_id),
            "counts_by_type": self.store.count_by_type(session_id),
            "counts_by_role": self.store.count_by_role(session_id),
            "recent_entries": [
                {
                    "entry_id": entry.entry_id,
                    "role": entry.role,
                    "memory_type": entry.memory_type,
                    "content": entry.content if len(entry.content) <= 160 else f"{entry.content[:160]}...",
                    "timestamp": entry.timestamp.isoformat(),
                }
                for entry in recent_entries
            ],
            "profile_facts": current_facts,
            "profile_fact_history": fact_history,
        }

    def enforce_working_limits(self, session_id: str) -> None:
        self.store.purge_expired_working(
            session_id,
            ttl_minutes=self.working_ttl_minutes,
            keep_latest_n=min(4, self.working_max_entries),
        )
        self.store.trim_working_entries(session_id, max_entries=self.working_max_entries)

    def _ingest_profile_facts(self, record: MemoryRecord) -> None:
        if not self.profile_enabled or record.role == "assistant":
            return

        facts = extract_profile_facts(record.content)
        for fact in facts[: self.profile_max_facts]:
            self.store.upsert_fact(
                user_id=record.user_id,
                session_id=record.session_id,
                fact_key=fact.fact_key,
                fact_value=fact.fact_value,
                confidence=fact.confidence,
                source_entry_id=record.entry_id,
            )

    @staticmethod
    def _format_history_line(item: StoredProfileFactHistory) -> str:
        marker = "当前" if item.is_current else "历史"
        date_text = item.recorded_at.strftime("%Y-%m-%d %H:%M")
        return f"{marker} | {date_text} | {format_profile_fact_line(item.fact_key, item.fact_value)}"
