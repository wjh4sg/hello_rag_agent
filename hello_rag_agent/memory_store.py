from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import re
import sqlite3
from threading import RLock
from typing import Any

from hello_rag_agent.memory_profile import format_profile_fact_line


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
FACT_QUERY_HINTS: dict[str, tuple[str, ...]] = {
    "floor_type": ("地面", "地板", "木地板"),
    "pets": ("宠物", "猫", "狗", "毛发"),
    "home_size": ("户型", "面积", "大户型", "小户型"),
    "noise_preference": ("偏好", "在意", "诉求", "静音"),
    "app_stability_priority": ("偏好", "在意", "诉求", "APP", "稳定"),
    "anti_tangle_priority": ("偏好", "在意", "诉求", "防缠绕"),
    "carpet_presence": ("地毯",),
    "maintenance_status": ("维护", "保养", "清理", "清洗", "更换", "滤网", "滚刷", "边刷"),
    "recent_issue": ("问题", "异常", "失败", "掉线", "故障", "回充"),
}


@dataclass(frozen=True)
class MemoryRecord:
    entry_id: str
    user_id: str
    session_id: str
    role: str
    content: str
    memory_type: str
    timestamp: datetime
    importance: float
    source: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class StoredProfileFact:
    fact_id: str
    user_id: str
    session_id: str
    fact_key: str
    fact_value: str
    confidence: float
    source_entry_id: str | None
    updated_at: datetime

    @property
    def content(self) -> str:
        return format_profile_fact_line(self.fact_key, self.fact_value)


@dataclass(frozen=True)
class StoredProfileFactHistory:
    history_id: str
    fact_id: str | None
    user_id: str
    session_id: str
    fact_key: str
    fact_value: str
    confidence: float
    source_entry_id: str | None
    recorded_at: datetime
    is_current: bool

    @property
    def content(self) -> str:
        return format_profile_fact_line(self.fact_key, self.fact_value)


class SQLiteMemoryStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._init_db()

    def add_entry(self, record: MemoryRecord) -> None:
        with self._lock, self._connect() as conn:
            payload = (
                record.entry_id,
                record.user_id,
                record.session_id,
                record.role,
                record.content,
                record.memory_type,
                record.timestamp.isoformat(),
                float(record.importance),
                record.source,
                json.dumps(record.metadata, ensure_ascii=False),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_entries (
                    entry_id, user_id, session_id, role, content, memory_type,
                    timestamp, importance, source, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )

    def get_entry(self, entry_id: str) -> MemoryRecord | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT entry_id, user_id, session_id, role, content, memory_type,
                       timestamp, importance, source, metadata_json
                FROM memory_entries
                WHERE entry_id = ?
                """,
                (entry_id,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def update_entry(
        self,
        entry_id: str,
        *,
        content: str | None = None,
        importance: float | None = None,
        memory_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord | None:
        current = self.get_entry(entry_id)
        if current is None:
            return None

        updated = MemoryRecord(
            entry_id=current.entry_id,
            user_id=current.user_id,
            session_id=current.session_id,
            role=current.role,
            content=content.strip() if isinstance(content, str) and content.strip() else current.content,
            memory_type=memory_type.strip().lower() if isinstance(memory_type, str) and memory_type.strip() else current.memory_type,
            timestamp=current.timestamp,
            importance=float(importance) if importance is not None else current.importance,
            source=current.source,
            metadata=dict(metadata) if metadata is not None else dict(current.metadata),
        )
        self.add_entry(updated)
        return updated

    def remove_entry(self, entry_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cursor = conn.execute("DELETE FROM memory_entries WHERE entry_id = ?", (entry_id,))
        return cursor.rowcount > 0

    def list_recent_entries(self, session_id: str, limit: int = 10) -> list[MemoryRecord]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT entry_id, user_id, session_id, role, content, memory_type,
                       timestamp, importance, source, metadata_json
                FROM memory_entries
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get_user_id_for_session(self, session_id: str) -> str | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT user_id
                FROM memory_entries
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        user_id = str(row["user_id"]).strip()
        return user_id or None

    def count_entries(self, session_id: str) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS count FROM memory_entries WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return int(row["count"]) if row else 0

    def count_by_type(self, session_id: str) -> dict[str, int]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT memory_type, COUNT(*) AS count
                FROM memory_entries
                WHERE session_id = ?
                GROUP BY memory_type
                """,
                (session_id,),
            ).fetchall()
        return {str(row["memory_type"]): int(row["count"]) for row in rows}

    def count_by_role(self, session_id: str) -> dict[str, int]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, COUNT(*) AS count
                FROM memory_entries
                WHERE session_id = ?
                GROUP BY role
                """,
                (session_id,),
            ).fetchall()
        return {str(row["role"]): int(row["count"]) for row in rows}

    def search_entries(
        self,
        *,
        session_id: str,
        query: str,
        top_k: int,
        memory_types: list[str] | None = None,
    ) -> list[tuple[float, MemoryRecord]]:
        normalized_query = query.strip()
        if not normalized_query:
            return []

        sql = """
            SELECT entry_id, user_id, session_id, role, content, memory_type,
                   timestamp, importance, source, metadata_json
            FROM memory_entries
            WHERE session_id = ?
        """
        params: list[Any] = [session_id]
        if memory_types:
            placeholders = ", ".join("?" for _ in memory_types)
            sql += f" AND memory_type IN ({placeholders})"
            params.extend(memory_types)
        sql += " ORDER BY timestamp DESC"

        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        records = [self._row_to_record(row) for row in rows]
        query_tokens = self._expand_query_tokens(normalized_query)
        scored: list[tuple[float, MemoryRecord]] = []
        now = datetime.now()
        for record in records:
            content_tokens = self._tokenize(record.content)
            if not content_tokens:
                continue

            overlap = sum(
                min(query_tokens.count(token), content_tokens.count(token))
                for token in set(query_tokens)
            )
            if overlap == 0 and normalized_query not in record.content:
                continue

            age_seconds = max((now - record.timestamp).total_seconds(), 0.0)
            recency_bonus = pow(2.718281828, -age_seconds / 3600.0)
            exact_bonus = 1.5 if normalized_query in record.content else 0.0
            role_bonus = 0.2 if record.role == "assistant" else 0.0
            type_bonus = 0.15 if record.memory_type == "working" else 0.05
            score = overlap * 2.0 + recency_bonus + exact_bonus + role_bonus + type_bonus + record.importance
            scored.append((score, record))

        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]

    def upsert_fact(
        self,
        *,
        user_id: str,
        session_id: str,
        fact_key: str,
        fact_value: str,
        confidence: float,
        source_entry_id: str | None = None,
    ) -> StoredProfileFact:
        now = datetime.now().isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE memory_fact_history
                SET is_current = 0
                WHERE user_id = ? AND fact_key = ? AND is_current = 1
                """,
                (user_id, fact_key),
            )
            existing = conn.execute(
                """
                SELECT fact_id, user_id, session_id, fact_key, fact_value,
                       confidence, source_entry_id, updated_at
                FROM memory_facts_current
                WHERE user_id = ? AND fact_key = ?
                """,
                (user_id, fact_key),
            ).fetchone()

            if existing is None:
                conn.execute(
                    """
                    INSERT INTO memory_facts_current (
                        fact_id, user_id, session_id, fact_key, fact_value,
                        confidence, source_entry_id, updated_at
                    ) VALUES (lower(hex(randomblob(16))), ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        session_id,
                        fact_key,
                        fact_value,
                        float(confidence),
                        source_entry_id,
                        now,
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE memory_facts_current
                    SET session_id = ?,
                        fact_value = ?,
                        confidence = ?,
                        source_entry_id = ?,
                        updated_at = ?
                    WHERE user_id = ? AND fact_key = ?
                    """,
                    (
                        session_id,
                        fact_value,
                        float(confidence),
                        source_entry_id,
                        now,
                        user_id,
                        fact_key,
                    ),
                )

            current_row = conn.execute(
                """
                SELECT fact_id, user_id, session_id, fact_key, fact_value,
                       confidence, source_entry_id, updated_at
                FROM memory_facts_current
                WHERE user_id = ? AND fact_key = ?
                """,
                (user_id, fact_key),
            ).fetchone()

            conn.execute(
                """
                INSERT INTO memory_fact_history (
                    history_id, fact_id, user_id, session_id, fact_key, fact_value,
                    confidence, source_entry_id, recorded_at, is_current
                ) VALUES (lower(hex(randomblob(16))), ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(current_row["fact_id"]),
                    user_id,
                    session_id,
                    fact_key,
                    fact_value,
                    float(confidence),
                    source_entry_id,
                    now,
                    1,
                ),
            )

        return self._row_to_fact(current_row)

    def get_facts(self, user_id: str, limit: int | None = None) -> list[StoredProfileFact]:
        sql = """
            SELECT fact_id, user_id, session_id, fact_key, fact_value,
                   confidence, source_entry_id, updated_at
            FROM memory_facts_current
            WHERE user_id = ?
            ORDER BY confidence DESC, updated_at DESC
        """
        params: list[Any] = [user_id]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_fact(row) for row in rows]

    def search_facts(self, *, user_id: str, query: str, top_k: int) -> list[tuple[float, StoredProfileFact]]:
        normalized_query = query.strip()
        if not normalized_query:
            return []

        facts = self.get_facts(user_id)
        query_tokens = self._expand_query_tokens(normalized_query)
        scored: list[tuple[float, StoredProfileFact]] = []
        for fact in facts:
            text = fact.content
            text_tokens = self._tokenize(text)
            overlap = sum(
                min(query_tokens.count(token), text_tokens.count(token))
                for token in set(query_tokens)
            )
            if overlap == 0 and normalized_query not in text:
                continue

            exact_bonus = 1.0 if normalized_query in text else 0.0
            fact_bonus = self._fact_query_bonus(fact.fact_key, normalized_query)
            score = overlap * 2.5 + exact_bonus + fact_bonus + fact.confidence
            scored.append((score, fact))

        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]

    def fact_count(self, user_id: str) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS count FROM memory_facts_current WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return int(row["count"]) if row else 0

    def list_fact_history(
        self,
        user_id: str,
        *,
        fact_key: str | None = None,
        limit: int = 20,
    ) -> list[StoredProfileFactHistory]:
        sql = """
            SELECT history_id, fact_id, user_id, session_id, fact_key, fact_value,
                   confidence, source_entry_id, recorded_at, is_current
            FROM memory_fact_history
            WHERE user_id = ?
        """
        params: list[Any] = [user_id]
        if fact_key is not None:
            sql += " AND fact_key = ?"
            params.append(fact_key)
        sql += " ORDER BY recorded_at DESC LIMIT ?"
        params.append(limit)

        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_fact_history(row) for row in rows]

    def clear_session(self, session_id: str) -> int:
        with self._lock, self._connect() as conn:
            count_row = conn.execute(
                "SELECT COUNT(*) AS count FROM memory_entries WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            conn.execute("DELETE FROM memory_entries WHERE session_id = ?", (session_id,))
        return int(count_row["count"]) if count_row else 0

    def clear_user_facts(self, user_id: str) -> int:
        with self._lock, self._connect() as conn:
            count_row = conn.execute(
                "SELECT COUNT(*) AS count FROM memory_facts_current WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            conn.execute("DELETE FROM memory_facts_current WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM memory_fact_history WHERE user_id = ?", (user_id,))
        return int(count_row["count"]) if count_row else 0

    def purge_expired_working(self, session_id: str, ttl_minutes: int, keep_latest_n: int) -> int:
        cutoff = (datetime.now() - timedelta(minutes=ttl_minutes)).isoformat()
        with self._lock, self._connect() as conn:
            recent_rows = conn.execute(
                """
                SELECT entry_id
                FROM memory_entries
                WHERE session_id = ? AND memory_type = 'working'
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (session_id, keep_latest_n),
            ).fetchall()
            protected_ids = {str(row["entry_id"]) for row in recent_rows}

            expired_rows = conn.execute(
                """
                SELECT entry_id
                FROM memory_entries
                WHERE session_id = ?
                  AND memory_type = 'working'
                  AND timestamp < ?
                """,
                (session_id, cutoff),
            ).fetchall()
            expired_ids = [
                str(row["entry_id"])
                for row in expired_rows
                if str(row["entry_id"]) not in protected_ids
            ]
            if not expired_ids:
                return 0

            placeholders = ", ".join("?" for _ in expired_ids)
            conn.execute(
                f"DELETE FROM memory_entries WHERE entry_id IN ({placeholders})",
                expired_ids,
            )
        return len(expired_ids)

    def trim_working_entries(self, session_id: str, max_entries: int) -> int:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT entry_id, timestamp, importance
                FROM memory_entries
                WHERE session_id = ? AND memory_type = 'working'
                ORDER BY timestamp DESC
                """,
                (session_id,),
            ).fetchall()

            if len(rows) <= max_entries:
                return 0

            now = datetime.now()
            scored: list[tuple[float, str]] = []
            for row in rows:
                timestamp = datetime.fromisoformat(str(row["timestamp"]))
                importance = float(row["importance"])
                age_seconds = max((now - timestamp).total_seconds(), 0.0)
                recency_bonus = pow(2.718281828, -age_seconds / 3600.0)
                score = importance + recency_bonus
                scored.append((score, str(row["entry_id"])))

            scored.sort(key=lambda item: item[0], reverse=True)
            remove_ids = [entry_id for _, entry_id in scored[max_entries:]]
            if not remove_ids:
                return 0

            placeholders = ", ".join("?" for _ in remove_ids)
            conn.execute(
                f"DELETE FROM memory_entries WHERE entry_id IN ({placeholders})",
                remove_ids,
            )
        return len(remove_ids)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_entries (
                    entry_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT '',
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    importance REAL NOT NULL,
                    source TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_facts_current (
                    fact_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    fact_key TEXT NOT NULL,
                    fact_value TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source_entry_id TEXT,
                    updated_at TEXT NOT NULL,
                    UNIQUE(user_id, fact_key)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_fact_history (
                    history_id TEXT PRIMARY KEY,
                    fact_id TEXT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    fact_key TEXT NOT NULL,
                    fact_value TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source_entry_id TEXT,
                    recorded_at TEXT NOT NULL,
                    is_current INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            self._ensure_column(conn, "memory_entries", "user_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "memory_facts_current", "user_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "memory_facts_current", "session_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "memory_fact_history", "fact_id", "TEXT")
            self._ensure_column(conn, "memory_fact_history", "user_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "memory_fact_history", "session_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "memory_fact_history", "recorded_at", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "memory_fact_history", "is_current", "INTEGER NOT NULL DEFAULT 1")
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_entries_session_time
                ON memory_entries(session_id, timestamp DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_entries_session_type
                ON memory_entries(session_id, memory_type)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_entries_user_time
                ON memory_entries(user_id, timestamp DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_facts_current_user_key
                ON memory_facts_current(user_id, fact_key)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_fact_history_user_time
                ON memory_fact_history(user_id, recorded_at DESC)
                """
            )

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, column_def: str) -> None:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        existing = {str(row["name"]) for row in rows}
        if column_name in existing:
            return
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]

    def _expand_query_tokens(self, query: str) -> list[str]:
        tokens = self._tokenize(query)
        expanded = list(tokens)
        lowered_query = query.lower()

        def extend_if_matches(markers: tuple[str, ...], additions: tuple[str, ...]) -> None:
            if any(marker.lower() in lowered_query for marker in markers):
                expanded.extend(item.lower() for item in additions)

        extend_if_matches(
            ("问题", "异常", "故障", "失败"),
            ("问题", "异常", "失败", "掉线", "断连", "不稳定", "回充"),
        )
        extend_if_matches(
            ("维护", "保养", "做过什么"),
            ("维护", "保养", "清洗", "清理", "更换", "滤网", "滚刷", "边刷"),
        )
        extend_if_matches(
            ("偏好", "在意", "诉求"),
            ("偏好", "静音", "防缠绕", "app", "稳定性"),
        )
        extend_if_matches(
            ("宠物", "毛发"),
            ("宠物", "猫", "狗", "长毛猫", "短毛猫"),
        )
        extend_if_matches(
            ("地面", "地板"),
            ("地面", "木地板", "瓷砖", "地砖", "大理石"),
        )
        return expanded

    @staticmethod
    def _fact_query_bonus(fact_key: str, query: str) -> float:
        hints = FACT_QUERY_HINTS.get(fact_key, ())
        if not hints:
            return 0.0
        lowered_query = query.lower()
        if any(hint.lower() in lowered_query for hint in hints):
            return 1.2
        return 0.0

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        metadata_raw = row["metadata_json"] if row["metadata_json"] else "{}"
        metadata = json.loads(str(metadata_raw))
        return MemoryRecord(
            entry_id=str(row["entry_id"]),
            user_id=str(row["user_id"]),
            session_id=str(row["session_id"]),
            role=str(row["role"]),
            content=str(row["content"]),
            memory_type=str(row["memory_type"]),
            timestamp=datetime.fromisoformat(str(row["timestamp"])),
            importance=float(row["importance"]),
            source=str(row["source"]),
            metadata=dict(metadata),
        )

    @staticmethod
    def _row_to_fact(row: sqlite3.Row) -> StoredProfileFact:
        return StoredProfileFact(
            fact_id=str(row["fact_id"]),
            user_id=str(row["user_id"]),
            session_id=str(row["session_id"]),
            fact_key=str(row["fact_key"]),
            fact_value=str(row["fact_value"]),
            confidence=float(row["confidence"]),
            source_entry_id=str(row["source_entry_id"]) if row["source_entry_id"] is not None else None,
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
        )

    @staticmethod
    def _row_to_fact_history(row: sqlite3.Row) -> StoredProfileFactHistory:
        return StoredProfileFactHistory(
            history_id=str(row["history_id"]),
            fact_id=str(row["fact_id"]) if row["fact_id"] is not None else None,
            user_id=str(row["user_id"]),
            session_id=str(row["session_id"]),
            fact_key=str(row["fact_key"]),
            fact_value=str(row["fact_value"]),
            confidence=float(row["confidence"]),
            source_entry_id=str(row["source_entry_id"]) if row["source_entry_id"] is not None else None,
            recorded_at=datetime.fromisoformat(str(row["recorded_at"])),
            is_current=bool(int(row["is_current"])),
        )
