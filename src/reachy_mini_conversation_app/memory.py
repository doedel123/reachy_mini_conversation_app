from __future__ import annotations

import json
import sqlite3
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Iterable

from reachy_mini_conversation_app.config import config


logger = logging.getLogger(__name__)

DEFAULT_MEMORY_CATEGORY = "other"
LAST_ACTIVE_MEMORY_USER_KEY = "last_active_user_id"
MEMORY_CATEGORIES = (
    "identity",
    "preference",
    "relationship",
    "routine",
    "project",
    "context",
    "other",
)


def _utcnow_iso() -> str:
    """Return a compact UTC timestamp string."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _normalize_tags(tags: Iterable[str] | None) -> list[str]:
    """Normalize tags to a stable, deduplicated list."""
    if tags is None:
        return []

    normalized = {
        tag.strip()
        for tag in tags
        if isinstance(tag, str) and tag.strip()
    }
    return sorted(normalized)


def _normalize_category(category: str | None) -> str:
    """Map unknown categories to the default category."""
    value = (category or DEFAULT_MEMORY_CATEGORY).strip().lower()
    return value if value in MEMORY_CATEGORIES else DEFAULT_MEMORY_CATEGORY


def _resolve_memory_db_path(raw_path: str | Path | None = None) -> Path:
    """Resolve the configured memory DB path."""
    candidate = Path(raw_path or config.REACHY_MINI_MEMORY_DB_PATH).expanduser()
    return candidate if candidate.is_absolute() else Path.cwd() / candidate


def get_default_memory_user_id() -> str:
    """Return the configured default memory identity."""
    value = (config.REACHY_MINI_MEMORY_USER_ID or "default").strip()
    return value or "default"


def normalize_memory_user_id(raw_value: str | None) -> str:
    """Normalize a spoken or configured identity into a stable user ID."""
    value = (raw_value or "").strip().lower()
    if not value:
        return get_default_memory_user_id()

    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"[\s_-]+", "-", value).strip("-")
    return value or get_default_memory_user_id()


@dataclass
class MemoryEntry:
    """A persisted memory row."""

    id: int
    user_id: str
    category: str
    fact: str
    tags: list[str]
    created_at: str
    updated_at: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return asdict(self)


class MemoryStore:
    """SQLite-backed memory store for durable user facts."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = _resolve_memory_db_path(db_path)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection with row access by name."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        """Create the memories table if needed."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    fact TEXT NOT NULL,
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    tags_text TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    forgotten_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_active_user_updated
                ON memories(user_id, forgotten_at, updated_at DESC)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
        """Convert a SQLite row into a MemoryEntry."""
        tags_raw = row["tags_json"] or "[]"
        try:
            tags = json.loads(tags_raw)
        except json.JSONDecodeError:
            tags = []

        return MemoryEntry(
            id=int(row["id"]),
            user_id=str(row["user_id"]),
            category=str(row["category"]),
            fact=str(row["fact"]),
            tags=[str(tag) for tag in tags if isinstance(tag, str)],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def remember_fact(
        self,
        *,
        user_id: str,
        fact: str,
        category: str | None = None,
        tags: Iterable[str] | None = None,
    ) -> tuple[str, MemoryEntry]:
        """Insert or update a durable fact."""
        normalized_fact = fact.strip()
        if not normalized_fact:
            raise ValueError("fact must be a non-empty string")

        normalized_category = _normalize_category(category)
        normalized_tags = _normalize_tags(tags)
        now = _utcnow_iso()

        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT *
                FROM memories
                WHERE user_id = ?
                  AND category = ?
                  AND lower(fact) = lower(?)
                  AND forgotten_at IS NULL
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (user_id, normalized_category, normalized_fact),
            ).fetchone()

            if existing is None:
                cursor = conn.execute(
                    """
                    INSERT INTO memories(user_id, category, fact, tags_json, tags_text, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        normalized_category,
                        normalized_fact,
                        json.dumps(normalized_tags),
                        " ".join(normalized_tags),
                        now,
                        now,
                    ),
                )
                row = conn.execute("SELECT * FROM memories WHERE id = ?", (cursor.lastrowid,)).fetchone()
                assert row is not None
                return "saved", self._row_to_entry(row)

            merged_tags = _normalize_tags([*json.loads(existing["tags_json"] or "[]"), *normalized_tags])
            conn.execute(
                """
                UPDATE memories
                SET tags_json = ?, tags_text = ?, updated_at = ?
                WHERE id = ?
                """,
                (json.dumps(merged_tags), " ".join(merged_tags), now, existing["id"]),
            )
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (existing["id"],)).fetchone()
            assert row is not None
            return "updated", self._row_to_entry(row)

    def recall_memories(
        self,
        *,
        user_id: str,
        query: str | None = None,
        category: str | None = None,
        limit: int = 5,
    ) -> list[MemoryEntry]:
        """Return active memories ordered by recency."""
        sql = [
            """
            SELECT *
            FROM memories
            WHERE user_id = ?
              AND forgotten_at IS NULL
            """
        ]
        params: list[Any] = [user_id]

        normalized_category = _normalize_category(category) if category else None
        if normalized_category:
            sql.append("AND category = ?")
            params.append(normalized_category)

        if query and query.strip():
            pattern = f"%{query.strip().lower()}%"
            sql.append("AND (lower(fact) LIKE ? OR lower(tags_text) LIKE ? OR lower(category) LIKE ?)")
            params.extend([pattern, pattern, pattern])

        sql.append("ORDER BY updated_at DESC LIMIT ?")
        params.append(max(1, min(int(limit), 50)))

        with self._connect() as conn:
            rows = conn.execute("\n".join(sql), params).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def forget_memories(
        self,
        *,
        user_id: str,
        memory_id: int | None = None,
        query: str | None = None,
        category: str | None = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Mark one or more memories as forgotten and return what changed."""
        if memory_id is None and not (query and query.strip()):
            raise ValueError("memory_id or query is required")

        with self._connect() as conn:
            if memory_id is not None:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM memories
                    WHERE id = ?
                      AND user_id = ?
                      AND forgotten_at IS NULL
                    """,
                    (memory_id, user_id),
                ).fetchall()
            else:
                pattern = f"%{query.strip().lower()}%"
                sql = [
                    """
                    SELECT *
                    FROM memories
                    WHERE user_id = ?
                      AND forgotten_at IS NULL
                      AND (lower(fact) LIKE ? OR lower(tags_text) LIKE ? OR lower(category) LIKE ?)
                    """
                ]
                params: list[Any] = [user_id, pattern, pattern, pattern]
                if category:
                    sql.append("AND category = ?")
                    params.append(_normalize_category(category))
                sql.append("ORDER BY updated_at DESC LIMIT ?")
                params.append(max(1, min(int(limit), 50)))
                rows = conn.execute("\n".join(sql), params).fetchall()

            entries = [self._row_to_entry(row) for row in rows]
            if entries:
                now = _utcnow_iso()
                conn.executemany(
                    "UPDATE memories SET forgotten_at = ?, updated_at = ? WHERE id = ?",
                    [(now, now, entry.id) for entry in entries],
                )
        return entries

    def format_for_prompt(self, *, user_id: str, limit: int = 12) -> str:
        """Format active memories as a compact prompt section."""
        entries = self.recall_memories(user_id=user_id, limit=limit)
        if not entries:
            return ""

        lines = [
            f"- [{entry.id}] {entry.category}: {entry.fact}"
            + (f" (tags: {', '.join(entry.tags)})" if entry.tags else "")
            for entry in entries
        ]
        return "## STORED MEMORY\nKnown durable facts from earlier conversations:\n" + "\n".join(lines)

    def get_last_active_user_id(self, *, default: str | None = None) -> str:
        """Return the most recently identified memory user ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM memory_state WHERE key = ?",
                (LAST_ACTIVE_MEMORY_USER_KEY,),
            ).fetchone()

        if row is None:
            return normalize_memory_user_id(default)
        return normalize_memory_user_id(str(row["value"]))

    def set_last_active_user_id(self, user_id: str) -> str:
        """Persist the most recently identified memory user ID."""
        normalized_user_id = normalize_memory_user_id(user_id)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_state(key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (LAST_ACTIVE_MEMORY_USER_KEY, normalized_user_id, _utcnow_iso()),
            )
        return normalized_user_id
