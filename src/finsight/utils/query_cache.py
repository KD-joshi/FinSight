"""Session-scoped cache for final RAG answers and their source documents."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
import unicodedata
from pathlib import Path
from typing import Any

from config.settings import settings

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60
_MAX_CACHE_ENTRIES = 1000


class QueryCache:
    """Persist final answers, isolated by session, corpus revision and prompts."""

    def __init__(self, cache_file: Path | None = None) -> None:
        self.cache_file = cache_file

    def _path(self) -> Path:
        return self.cache_file or (settings.cache_dir / "session_query_cache.sqlite")

    def _connect(self) -> sqlite3.Connection:
        path = self._path()
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS scope_revisions (
                scope_id TEXT PRIMARY KEY,
                revision INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS query_entries (
                cache_key TEXT PRIMARY KEY,
                scope_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
                answer TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_query_entries_created_at
                ON query_entries(created_at);
            """
        )
        return connection

    @staticmethod
    def _normalized_query(query: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", query).casefold().split())

    def _key(
        self,
        query: str,
        scope_id: str,
        revision: int,
        prompt_fingerprint: str,
        model_fingerprint: str,
    ) -> str:
        payload = json.dumps(
            [scope_id, revision, self._normalized_query(query), prompt_fingerprint, model_fingerprint],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def lookup(
        self,
        query: str,
        *,
        scope_id: str | None,
        prompt_fingerprint: str,
        model_fingerprint: str,
        bypass: bool = False,
    ) -> tuple[int | None, dict[str, Any] | None]:
        """Return the observed scope revision and a matching entry, if present."""
        if not scope_id or not query.strip():
            return None, None

        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            row = connection.execute(
                "SELECT revision FROM scope_revisions WHERE scope_id = ?",
                (scope_id,),
            ).fetchone()
            revision = int(row["revision"]) if row else 0
            if bypass:
                return revision, None
            key = self._key(query, scope_id, revision, prompt_fingerprint, model_fingerprint)
            entry = connection.execute(
                "SELECT answer, sources_json, created_at FROM query_entries WHERE cache_key = ?",
                (key,),
            ).fetchone()

            if not entry:
                return revision, None
            if time.time() - float(entry["created_at"]) > _CACHE_TTL_SECONDS:
                connection.execute("DELETE FROM query_entries WHERE cache_key = ?", (key,))
                connection.commit()
                return revision, None

            try:
                sources = json.loads(entry["sources_json"])
                if not isinstance(sources, list):
                    raise ValueError("Cached sources must be a list")
            except (TypeError, ValueError, json.JSONDecodeError):
                connection.execute("DELETE FROM query_entries WHERE cache_key = ?", (key,))
                connection.commit()
                return revision, None

            logger.info("Session query cache hit for scope %s", scope_id)
            return revision, {"answer": entry["answer"], "documents": sources}
        except Exception:
            logger.warning("Query cache lookup failed; continuing without cache.", exc_info=True)
            return None, None
        finally:
            if connection is not None:
                connection.close()

    def get_scope_revision(self, scope_id: str | None) -> int | None:
        """Read the current revision after an ingestion operation."""
        if not scope_id:
            return None

        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            row = connection.execute(
                "SELECT revision FROM scope_revisions WHERE scope_id = ?",
                (scope_id,),
            ).fetchone()
            return int(row["revision"]) if row else 0
        except Exception:
            logger.warning("Could not read cache revision for scope %s.", scope_id, exc_info=True)
            return None
        finally:
            if connection is not None:
                connection.close()

    def set(
        self,
        query: str,
        answer: str,
        documents: list[dict[str, Any]],
        *,
        scope_id: str | None,
        expected_revision: int | None,
        prompt_fingerprint: str,
        model_fingerprint: str,
    ) -> None:
        """Save a final answer only if the session corpus has not changed."""
        if not scope_id or expected_revision is None or not answer.strip() or not documents:
            return

        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT revision FROM scope_revisions WHERE scope_id = ?",
                (scope_id,),
            ).fetchone()
            current_revision = int(row["revision"]) if row else 0
            if current_revision != expected_revision:
                logger.info("Skipping cache write because scope %s changed during generation.", scope_id)
                connection.rollback()
                return

            key = self._key(query, scope_id, current_revision, prompt_fingerprint, model_fingerprint)
            connection.execute(
                """
                INSERT OR REPLACE INTO query_entries
                    (cache_key, scope_id, revision, answer, sources_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    scope_id,
                    current_revision,
                    answer,
                    json.dumps(documents, ensure_ascii=False, default=str),
                    time.time(),
                ),
            )
            connection.execute(
                "DELETE FROM query_entries WHERE created_at < ?",
                (time.time() - _CACHE_TTL_SECONDS,),
            )
            connection.execute(
                "DELETE FROM query_entries WHERE cache_key IN "
                "(SELECT cache_key FROM query_entries ORDER BY created_at DESC LIMIT -1 OFFSET ?)",
                (_MAX_CACHE_ENTRIES,),
            )
            connection.commit()
        except Exception:
            if connection is not None:
                connection.rollback()
            logger.warning("Query cache write failed; continuing without cache.", exc_info=True)
        finally:
            if connection is not None:
                connection.close()

    def bump_scope_revision(self, scope_id: str | None) -> None:
        """Invalidate prior answers after documents in a session are changed."""
        if not scope_id:
            return

        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            connection.execute(
                """
                INSERT INTO scope_revisions (scope_id, revision) VALUES (?, 1)
                ON CONFLICT(scope_id) DO UPDATE SET revision = revision + 1
                """,
                (scope_id,),
            )
            connection.commit()
        except Exception:
            if connection is not None:
                connection.rollback()
            logger.warning("Could not advance cache revision for scope %s.", scope_id, exc_info=True)
        finally:
            if connection is not None:
                connection.close()


query_cache = QueryCache()
