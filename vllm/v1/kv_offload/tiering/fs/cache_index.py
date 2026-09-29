# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent index and bounded-storage policy for filesystem KV caches."""

from __future__ import annotations

import contextlib
import logging
import os
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable

try:
    import fcntl
except ImportError:  # pragma: no cover - the filesystem tier is Linux-first.
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

PROBATIONARY = 0
PROTECTED = 1
_LEASE_TTL_NS = 24 * 60 * 60 * 1_000_000_000


def default_group_key(root_dir: str, path: str) -> str:
    """Return the logical cache group for a block path.

    FileMapper puts every rank and KV group for one content hash below the
    same model namespace.  Removing the ``_r<rank>`` suffix means cleanup
    evicts all rank/group files for a hash together.
    """
    root = os.path.abspath(root_dir)
    absolute = os.path.abspath(path)
    try:
        relative = os.path.relpath(absolute, root)
    except ValueError:
        relative = absolute
    parts = relative.split(os.sep)
    if len(parts) >= 4 and parts[-1].endswith(".bin"):
        namespace = parts[0]
        rank_marker = namespace.rsplit("_r", 1)
        if len(rank_marker) == 2 and rank_marker[1].isdigit():
            namespace = rank_marker[0]
        return f"{namespace}/{parts[-1][:-4]}"
    return relative


class FileSystemCacheIndex:
    """SQLite index used to bound filesystem KV cache storage.

    The index is deliberately independent of the vLLM scheduler.  A process
    can restart or another process can share the same root without losing
    recency information.  SQLite coordinates metadata writes; a separate
    advisory lock serializes physical cleanup across processes.
    """

    def __init__(
        self,
        root_dir: str,
        group_key_fn: Callable[[str], str] | None = None,
    ) -> None:
        self.root_dir = os.path.abspath(root_dir)
        self.db_path = os.path.join(self.root_dir, ".vllm_fs_cache.sqlite3")
        self.lock_path = os.path.join(self.root_dir, ".vllm_fs_cache.lock")
        self._group_key_fn = group_key_fn or (
            lambda path: default_group_key(self.root_dir, path)
        )
        self._lock = threading.RLock()
        self._closed = False
        self._conn = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS entries (
                path TEXT PRIMARY KEY,
                group_key TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                queue INTEGER NOT NULL CHECK(queue IN (0, 1)),
                last_access_ns INTEGER NOT NULL,
                hit_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS entries_eviction
                ON entries(queue, last_access_ns);
            CREATE INDEX IF NOT EXISTS entries_group
                ON entries(group_key);
            CREATE TABLE IF NOT EXISTS leases (
                path TEXT NOT NULL,
                owner TEXT NOT NULL,
                expires_ns INTEGER NOT NULL,
                PRIMARY KEY(path, owner)
            );
            CREATE INDEX IF NOT EXISTS leases_expiry
                ON leases(expires_ns);
            """
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._conn.close()
            self._closed = True

    def reconcile(self) -> None:
        """Synchronize indexed entries with completed ``.bin`` files."""
        actual: dict[str, tuple[int, int, str]] = {}
        for directory, _, names in os.walk(self.root_dir):
            for name in names:
                if not name.endswith(".bin"):
                    continue
                path = os.path.abspath(os.path.join(directory, name))
                try:
                    stat = os.stat(path)
                except OSError:
                    continue
                actual[path] = (
                    int(stat.st_size),
                    int(stat.st_mtime_ns),
                    self._group_key_fn(path),
                )

        with self._lock:
            self._begin()
            try:
                indexed = {
                    row[0]: row[1]
                    for row in self._conn.execute(
                        "SELECT path, size_bytes FROM entries"
                    )
                }
                for path in indexed.keys() - actual.keys():
                    self._conn.execute("DELETE FROM entries WHERE path = ?", (path,))
                for path, (size, mtime_ns, group_key) in actual.items():
                    if path in indexed:
                        if indexed[path] != size:
                            self._conn.execute(
                                "UPDATE entries SET size_bytes = ? WHERE path = ?",
                                (size, path),
                            )
                    else:
                        self._conn.execute(
                            """
                            INSERT INTO entries
                                (path, group_key, size_bytes, queue, last_access_ns)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (path, group_key, size, PROBATIONARY, mtime_ns),
                        )
                self._conn.execute(
                    "DELETE FROM leases WHERE expires_ns <= ?", (time.time_ns(),)
                )
                self._conn.execute(
                    "DELETE FROM leases WHERE path NOT IN (SELECT path FROM entries)"
                )
                self._commit()
            except Exception:
                self._rollback()
                raise

    def record_store(self, paths: Iterable[str]) -> None:
        """Register files written by a successful store job."""
        now = time.time_ns()
        rows = self._existing_rows(paths)
        if not rows:
            return
        with self._lock:
            self._begin()
            try:
                for path, size in rows:
                    self._conn.execute(
                        """
                        INSERT INTO entries
                            (path, group_key, size_bytes, queue, last_access_ns)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(path) DO UPDATE SET size_bytes = excluded.size_bytes
                        """,
                        (path, self._group_key_fn(path), size, PROBATIONARY, now),
                    )
                self._commit()
            except Exception:
                self._rollback()
                raise

    def mark_access(self, paths: Iterable[str], promote: bool = True) -> None:
        """Update recency and optionally promote entries to protected queue."""
        unique_paths = tuple(dict.fromkeys(os.path.abspath(path) for path in paths))
        if not unique_paths:
            return
        now = time.time_ns()
        with self._lock:
            self._begin()
            try:
                for path in unique_paths:
                    if promote:
                        self._conn.execute(
                            """
                            UPDATE entries
                            SET queue = ?, last_access_ns = ?, hit_count = hit_count + 1
                            WHERE path = ?
                            """,
                            (PROTECTED, now, path),
                        )
                    else:
                        self._conn.execute(
                            """
                            UPDATE entries
                            SET last_access_ns = ?, hit_count = hit_count + 1
                            WHERE path = ?
                            """,
                            (now, path),
                        )
                self._commit()
            except Exception:
                self._rollback()
                raise

    def reserve(
        self,
        paths: Iterable[str],
        owner: str,
        ttl_ns: int = _LEASE_TTL_NS,
    ) -> None:
        """Lease paths while a lookup or asynchronous transfer is active."""
        unique_paths = tuple(dict.fromkeys(os.path.abspath(path) for path in paths))
        if not unique_paths:
            return
        expires_ns = time.time_ns() + ttl_ns
        with self._lock:
            self._begin()
            try:
                self._conn.execute(
                    "DELETE FROM leases WHERE expires_ns <= ?", (time.time_ns(),)
                )
                self._conn.executemany(
                    """
                    INSERT INTO leases(path, owner, expires_ns)
                    VALUES (?, ?, ?)
                    ON CONFLICT(path, owner) DO UPDATE SET
                        expires_ns = excluded.expires_ns
                    """,
                    ((path, owner, expires_ns) for path in unique_paths),
                )
                self._commit()
            except Exception:
                self._rollback()
                raise

    def release(self, owner: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM leases WHERE owner = ?", (owner,))

    def remove_missing(self, paths: Iterable[str]) -> None:
        missing = [
            os.path.abspath(path)
            for path in dict.fromkeys(paths)
            if not os.path.exists(path)
        ]
        if not missing:
            return
        with self._lock:
            self._conn.executemany(
                "DELETE FROM entries WHERE path = ?", ((p,) for p in missing)
            )
            self._conn.executemany(
                "DELETE FROM leases WHERE path = ?", ((p,) for p in missing)
            )

    def total_bytes(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(size_bytes), 0) FROM entries"
            ).fetchone()
        return int(row[0]) if row is not None else 0

    def cleanup(
        self,
        max_disk_bytes: int,
        target_disk_bytes: int,
        min_disk_free_bytes: int,
    ) -> tuple[int, int]:
        """Evict complete groups when a quota or free-space threshold is hit.

        Returns ``(groups_removed, bytes_removed)``.  Probationary groups are
        considered before protected groups, and groups with active leases are
        skipped.
        """
        if not max_disk_bytes and not min_disk_free_bytes:
            return 0, 0
        lock_file = None
        if fcntl is not None:
            try:
                lock_file = open(self.lock_path, "a+")
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                if lock_file is not None:
                    lock_file.close()
                return 0, 0
            except OSError as exc:
                if lock_file is not None:
                    lock_file.close()
                logger.warning(
                    "Unable to lock filesystem KV cache cleanup: %s", exc
                )
                return 0, 0

        removed_groups = 0
        removed_bytes = 0
        try:
            self.reconcile()
            total = self.total_bytes()
            free = self._free_bytes()
            quota_triggered = bool(max_disk_bytes and total > max_disk_bytes)
            free_space_triggered = bool(
                min_disk_free_bytes and free < min_disk_free_bytes
            )
            if not quota_triggered and not free_space_triggered:
                return 0, 0

            with self._lock:
                now = time.time_ns()
                self._conn.execute(
                    "DELETE FROM leases WHERE expires_ns <= ?", (now,)
                )
                groups = self._conn.execute(
                    """
                    SELECT group_key, MAX(queue) AS queue,
                           MAX(last_access_ns) AS last_access_ns
                    FROM entries
                    GROUP BY group_key
                    ORDER BY queue ASC, last_access_ns ASC, group_key ASC
                    """
                ).fetchall()

            for candidate in groups:
                above_target = quota_triggered and total > target_disk_bytes
                below_free_space_target = (
                    free_space_triggered and free < min_disk_free_bytes
                )
                if not above_target and not below_free_space_target:
                    break
                group_key = str(candidate[0])

                with self._lock:
                    self._begin()
                    try:
                        leased = self._conn.execute(
                            """
                            SELECT 1
                            FROM leases AS l
                            JOIN entries AS e ON e.path = l.path
                            WHERE e.group_key = ? AND l.expires_ns > ?
                            LIMIT 1
                            """,
                            (group_key, time.time_ns()),
                        ).fetchone()
                        paths = [
                            (str(path), int(size))
                            for path, size in self._conn.execute(
                                "SELECT path, size_bytes FROM entries "
                                "WHERE group_key = ?",
                                (group_key,),
                            ).fetchall()
                        ]
                        if leased or not paths:
                            self._commit()
                            continue

                        deleted: list[str] = []
                        for path, _ in paths:
                            try:
                                os.remove(path)
                            except FileNotFoundError:
                                deleted.append(path)
                            except OSError as exc:
                                logger.warning(
                                    "Failed to evict filesystem KV block %s: %s",
                                    path,
                                    exc,
                                )
                            else:
                                deleted.append(path)
                        if deleted:
                            self._conn.executemany(
                                "DELETE FROM entries WHERE path = ?",
                                ((p,) for p in deleted),
                            )
                            self._conn.executemany(
                                "DELETE FROM leases WHERE path = ?",
                                ((p,) for p in deleted),
                            )
                        self._commit()
                    except Exception:
                        self._rollback()
                        raise

                if deleted:
                    removed_groups += 1
                    group_removed_bytes = sum(
                        size for path, size in paths if path in deleted
                    )
                    removed_bytes += group_removed_bytes
                    total -= group_removed_bytes
                    if free_space_triggered:
                        free = self._free_bytes()
        finally:
            if lock_file is not None:
                with contextlib.suppress(OSError):
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
        if removed_groups:
            logger.info(
                "Filesystem KV cache cleanup removed %d group(s), %.2f GiB",
                removed_groups,
                removed_bytes / (1024**3),
            )
        return removed_groups, removed_bytes

    def _existing_rows(self, paths: Iterable[str]) -> list[tuple[str, int]]:
        rows = []
        for path in dict.fromkeys(os.path.abspath(path) for path in paths):
            try:
                rows.append((path, int(os.stat(path).st_size)))
            except OSError:
                continue
        return rows

    def _free_bytes(self) -> int:
        try:
            stat = os.statvfs(self.root_dir)
        except OSError:
            return 0
        return int(stat.f_bavail * stat.f_frsize)

    def _begin(self) -> None:
        self._conn.execute("BEGIN IMMEDIATE")

    def _commit(self) -> None:
        self._conn.execute("COMMIT")

    def _rollback(self) -> None:
        with contextlib.suppress(sqlite3.Error):
            self._conn.execute("ROLLBACK")
