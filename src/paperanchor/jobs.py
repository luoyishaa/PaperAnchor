"""Durable local jobs for work that should outlive one HTTP request."""

from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from threading import Event, Thread
from typing import Callable

from .semantic import EmbeddingProvider, SemanticIndex


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Job:
    id: int
    kind: str
    state: str
    completed: int
    total: int
    error: str | None
    created_at: str
    updated_at: str


class JobStore:
    def __init__(self, database_path: Path):
        self.database_path = database_path
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS jobs (
                     id INTEGER PRIMARY KEY,
                     kind TEXT NOT NULL,
                     state TEXT NOT NULL CHECK(state IN
                       ('queued', 'running', 'succeeded', 'failed')),
                     completed INTEGER NOT NULL DEFAULT 0,
                     total INTEGER NOT NULL DEFAULT 0,
                     error TEXT,
                     created_at TEXT NOT NULL,
                     updated_at TEXT NOT NULL
                   )"""
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @staticmethod
    def _job(row: sqlite3.Row) -> Job:
        return Job(**dict(row))

    def get(self, job_id: int) -> Job | None:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            return self._job(row) if row else None

    def latest(self, kind: str) -> Job | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE kind=? ORDER BY id DESC LIMIT 1", (kind,)
            ).fetchone()
            return self._job(row) if row else None

    def enqueue(self, kind: str) -> Job:
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                """SELECT * FROM jobs WHERE kind=? AND state IN ('queued', 'running')
                   ORDER BY id DESC LIMIT 1""", (kind,)
            ).fetchone()
            if row:
                return self._job(row)
            now = _now()
            cursor = connection.execute(
                """INSERT INTO jobs (kind, state, completed, total, error, created_at, updated_at)
                   VALUES (?, 'queued', 0, 0, NULL, ?, ?)""", (kind, now, now)
            )
            job_id = cursor.lastrowid
        return self.get(job_id)

    def recover(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "UPDATE jobs SET state='queued', updated_at=? WHERE state='running'",
                (_now(),),
            )

    def claim_next(self) -> Job | None:
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT id FROM jobs WHERE state='queued' ORDER BY id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "UPDATE jobs SET state='running', updated_at=? WHERE id=?",
                (_now(), row["id"]),
            )
            job_id = row["id"]
        return self.get(job_id)

    def progress(self, job_id: int, completed: int, total: int) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """UPDATE jobs SET completed=?, total=?, updated_at=?
                   WHERE id=? AND state='running'""",
                (completed, total, _now(), job_id),
            )

    def finish(self, job_id: int, *, error: str | None = None) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "UPDATE jobs SET state=?, error=?, updated_at=? WHERE id=?",
                ("failed" if error else "succeeded", error, _now(), job_id),
            )


class SemanticWorker:
    """One worker prevents two model downloads or embedding builds racing."""

    def __init__(self, store: JobStore, index: SemanticIndex,
                 provider_factory: Callable[[], EmbeddingProvider]):
        self.store = store
        self.index = index
        self.provider_factory = provider_factory
        self._stop = Event()
        self._wake = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.store.recover()
        self._stop.clear()
        self._thread = Thread(target=self._run, daemon=True, name="paperanchor-semantic")
        self._thread.start()

    def enqueue(self) -> Job:
        job = self.store.enqueue("semantic")
        self._wake.set()
        return job

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            job = self.store.claim_next()
            if job is None:
                self._wake.wait(timeout=1)
                self._wake.clear()
                continue
            try:
                provider = self.provider_factory()
                result = self.index.build(
                    provider,
                    progress=lambda done, total: self.store.progress(job.id, done, total),
                )
                self.store.progress(job.id, result.indexed, result.total)
                self.store.finish(job.id)
            except Exception as error:
                self.store.finish(job.id, error=str(error)[:1000])


def job_data(job: Job) -> dict:
    return asdict(job)
