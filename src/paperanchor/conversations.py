"""Local conversation history scoped to one research space."""

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3
from uuid import uuid4


from .database import Database


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Conversation:
    id: str
    space_id: int
    title: str
    updated_at: str
    turn_count: int


@dataclass(frozen=True)
class ConversationTurn:
    id: int
    conversation_id: str
    question: str
    standalone_question: str
    answer: dict
    created_at: str


class ConversationStore:
    def __init__(self, database_path: Path):
        self.database_path = database_path
        self.database = Database(database_path)
        with self.database.connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    space_id INTEGER NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    question TEXT NOT NULL,
                    standalone_question TEXT NOT NULL,
                    answer_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS conversation_turns_recent
                ON conversation_turns(conversation_id,id DESC);
            """)

    def _connect(self) -> sqlite3.Connection:
        return self.database.connect_only()

    def create(self, space_id: int, title: str) -> Conversation:
        identifier = uuid4().hex
        now = _now()
        name = title.strip()[:100] or "New research chat"
        with closing(self._connect()) as connection, connection:
            connection.execute("INSERT INTO conversations(id,space_id,title,updated_at) "
                               "VALUES (?,?,?,?)",
                               (identifier, space_id, name, now))
        return Conversation(identifier, space_id, name, now, 0)

    def get(self, identifier: str, *, space_id: int | None = None) -> Conversation | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """SELECT c.*, COUNT(t.id) AS turn_count FROM conversations AS c
                   LEFT JOIN conversation_turns AS t ON t.conversation_id=c.id
                   WHERE c.id=? AND (? IS NULL OR c.space_id=?) GROUP BY c.id""",
                (identifier, space_id, space_id),
            ).fetchone()
            return Conversation(**dict(row)) if row else None

    def list(self, space_id: int) -> tuple[Conversation, ...]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """SELECT c.*,COUNT(t.id) AS turn_count FROM conversations AS c
                   LEFT JOIN conversation_turns AS t ON t.conversation_id=c.id
                   WHERE c.space_id=? GROUP BY c.id ORDER BY c.updated_at DESC LIMIT 30""",
                (space_id,),
            ).fetchall()
            return tuple(Conversation(**dict(row)) for row in rows)

    def turns(self, identifier: str, *, limit: int = 100) -> tuple[ConversationTurn, ...]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM conversation_turns WHERE conversation_id=? "
                "ORDER BY id DESC LIMIT ?", (identifier, limit),
            ).fetchall()
            return tuple(ConversationTurn(
                row["id"], row["conversation_id"], row["question"],
                row["standalone_question"], json.loads(row["answer_json"]),
                row["created_at"],
            ) for row in reversed(rows))

    def append(self, identifier: str, question: str, standalone_question: str,
               answer: dict) -> ConversationTurn:
        now = _now()
        encoded = json.dumps(answer, ensure_ascii=False)
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """INSERT INTO conversation_turns
                   (conversation_id,question,standalone_question,answer_json,created_at)
                   VALUES (?,?,?,?,?)""",
                (identifier, question, standalone_question, encoded, now),
            )
            connection.execute("UPDATE conversations SET updated_at=? WHERE id=?",
                               (now, identifier))
            turn_id = cursor.lastrowid
        return ConversationTurn(turn_id, identifier, question, standalone_question,
                                answer, now)
