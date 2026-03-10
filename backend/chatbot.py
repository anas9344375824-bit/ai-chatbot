from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError


class ChatbotProviderError(RuntimeError):
    """Raised when the upstream AI provider fails."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ConversationStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._lock:
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_messages_session_id_id
                    ON messages (session_id, id)
                    """
                )
                connection.commit()

    def add_message(self, session_id: str, role: str, content: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO messages (session_id, role, content, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (session_id, role, content, timestamp),
                )
                connection.commit()

    def get_recent_messages(self, session_id: str, limit: int = 20) -> list[dict[str, str]]:
        with self._lock:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT role, content
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (session_id, limit),
                ).fetchall()

        return [
            {"role": row["role"], "content": row["content"]}
            for row in reversed(rows)
        ]

    def get_history(self, session_id: str, limit: int = 100) -> list[dict[str, str]]:
        with self._lock:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT role, content, created_at
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (session_id, limit),
                ).fetchall()

        history = [
            {
                "role": row["role"],
                "content": row["content"],
                "created_at": row["created_at"],
            }
            for row in reversed(rows)
        ]
        return history


class ChatbotService:
    MAX_CONTEXT_MESSAGES = 20

    def __init__(
        self,
        api_key: str,
        model: str,
        db_path: str,
        system_prompt: str,
        base_url: str | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required.")

        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.system_prompt = system_prompt
        self.store = ConversationStore(db_path=db_path)

    async def generate_response(self, session_id: str, user_message: str) -> str:
        normalized_message = user_message.strip()
        if not normalized_message:
            raise ValueError("Message cannot be empty.")

        self.store.add_message(session_id=session_id, role="user", content=normalized_message)
        recent_messages = self.store.get_recent_messages(
            session_id=session_id,
            limit=self.MAX_CONTEXT_MESSAGES,
        )

        messages = [{"role": "system", "content": self.system_prompt}, *recent_messages]

        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
            )
        except RateLimitError as exc:
            raise ChatbotProviderError(
                "AI provider is rate limiting requests. Please retry shortly.",
                status_code=429,
            ) from exc
        except APIConnectionError as exc:
            raise ChatbotProviderError(
                "Could not connect to AI provider.",
                status_code=503,
            ) from exc
        except APIStatusError as exc:
            if exc.status_code == 401:
                raise ChatbotProviderError(
                    "OpenAI authentication failed. Check OPENAI_API_KEY in backend/.env and restart the backend.",
                    status_code=401,
                ) from exc
            if exc.status_code == 403:
                raise ChatbotProviderError(
                    "OpenAI request was forbidden. Verify your project, organization, and API permissions.",
                    status_code=403,
                ) from exc
            if exc.status_code == 404:
                raise ChatbotProviderError(
                    f"Configured model '{self.model}' was not found. Update OPENAI_MODEL in backend/.env.",
                    status_code=404,
                ) from exc
            raise ChatbotProviderError(
                f"AI provider returned HTTP {exc.status_code}.",
                status_code=exc.status_code,
            ) from exc
        except Exception as exc:
            raise ChatbotProviderError(
                "Unexpected AI provider error.",
                status_code=503,
            ) from exc

        assistant_reply = self._extract_text(completion)
        self.store.add_message(session_id=session_id, role="assistant", content=assistant_reply)
        return assistant_reply

    def get_history(self, session_id: str, limit: int = 100) -> list[dict[str, str]]:
        return self.store.get_history(session_id=session_id, limit=limit)

    @staticmethod
    def _extract_text(completion: Any) -> str:
        if not getattr(completion, "choices", None):
            return "I do not have a response right now. Please try again."

        first_choice = completion.choices[0]
        message = getattr(first_choice, "message", None)
        content = getattr(message, "content", None)

        if isinstance(content, str):
            cleaned = content.strip()
            return cleaned or "I do not have a response right now. Please try again."

        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                part = getattr(item, "text", None)
                if isinstance(part, str) and part.strip():
                    text_parts.append(part.strip())
            combined = " ".join(text_parts).strip()
            return combined or "I do not have a response right now. Please try again."

        return "I do not have a response right now. Please try again."
