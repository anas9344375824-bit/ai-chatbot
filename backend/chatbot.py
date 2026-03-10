from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import httpx
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


class HuggingFaceChatbotService:
    MAX_CONTEXT_MESSAGES = 20
    DEFAULT_MAX_NEW_TOKENS = 256
    DEFAULT_TEMPERATURE = 0.7

    def __init__(
        self,
        api_token: str,
        model: str,
        db_path: str,
        system_prompt: str,
        api_url: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not model and not api_url:
            raise ValueError("HUGGINGFACE_MODEL or HUGGINGFACE_API_URL is required.")

        self.api_token = api_token
        self.model = model
        self.api_url = api_url or f"https://api-inference.huggingface.co/models/{model}"
        self.system_prompt = system_prompt
        self.timeout_seconds = timeout_seconds
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

        prompt = self._build_prompt(recent_messages)
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.DEFAULT_MAX_NEW_TOKENS,
                "temperature": self.DEFAULT_TEMPERATURE,
                "return_full_text": False,
            },
        }

        headers = {"Accept": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(self.api_url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise ChatbotProviderError(
                "Hugging Face request timed out. Please retry shortly.",
                status_code=504,
            ) from exc
        except httpx.RequestError as exc:
            raise ChatbotProviderError(
                "Could not connect to Hugging Face.",
                status_code=503,
            ) from exc

        if response.status_code == 401:
            raise ChatbotProviderError(
                "Hugging Face authentication failed. Check HUGGINGFACE_API_TOKEN in backend/.env.",
                status_code=401,
            )
        if response.status_code == 403:
            raise ChatbotProviderError(
                "Hugging Face request was forbidden. Verify your token permissions.",
                status_code=403,
            )
        if response.status_code == 404:
            raise ChatbotProviderError(
                "Hugging Face model was not found. Update HUGGINGFACE_MODEL or HUGGINGFACE_API_URL.",
                status_code=404,
            )
        if response.status_code == 429:
            raise ChatbotProviderError(
                "Hugging Face is rate limiting requests. Please retry shortly.",
                status_code=429,
            )
        if response.status_code >= 400:
            raise ChatbotProviderError(
                f"Hugging Face returned HTTP {response.status_code}.",
                status_code=response.status_code,
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise ChatbotProviderError(
                "Invalid response from Hugging Face.",
                status_code=502,
            ) from exc

        if isinstance(data, dict) and "error" in data:
            error_message = str(data.get("error", "Unknown Hugging Face error"))
            if "loading" in error_message.lower():
                raise ChatbotProviderError(
                    "Hugging Face model is loading. Please retry shortly.",
                    status_code=503,
                )
            raise ChatbotProviderError(
                f"Hugging Face error: {error_message}",
                status_code=503,
            )

        assistant_reply = self._extract_text(data, prompt)
        self.store.add_message(session_id=session_id, role="assistant", content=assistant_reply)
        return assistant_reply

    def get_history(self, session_id: str, limit: int = 100) -> list[dict[str, str]]:
        return self.store.get_history(session_id=session_id, limit=limit)

    def _build_prompt(self, messages: list[dict[str, str]]) -> str:
        prompt_lines: list[str] = []
        if self.system_prompt:
            prompt_lines.append(f"System: {self.system_prompt}")
        for message in messages:
            role = message.get("role", "user")
            label = "Assistant" if role == "assistant" else "User"
            prompt_lines.append(f"{label}: {message.get('content', '')}")
        prompt_lines.append("Assistant:")
        return "\n".join(prompt_lines).strip()

    @staticmethod
    def _extract_text(payload: Any, prompt: str) -> str:
        generated_text: str | None = None

        if isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, dict):
                generated_text = first.get("generated_text")
            elif isinstance(first, str):
                generated_text = first
        elif isinstance(payload, dict):
            if isinstance(payload.get("generated_text"), str):
                generated_text = payload["generated_text"]

        if isinstance(generated_text, str):
            cleaned = generated_text.strip()
            if prompt and cleaned.startswith(prompt):
                cleaned = cleaned[len(prompt) :].lstrip()
            return cleaned or "I do not have a response right now. Please try again."

        return "I do not have a response right now. Please try again."
