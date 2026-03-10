from __future__ import annotations

import logging
import os
from collections import defaultdict, deque
from pathlib import Path
from threading import Lock
from time import monotonic
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

try:
    from .chatbot import ChatbotProviderError, ChatbotService, HuggingFaceChatbotService
except ImportError:
    from chatbot import ChatbotProviderError, ChatbotService, HuggingFaceChatbotService


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ai-chatbot")


def parse_allowed_origins(raw_value: str) -> list[str]:
    origins = [origin.strip() for origin in raw_value.split(",") if origin.strip()]
    return origins or ["*"]


def parse_positive_int(raw_value: str, default: int) -> int:
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def is_placeholder_key(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized in {
        "",
        "your_openai_api_key_here",
        "sk-placeholder",
        "changeme",
    }


def is_local_ollama_base_url(value: str) -> bool:
    normalized = value.strip().lower()
    return (
        normalized.startswith("http://localhost:11434")
        or normalized.startswith("http://127.0.0.1:11434")
    )


def is_render_environment() -> bool:
    return os.getenv("RENDER", "").strip().lower() == "true"

def normalize_provider(raw_value: str) -> str:
    normalized = raw_value.strip().lower()
    if normalized in {"huggingface", "hugging_face", "hf"}:
        return "huggingface"
    return "openai"

def resolve_huggingface_api_url(model: str, api_url: str) -> str:
    if api_url:
        return api_url
    if model:
        return f"https://api-inference.huggingface.co/models/{model}"
    return ""

def provider_setup_hint() -> str:
    if CHAT_PROVIDER == "huggingface":
        return (
            "Set HUGGINGFACE_MODEL or HUGGINGFACE_API_URL (and HUGGINGFACE_API_TOKEN if required) "
            "in backend/.env, then restart the backend."
        )
    return (
        "Set OPENAI_API_KEY (cloud) or OPENAI_BASE_URL for local Ollama in backend/.env, "
        "then restart the backend."
    )


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
CHAT_PROVIDER = normalize_provider(os.getenv("CHAT_PROVIDER", "openai"))
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "").strip()
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "microsoft/DialoGPT-medium").strip()
HUGGINGFACE_API_URL = os.getenv("HUGGINGFACE_API_URL", "").strip()
HUGGINGFACE_TIMEOUT_SECONDS = parse_positive_int(
    os.getenv("HUGGINGFACE_TIMEOUT_SECONDS", "30"),
    30,
)
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful AI assistant for website visitors. Keep responses concise and practical.",
).strip()
DB_PATH = os.getenv("DB_PATH", "chat_history.db").strip()
RATE_LIMIT_REQUESTS = parse_positive_int(os.getenv("RATE_LIMIT_REQUESTS", "30"), 30)
RATE_LIMIT_WINDOW_SECONDS = parse_positive_int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"), 60)
ALLOWED_ORIGINS = parse_allowed_origins(os.getenv("ALLOWED_ORIGINS", "*"))

effective_api_key = OPENAI_API_KEY
using_local_ollama = is_local_ollama_base_url(OPENAI_BASE_URL)
running_on_render = is_render_environment()
resolved_huggingface_api_url = resolve_huggingface_api_url(
    HUGGINGFACE_MODEL,
    HUGGINGFACE_API_URL,
)

if CHAT_PROVIDER == "openai":
    if using_local_ollama and running_on_render:
        logger.error(
            "OPENAI_BASE_URL is set to localhost on Render. "
            "Use a real provider endpoint or clear OPENAI_BASE_URL and set OPENAI_API_KEY."
        )
    elif not effective_api_key and using_local_ollama:
        effective_api_key = "ollama"
        logger.info("Using local Ollama mode with OPENAI_BASE_URL=%s", OPENAI_BASE_URL)


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[a-zA-Z0-9_-]+$",
    )


class ChatResponse(BaseModel):
    session_id: str
    response: str


class HistoryMessage(BaseModel):
    role: str
    content: str
    created_at: str


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[HistoryMessage]


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, deque[float]] = defaultdict(deque)
        self.lock = Lock()

    def allow(self, client_id: str) -> tuple[bool, int]:
        now = monotonic()
        with self.lock:
            bucket = self.requests[client_id]
            while bucket and (now - bucket[0]) > self.window_seconds:
                bucket.popleft()

            if len(bucket) >= self.max_requests:
                retry_after = max(1, int(self.window_seconds - (now - bucket[0])) + 1)
                return False, retry_after

            bucket.append(now)
            return True, 0


chatbot_service: ChatbotService | HuggingFaceChatbotService | None = None
if CHAT_PROVIDER == "huggingface":
    if not resolved_huggingface_api_url:
        logger.error(
            "HUGGINGFACE_MODEL or HUGGINGFACE_API_URL must be set when CHAT_PROVIDER=huggingface."
        )
    else:
        chatbot_service = HuggingFaceChatbotService(
            api_token=HUGGINGFACE_API_TOKEN,
            model=HUGGINGFACE_MODEL,
            db_path=DB_PATH,
            system_prompt=SYSTEM_PROMPT,
            api_url=HUGGINGFACE_API_URL or None,
            timeout_seconds=float(HUGGINGFACE_TIMEOUT_SECONDS),
        )
else:
    if using_local_ollama and running_on_render:
        logger.error("Chat provider disabled because localhost Ollama is unreachable on Render.")
    elif is_placeholder_key(effective_api_key):
        logger.error(
            "OPENAI_API_KEY is missing or placeholder. /chat will return 503 until a real key is configured."
        )
    else:
        chatbot_service = ChatbotService(
            api_key=effective_api_key,
            model=OPENAI_MODEL,
            db_path=DB_PATH,
            system_prompt=SYSTEM_PROMPT,
            base_url=OPENAI_BASE_URL or None,
        )

rate_limiter = RateLimiter(
    max_requests=RATE_LIMIT_REQUESTS,
    window_seconds=RATE_LIMIT_WINDOW_SECONDS,
)

app = FastAPI(
    title="AI Chatbot API",
    description="FastAPI backend for a website chatbot powered by OpenAI.",
    version="1.0.0",
)

allow_all_origins = ALLOWED_ORIGINS == ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all_origins else ALLOWED_ORIGINS,
    allow_credentials=not allow_all_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.method != "OPTIONS" and request.url.path == "/chat":
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            client_id = forwarded_for.split(",")[0].strip()
        else:
            client_id = request.client.host if request.client else "unknown"

        is_allowed, retry_after = rate_limiter.allow(client_id)
        if not is_allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please try again later."},
                headers={"Retry-After": str(retry_after)},
            )

    try:
        return await call_next(request)
    except Exception:
        logger.exception("Unhandled server error")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error."},
        )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Invalid request payload.",
            "errors": exc.errors(),
        },
    )


@app.get("/health")
async def health_check():
    provider_model = HUGGINGFACE_MODEL if CHAT_PROVIDER == "huggingface" else OPENAI_MODEL
    provider_base_url = (
        resolved_huggingface_api_url
        if CHAT_PROVIDER == "huggingface"
        else (OPENAI_BASE_URL or "https://api.openai.com/v1")
    )
    return {
        "status": "ok",
        "provider": CHAT_PROVIDER,
        "provider_configured": chatbot_service is not None,
        "openai_configured": chatbot_service is not None and CHAT_PROVIDER == "openai",
        "model": provider_model,
        "base_url": provider_base_url,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest):
    if chatbot_service is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Server is not configured for chat provider access. "
                f"{provider_setup_hint()}"
            ),
        )

    session_id = payload.session_id or uuid4().hex

    try:
        assistant_response = await chatbot_service.generate_response(
            session_id=session_id,
            user_message=payload.message,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ChatbotProviderError as exc:
        logger.warning("Chat provider failure: %s", exc)
        http_status = exc.status_code if isinstance(exc.status_code, int) else 503
        raise HTTPException(status_code=http_status, detail=str(exc)) from exc
    except Exception:
        logger.exception("Unexpected failure while generating chat response")
        raise HTTPException(status_code=500, detail="Failed to generate response.") from None

    return ChatResponse(session_id=session_id, response=assistant_response)


@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    if chatbot_service is None:
        raise HTTPException(
            status_code=503,
            detail="Chat history is unavailable until a chat provider is configured.",
        )
    messages = chatbot_service.get_history(session_id=session_id, limit=200)
    return HistoryResponse(session_id=session_id, messages=messages)


frontend_dir_env = os.getenv("FRONTEND_DIR", "").strip()
frontend_dir_candidates = [
    Path(frontend_dir_env).resolve() if frontend_dir_env else None,
    BASE_DIR / "frontend",
    BASE_DIR.parent / "frontend",
    Path.cwd() / "frontend",
]
frontend_dir = next(
    (
        path
        for path in frontend_dir_candidates
        if path is not None and path.is_dir() and (path / "index.html").is_file()
    ),
    None,
)

@app.get("/", include_in_schema=False)
def home():
    if frontend_dir is not None:
        return FileResponse(frontend_dir / "index.html")
    return {"message": "AI Chatbot API is running"}


if frontend_dir is not None:
    logger.info("Serving frontend from %s", frontend_dir)
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    logger.warning("No frontend directory found. Serving API-only endpoints.")
