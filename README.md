# AI Chatbot (FastAPI + OpenAI + Web Frontend)

A production-ready starter chatbot for websites with:
- FastAPI backend (`/chat` JSON API)
- OpenAI-powered responses
- SQLite conversation history
- Basic per-IP rate limiting
- Error handling with clear API errors
- Plain HTML/CSS/JavaScript frontend (ChatGPT-like layout)

## Project Structure

```text
ai-chatbot/
|-- backend/
|   |-- main.py
|   |-- chatbot.py
|   |-- requirements.txt
|   `-- .env.example
|-- frontend/
|   |-- index.html
|   |-- script.js
|   `-- style.css
`-- README.md
```

## 1. Install and Configure

### Prerequisites
- Python 3.10+
- OpenAI API key

### Backend setup

```bash
cd backend
python -m venv .venv
```

Activate the virtual environment:

Windows (PowerShell):
```bash
.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env` from the example:

```bash
cp .env.example .env
```

On Windows PowerShell if `cp` is unavailable:

```bash
Copy-Item .env.example .env
```

Edit `backend/.env` and set:
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` (optional, for local Ollama mode)
- `OPENAI_MODEL` (default: `gpt-4o-mini`)
- `ALLOWED_ORIGINS` (use explicit frontend origin in production)

## 2. Run the Backend Server

From the `backend` directory:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API will be available at:
- `http://localhost:8000/chat`
- `http://localhost:8000/health`
- `http://localhost:8000/history/{session_id}`

`/health` now includes OpenAI configuration status:

```json
{
  "status": "ok",
  "openai_configured": true,
  "model": "gpt-4o-mini"
}
```

## 3. Run the Frontend

From the `frontend` directory:

```bash
python -m http.server 5500
```

Open:
- `http://localhost:5500`

## 4. Use Without OpenAI Key (Ollama Local)

If you do not have an OpenAI API key, use Ollama locally.

1. Install Ollama (Windows):
   - Download and install from: https://ollama.com/download/windows
2. Verify installation:

```bash
ollama --version
```

3. Pull a local model:

```bash
ollama pull llama3.2
```

4. Set `backend/.env` for local mode:

```env
OPENAI_API_KEY=
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=llama3.2
```

5. Restart backend:

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

6. Check health:

```bash
curl http://localhost:8000/health
```

Expected `base_url` is `http://localhost:11434/v1` and `openai_configured` is `true`.

## 5. Connect Frontend to Backend

Frontend calls:
- `POST http://localhost:8000/chat`

If your backend runs on another URL, change this line in `frontend/script.js`:

```javascript
const API_BASE_URL = window.CHAT_API_BASE_URL || "http://localhost:8000";
```

Also update backend CORS in `backend/.env`:

```env
ALLOWED_ORIGINS=http://localhost:5500
```

For development only, you can use:

```env
ALLOWED_ORIGINS=*
```

## API Contract

### `POST /chat`

Request JSON:

```json
{
  "message": "Hello, can you help me?",
  "session_id": "optional_existing_session_id"
}
```

Response JSON:

```json
{
  "session_id": "c2fd1e83ef1148bbba2b69472de4f54f",
  "response": "Of course. What would you like help with?"
}
```

### `GET /history/{session_id}`

Returns stored message history from SQLite:

```json
{
  "session_id": "c2fd1e83ef1148bbba2b69472de4f54f",
  "messages": [
    {
      "role": "user",
      "content": "Hello",
      "created_at": "2026-03-08T12:00:00.000000+00:00"
    },
    {
      "role": "assistant",
      "content": "Hi there!",
      "created_at": "2026-03-08T12:00:01.000000+00:00"
    }
  ]
}
```

## Notes for Production

- Set strict `ALLOWED_ORIGINS` (do not use `*`).
- Run with a production ASGI setup (for example Gunicorn + Uvicorn workers).
- Put the API behind HTTPS and a reverse proxy.
- Store secrets in environment variables or a secret manager.
- Replace basic in-memory rate limiting with Redis-backed rate limiting if running multiple instances.

## Troubleshooting

- Frontend shows `API key missing` or `API key error`:
  - Set a valid `OPENAI_API_KEY` in `backend/.env`.
  - Restart backend server.
- Backend `/chat` returns `401`:
  - The OpenAI key is invalid/revoked or linked to wrong project/organization permissions.
- Backend `/chat` returns `404` with model error:
  - Update `OPENAI_MODEL` in `backend/.env` to an available model for your account.
