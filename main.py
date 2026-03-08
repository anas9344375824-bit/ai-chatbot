from fastapi.routing import APIRoute

from backend.main import app


def _has_get_root_route() -> bool:
    for route in app.routes:
        if isinstance(route, APIRoute) and route.path == "/" and "GET" in route.methods:
            return True
    return False


if not _has_get_root_route():
    @app.get("/", include_in_schema=False)
    def home():
        return {"message": "AI Chatbot API is running"}
