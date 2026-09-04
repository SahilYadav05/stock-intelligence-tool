"""Dependency-light HTTP authentication, limits, and response hardening."""

from __future__ import annotations

from collections import defaultdict, deque
from hmac import compare_digest
from time import monotonic

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from nifty_terminal.settings import Settings


PUBLIC_PATHS = frozenset({"/api/v1/live"})


class SecurityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, settings: Settings) -> None:
        super().__init__(app)
        self.settings = settings
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        rejection = self._validate(request)
        if rejection is not None:
            response: Response = rejection
        else:
            response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        if self.settings.environment == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

    def _validate(self, request: Request) -> JSONResponse | None:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > self.settings.request_body_limit_bytes:
                    return _error(413, "REQUEST_BODY_TOO_LARGE")
            except ValueError:
                return _error(400, "INVALID_CONTENT_LENGTH")

        client = request.client.host if request.client else "unknown"
        now = monotonic()
        window = self._requests[client]
        while window and window[0] <= now - 60:
            window.popleft()
        if len(window) >= self.settings.requests_per_minute:
            return _error(429, "RATE_LIMIT_EXCEEDED")
        window.append(now)

        if request.method == "OPTIONS":
            return None
        if self.settings.api_auth_mode == "bearer" and request.url.path not in PUBLIC_PATHS:
            authorization = request.headers.get("authorization", "")
            scheme, _, supplied = authorization.partition(" ")
            expected = self.settings.api_auth_token or ""
            if scheme.lower() != "bearer" or not supplied or not compare_digest(supplied, expected):
                return _error(401, "AUTHENTICATION_REQUIRED")
        return None


def valid_websocket_request(websocket, settings: Settings) -> bool:
    origin = websocket.headers.get("origin")
    if origin and origin.rstrip("/") not in settings.api_allowed_origins:
        return False
    if settings.api_auth_mode != "bearer":
        return True
    authorization = websocket.headers.get("authorization", "")
    scheme, _, supplied = authorization.partition(" ")
    return scheme.lower() == "bearer" and bool(supplied) and compare_digest(
        supplied, settings.api_auth_token or ""
    )


def _error(status: int, code: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": {"code": code}})
