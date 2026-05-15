import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from app.api.routes_workbench import router as workbench_router
from app.core.config import settings
from app.core.database import Base, engine, check_app_db_connection
from app.core.logging_utils import configure_logging
from app.core.rate_limit import InMemoryRateLimiter
from app.core.errors import (
    WorkbenchValidationError,
    WorkbenchConnectionError,
    format_error_response,
)

logger = logging.getLogger(__name__)

configure_logging(settings.log_json)

rate_limiter = InMemoryRateLimiter(
    limit=settings.rate_limit_requests,
    window_seconds=settings.rate_limit_window_seconds,
)


def _initialize_app_tables() -> None:
    """
    Create application tables (anomaly_workbench_runs, etc.) in the
    PostgreSQL app database if they do not yet exist.

    This is idempotent — safe to call on every startup.
    """
    health = check_app_db_connection()
    if not health["connected"]:
        logger.warning(
            "App DB not reachable at startup (%s). "
            "Table initialisation skipped — the app will retry on first request.",
            health.get("error", "unknown error"),
        )
        return

    try:
        Base.metadata.create_all(bind=engine)
        logger.info("App DB tables verified / created successfully.")
    except SQLAlchemyError as exc:
        logger.warning("App DB table initialisation skipped: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle hook."""
    logger.info(
        "Starting %s — app DB: %s",
        settings.app_name,
        # Hide password from log
        settings.db_url.split("@")[-1] if "@" in settings.db_url else settings.db_url,
    )
    _initialize_app_tables()
    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version="2.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow the Vite dev server and any same-origin production build
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        # Add your production origin here, e.g. "https://tulip.example.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request-ID middleware — adds X-Request-ID tracing to every request
# ---------------------------------------------------------------------------
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.debug(
        "[%s] %s %s → %s  (%.1f ms)",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.middleware("http")
async def rate_limit_requests(request: Request, call_next):
    if not settings.rate_limit_enabled or request.url.path in {"/health"}:
        return await call_next(request)

    forwarded_for = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    client_host = forwarded_for or (request.client.host if request.client else "unknown")
    key = f"{client_host}:{request.url.path}"
    allowed, remaining, retry_after = rate_limiter.allow(key)
    if not allowed:
        logger.warning("Rate limit exceeded for client=%s path=%s", client_host, request.url.path)
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please retry later."},
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(settings.rate_limit_requests),
                "X-RateLimit-Remaining": "0",
            },
        )

    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(settings.rate_limit_requests)
    response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
    return response


# ---------------------------------------------------------------------------
# Global exception handler — standardizes all error responses
# ---------------------------------------------------------------------------
@app.middleware("http")
async def exception_handler_middleware(request: Request, call_next):
    """Catch unhandled exceptions and format consistent error responses."""
    try:
        response = await call_next(request)
        return response
    except WorkbenchValidationError as exc:
        logger.warning("[%s] Validation error: %s", getattr(request.state, "request_id", "?"), exc.message)
        return JSONResponse(status_code=400, content=exc.to_http_detail())
    except WorkbenchConnectionError as exc:
        logger.error("[%s] Connection error: %s", getattr(request.state, "request_id", "?"), exc.message)
        return JSONResponse(status_code=503, content=exc.to_http_detail())
    except ConnectionError as exc:
        logger.error("[%s] Upstream connection error: %s", getattr(request.state, "request_id", "?"), str(exc))
        detail = format_error_response(exc, 503)
        return JSONResponse(status_code=503, content=detail)
    except ValueError as exc:
        logger.warning("[%s] Value error: %s", getattr(request.state, "request_id", "?"), str(exc))
        detail = format_error_response(exc, 400)
        return JSONResponse(status_code=400, content=detail)
    except Exception as exc:
        logger.exception("[%s] Unhandled exception", getattr(request.state, "request_id", "?"))
        detail = format_error_response(exc, 500)
        return JSONResponse(status_code=500, content=detail)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health", tags=["ops"])
def health_check():
    """Lightweight liveness probe."""
    app_db = check_app_db_connection()
    return {
        "status": "ok" if app_db["connected"] else "degraded",
        "app_db_connected": app_db["connected"],
        "app_db_error": app_db.get("error"),
    }


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(workbench_router)
