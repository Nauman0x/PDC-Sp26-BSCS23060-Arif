"""
StudySync FastAPI Application — Entry Point

Wires together:
  - The FastAPI application instance
  - The StudentIDMiddleware (mandatory X-Student-ID header on every response)
  - The document router (LLM generation endpoints)
  - A /health endpoint for quick liveness checks and circuit-breaker observability
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from middleware.student_id_middleware import StudentIDMiddleware
from routers.document import router as document_router
from services.llm_service import get_circuit_breaker

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="StudySync API",
    description=(
        "Minimal FastAPI backend for the PDC assignment: "
        "demonstrates resilient distributed systems patterns — specifically "
        "the Circuit Breaker + Fallback pattern for an external LLM API."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

# CORS — permissive for local development / grading environment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inject X-Student-ID: BSCS23060 on every response (grading requirement)
app.add_middleware(StudentIDMiddleware)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(document_router)

# ---------------------------------------------------------------------------
# Health / observability endpoint
# ---------------------------------------------------------------------------

@app.get("/health", tags=["observability"], summary="Liveness check + circuit state")
async def health() -> dict:
    """
    Returns application health and the current circuit-breaker state.

    Useful for quickly checking whether the LLM circuit is OPEN (degraded)
    or CLOSED (healthy) without making a real LLM call.

    Example response when healthy::

        {
            "status": "ok",
            "circuit_breaker": {
                "state": "closed",
                "failure_count": 0
            }
        }

    Example response when circuit is open::

        {
            "status": "degraded",
            "circuit_breaker": {
                "state": "open",
                "failure_count": 3
            }
        }
    """
    cb = get_circuit_breaker()
    cb_state = cb.state.value
    return {
        "status": "degraded" if cb_state == "open" else "ok",
        "circuit_breaker": {
            "state": cb_state,
            "failure_count": cb._failure_count,
        },
    }
