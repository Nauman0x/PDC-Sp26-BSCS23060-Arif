"""
Document router: exposes the LLM-backed document generation endpoint.

This router contains two routes that map directly to the assignment's
"before vs. after" demo:

  POST /api/document/generate
      The resilient endpoint that uses the circuit breaker and fallback.
      This is the "after" — it stays responsive even when the LLM is down.

  POST /api/document/generate-naive
      The broken endpoint that calls the LLM with no timeout or circuit
      breaker.  This is the "before" — it hangs the server when the LLM
      stops responding.

Both routes return the ``X-Student-ID`` header via the global middleware;
the circuit state is also surfaced in the response body for observability.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from services.llm_service import generate_with_fallback, naive_call_llm

router = APIRouter(prefix="/api/document", tags=["document"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    """
    Payload for a document-generation request.

    :param prompt: The user's free-text prompt that will be sent to the LLM.
    """

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Free-text prompt to send to the LLM.",
        examples=["Summarise the key concepts of distributed systems in 3 bullet points."],
    )


class GenerateResponse(BaseModel):
    """
    Response from the resilient document-generation endpoint.

    :param text: The generated text — either from the LLM or the fallback.
    :param is_fallback: ``True`` when the LLM was unavailable and a static
        fallback was returned instead.
    :param circuit_state: Current state of the circuit breaker
        (``"closed"``, ``"open"``, or ``"half_open"``).
    """

    text: str
    is_fallback: bool
    circuit_state: str


class NaiveGenerateResponse(BaseModel):
    """
    Response from the naive (unprotected) document-generation endpoint.

    :param text: The raw text returned by the LLM.
    """

    text: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/generate",
    response_model=GenerateResponse,
    summary="Generate document content (circuit-breaker protected)",
    description=(
        "Sends the prompt to the external LLM API through a Circuit Breaker. "
        "If the LLM is slow or unavailable the circuit opens and a fallback "
        "response is returned immediately — the server never hangs."
    ),
)
async def generate_document(body: GenerateRequest) -> GenerateResponse:
    """
    Resilient document-generation endpoint.

    Delegates to :func:`~services.llm_service.generate_with_fallback` which
    wraps the LLM call in a circuit breaker with a 5-second timeout.

    When the LLM is healthy:
        Returns ``is_fallback: false`` and the real generated text.

    When the LLM is down (circuit CLOSED but individual call times out):
        Returns ``is_fallback: true`` and the static fallback message.
        The circuit breaker increments its failure counter.

    When the circuit is OPEN:
        Returns ``is_fallback: true`` instantly — no network call is made.

    :param body: The generation request containing the prompt.
    :return: A :class:`GenerateResponse` with the text and circuit metadata.
    """
    result = await generate_with_fallback(body.prompt)
    return GenerateResponse(
        text=result.text,
        is_fallback=result.is_fallback,
        circuit_state=result.circuit_state.value,
    )


@router.post(
    "/generate-naive",
    response_model=NaiveGenerateResponse,
    summary="Generate document content (NAIVE — no protection)",
    description=(
        "Sends the prompt to the LLM with NO timeout and NO circuit breaker. "
        "If the LLM hangs, this endpoint blocks indefinitely and causes the "
        "entire server to become unresponsive. Included to demonstrate the "
        "failure mode in the 'before' portion of the demo."
    ),
)
async def generate_document_naive(body: GenerateRequest) -> NaiveGenerateResponse:
    """
    Unprotected document-generation endpoint (demonstrates the bug).

    Calls the LLM directly via :func:`~services.llm_service.naive_call_llm`
    with no timeout and no circuit breaker.  When the LLM stops responding
    this coroutine suspends forever, consuming the server's worker and making
    every other concurrent request stall.

    This endpoint is provided solely for the "before" portion of the
    assignment demo.  Do NOT use in production.

    :param body: The generation request containing the prompt.
    :return: A :class:`NaiveGenerateResponse` with the raw LLM text.
    """
    text = await naive_call_llm(body.prompt)
    return NaiveGenerateResponse(text=text)
