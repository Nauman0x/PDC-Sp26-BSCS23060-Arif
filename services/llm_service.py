"""
LLM Service: wraps an external LLM API call with a Circuit Breaker and fallback.

Architecture overview
---------------------
The naive implementation (illustrated in NaiveLLMService) makes a plain
``httpx.AsyncClient.post()`` call with no timeout.  When the LLM API hangs,
the coroutine suspends indefinitely and every subsequent request that hits the
same endpoint also hangs — effectively taking the entire application down for
all users.  This is the fault described in Problem 3 of the assignment.

The resilient implementation (LLMService) wraps every outbound call in:

  1. A hard 5-second ``httpx`` timeout  — so a single request never blocks
     for more than 5 s regardless of the downstream service's behaviour.
  2. A CircuitBreaker  — after ``FAILURE_THRESHOLD`` consecutive timeouts /
     errors the circuit opens and all subsequent callers receive an instant
     fallback without touching the LLM API at all.
  3. A ``generate_with_fallback`` façade  — catches both ``CircuitOpenError``
     (circuit is open) and ``httpx.TimeoutException`` (individual timeout) and
     returns a pre-defined fallback response so the rest of the application
     never has to deal with LLM-specific exceptions.

The LLM_URL constant points to the mock server started by ``test_naive.py``
and ``test_circuit_breaker.py``, but can be overridden by the ``LLM_URL``
environment variable for a real deployment.
"""

import os
from dataclasses import dataclass

import httpx

from circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# The external LLM endpoint.  Override with the LLM_URL env-var in production.
LLM_URL: str = os.getenv("LLM_URL", "http://localhost:9999/v1/generate")

# Hard network timeout (seconds).  Prevents a single slow response from
# blocking the event loop indefinitely.
REQUEST_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT", "5.0"))

# Number of consecutive failures before the circuit trips to OPEN.
FAILURE_THRESHOLD: int = int(os.getenv("CB_FAILURE_THRESHOLD", "3"))

# Seconds the circuit stays OPEN before allowing a single probe request.
RECOVERY_TIMEOUT: float = float(os.getenv("CB_RECOVERY_TIMEOUT", "30.0"))

# Text returned to the caller when the LLM is unavailable.
FALLBACK_TEXT: str = (
    "The AI assistant is temporarily unavailable. "
    "Your request has been queued and will be processed when service is restored. "
    "Please try again in a few minutes."
)


# ---------------------------------------------------------------------------
# Shared circuit-breaker instance (module-level singleton)
# ---------------------------------------------------------------------------

# A single instance is shared across all requests in the process so the failure
# counter accumulates correctly across concurrent calls.
_circuit_breaker: CircuitBreaker = CircuitBreaker(
    failure_threshold=FAILURE_THRESHOLD,
    recovery_timeout=RECOVERY_TIMEOUT,
    name="LLMCircuitBreaker",
)


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """
    Structured response returned by :func:`generate_with_fallback`.

    :param text: The generated text, or the fallback message if the LLM was
        unavailable.
    :param is_fallback: ``True`` when the fallback path was activated (the LLM
        was not reached or failed).
    :param circuit_state: The circuit breaker state at the time of the response.
        Exposed in the API so callers/operators can observe the system health.
    """

    text: str
    is_fallback: bool
    circuit_state: CircuitState


# ---------------------------------------------------------------------------
# Naive implementation (the bug — for the "before" demo)
# ---------------------------------------------------------------------------

async def naive_call_llm(prompt: str) -> str:
    """
    Make a synchronous-style LLM call with **no timeout and no circuit breaker**.

    This is intentionally broken.  When the LLM API hangs, this coroutine
    suspends forever, holding the event-loop worker and preventing all other
    concurrent requests from being served.

    Used exclusively by ``test_naive.py`` to demonstrate the failure mode.

    :param prompt: The user's prompt to send to the LLM.
    :return: The generated text from the LLM API.
    :raises httpx.TimeoutException: Never — there is no timeout configured.
    """
    async with httpx.AsyncClient(timeout=None) as client:
        # timeout=None explicitly disables all timeouts — this will block
        # indefinitely if the server hangs, which is the bug being demonstrated
        response = await client.post(LLM_URL, json={"prompt": prompt})
        response.raise_for_status()
        return response.json()["text"]


# ---------------------------------------------------------------------------
# Resilient implementation (the fix)
# ---------------------------------------------------------------------------

async def _call_llm_with_timeout(prompt: str) -> str:
    """
    Inner LLM call with a hard network timeout.

    This is the raw callable passed into the circuit breaker.  It is kept
    separate from the circuit-breaker logic so both can be tested independently.

    :param prompt: The user's prompt to send to the LLM.
    :return: The generated text from the LLM API.
    :raises httpx.TimeoutException: When the LLM API does not respond within
        ``REQUEST_TIMEOUT`` seconds.
    :raises httpx.HTTPStatusError: When the LLM API returns a non-2xx status.
    """
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.post(LLM_URL, json={"prompt": prompt})
        response.raise_for_status()
        return response.json()["text"]


async def call_llm(prompt: str) -> str:
    """
    Call the LLM API through the circuit breaker.

    Delegates to :func:`_call_llm_with_timeout` via the shared
    :data:`_circuit_breaker`.  If the circuit is OPEN, raises
    :exc:`CircuitOpenError` immediately without making a network call.

    :param prompt: The user's prompt.
    :return: The raw text response from the LLM.
    :raises CircuitOpenError: When the circuit is OPEN.
    :raises httpx.TimeoutException: When the call times out (circuit records
        this as a failure internally).
    """
    return await _circuit_breaker.call(_call_llm_with_timeout, prompt)


async def generate_with_fallback(prompt: str) -> LLMResponse:
    """
    High-level entry point for generating LLM content with automatic fallback.

    Tries to call the LLM API via the circuit breaker.  If the circuit is open
    OR the individual request times out, returns a pre-defined fallback message
    instead of propagating the exception — the API endpoint stays responsive for
    all users even when the LLM is down.

    :param prompt: The user's prompt.
    :return: An :class:`LLMResponse` containing either the LLM-generated text
        or the fallback message, plus metadata about whether fallback was used
        and the current circuit-breaker state.

    Example::

        result = await generate_with_fallback("Summarise this article...")
        if result.is_fallback:
            log.warning("LLM unavailable, served fallback")
        return {"text": result.text, "circuit": result.circuit_state}
    """
    try:
        text = await call_llm(prompt)
        return LLMResponse(
            text=text,
            is_fallback=False,
            circuit_state=_circuit_breaker.state,
        )

    except CircuitOpenError:
        # Circuit is open — LLM is presumed down, return fallback immediately
        return LLMResponse(
            text=FALLBACK_TEXT,
            is_fallback=True,
            circuit_state=CircuitState.OPEN,
        )

    except (httpx.TimeoutException, httpx.RequestError, httpx.HTTPStatusError):
        # Individual request failed but the circuit may not be open yet —
        # still serve the fallback so this request does not surface a 500.
        return LLMResponse(
            text=FALLBACK_TEXT,
            is_fallback=True,
            circuit_state=_circuit_breaker.state,
        )


def get_circuit_breaker() -> CircuitBreaker:
    """
    Return the module-level circuit breaker instance.

    Exposed so tests can inspect or reset the breaker state between runs.

    :return: The shared :class:`~circuit_breaker.CircuitBreaker` instance.
    """
    return _circuit_breaker
