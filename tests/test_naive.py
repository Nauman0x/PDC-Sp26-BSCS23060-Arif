"""
BEFORE: Demonstrates the fault-tolerance bug in the naive implementation.

This test file shows what happens when an LLM endpoint hangs and there is
NO circuit breaker or timeout protecting the application.

What this test proves
---------------------
When the LLM API stops responding, the naive endpoint (POST /api/document/generate-naive)
blocks for the full duration of the network call — potentially minutes.  Even
with a short asyncio timeout wrapping the test itself we can observe that:

  1. The naive call does NOT return within an acceptable window.
  2. The server would be unresponsive to ALL other requests for the same period.
  3. There is no fallback — the caller receives no response at all.

How to interpret the output
----------------------------
Run with: pytest tests/test_naive.py -v -s

You will see:

    BEFORE (naive):  Calling the unprotected endpoint...
    BEFORE (naive):  Request timed out or hung as expected — server is blocked!
    PASSED

This is the "FAIL" state — the pass simply means the test correctly
detected the hang.  The test would FAIL (incorrectly) if the naive
endpoint somehow returned quickly.

Run: pytest tests/test_circuit_breaker.py -v -s  afterwards to see the fix.
"""

import asyncio

import pytest
import respx
import httpx
from httpx import AsyncClient, ASGITransport

from main import app


# ---------------------------------------------------------------------------
# Shared mock URL (must match LLM_URL env-var / default in llm_service.py)
# ---------------------------------------------------------------------------

LLM_URL = "http://localhost:9999/v1/generate"

# How many seconds we wait before declaring the naive call "hung"
HANG_DETECTION_TIMEOUT = 3.0


# ---------------------------------------------------------------------------
# Helper: a mock LLM server that never responds (simulates the 60s hang)
# ---------------------------------------------------------------------------

async def _hanging_llm_handler(request: httpx.Request) -> httpx.Response:
    """
    Simulates an LLM API that accepts the connection but never sends a response.
    Sleeps for 60 seconds — far longer than any reasonable test budget.
    """
    await asyncio.sleep(60)
    return httpx.Response(200, json={"text": "this will never be reached"})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_naive_endpoint_hangs_when_llm_is_down():
    """
    BEFORE state: proves the naive endpoint blocks indefinitely.

    The mock LLM sleeps for 60 seconds.  We wrap the API call in a 3-second
    asyncio timeout.  The naive endpoint has no internal timeout so it will
    still be waiting when our external deadline fires — confirming the bug.

    The test PASSES when the endpoint fails to respond within the detection
    window (i.e., the hang is confirmed).
    """
    print("\n\nBEFORE (naive):  Calling the unprotected endpoint...")
    print("BEFORE (naive):  The mock LLM will hang for 60s — watch the server block.\n")

    # assert_all_called=False: when wait_for cancels the task, the request is
    # torn down mid-flight so respx may not record it as "called" — that is
    # fine, the cancellation itself is what we are asserting.
    async with respx.mock(base_url="http://localhost:9999", assert_all_called=False) as mock:
            # Register the hanging handler for the LLM endpoint
            mock.post("/v1/generate").mock(side_effect=_hanging_llm_handler)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:

                with pytest.raises(asyncio.TimeoutError):
                    # Give the naive call only HANG_DETECTION_TIMEOUT seconds.
                    # asyncio.wait_for raises TimeoutError when the deadline fires —
                    # the naive call will not finish in time, confirming the bug.
                    await asyncio.wait_for(
                        client.post(
                            "/api/document/generate-naive",
                            json={"prompt": "Tell me about distributed systems"},
                        ),
                        timeout=HANG_DETECTION_TIMEOUT,
                    )

    print("BEFORE (naive):  Request timed out / hung as expected — the server would be blocked!")
    print("BEFORE (naive):  Every other user's request is now also stalled.\n")


@pytest.mark.asyncio
async def test_naive_endpoint_has_no_x_student_id_on_timeout():
    """
    Secondary observation: because the naive endpoint hangs, the middleware
    never gets to stamp the X-Student-ID header — no response is delivered at all.

    This test documents the behaviour rather than asserting a header value;
    it confirms an asyncio.TimeoutError is raised before any response arrives.
    """
    async with respx.mock(base_url="http://localhost:9999", assert_all_called=False) as mock:
        mock.post("/v1/generate").mock(side_effect=_hanging_llm_handler)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    client.post(
                        "/api/document/generate-naive",
                        json={"prompt": "What is the CAP theorem?"},
                    ),
                    timeout=HANG_DETECTION_TIMEOUT,
                )
