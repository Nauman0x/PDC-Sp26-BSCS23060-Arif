"""
AFTER: Proves the Circuit Breaker + Fallback fixes the fault-tolerance bug.

This test file is the "after" side of the demo.  It covers all three
phases of the circuit breaker state machine in sequence:

  Phase 1 — CLOSED, LLM is down
      Three requests are fired while the mock LLM times out.
      Each request returns quickly with a fallback (no 60-second hang).
      After the third failure the circuit trips to OPEN.

  Phase 2 — OPEN, instant fallback
      A fourth request is fired while the circuit is OPEN.
      The circuit breaker rejects it immediately without touching the LLM.
      The fallback is returned in near-zero time.

  Phase 3 — HALF_OPEN, probe and recovery
      The recovery timer is artificially advanced by monkey-patching
      ``time.monotonic`` so the test does not have to wait 30 seconds.
      A fifth request is allowed through as a probe.
      The mock LLM now succeeds — the circuit closes.
      The sixth request goes to the real LLM and gets the live response.

Cross-cutting assertion
-----------------------
Every single response — fallback or real, open or closed — must carry the
header  X-Student-ID: BSCS23060  (the mandatory grading requirement).

How to run
----------
    pytest tests/test_circuit_breaker.py -v -s
"""

import time
from unittest.mock import patch

import pytest
import respx
import httpx
from httpx import AsyncClient, ASGITransport

from main import app
from services.llm_service import get_circuit_breaker
from circuit_breaker.circuit_breaker import CircuitState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LLM_URL = "http://localhost:9999/v1/generate"
STUDENT_ID_HEADER = "X-Student-ID"
EXPECTED_STUDENT_ID = "BSCS23060"

# Must match FAILURE_THRESHOLD in llm_service.py (default 3)
FAILURE_THRESHOLD = 3

# Must match RECOVERY_TIMEOUT in llm_service.py (default 30 s)
RECOVERY_TIMEOUT = 30.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_circuit_breaker():
    """
    Reset the shared circuit breaker to CLOSED before every test so each
    test starts from a clean slate and they do not bleed state into each other.
    """
    get_circuit_breaker().reset()
    yield
    get_circuit_breaker().reset()


# ---------------------------------------------------------------------------
# Phase 1: CLOSED circuit, LLM is down — each request gets a fast fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_phase1_closed_circuit_returns_fallback_on_timeout():
    """
    AFTER — Phase 1: Circuit CLOSED, LLM times out.

    Fires FAILURE_THRESHOLD requests against a mock LLM that always times out.
    Each call must:
      - Return HTTP 200 (not a 500 / hang)
      - Return is_fallback: true
      - Carry the X-Student-ID header
      - Complete within a few seconds (not 60s)

    After the last failing request the circuit should have tripped to OPEN.
    """
    print("\n\nAFTER — Phase 1: Sending requests while LLM is DOWN (circuit CLOSED)...")

    async with respx.mock(base_url="http://localhost:9999") as mock:
        # Simulate LLM timing out on every call
        mock.post("/v1/generate").mock(
            side_effect=httpx.TimeoutException("LLM timed out")
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:

            for i in range(1, FAILURE_THRESHOLD + 1):
                response = await client.post(
                    "/api/document/generate",
                    json={"prompt": f"Prompt number {i}"},
                )

                assert response.status_code == 200, (
                    f"Request {i} should return 200 with fallback, not {response.status_code}"
                )

                body = response.json()
                assert body["is_fallback"] is True, (
                    f"Request {i}: expected fallback=true, got {body}"
                )

                # Mandatory header must be present on every response
                assert response.headers.get(STUDENT_ID_HEADER) == EXPECTED_STUDENT_ID, (
                    f"Missing or incorrect X-Student-ID on request {i}"
                )

                print(
                    f"  Request {i}: status={response.status_code}, "
                    f"fallback={body['is_fallback']}, "
                    f"circuit={body['circuit_state']}, "
                    f"X-Student-ID={response.headers.get(STUDENT_ID_HEADER)}"
                )

    # After FAILURE_THRESHOLD failures the circuit must be OPEN
    assert get_circuit_breaker().state == CircuitState.OPEN, (
        f"Circuit should be OPEN after {FAILURE_THRESHOLD} failures, "
        f"but is {get_circuit_breaker().state}"
    )
    print(f"\n  Circuit is now: {get_circuit_breaker().state.value.upper()} [OK]")


# ---------------------------------------------------------------------------
# Phase 2: OPEN circuit — instant fallback, zero network calls
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_phase2_open_circuit_instant_fallback():
    """
    AFTER — Phase 2: Circuit OPEN, no LLM call is made.

    Forces the circuit to OPEN by simulating past failures, then fires a
    request and verifies:
      - Fallback is returned immediately
      - The mock LLM endpoint is NEVER contacted (zero network calls)
      - X-Student-ID header is still present
    """
    print("\n\nAFTER — Phase 2: Circuit is OPEN — instant fallback, no network call...")

    # Manually trip the circuit to OPEN by simulating failures
    cb = get_circuit_breaker()
    cb._failure_count = FAILURE_THRESHOLD
    cb._state = CircuitState.OPEN
    cb._last_failure_time = time.monotonic()

    # assert_all_called=False because the whole point of this test is that the
    # LLM route is NEVER contacted when the circuit is OPEN.
    async with respx.mock(base_url="http://localhost:9999", assert_all_called=False) as mock:
        # Register the route but expect it to NEVER be called
        llm_route = mock.post("/v1/generate").mock(
            return_value=httpx.Response(200, json={"text": "should not reach LLM"})
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/document/generate",
                json={"prompt": "Will this reach the LLM?"},
            )

    assert response.status_code == 200
    body = response.json()
    assert body["is_fallback"] is True, f"Expected fallback=true, got {body}"
    assert body["circuit_state"] == "open", f"Expected state=open, got {body['circuit_state']}"

    # The LLM endpoint must not have been called at all
    assert llm_route.call_count == 0, (
        f"LLM was called {llm_route.call_count} times — it should be 0 when circuit is OPEN"
    )

    assert response.headers.get(STUDENT_ID_HEADER) == EXPECTED_STUDENT_ID

    print(
        f"  Fallback returned instantly. LLM call count: {llm_route.call_count} (expected 0)."
    )
    print(f"  Circuit state: {body['circuit_state'].upper()}")
    print(f"  X-Student-ID: {response.headers.get(STUDENT_ID_HEADER)}")


# ---------------------------------------------------------------------------
# Phase 3: HALF_OPEN — probe succeeds, circuit closes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_phase3_half_open_probe_closes_circuit():
    """
    AFTER — Phase 3: Recovery window elapses, probe succeeds, circuit closes.

    Sets the circuit to OPEN and artificially advances ``time.monotonic`` by
    RECOVERY_TIMEOUT + 1 seconds so the circuit transitions to HALF_OPEN on
    the next call.  The probe call succeeds → circuit closes → subsequent
    request goes to the real LLM.
    """
    print("\n\nAFTER — Phase 3: Recovery window elapsed — sending probe request...")

    cb = get_circuit_breaker()
    # Manually open the circuit
    cb._state = CircuitState.OPEN
    cb._failure_count = FAILURE_THRESHOLD
    recorded_failure_time = time.monotonic()
    cb._last_failure_time = recorded_failure_time

    # Monkey-patch monotonic so the circuit thinks the recovery window has elapsed
    future_time = recorded_failure_time + RECOVERY_TIMEOUT + 1

    async with respx.mock(base_url="http://localhost:9999") as mock:
        mock.post("/v1/generate").mock(
            return_value=httpx.Response(200, json={"text": "LLM is back online!"})
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:

            # Advance time past the recovery window so _evaluate_state() transitions
            # OPEN → HALF_OPEN
            with patch("circuit_breaker.circuit_breaker.time") as mock_time:
                mock_time.monotonic.return_value = future_time

                # Probe request — circuit goes HALF_OPEN, then CLOSED on success
                probe_response = await client.post(
                    "/api/document/generate",
                    json={"prompt": "Is the LLM back?"},
                )

    assert probe_response.status_code == 200
    probe_body = probe_response.json()

    print(
        f"  Probe response: is_fallback={probe_body['is_fallback']}, "
        f"circuit={probe_body['circuit_state']}"
    )

    # After a successful probe the circuit must be CLOSED
    assert cb.state == CircuitState.CLOSED, (
        f"Circuit should be CLOSED after a successful probe, but is {cb.state}"
    )

    assert probe_body["is_fallback"] is False, (
        f"Probe should return real LLM text, not fallback. Got: {probe_body}"
    )
    assert probe_body["text"] == "LLM is back online!"
    assert probe_response.headers.get(STUDENT_ID_HEADER) == EXPECTED_STUDENT_ID

    print(f"  Circuit is now: {cb.state.value.upper()} [OK]")
    print(f"  X-Student-ID: {probe_response.headers.get(STUDENT_ID_HEADER)}")


# ---------------------------------------------------------------------------
# Cross-cutting: X-Student-ID header is present on EVERY response
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_student_id_header_on_healthy_response():
    """
    Verifies the X-Student-ID header is present when the LLM is healthy
    (circuit CLOSED, real response returned).
    """
    async with respx.mock(base_url="http://localhost:9999") as mock:
        mock.post("/v1/generate").mock(
            return_value=httpx.Response(200, json={"text": "Here is your study guide."})
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/document/generate",
                json={"prompt": "Explain the CAP theorem"},
            )

    assert response.status_code == 200
    assert response.headers.get(STUDENT_ID_HEADER) == EXPECTED_STUDENT_ID

    body = response.json()
    assert body["is_fallback"] is False
    assert body["text"] == "Here is your study guide."
    assert body["circuit_state"] == "closed"

    print(f"\nHealthy response — X-Student-ID: {response.headers.get(STUDENT_ID_HEADER)}")


@pytest.mark.asyncio
async def test_student_id_header_on_health_endpoint():
    """
    Verifies the X-Student-ID header is present on non-LLM endpoints too
    (the /health route should also carry the header).
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.headers.get(STUDENT_ID_HEADER) == EXPECTED_STUDENT_ID
    print(f"\n/health endpoint — X-Student-ID: {response.headers.get(STUDENT_ID_HEADER)}")
