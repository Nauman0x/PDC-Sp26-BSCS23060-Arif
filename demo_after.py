"""
DEMO — AFTER: The circuit breaker returns a fallback immediately.

Run this script after demo_before.py to show the fix for your video.

What happens:
  1. The same mock "hanging LLM" server starts on port 9999.
  2. Three requests are fired through the circuit breaker (5s timeout each).
     Each one returns within ~5 seconds with a fallback — the server is NEVER blocked.
  3. After 3 failures, the circuit OPENS. The 4th request returns INSTANTLY
     (< 1ms) — the LLM is not even contacted.
  4. The mock LLM "recovers". After the recovery window, a probe closes the circuit.

Total runtime: about 16 seconds (3 x 5s timeout + instant phase 2 + instant phase 3).
"""

import asyncio
import os
import socketserver
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

# Point the service at our local mock server
os.environ["LLM_URL"] = "http://localhost:9999/v1/generate"
# Use a short recovery timeout so the demo does not take 30 seconds
os.environ["CB_RECOVERY_TIMEOUT"] = "5"

# Import AFTER setting env vars so the service picks them up
from services.llm_service import generate_with_fallback, get_circuit_breaker

# ---------------------------------------------------------------------------
# Mock server: starts hanging, then "recovers" after a flag is set
# ---------------------------------------------------------------------------

_llm_recovered = False


class SmartMockLLMHandler(BaseHTTPRequestHandler):
    """Hangs until _llm_recovered is True, then responds normally."""

    def do_POST(self):
        if _llm_recovered:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"text": "LLM is back online! Here is your study guide."}')
        else:
            # Simulate a hung LLM — sleep long enough to trigger the 5s timeout
            time.sleep(30)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"text": "too late"}')

    def log_message(self, format, *args):
        pass


class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """Each request is handled in its own thread so sleeping handlers
    do not block new incoming connections."""
    daemon_threads = True


def start_mock_server():
    server = ThreadedHTTPServer(("localhost", 9999), SmartMockLLMHandler)
    server.serve_forever()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

async def run_demo():
    global _llm_recovered

    cb = get_circuit_breaker()

    print()
    print("=" * 60)
    print("  AFTER: Circuit Breaker + Fallback demo")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Phase 1: Circuit CLOSED, LLM is down
    # ------------------------------------------------------------------
    print()
    print("  PHASE 1 — Circuit CLOSED, LLM is DOWN")
    print("  Firing 3 requests (each will timeout after 5s and return fallback)...")
    print()

    for i in range(1, 4):
        t0 = time.monotonic()
        result = await generate_with_fallback(f"Prompt {i}: explain distributed systems")
        elapsed = time.monotonic() - t0

        print(
            f"  Request {i}: returned in {elapsed:.1f}s | "
            f"fallback={result.is_fallback} | "
            f"circuit={result.circuit_state.value}"
        )

    print()
    print(f"  Circuit state after 3 failures: {cb.state.value.upper()}")

    # ------------------------------------------------------------------
    # Phase 2: Circuit OPEN — instant fallback, zero network calls
    # ------------------------------------------------------------------
    print()
    print("  PHASE 2 — Circuit OPEN")
    print("  Firing request while circuit is OPEN (should return INSTANTLY)...")
    print()

    t0 = time.monotonic()
    result = await generate_with_fallback("Prompt 4: what is the CAP theorem?")
    elapsed = time.monotonic() - t0

    print(
        f"  Request 4: returned in {elapsed*1000:.1f}ms | "
        f"fallback={result.is_fallback} | "
        f"circuit={result.circuit_state.value}"
    )
    print(f"  (LLM was NOT contacted — circuit rejected the call in {elapsed*1000:.1f}ms)")

    # ------------------------------------------------------------------
    # Phase 3: LLM "recovers", circuit HALF_OPEN → CLOSED
    # ------------------------------------------------------------------
    print()
    recovery = int(os.environ["CB_RECOVERY_TIMEOUT"])
    print(f"  PHASE 3 — Waiting {recovery}s for recovery window to elapse...")
    print(f"  (In production this is 30s; set to {recovery}s for this demo)")

    _llm_recovered = True  # tell the mock server to start responding normally
    await asyncio.sleep(recovery + 1)

    print()
    print("  Recovery window elapsed. Firing probe request (HALF_OPEN)...")
    print()

    t0 = time.monotonic()
    result = await generate_with_fallback("Probe: is the LLM back?")
    elapsed = time.monotonic() - t0

    print(
        f"  Probe:      returned in {elapsed:.1f}s | "
        f"fallback={result.is_fallback} | "
        f"circuit={cb.state.value}"
    )
    if not result.is_fallback:
        print(f"  LLM text:  \"{result.text}\"")

    print()
    print(f"  Circuit state: {cb.state.value.upper()}")
    print()
    print("=" * 60)
    print("  DONE: Server stayed responsive throughout.")
    print("  Even when the LLM was down, every request returned a")
    print("  fallback in under 6 seconds — not 60.")
    print("=" * 60)
    print()


if __name__ == "__main__":
    thread = threading.Thread(target=start_mock_server, daemon=True)
    thread.start()

    print()
    print("  [mock-llm server] Started on http://localhost:9999")
    print("  [mock-llm server] Currently: HANGING (simulating LLM outage)")

    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        print("\n  Demo interrupted.")
