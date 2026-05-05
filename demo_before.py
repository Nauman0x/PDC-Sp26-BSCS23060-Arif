"""
DEMO — BEFORE: The real FastAPI server visibly blocks under a hung LLM.

What this script does
---------------------
1. Starts a "hanging LLM" mock server on port 9999 — it accepts connections
   but sleeps 60 s before replying, simulating an overloaded/crashed LLM API.
2. Starts the REAL FastAPI app (uvicorn) on port 8000.
3. Fires 5 concurrent user requests to POST /api/document/generate-naive.
4. Displays a live elapsed-time counter while ALL 5 requests hang with no response.
5. On Ctrl+C, prints a summary showing 0 out of 5 users were served.

This is the realistic failure mode: the server is running, it accepted every
request, but no user ever gets a response back.

Run:
    python demo_before.py

Then press Ctrl+C after ~10 seconds to stop.
"""

import asyncio
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver

import httpx

FASTAPI_PORT = 8000
MOCK_LLM_PORT = 9999
NUM_USERS = 5
LLM_HANG_SECONDS = 60


# ---------------------------------------------------------------------------
# Mock LLM server: accepts connections, sleeps, never replies in time
# ---------------------------------------------------------------------------

class HangingLLMHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        time.sleep(LLM_HANG_SECONDS)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"text": "too late"}')

    def log_message(self, format, *args):
        pass  # suppress noisy HTTP logs


class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True


def start_mock_llm():
    server = ThreadedHTTPServer(("localhost", MOCK_LLM_PORT), HangingLLMHandler)
    server.serve_forever()


# ---------------------------------------------------------------------------
# Concurrent users: all fire at the same time, all hang
# ---------------------------------------------------------------------------

results = {}   # user_id -> "pending" | "responded" | "error"


async def user_request(user_id: int, client: httpx.AsyncClient):
    """Simulate one user hitting the unprotected endpoint."""
    results[user_id] = "pending"
    try:
        response = await client.post(
            f"http://localhost:{FASTAPI_PORT}/api/document/generate-naive",
            json={"prompt": f"User {user_id}: explain distributed systems"},
            timeout=None,  # no timeout — mirrors the naive implementation
        )
        results[user_id] = f"responded ({response.status_code})"
    except Exception as exc:
        results[user_id] = f"error: {exc}"


async def fire_concurrent_users():
    """Launch all user requests simultaneously and let them hang."""
    async with httpx.AsyncClient() as client:
        tasks = [
            asyncio.create_task(user_request(i, client))
            for i in range(1, NUM_USERS + 1)
        ]
        await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# Live status display: runs in a thread, prints a counter every second
# ---------------------------------------------------------------------------

_stop_display = threading.Event()


def live_status():
    start = time.monotonic()
    while not _stop_display.is_set():
        elapsed = int(time.monotonic() - start)
        pending = sum(1 for v in results.values() if v == "pending")
        served  = sum(1 for v in results.values() if v != "pending")

        # Overwrite the same line with a live counter
        print(
            f"\r  [{elapsed:>3}s elapsed]  "
            f"Requests sent: {NUM_USERS}  |  "
            f"Responses received: {served}  |  "
            f"Still waiting: {pending}   ",
            end="",
            flush=True,
        )
        time.sleep(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("=" * 64)
    print("  BEFORE: Naive LLM endpoint — no timeout, no circuit breaker")
    print("=" * 64)
    print()

    # Step 1: start hanging mock LLM
    t = threading.Thread(target=start_mock_llm, daemon=True)
    t.start()
    print(f"  [mock-llm]  Hanging LLM server started on port {MOCK_LLM_PORT}")
    print(f"  [mock-llm]  Will sleep {LLM_HANG_SECONDS}s before replying to any request")
    print()

    # Step 2: start the real FastAPI server (uvicorn) as a subprocess
    print(f"  [server]    Starting FastAPI on http://localhost:{FASTAPI_PORT} ...")
    server_proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "main:app",
            "--port", str(FASTAPI_PORT),
            "--log-level", "warning",   # suppress per-request logs for cleaner output
        ],
        env={**__import__("os").environ, "LLM_URL": f"http://localhost:{MOCK_LLM_PORT}/v1/generate"},
    )

    time.sleep(2)  # wait for uvicorn to be ready
    print(f"  [server]    FastAPI is up.")
    print()
    print(f"  Firing {NUM_USERS} concurrent user requests to /api/document/generate-naive ...")
    print(f"  (Watch: none of them will get a response)")
    print()

    # Step 3: start the live status display
    display_thread = threading.Thread(target=live_status, daemon=True)
    display_thread.start()

    # Step 4: fire concurrent users — this will hang
    try:
        asyncio.run(fire_concurrent_users())
    except KeyboardInterrupt:
        pass
    finally:
        _stop_display.set()
        server_proc.terminate()

    print()
    print()

    # Step 5: summary
    served  = sum(1 for v in results.values() if v != "pending")
    pending = sum(1 for v in results.values() if v == "pending")
    print(f"  RESULT: {served} / {NUM_USERS} users received a response.")
    print(f"          {pending} users are still waiting (or were cut short by Ctrl+C).")
    print()
    print("  This is the bug. Now run:  python demo_after.py")
    print()
