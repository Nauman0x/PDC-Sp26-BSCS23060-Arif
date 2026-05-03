Nauman Arif | BSCS23060

# PDC-Sp26-BSCS23060-Arif

**Course:** Parallel and Distributed Computing (PDC)  
**Assignment:** Building Resilient Distributed Systems  
**Student ID:** BSCS23060  
**Problem Implemented:** Problem 3 — Circuit Breaker + Fallback for the LLM API

---

## What This Project Does

This is a minimal FastAPI backend simulating the StudySync application.  
It demonstrates the **Circuit Breaker pattern** applied to an external LLM API call.

**The bug (before):** `POST /api/document/generate-naive` calls the LLM with no timeout and no
circuit breaker. When the LLM hangs, this endpoint blocks indefinitely, making the entire server
unresponsive for all users.

**The fix (after):** `POST /api/document/generate` wraps the same call in a 3-state circuit
breaker (CLOSED → OPEN → HALF_OPEN) with a 5-second timeout. When the LLM is down, callers
receive an instant fallback response. Every response also carries the mandatory `X-Student-ID: BSCS23060` header.

---

## Project Structure

```
.
├── main.py                            # FastAPI app entry point
├── middleware/
│   └── student_id_middleware.py       # Injects X-Student-ID header on every response
├── circuit_breaker/
│   └── circuit_breaker.py             # CLOSED/OPEN/HALF_OPEN state machine
├── services/
│   └── llm_service.py                 # LLM call wrapped with circuit breaker + fallback
├── routers/
│   └── document.py                    # /api/document/generate and /generate-naive endpoints
├── tests/
│   ├── test_naive.py                  # BEFORE: proves the naive endpoint hangs
│   └── test_circuit_breaker.py        # AFTER: proves all 3 circuit breaker phases work
├── report/
│   └── report.md                      # Parts 1 & 2 — export to PDF for submission
├── requirements.txt
└── pytest.ini
```

---

## Prerequisites

- Python 3.10 or higher
- pip

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/[YourUsername]/PDC-Sp26-BSCS23060-Arif.git
cd PDC-Sp26-BSCS23060-Arif

# 2. (Recommended) Create a virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

### Useful endpoints

| Method | URL | Description |
|--------|-----|-------------|
| GET | `/health` | Liveness check + circuit breaker state |
| POST | `/api/document/generate` | Protected endpoint (circuit breaker + fallback) |
| POST | `/api/document/generate-naive` | Unprotected endpoint (demonstrates the bug) |

Example request body for both POST endpoints:
```json
{ "prompt": "Explain the CAP theorem in simple terms." }
```

### Verifying the X-Student-ID header

```bash
curl -s -D - -o /dev/null -X POST http://localhost:8000/api/document/generate \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"hello\"}"
```

You should see `X-Student-ID: BSCS23060` in the response headers.

---

## Running the Tests

### BEFORE demo (shows the bug)

```bash
pytest tests/test_naive.py -v -s
```

Expected output: both tests PASS, confirming the naive endpoint hangs (TimeoutError is raised
before the server returns a response).

### AFTER demo (proves the fix)

```bash
pytest tests/test_circuit_breaker.py -v -s
```

Expected output: all 5 tests PASS, showing:
- Phase 1: Circuit CLOSED, LLM down → fast fallback on every request, circuit opens after 3 failures
- Phase 2: Circuit OPEN → instant fallback, LLM never contacted (0 network calls)
- Phase 3: Recovery window elapsed → probe succeeds → circuit closes, real LLM response returned
- X-Student-ID: BSCS23060 present on every response

### Full suite

```bash
pytest tests/ -v -s
```

Expected: **7 passed** in ~7 seconds.

---

## Environment Variables (optional overrides)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_URL` | `http://localhost:9999/v1/generate` | External LLM endpoint |
| `LLM_TIMEOUT` | `5.0` | Request timeout in seconds |
| `CB_FAILURE_THRESHOLD` | `3` | Failures before circuit opens |
| `CB_RECOVERY_TIMEOUT` | `30.0` | Seconds before HALF_OPEN probe |
