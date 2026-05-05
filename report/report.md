# Building Resilient Distributed Systems
**Nauman Arif** | **BSCS23060** | **Spring 2026**  
**Course:** Parallel and Distributed Computing

---

## Part 1: Analysis

### Problem 1 — Lost Update (Synchronization)

The root cause is the absence of a version guard in the database write path. When two users
concurrently fetch a document, they both receive the same current state (e.g., `version: 5`).
Each then independently modifies it and issues an `UPDATE` statement to the database. Because
neither statement carries a version predicate, both succeed: the database executes
`SET content = "User A's edit"` and then `SET content = "User B's edit"`, silently discarding
User A's work. This is the classic **lost update anomaly** — a write-write conflict that
occurs when two transactions read the same value, compute independent updates, and then write
back without checking whether the value changed in the interim. The failure happens at the
boundary between the API layer (which holds no locks) and the database layer (which has no
optimistic version check), resulting in a last-writer-wins outcome that is invisible to the
losing writer.

### Problem 2 — Dropped Webhook (Coordination)

The root cause is that the webhook handler is **not idempotent** and has **no durable delivery
guarantee**. Clerk sends a cancellation event over HTTP exactly once per trigger. If a transient
network blip causes that request to drop or time out before the handler acknowledges it, the
event is lost permanently — Clerk has no built-in retry loop in the naive integration. The
handler also maintains no event log, so there is no way to detect the gap or replay the missed
cancellation. The user therefore remains marked as premium in the application database despite
having cancelled, creating an indefinitely inconsistent state between the identity provider and
the application. The underlying distributed systems principle being violated is **at-least-once
delivery**: without retries and idempotency, the system operates at **at-most-once**, which
tolerates data loss on network failure.

### Problem 3 — Synchronous LLM Call (Fault Tolerance)

The root cause is an unconstrained blocking call to an external service with no timeout and no
circuit breaker. Although FastAPI is built on an async event loop (`asyncio`), a coroutine that
is suspended waiting for a network response still occupies a logical "slot" in the worker. When
the LLM API hangs for 60 seconds, the coroutine is suspended for 60 seconds. Concurrent
requests that hit the same endpoint queue behind it — the event loop cannot make progress on
any of them because its I/O threads are occupied waiting for responses that never arrive. This
makes the external LLM API a **single point of failure**: one misbehaving dependency drags down
the entire application for all users. The absence of a timeout means the application cannot
proactively detect the failure; the absence of a circuit breaker means it cannot stop sending
requests to a known-broken dependency.

---

## Part 2: Design

### Fix 1 — Optimistic Locking for Concurrent Document Edits

Add a `version: INTEGER` column to the document table. Every read includes the version number.
Every write includes a `WHERE version = <last_seen_version>` predicate:

```sql
UPDATE documents
SET    content = ?, version = version + 1
WHERE  id = ? AND version = ?;
```

If the affected row count is 0, the application returns **HTTP 409 Conflict** — the version
has changed since the caller last read the document, meaning another writer committed first.
The client is instructed to re-fetch the latest version and reapply their edit. This approach
requires no locks in the database and adds negligible latency on the happy path.

#### UML Sequence Diagram — Concurrent Edit Resolution

```
  User A                  API Server              Database
    |                         |                       |
    |--- GET /doc/1 --------> |                       |
    |                         |--- SELECT ... ------> |
    |                         | <-- {content, v:5} -- |
    | <-- {content, v:5} ---- |                       |
    |                         |                       |
  User B                      |                       |
    |--- GET /doc/1 --------> |                       |
    |                         |--- SELECT ... ------> |
    |                         | <-- {content, v:5} -- |
    | <-- {content, v:5} ---- |                       |
    |                         |                       |
  User A                      |                       |
    |--- PUT /doc/1 {v:5} --> |                       |
    |                         |-- UPDATE ... v=5 --> |
    |                         | <-- 1 row updated --- |
    | <-- 200 OK {v:6} ------ |                       |
    |                         |                       |
  User B                      |                       |
    |--- PUT /doc/1 {v:5} --> |                       |
    |                         |-- UPDATE ... v=5 --> |
    |                         | <-- 0 rows updated -- |  (version is now 6)
    | <-- 409 Conflict ------- |                       |
    | (re-fetch and retry)     |                       |
```

### Fix 2 — Fault-Tolerant Webhook Handler

Use a two-part defence:

1. **Idempotency key:** Clerk attaches a unique `svix-id` to every webhook delivery. Store each
   processed `svix-id` in a `webhook_events` table with the received timestamp and outcome. On
   arrival, check for the key before processing. If found, return HTTP 200 immediately — the
   event was already handled. This makes the handler safe to call multiple times.

2. **Retry with exponential backoff:** Configure Clerk's webhook settings to retry failed
   deliveries up to 5 times with exponential backoff (e.g. 5 s, 10 s, 20 s, 40 s, 80 s).
   Because the handler is now idempotent, retries cannot cause double-processing.

3. **Dead-letter queue (DLQ):** For events that exhaust all retries, publish them to a Redis
   stream or a dedicated `failed_events` table. An operator can inspect and manually replay them,
   or an automated job can retry on a slower schedule. This ensures no cancellation is silently
   dropped — it either succeeds or lands in a visible, recoverable queue.

### Fix 3 — Circuit Breaker + Fallback for the LLM API

Wrap every LLM call in a three-state circuit breaker:

```
  CLOSED --(failures >= threshold)--> OPEN --(recovery timeout)--> HALF_OPEN
    ^                                                                   |
    +--------------------(probe succeeds)------------------------------+
    HALF_OPEN --(probe fails)--> OPEN
```

- **CLOSED:** Normal operation. Each call is forwarded to the LLM with a 5-second timeout.
  A failure increments a consecutive-failure counter. After `N` failures (default 3) the
  circuit trips to OPEN.
- **OPEN:** All calls are rejected immediately with a `CircuitOpenError`. The caller activates
  a fallback response (a cached answer or a graceful "service unavailable" message). No network
  call is made, so the event loop is never blocked.
- **HALF_OPEN:** After a configurable recovery window (default 30 s), one probe request is
  allowed through. Success closes the circuit; failure re-opens it and resets the timer.

The fallback ensures the API endpoint returns a meaningful response to the user even when the
LLM is down, maintaining availability at the cost of freshness.

### CAP Theorem Trade-offs

The three fixes make different positions along the CAP spectrum:

| Fix | CAP Position | Reasoning |
|-----|-------------|-----------|
| Optimistic Locking | Favours **Consistency** | A losing write is rejected rather than applied silently. The system sacrifices availability (the losing writer gets a 409 and must retry) to guarantee that the database never holds a corrupted merged state. Under a network partition, writes are blocked until the partition heals. |
| Idempotent Webhook | Favours **Availability** | By storing idempotency keys and using retries, the system tolerates transient network partitions without losing events. The trade-off is a small latency cost (deduplication lookup) and eventual consistency — there is a window between a cancellation event being sent and it being processed. |
| Circuit Breaker | Favours **Availability** | When the LLM is unreachable the system returns a stale/static fallback rather than hanging or returning an error. Users remain served (available) at the cost of consistency — the content they receive is not freshly generated. Latency is dramatically reduced for failing requests (milliseconds instead of 60 seconds). |

There is no single "correct" position; the right trade-off depends on the business requirement.
Billing accuracy (webhooks) demands strong consistency. User-facing content generation
(LLM) can tolerate eventual consistency to maintain responsiveness. Document collaboration
sits in between — optimistic locking balances usability with correctness by allowing concurrent
reads while rejecting conflicting writes cleanly.
