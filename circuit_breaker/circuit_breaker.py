"""
Circuit Breaker implementation for protecting external service calls.

The Circuit Breaker pattern prevents a failing downstream dependency (e.g. an LLM API)
from cascading its failure across the entire application.  Instead of letting every
caller block for a full timeout on every request, the breaker "opens" after a
configurable number of consecutive failures and immediately rejects further calls
until a recovery window has elapsed.

State machine:

    CLOSED ──(failures >= threshold)──► OPEN ──(recovery_timeout elapsed)──► HALF_OPEN
      ▲                                                                            │
      └─────────────────(probe succeeds)──────────────────────────────────────────┘
      HALF_OPEN ──(probe fails)──► OPEN  (reset timer)

States:
    CLOSED    — Normal operation.  All calls are forwarded to the real service.
    OPEN      — Service is considered down.  All calls fail immediately with
                CircuitOpenError so callers can use a fallback without waiting.
    HALF_OPEN — A single probe call is allowed through to test whether the
                downstream service has recovered.  Success closes the circuit;
                failure re-opens it and resets the recovery timer.
"""

import asyncio
import time
from enum import Enum
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


class CircuitState(str, Enum):
    """The three possible states of the circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """
    Raised when a call is attempted while the circuit is in the OPEN state.

    Callers should catch this and activate their fallback path rather than
    waiting for the underlying service, which is presumed to be unavailable.

    Example::

        try:
            result = await circuit_breaker.call(llm_request)
        except CircuitOpenError:
            return FALLBACK_RESPONSE
    """


class CircuitBreaker:
    """
    Async-safe, three-state circuit breaker.

    All state mutations are serialised through an ``asyncio.Lock`` so the
    breaker is safe for concurrent coroutines in a single event loop (the
    normal FastAPI execution model).

    :param failure_threshold: Number of consecutive failures required to trip
        the circuit from CLOSED to OPEN.
    :param recovery_timeout: Seconds the breaker stays OPEN before moving to
        HALF_OPEN and allowing a single probe request.
    :param half_open_max_calls: How many probe calls are permitted while
        HALF_OPEN before a decision is made.  Keeping this at 1 (the default)
        keeps the logic simple and predictable.
    :param name: Optional human-readable label used in log messages.

    Example::

        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)

        async def protected_call():
            try:
                return await breaker.call(fetch_from_llm, prompt)
            except CircuitOpenError:
                return "LLM unavailable — here is a cached answer."
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
        name: str = "CircuitBreaker",
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.name = name

        # Mutable state — always access under _lock
        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._last_failure_time: float = 0.0
        self._half_open_calls: int = 0

        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        """Return the current circuit state (read-only snapshot)."""
        return self._state

    async def call(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Attempt to call ``func(*args, **kwargs)`` through the circuit breaker.

        Behaviour by state:

        * **CLOSED** — forwards the call; on failure increments the counter
          and may trip the circuit to OPEN.
        * **OPEN** — checks whether ``recovery_timeout`` has elapsed; if so
          transitions to HALF_OPEN and allows the call as a probe; otherwise
          raises :exc:`CircuitOpenError` immediately.
        * **HALF_OPEN** — allows up to ``half_open_max_calls`` probe calls.
          A successful probe closes the circuit; a failure re-opens it.

        :param func: An async callable to protect.
        :param args: Positional arguments forwarded to ``func``.
        :param kwargs: Keyword arguments forwarded to ``func``.
        :raises CircuitOpenError: When the circuit is OPEN and the recovery
            window has not yet elapsed.
        :return: The return value of ``func``.
        """
        async with self._lock:
            current_state = self._evaluate_state()

            if current_state == CircuitState.OPEN:
                # Fast-fail — do not touch the downstream service
                raise CircuitOpenError(
                    f"[{self.name}] Circuit is OPEN. "
                    f"Service unavailable until recovery window expires."
                )

            if current_state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1

        # Execute outside the lock so other coroutines are not blocked while
        # the (potentially slow) downstream call is in flight.
        try:
            result = await func(*args, **kwargs)
        except Exception as exc:
            async with self._lock:
                self._on_failure()
            raise exc

        async with self._lock:
            self._on_success()

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_state(self) -> CircuitState:
        """
        Check whether the OPEN → HALF_OPEN transition is due and apply it.

        Called while the lock is held.  Returns the effective state after
        any automatic transition.
        """
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.recovery_timeout:
                # Recovery window elapsed — let one probe through
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
        return self._state

    def _on_success(self) -> None:
        """
        Record a successful call.

        * If HALF_OPEN: the probe succeeded — close the circuit and reset counters.
        * If CLOSED: reset the consecutive failure counter (partial recovery).

        Called while the lock is held.
        """
        self._failure_count = 0
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED
            self._half_open_calls = 0

    def _on_failure(self) -> None:
        """
        Record a failed call and potentially trip or re-open the circuit.

        * If CLOSED and failures reach threshold → trip to OPEN.
        * If HALF_OPEN → the probe failed, reopen the circuit and reset the timer.

        Called while the lock is held.
        """
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._state == CircuitState.HALF_OPEN:
            # Probe failed — back to OPEN, reset recovery timer
            self._state = CircuitState.OPEN
            self._half_open_calls = 0
            return

        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN

    def reset(self) -> None:
        """
        Manually reset the circuit breaker to CLOSED with a clean slate.

        Useful in tests to restore the breaker to a known state between test
        cases without creating a new instance.
        """
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
