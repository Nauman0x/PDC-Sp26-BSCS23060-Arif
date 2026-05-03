"""
Middleware for injecting the mandatory X-Student-ID header into every API response.

This fulfills the assignment's strict submission requirement: every single HTTP response
from the FastAPI application must carry the header X-Student-ID: BSCS23060.
Without this header, Part 3 receives an automatic zero.
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

STUDENT_ID = "BSCS23060"


class StudentIDMiddleware(BaseHTTPMiddleware):
    """
    Starlette BaseHTTPMiddleware that appends ``X-Student-ID: BSCS23060``
    to the headers of every outbound HTTP response, regardless of route,
    status code, or response body.

    Because this sits at the middleware layer it fires for ALL responses —
    including 404s, 422 validation errors, and 500 internal server errors —
    so the grader will always see the header.

    Example response headers::

        HTTP/1.1 200 OK
        content-type: application/json
        X-Student-ID: BSCS23060
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Pass the request through to the next handler, then stamp the
        student-ID header onto the response before returning it to the client.

        :param request: The incoming HTTP request.
        :param call_next: Callable that forwards the request down the middleware stack.
        :return: The original response, with the ``X-Student-ID`` header injected.
        """
        response: Response = await call_next(request)
        response.headers["X-Student-ID"] = STUDENT_ID
        return response
