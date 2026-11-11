"""
Lightweight in-memory rate limiting for the FastAPI server (no extra dependencies).

- Per-client sliding-window limits, applied as FastAPI dependencies per endpoint.
- A global concurrency cap for the expensive inference endpoint.

Limits are configurable through environment variables:
    RATE_LIMIT_ENABLED         (default "1"; set "0" to disable)
    RATE_LIMIT_PREDICT         requests/minute/client for /api/predict   (default 5)
    RATE_LIMIT_REPORT          requests/minute/client for /api/report/pdf (default 20)
    RATE_LIMIT_DEFAULT         requests/minute/client for other endpoints (default 60)
    MAX_CONCURRENT_PREDICTIONS simultaneous inference jobs server-wide   (default 1)
    TRUST_PROXY_HEADERS        "1" to key clients by X-Forwarded-For (only behind a trusted proxy)

Note: state is per-process. If running multiple workers, use a shared store (e.g. Redis).
"""

import os
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager

from fastapi import HTTPException, Request

WINDOW_SECONDS = 60.0


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except ValueError:
        return default


ENABLED = os.environ.get("RATE_LIMIT_ENABLED", "1") != "0"
TRUST_PROXY = os.environ.get("TRUST_PROXY_HEADERS", "0") == "1"
MAX_CONCURRENT_PREDICTIONS = _env_int("MAX_CONCURRENT_PREDICTIONS", 1)


def client_key(request: Request) -> str:
    if TRUST_PROXY:
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class SlidingWindowLimiter:
    """Allows `limit` requests per `window` seconds for each client key."""

    def __init__(self, name: str, limit: int, window: float = WINDOW_SECONDS):
        self.name = name
        self.limit = limit
        self.window = window
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()
        self._last_sweep = time.monotonic()

    def _sweep(self, now: float) -> None:
        # Periodically drop idle clients so memory doesn't grow unbounded.
        if now - self._last_sweep < self.window:
            return
        self._last_sweep = now
        for key in [k for k, q in self._hits.items() if not q or now - q[-1] > self.window]:
            del self._hits[key]

    def check(self, key: str):
        """Returns (allowed, retry_after_seconds, remaining)."""
        now = time.monotonic()
        with self._lock:
            self._sweep(now)
            q = self._hits[key]
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False, max(1, int(self.window - (now - q[0])) + 1), 0
            q.append(now)
            return True, 0, self.limit - len(q)

    def dependency(self):
        async def _dep(request: Request):
            if not ENABLED:
                return
            allowed, retry_after, _ = self.check(client_key(request))
            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded for {self.name}: max {self.limit} requests "
                           f"per minute. Retry in {retry_after}s.",
                    headers={"Retry-After": str(retry_after)},
                )
        return _dep


predict_limit = SlidingWindowLimiter("analysis", _env_int("RATE_LIMIT_PREDICT", 5)).dependency()
report_limit = SlidingWindowLimiter("report generation", _env_int("RATE_LIMIT_REPORT", 20)).dependency()
default_limit = SlidingWindowLimiter("this endpoint", _env_int("RATE_LIMIT_DEFAULT", 60)).dependency()

_inference_slots = threading.BoundedSemaphore(MAX_CONCURRENT_PREDICTIONS)


@contextmanager
def inference_slot():
    """Reject with 503 instead of queueing unboundedly when the server is saturated."""
    if not ENABLED:
        yield
        return
    if not _inference_slots.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="Server is busy with other analyses. Please retry shortly.",
            headers={"Retry-After": "10"},
        )
    try:
        yield
    finally:
        _inference_slots.release()
