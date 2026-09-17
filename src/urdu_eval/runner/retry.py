"""Retry with exponential backoff and rate limit handling."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")
logger = logging.getLogger("urdu_eval.retry")


def execute_with_retry(
    fn: Callable[..., T],
    *args: Any,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    **kwargs: Any,
) -> T:
    """Execute a function with exponential backoff on transient errors and HTTP 429/500."""
    delay = initial_delay
    last_exception: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exception = exc
            err_msg = str(exc).lower()

            # Check if likely transient or rate limited
            is_transient = any(
                code in err_msg
                for code in [
                    "429",
                    "rate limit",
                    "500",
                    "502",
                    "503",
                    "504",
                    "timeout",
                    "connection",
                ]
            )

            if attempt == max_retries or not is_transient:
                raise exc

            sleep_time = delay * (1.0 + random.uniform(0, 0.2)) if jitter else delay
            logger.warning(
                "Transient error on attempt %d/%d: %s. Retrying in %.2fs...",
                attempt,
                max_retries,
                exc,
                sleep_time,
            )
            time.sleep(sleep_time)
            delay *= backoff_factor

    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected retry termination")
