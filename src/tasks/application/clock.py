from datetime import datetime, timezone
from typing import Callable

# Use cases take the current instant from an injected clock so tests control time.
Clock = Callable[[], datetime]

UTC = timezone.utc


def utc_now() -> datetime:
    return datetime.now(UTC)
