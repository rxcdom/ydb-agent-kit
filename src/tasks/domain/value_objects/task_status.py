from enum import Enum


class TaskStatus(str, Enum):
    """Lifecycle state of a task. Only ``DONE`` carries a completion instant."""

    OPEN = "open"
    DONE = "done"
    CANCELLED = "cancelled"
