from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MigrationRecord:
    """One row of the ``schema_migrations`` bookkeeping table."""

    version: str
    module_name: str
    description: str
    applied_at: dt.datetime
    checksum: str
    applied_by: str
    status: str
    error_message: Optional[str]
