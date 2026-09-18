"""Lifecycle of the YDB driver and its query session pool.

One ``YDBConnection`` exists per process. It is opened by the composition root
(as a dependency-injection resource) or by a command-line entry point, handed
to whoever needs the pool or the driver, and closed by the same owner.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable

import ydb

from src.shared.infrastructure.database.ydb.settings import YDBSettings

logger = logging.getLogger(__name__)

Sleep = Callable[[float], Awaitable[None]]


@dataclass
class YDBConnection:
    """An open driver together with the session pool built on it."""

    driver: ydb.aio.Driver
    pool: ydb.aio.QuerySessionPool
    database: str

    async def close(self) -> None:
        """Stop the pool first, then the driver it runs on."""
        try:
            await self.pool.stop()
        finally:
            await self.driver.stop()


async def _wait_until_ready(driver: ydb.aio.Driver, settings: YDBSettings, sleep: Sleep) -> None:
    last_error: BaseException | None = None

    for attempt in range(1, settings.connect_attempts + 1):
        try:
            await driver.wait(timeout=settings.connect_timeout_seconds, fail_fast=False)
        except (ydb.Error, TimeoutError, ConnectionError) as error:
            last_error = error
            logger.warning(
                "YDB connection attempt %d/%d failed: %s: %s",
                attempt,
                settings.connect_attempts,
                type(error).__name__,
                error,
            )
            if attempt == 1:
                details = driver.discovery_debug_details()
                if details:
                    logger.warning("YDB discovery details: %s", details)
            if attempt < settings.connect_attempts:
                await sleep(settings.connect_retry_delay_seconds)
        else:
            logger.info(
                "Connected to YDB at %s%s (attempt %d/%d)",
                settings.endpoint,
                settings.database,
                attempt,
                settings.connect_attempts,
            )
            return

    raise ConnectionError(
        f"Failed to connect to YDB at {settings.endpoint} (database {settings.database}) after "
        f"{settings.connect_attempts} attempts of {settings.connect_timeout_seconds}s each. "
        f"Last error: {type(last_error).__name__}: {last_error}. "
        f"Discovery details: {driver.discovery_debug_details()}"
    ) from last_error


async def open_ydb_connection(
    settings: YDBSettings, sleep: Sleep = asyncio.sleep
) -> YDBConnection:
    """Create the driver, wait until it is ready and build the session pool.

    The local YDB container needs no credentials, so the driver is always
    anonymous. A driver that never becomes ready is stopped before the error
    propagates, so a failed start leaves nothing behind.
    """
    driver = ydb.aio.Driver(
        ydb.DriverConfig(
            endpoint=settings.endpoint,
            database=settings.database,
            credentials=ydb.AnonymousCredentials(),
        )
    )
    try:
        await _wait_until_ready(driver, settings, sleep)
    except BaseException:
        await driver.stop()
        raise

    pool = ydb.aio.QuerySessionPool(driver, size=settings.pool_size)
    return YDBConnection(driver=driver, pool=pool, database=settings.database)
