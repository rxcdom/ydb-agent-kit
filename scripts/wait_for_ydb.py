#!/usr/bin/env python3
"""Block until the YDB instance answers a query.

A healthy container process is not enough: the local YDB image accepts
connections before its storage pools are initialised, and DDL issued in that
gap fails. This probe waits for the driver and then runs ``SELECT 1``.
"""
import asyncio
import os
import sys

import ydb

ENDPOINT = os.getenv("YDB_ENDPOINT", "grpc://ydb:2136")
DATABASE = os.getenv("YDB_DATABASE", "/local")
MAX_ATTEMPTS = int(os.getenv("YDB_WAIT_MAX_ATTEMPTS", "60"))
RETRY_INTERVAL_SECONDS = float(os.getenv("YDB_WAIT_RETRY_INTERVAL_SECONDS", "2"))
DISABLE_DISCOVERY = os.getenv("YDB_DISABLE_DISCOVERY", "").strip().lower() in {"1", "true", "yes", "on"}
DRIVER_WAIT_TIMEOUT_SECONDS = 5


async def _probe(driver: ydb.aio.Driver) -> None:
    await driver.wait(timeout=DRIVER_WAIT_TIMEOUT_SECONDS, fail_fast=False)
    pool = ydb.aio.QuerySessionPool(driver, size=1)
    try:
        await pool.execute_with_retries("SELECT 1 AS one;")
    finally:
        await pool.stop()


async def wait_for_ydb() -> int:
    print(f"Waiting for YDB at {ENDPOINT} (database {DATABASE})")
    driver = ydb.aio.Driver(
        ydb.DriverConfig(
            endpoint=ENDPOINT,
            database=DATABASE,
            credentials=ydb.AnonymousCredentials(),
            disable_discovery=DISABLE_DISCOVERY,
        )
    )
    try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                await _probe(driver)
            except (ydb.Error, TimeoutError, ConnectionError) as error:
                print(f"  attempt {attempt}/{MAX_ATTEMPTS}: not ready ({type(error).__name__}: {error})")
                await asyncio.sleep(RETRY_INTERVAL_SECONDS)
            else:
                print(f"YDB is ready (attempt {attempt}/{MAX_ATTEMPTS})")
                return 0
        print(f"YDB did not become ready after {MAX_ATTEMPTS} attempts")
        return 1
    finally:
        await driver.stop()


if __name__ == "__main__":
    sys.exit(asyncio.run(wait_for_ydb()))
