import time
from contextlib import contextmanager
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool

from certamen.gui.auth.config import (
    DB_HOST,
    DB_NAME,
    DB_PASSWORD,
    DB_POOL_MAX_SIZE,
    DB_POOL_MIN_SIZE,
    DB_PORT,
    DB_USER,
    SKIP_DB_INIT,
)
from certamen.logging import get_contextual_logger

logger = get_contextual_logger(__name__)

if SKIP_DB_INIT:
    logger.info(
        "Skipping database pool initialization because SKIP_DB_INIT is set"
    )
    db_pool = None
else:
    try:
        db_pool = SimpleConnectionPool(
            DB_POOL_MIN_SIZE,
            DB_POOL_MAX_SIZE,
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
        )
        logger.info(
            "Database pool initialized: %s:%s/%s", DB_HOST, DB_PORT, DB_NAME
        )
    except Exception as e:
        logger.warning("Failed to initialize database pool: %s", e)
        db_pool = None


def get_db() -> Any:
    if db_pool is None:
        raise RuntimeError("Database pool is not initialized")
    try:
        logger.debug(
            "Getting connection from pool: min=%d max=%d",
            DB_POOL_MIN_SIZE,
            DB_POOL_MAX_SIZE,
        )
        conn = db_pool.getconn(timeout=10)
        if conn is None:
            raise psycopg2.pool.PoolError("Failed to get connection from pool")
        logger.debug("Connection acquired from pool")
        return conn
    except Exception as e:
        logger.exception("Failed to get database connection: %s", e)
        raise


def put_db(conn: Any) -> None:
    if db_pool is None:
        raise RuntimeError("Database pool is not initialized")
    db_pool.putconn(conn)
    logger.debug("Connection returned to pool")


def _safe_rollback(conn: Any) -> None:
    """Safely rollback transaction with error suppression."""
    if conn:
        try:
            conn.rollback()
        except Exception as rollback_err:
            logger.debug("Failed to rollback: %s", rollback_err)


def _safe_connection_return(conn: Any) -> None:
    """Safely return connection to pool with fallback close."""
    if conn:
        try:
            put_db(conn)
        except Exception as e:
            logger.critical("Failed to return connection to pool: %s", e)
            try:
                conn.close()
            except Exception as close_err:
                logger.debug(
                    "Failed to close connection after pool return failure: %s",
                    close_err,
                )


@contextmanager
def db_transaction(write: bool = False) -> Any:
    """Context manager for database transactions with automatic cleanup.

    Args:
        write: If True, commits the transaction. If False, read-only (no commit).

    Yields:
        Database connection with cursor factory set to RealDictCursor

    Raises:
        RuntimeError: If database pool is not initialized
        psycopg2.pool.PoolError: If connection pool error occurs
        Exception: For any other database errors
    """
    conn = None
    start_time = time.time()
    try:
        conn = get_db()
        if not conn:
            raise Exception("Failed to get database connection")

        yield conn

        if write:
            conn.commit()
            duration_ms = (time.time() - start_time) * 1000
            logger.debug(
                "Transaction completed: type=write duration_ms=%.1f",
                duration_ms,
            )
        else:
            duration_ms = (time.time() - start_time) * 1000
            logger.debug(
                "Transaction completed: type=read duration_ms=%.1f",
                duration_ms,
            )

    except psycopg2.pool.PoolError as e:
        logger.exception("Connection pool error: %s", e)
        _safe_rollback(conn)
        raise
    except Exception as e:
        logger.exception("Database transaction error: %s", e)
        _safe_rollback(conn)
        raise
    finally:
        _safe_connection_return(conn)


def query_db(query: str, args: tuple[Any, ...] = (), one: bool = False) -> Any:
    """Execute a read-only SELECT query.

    Args:
        query: SQL SELECT query
        args: Query parameters
        one: If True, return single row or None. If False, return all rows.

    Returns:
        Single row dict (if one=True) or list of dicts (if one=False)

    Raises:
        ValueError: If query contains write operations or multiple statements
        RuntimeError: If database pool is not initialized
    """
    import re

    query_cleaned = re.sub(r"--.*?(\n|$)", " ", query)
    query_cleaned = re.sub(r"/\*.*?\*/", " ", query_cleaned, flags=re.DOTALL)
    query_cleaned = query_cleaned.strip().upper()

    write_keywords = [
        "INSERT",
        "UPDATE",
        "DELETE",
        "CREATE",
        "DROP",
        "ALTER",
        "TRUNCATE",
    ]

    for keyword in write_keywords:
        if keyword in query_cleaned:
            raise ValueError(
                f"query_db() detected a write operation containing "
                f"'{keyword}'. Use execute_write_transaction() instead."
            )

    if ";" in query and query.rstrip().rstrip(";").count(";") > 0:
        raise ValueError(
            "query_db() detected multiple SQL statements (semicolon found). "
            "Only single SELECT statements are allowed."
        )

    with db_transaction(write=False) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query_start = time.time()
            cur.execute(query, args)
            rv = cur.fetchall()
            query_duration_ms = (time.time() - query_start) * 1000
            row_count = len(rv) if rv else 0
            logger.debug(
                "Query executed: type=SELECT duration_ms=%.1f rows=%d",
                query_duration_ms,
                row_count,
            )
            return (rv[0] if rv else None) if one else rv


def execute_write_transaction(
    query: str,
    args: tuple[Any, ...] = (),
    fetch_results: bool = False,
    one: bool = False,
) -> Any:
    """Execute a write transaction (INSERT, UPDATE, DELETE, etc).

    Args:
        query: SQL write query
        args: Query parameters
        fetch_results: If True, fetch and return results after execution
        one: If True and fetch_results=True, return single row or None

    Returns:
        If fetch_results=True: Single row dict or list of dicts
        If fetch_results=False: Number of affected rows

    Raises:
        RuntimeError: If database pool is not initialized
    """
    import re

    query_cleaned = re.sub(r"--.*?(\n|$)", " ", query)
    query_cleaned = re.sub(r"/\*.*?\*/", " ", query_cleaned, flags=re.DOTALL)
    query_cleaned = query_cleaned.strip().upper()

    query_type = "WRITE"
    for keyword in [
        "INSERT",
        "UPDATE",
        "DELETE",
        "CREATE",
        "DROP",
        "ALTER",
        "TRUNCATE",
    ]:
        if keyword in query_cleaned:
            query_type = keyword
            break

    with db_transaction(write=True) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query_start = time.time()
            cur.execute(query, args)
            query_duration_ms = (time.time() - query_start) * 1000

            if fetch_results:
                rv = cur.fetchall()
                row_count = len(rv) if rv else 0
                logger.debug(
                    "Query executed: type=%s duration_ms=%.1f rows=%d",
                    query_type,
                    query_duration_ms,
                    row_count,
                )
                return (rv[0] if rv else None) if one else rv

            affected_rows = cur.rowcount
            logger.debug(
                "Query executed: type=%s duration_ms=%.1f affected_rows=%d",
                query_type,
                query_duration_ms,
                affected_rows,
            )
            return affected_rows


def cleanup_db_pool() -> None:
    global db_pool  # noqa: PLW0603
    if db_pool is not None:
        try:
            db_pool.closeall()
            logger.info("Database pool closed successfully")
        except Exception as e:
            logger.exception("Error closing database pool: %s", e)
        finally:
            db_pool = None
