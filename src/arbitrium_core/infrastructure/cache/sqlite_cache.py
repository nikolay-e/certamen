"""Response caching for LLM API calls."""

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

from arbitrium_core.shared.logging import get_contextual_logger

logger = get_contextual_logger(__name__)


class ResponseCache:
    """Caches LLM responses to avoid redundant API calls and reduce costs.

    Uses SQLite to store responses keyed by model name, prompt, temperature,
    and max_tokens. This dramatically reduces costs during development,
    testing, and when re-running tournaments with similar questions.

    Args:
        db_path: Path to SQLite database file (default: arbitrium_cache.db)
        enabled: Whether caching is enabled (default: True)

    Example:
        ```python
        cache = ResponseCache("cache.db")

        # Check cache before API call
        cached = cache.get("gpt-4o", prompt, 0.7, 2048)
        if cached:
            response, cost = cached
            return ModelResponse(response, cost=cost)

        # ... make API call ...

        # Save to cache
        cache.set("gpt-4o", prompt, 0.7, 2048, response.content, response.cost)
        ```
    """

    def __init__(
        self, db_path: str | Path = "arbitrium_cache.db", enabled: bool = True
    ) -> None:
        self.enabled = enabled
        self._lock = threading.Lock()
        self._hit_count = 0
        self._miss_count = 0

        if not self.enabled:
            self.conn = None
            logger.info("Response cache disabled")
            return

        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False, timeout=5.0
        )
        self._setup_pragmas()
        self._create_table()
        logger.info(
            "Response cache initialized: database=%s, enabled=%s",
            self.db_path.absolute(),
            self.enabled,
        )

    def _is_active(self) -> bool:
        return self.enabled and self.conn is not None

    def _get_conn(self) -> sqlite3.Connection:
        if self.conn is None:
            raise RuntimeError("Cache connection is not initialized")
        return self.conn

    def _setup_pragmas(self) -> None:
        if not self._is_active():
            return
        conn = self._get_conn()

        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
        except sqlite3.Error as e:
            logger.warning("Failed to set SQLite pragmas: %s", e)

    def _create_table(self) -> None:
        if not self._is_active():
            return
        conn = self._get_conn()

        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS responses (
                    cache_key TEXT PRIMARY KEY,
                    response TEXT NOT NULL,
                    cost REAL NOT NULL,
                    timestamp INTEGER NOT NULL
                )
                """)
            conn.commit()
            logger.debug("Cache table created or verified")
        except sqlite3.Error as e:
            logger.error("Failed to create cache table: %s", e)
            raise

    def _hash_key(
        self, model_name: str, prompt: str, temperature: float, max_tokens: int
    ) -> str:
        key_data = {
            "model": model_name,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get(
        self, model_name: str, prompt: str, temperature: float, max_tokens: int
    ) -> tuple[str, float] | None:
        if not self._is_active():
            return None
        conn = self._get_conn()

        key = self._hash_key(model_name, prompt, temperature, max_tokens)

        try:
            with self._lock:
                cursor = conn.execute(
                    "SELECT response, cost FROM responses WHERE cache_key = ?",
                    (key,),
                )
                result = cursor.fetchone()

                if result:
                    self._hit_count += 1
                else:
                    self._miss_count += 1

            if result:
                logger.debug(
                    "Cache HIT: model=%s, temp=%.2f, tokens=%d, cost=$%.6f",
                    model_name,
                    temperature,
                    max_tokens,
                    result[1],
                )
                return (result[0], result[1])
            else:
                logger.debug(
                    "Cache MISS: model=%s, temp=%.2f, tokens=%d",
                    model_name,
                    temperature,
                    max_tokens,
                )
                return None
        except sqlite3.Error as e:
            logger.error(
                "Cache lookup error: model=%s, error=%s", model_name, e
            )
            return None

    def set(
        self,
        model_name: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        response: str,
        cost: float,
    ) -> None:
        if not self._is_active():
            return
        conn = self._get_conn()

        key = self._hash_key(model_name, prompt, temperature, max_tokens)
        timestamp = int(time.time())

        try:
            with self._lock:
                conn.execute(
                    "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?)",
                    (key, response, cost, timestamp),
                )
                conn.commit()

            logger.debug(
                "Cache SET: model=%s, temp=%.2f, tokens=%d, cost=$%.6f",
                model_name,
                temperature,
                max_tokens,
                cost,
            )
        except sqlite3.Error as e:
            logger.error(
                "Cache write error: model=%s, error=%s", model_name, e
            )

    def clear(self) -> None:
        if not self._is_active():
            return
        conn = self._get_conn()

        try:
            with self._lock:
                cursor = conn.execute("SELECT COUNT(*) FROM responses")
                count_before = cursor.fetchone()[0]
                conn.execute("DELETE FROM responses")
                conn.commit()

            logger.info("Cache cleared: %d entries removed", count_before)
            self._hit_count = 0
            self._miss_count = 0
        except sqlite3.Error as e:
            logger.error("Cache clear error: %s", e)
            raise

    def stats(self) -> dict[str, int | float]:
        if not self._is_active():
            return {
                "total_entries": 0,
                "hit_count": 0,
                "miss_count": 0,
                "hit_ratio": 0.0,
            }
        conn = self._get_conn()

        try:
            with self._lock:
                cursor = conn.execute("SELECT COUNT(*) FROM responses")
                count = cursor.fetchone()[0]
                hit_count = self._hit_count
                miss_count = self._miss_count

            total_requests = hit_count + miss_count
            hit_ratio = (
                hit_count / total_requests if total_requests > 0 else 0.0
            )

            stats_dict = {
                "total_entries": count,
                "hit_count": hit_count,
                "miss_count": miss_count,
                "hit_ratio": hit_ratio,
            }

            logger.info(
                "Cache statistics: entries=%d, hits=%d, misses=%d, hit_ratio=%.2f%%",
                count,
                hit_count,
                miss_count,
                hit_ratio * 100,
            )

            return stats_dict
        except sqlite3.Error as e:
            logger.error("Cache stats error: %s", e)
            with self._lock:
                hit_count = self._hit_count
                miss_count = self._miss_count
            return {
                "total_entries": 0,
                "hit_count": hit_count,
                "miss_count": miss_count,
                "hit_ratio": 0.0,
            }

    def close(self) -> None:
        if self.conn:
            with self._lock:
                try:
                    self.conn.close()
                    self.conn = None
                except sqlite3.Error:
                    pass

    def __enter__(self) -> "ResponseCache":
        return self

    def __exit__(
        self,
        _exc_type: type | None,
        _exc_val: BaseException | None,
        _exc_tb: object,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: S110
            # Suppress errors during interpreter shutdown
            pass
