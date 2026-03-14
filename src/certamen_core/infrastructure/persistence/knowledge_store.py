import asyncio
import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from certamen_core.domain.knowledge_map.builder import KnowledgeMap

_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_claims (
    id TEXT PRIMARY KEY,
    claim TEXT NOT NULL,
    confidence TEXT,
    source_model TEXT,
    source_question TEXT,
    tournament_id TEXT,
    created_at INTEGER
);

CREATE TABLE IF NOT EXISTS disagreements (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    positions TEXT,
    status TEXT,
    tournament_id TEXT,
    created_at INTEGER
);

CREATE TABLE IF NOT EXISTS exploration_branches (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    parent_question TEXT,
    explored INTEGER DEFAULT 0,
    priority REAL,
    created_at INTEGER
);
"""


class PersistentKnowledgeStore:
    def __init__(self, db_path: str = "certamen_knowledge.db") -> None:
        self._db_path = Path(db_path)
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        async with self._lock:
            if not self._initialized:
                await asyncio.to_thread(self._create_schema)
                self._initialized = True

    def _create_schema(self) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    async def store_knowledge_map(
        self, km: "KnowledgeMap", tournament_id: str
    ) -> None:
        await self.initialize()
        async with self._lock:
            await asyncio.to_thread(self._store_sync, km, tournament_id)

    def _store_sync(self, km: "KnowledgeMap", tournament_id: str) -> None:
        now = int(time.time())
        conn = sqlite3.connect(self._db_path)
        try:
            for item in km.consensus:
                conn.execute(
                    "INSERT OR IGNORE INTO knowledge_claims VALUES (?,?,?,?,?,?,?)",
                    (
                        str(uuid.uuid4()),
                        item.claim,
                        item.confidence,
                        km.champion_model,
                        km.question,
                        tournament_id,
                        now,
                    ),
                )
            for d in km.disagreements:
                conn.execute(
                    "INSERT OR IGNORE INTO disagreements VALUES (?,?,?,?,?,?)",
                    (
                        str(uuid.uuid4()),
                        d.topic,
                        json.dumps(d.positions),
                        d.resolution_status,
                        tournament_id,
                        now,
                    ),
                )
            for branch in km.exploration_branches:
                conn.execute(
                    "INSERT OR IGNORE INTO exploration_branches VALUES (?,?,?,?,?,?)",
                    (
                        str(uuid.uuid4()),
                        branch,
                        km.question,
                        0,
                        1.0,
                        now,
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    async def get_relevant_prior_knowledge(
        self, question: str, limit: int = 20
    ) -> list[str]:
        await self.initialize()
        async with self._lock:
            rows = await asyncio.to_thread(self._fetch_all_claims)

        if not rows:
            return []

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            claims = [row[0] for row in rows]
            doc_texts = [
                f"{row[0]} {row[1]}" if row[1] else row[0] for row in rows
            ]
            corpus = [question, *doc_texts]
            vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
            tfidf_matrix = vectorizer.fit_transform(corpus)
            similarities = cosine_similarity(
                tfidf_matrix[0:1], tfidf_matrix[1:]
            ).flatten()
            scored = [
                (float(score), claims[i])
                for i, score in enumerate(similarities)
                if score > 0.05
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            return [claim for _, claim in scored[:limit]]
        except ImportError:
            words = set(question.lower().split())
            scored_fallback = []
            for claim, source_question in rows:
                if source_question:
                    overlap = len(words & set(source_question.lower().split()))
                    if overlap > 0:
                        scored_fallback.append((overlap, claim))
            scored_fallback.sort(key=lambda x: x[0], reverse=True)
            return [claim for _, claim in scored_fallback[:limit]]

    def _fetch_all_claims(self) -> list[tuple[str, str]]:
        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.execute(
                "SELECT claim, source_question FROM knowledge_claims"
            )
            return cursor.fetchall()
        finally:
            conn.close()

    async def get_unexplored_branches(self, limit: int = 10) -> list[str]:
        await self.initialize()
        async with self._lock:
            return await asyncio.to_thread(self._fetch_unexplored, limit)

    def _fetch_unexplored(self, limit: int) -> list[str]:
        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.execute(
                "SELECT question FROM exploration_branches "
                "WHERE explored=0 ORDER BY priority DESC LIMIT ?",
                (limit,),
            )
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

    def _fetch_all_branches(self) -> list[str]:
        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.execute("SELECT question FROM exploration_branches")
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

    async def get_all_branch_questions(self) -> set[str]:
        await self.initialize()
        async with self._lock:
            rows = await asyncio.to_thread(self._fetch_all_branches)
        return set(rows)

    async def mark_branch_explored(self, question: str) -> None:
        await self.initialize()
        async with self._lock:
            await asyncio.to_thread(self._mark_explored_sync, question)

    def _mark_explored_sync(self, question: str) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                "UPDATE exploration_branches SET explored=1 WHERE question=?",
                (question,),
            )
            conn.commit()
        finally:
            conn.close()
