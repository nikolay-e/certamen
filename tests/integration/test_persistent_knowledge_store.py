import pytest

from certamen.domain.knowledge_map.builder import (
    ConsensusItem,
    KnowledgeMap,
)
from certamen.infrastructure.persistence.knowledge_store import (
    PersistentKnowledgeStore,
)


@pytest.mark.asyncio
async def test_store_and_retrieve(tmp_path):
    store = PersistentKnowledgeStore(str(tmp_path / "test.db"))
    claims = [
        ConsensusItem(
            claim="Microservices increase deployment flexibility",
            confidence="HIGH",
        ),
        ConsensusItem(
            claim="Team size is a critical factor for migration",
            confidence="MEDIUM",
        ),
        ConsensusItem(
            claim="Cost and complexity grow with service count",
            confidence="LOW",
        ),
    ]
    km = KnowledgeMap(
        question="Should we migrate to microservices?",
        consensus=claims,
        champion_model="gpt4",
    )
    await store.store_knowledge_map(km, "t1")
    results = await store.get_relevant_prior_knowledge(
        "What are the benefits of microservices architecture?"
    )
    assert len(results) > 0


@pytest.mark.asyncio
async def test_get_unexplored_branches(tmp_path):
    store = PersistentKnowledgeStore(str(tmp_path / "test.db"))
    km = KnowledgeMap(
        question="Q?",
        exploration_branches=["Follow-up A?", "Follow-up B?"],
        champion_model="m1",
    )
    await store.store_knowledge_map(km, "t1")
    branches = await store.get_unexplored_branches()
    assert len(branches) == 2


@pytest.mark.asyncio
async def test_mark_branch_explored(tmp_path):
    store = PersistentKnowledgeStore(str(tmp_path / "test.db"))
    km = KnowledgeMap(
        question="Q?",
        exploration_branches=["Follow-up A?"],
        champion_model="m1",
    )
    await store.store_knowledge_map(km, "t1")
    await store.mark_branch_explored("Follow-up A?")
    branches = await store.get_unexplored_branches()
    assert "Follow-up A?" not in branches
