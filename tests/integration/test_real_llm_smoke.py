"""Simple smoke tests with real LLM (ollama) - no mocks, no overengineering."""

import pytest

from arbitrium_core.infrastructure.llm.litellm_adapter import LiteLLMModel


@pytest.mark.skipif(
    "not config.getoption('--run-real-llm')",
    reason="Run with --run-real-llm to test actual LLM calls",
)
@pytest.mark.asyncio
async def test_ollama_basic_generation():
    """Test basic generation with ollama (local, free, fast)."""
    model = LiteLLMModel(
        model_key="test_ollama",
        model_name="ollama/llama3.2:3b",
        display_name="Ollama Llama3.2 3B",
        provider="ollama",
        temperature=0.7,
        max_tokens=100,
    )

    response = await model.generate("What is 2+2? Answer in one word.")

    assert not response.is_error(), f"Expected success, got: {response.error}"
    assert response.content, "Response content should not be empty"
    assert len(response.content) > 0
    assert response.cost >= 0


@pytest.mark.skipif(
    "not config.getoption('--run-real-llm')",
    reason="Run with --run-real-llm to test actual LLM calls",
)
@pytest.mark.asyncio
async def test_ollama_with_cache():
    """Test that response caching works with real LLM."""
    from arbitrium_core.infrastructure.cache.sqlite_cache import ResponseCache

    cache = ResponseCache(enabled=True)

    model = LiteLLMModel(
        model_key="test_cached",
        model_name="ollama/llama3.2:1b",
        display_name="Cached Ollama",
        provider="ollama",
        temperature=0.7,
        max_tokens=50,
        response_cache=cache,
    )

    prompt = "Say 'hello' in one word."

    response1 = await model.generate(prompt)
    assert not response1.is_error()

    response2 = await model.generate(prompt)
    assert not response2.is_error()
    assert response1.content == response2.content
