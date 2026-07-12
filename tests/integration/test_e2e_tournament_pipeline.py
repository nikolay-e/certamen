import re
from typing import Any

import pytest

from certamen.application.execution.async_executor import AsyncExecutor
from certamen.application.slim_loader import materialize_workflow
from certamen.application.workflow.nodes import register_all
from certamen.infrastructure.config.slim import SlimConfig
from certamen.infrastructure.serialization import WorkflowLoader
from certamen.ports.llm import BaseModel, ModelResponse


class _StubModel(BaseModel):
    def __init__(self, key: str) -> None:
        super().__init__(
            key, f"stub/{key}", key, "stub", 512, 0.7, context_window=8192
        )

    async def generate(self, prompt: str) -> ModelResponse:
        if (
            "numbered questions" in prompt
            or "numbered follow-up questions" in prompt
        ):
            return ModelResponse.create_success(
                "1. What evidence supports claim X?\n"
                "2. What did you omit about Y?",
                prompt_tokens=50,
                completion_tokens=20,
                total_tokens=70,
            )
        if re.search(
            r"criteria|/\s*10|assign.*score|rate each|evaluate", prompt, re.I
        ):
            labels = re.findall(
                r"(response\s*\d+|model\s*\d+|llm\s*\d+)", prompt, re.I
            )
            body = (
                "\n".join(
                    f"{lbl}: {7 + i % 3}/10"
                    for i, lbl in enumerate(dict.fromkeys(labels))
                )
                or "Response 1: 8/10\nResponse 2: 7/10"
            )
            return ModelResponse.create_success(
                body,
                prompt_tokens=len(prompt) // 4,
                completion_tokens=40,
                total_tokens=len(prompt) // 4 + 40,
            )
        return ModelResponse.create_success(
            "STUB ANSWER: concrete claim with numbers 42 and mechanism M. "
            "1. specifics disclosed. 2. caveat admitted.",
            prompt_tokens=len(prompt) // 4,
            completion_tokens=60,
            total_tokens=len(prompt) // 4 + 60,
        )

    async def generate_with_retry(
        self,
        prompt: str,
        max_attempts: int = 3,
        initial_delay: float | None = None,
        max_delay: float | None = None,
    ) -> ModelResponse:
        return await self.generate(prompt)


def _diamond_config() -> SlimConfig:
    return SlimConfig(
        question="What are the three most important principles for X? Be concise.",
        models={
            k: {"provider": "ollama", "model_name": f"ollama/{k}"}
            for k in ("a", "b", "c", "d")
        },
        workflow="diamond-tournament",
    )


@pytest.fixture
def _stubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    register_all()
    from certamen.infrastructure.llm.litellm_adapter import LiteLLMModel

    async def _from_config(
        cls: Any,
        model_key: str,
        model_config: dict[str, Any],
        response_cache: Any = None,
    ) -> _StubModel:
        return _StubModel(model_config.get("display_name") or model_key)

    monkeypatch.setattr(LiteLLMModel, "from_config", classmethod(_from_config))


async def _run_diamond() -> dict[str, Any]:
    workflow = materialize_workflow(_diamond_config())
    ed = WorkflowLoader.to_executor_format(workflow)
    result = await AsyncExecutor().execute(ed["nodes"], ed["edges"])
    return result


@pytest.mark.asyncio
async def test_diamond_pipeline_produces_output_end_to_end(
    _stubbed: None,
) -> None:
    result = await _run_diamond()
    assert result.get("error") is None
    outs = result["outputs"]

    # Interrogation knowledge is actually extracted AND wired forward (was a
    # permanent no-op: edge asked for a non-existent `insights` output).
    interro = outs.get("interrogate", {})
    assert interro.get("insights"), "interrogation insights must flow forward"
    assert len(interro.get("extracted_knowledge", {})) > 0

    # Divergence enriched every model.
    assert len(outs.get("diverge_improve", {}).get("improved", {})) == 4

    # Finalization produces real output (was empty / "[No model available]").
    synthesis = outs.get("synthesize", {}).get("synthesis", "")
    assert synthesis.strip()
    assert "No model available" not in synthesis
    assert "Synthesis failed" not in synthesis

    assert outs.get("knowledge_map", {}).get("markdown", "").strip()


@pytest.mark.asyncio
async def test_interrogation_runs_multiple_rounds(_stubbed: None) -> None:
    # diamond sets interrogate.rounds=3; multi-round follow-up must fire, so
    # every ordered pair of 4 models (12) yields accumulated Q&A.
    result = await _run_diamond()
    interro = result["outputs"].get("interrogate", {})
    assert len(interro.get("extracted_knowledge", {})) == 12
