import pytest

from certamen.application.execution.sync_executor import SyncExecutor
from certamen.application.workflow.nodes import register_all
from certamen.application.workflow.nodes.llm import TextNode
from certamen.domain.workflow.base import ExecutionContext


@pytest.fixture(autouse=True)
def _register_nodes() -> None:
    register_all()


class TestTextNodePagesTrap:
    async def test_texts_property_produces_output(self) -> None:
        node = TextNode("t", {"texts": ["hello world"], "separator": "\n"})
        out = await node.execute({}, ExecutionContext())
        assert out["output_text"] == "hello world"

    async def test_pages_without_input_yields_empty_output(self) -> None:
        # Documented trap: seed text placed in `pages:` instead of `texts:` is
        # ignored when no input_text is connected -> silent empty output that
        # still validates and "runs" successfully.
        node = TextNode("t", {"pages": [["seed"]], "separator": "\n"})
        out = await node.execute({}, ExecutionContext())
        assert out["output_text"] == ""


class TestFeedbackCycleTerminates:
    async def test_back_edge_is_bounded_feedback_not_an_infinite_hang(
        self,
    ) -> None:
        # A back-edge (b -> a) is treated as a bounded feedback loop, not a
        # hard error. The safety property that matters: it must TERMINATE
        # (capped by max_iterations) and return outputs, never hang/recurse
        # forever.
        nodes = [
            {"id": "a", "type": "simple/text", "properties": {"texts": ["x"]}},
            {"id": "b", "type": "simple/text", "properties": {"texts": ["y"]}},
        ]
        edges = [
            {
                "source": "a",
                "sourceHandle": "output_text",
                "target": "b",
                "targetHandle": "input_text",
            },
            {
                "source": "b",
                "sourceHandle": "output_text",
                "target": "a",
                "targetHandle": "input_text",
            },
        ]
        result = await SyncExecutor(verbose=False).execute(nodes, edges)
        assert "error" not in result
        assert result.get("outputs")


class TestEarlyTerminationOnGateDone:
    async def test_gate_done_terminates_before_max_iterations(self) -> None:
        # A flow/gate that resolves to a single-model champion emits done:True,
        # but the gate is NOT in the last execution layer. Termination must
        # scan ALL node_outputs (not just the final-layer tasks) to observe the
        # signal and stop early; otherwise the feedback loop runs to the
        # 20-iteration cap. Regression guard for the sync/async executor split.
        nodes = [
            {
                "id": "m1",
                "type": "simple/llm",
                "properties": {
                    "name": "M1",
                    "provider": "openai",
                    "model_name": "gpt-4o-mini",
                },
            },
            {"id": "models", "type": "tournament/models", "properties": {}},
            {"id": "gate", "type": "flow/gate", "properties": {}},
            {"id": "elim", "type": "tournament/eliminate", "properties": {}},
            {"id": "out", "type": "simple/text", "properties": {"texts": []}},
        ]
        edges = [
            {
                "source": "m1",
                "sourceHandle": "model_config",
                "target": "models",
                "targetHandle": "model_1",
            },
            {
                "source": "models",
                "sourceHandle": "models",
                "target": "gate",
                "targetHandle": "primary",
            },
            {
                "source": "gate",
                "sourceHandle": "models",
                "target": "elim",
                "targetHandle": "models",
            },
            {
                "source": "elim",
                "sourceHandle": "survivors",
                "target": "gate",
                "targetHandle": "feedback",
            },
            {
                "source": "gate",
                "sourceHandle": "champion",
                "target": "out",
                "targetHandle": "input_text",
            },
        ]
        result = await SyncExecutor(verbose=False).execute(nodes, edges)
        assert "error" not in result
        assert result["iterations"] == 1
        assert result["outputs"]["gate"]["done"] is True
