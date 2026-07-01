from pathlib import Path

import pytest
import yaml

from certamen.application.execution.sync_executor import SyncExecutor
from certamen.application.workflow.nodes import register_all
from certamen.application.workflow.nodes.llm import TextNode
from certamen.domain.errors import FatalError
from certamen.domain.workflow.base import ExecutionContext
from certamen.interfaces.cli.main import _validate_workflow_file


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

    async def test_model_config_dict_renders_display_name_not_raw_config(
        self,
    ) -> None:
        # gate.champion / eliminate.survivors emit raw model config dicts
        # (system_prompt, context_window, base_url, ...). Wiring one straight
        # into a simple/text output node (as the shipped
        # tournament-elimination.yml does for champion_output) must show the
        # display name, not dump every internal field.
        node = TextNode("t", {})
        champion = {
            "name": "Creative Writer",
            "provider": "ollama",
            "model_name": "ollama/llama3.2:3b",
            "system_prompt": "You are a creative science communicator.",
            "context_window": 131072,
        }
        out = await node.execute({"input_text": champion}, ExecutionContext())
        assert out["output_text"] == "Creative Writer"

    async def test_generic_dict_without_model_shape_still_dumps_fields(
        self,
    ) -> None:
        # Non-model dicts (debugging output, arbitrary node data) keep the
        # existing key/value dump behavior.
        node = TextNode("t", {})
        out = await node.execute(
            {"input_text": {"foo": "bar", "baz": 1}}, ExecutionContext()
        )
        assert out["output_text"] == "[foo]\nbar\n---\n[baz]\n1"


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


class TestValidateCatchesUnknownNodeType:
    def _write_workflow(self, tmp_path: Path, node_type: str) -> str:
        workflow = {
            "name": "Test",
            "version": "1.0",
            "nodes": [
                {"id": "a", "type": node_type, "properties": {}},
            ],
            "edges": [],
            "outputs": [],
        }
        path = tmp_path / "wf.yml"
        path.write_text(yaml.dump(workflow))
        return str(path)

    def test_rejects_unregistered_node_type(self, tmp_path: Path) -> None:
        # `workflow validate` previously only ran the YAML-schema check
        # (WorkflowLoader.load_from_file) and never checked node types
        # against the registry, so a typo'd/unknown type reported
        # "Workflow is valid" and only failed later at `workflow execute`.
        path = self._write_workflow(tmp_path, "nonexistent/nodetype")
        with pytest.raises(FatalError, match="unknown node type"):
            _validate_workflow_file(path)

    def test_accepts_registered_node_type(self, tmp_path: Path) -> None:
        path = self._write_workflow(tmp_path, "simple/text")
        _validate_workflow_file(path)  # must not raise
