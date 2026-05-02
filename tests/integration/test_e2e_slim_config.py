from pathlib import Path

import pytest
import yaml

from certamen.application.slim_loader import (
    load_and_materialize,
    materialize_workflow,
)
from certamen.domain.errors import ConfigurationError
from certamen.infrastructure.config.slim import (
    SlimConfig,
    is_slim_config,
    load_slim_config,
)
from certamen.workflows import (
    BUILTIN_WORKFLOWS,
    list_builtin_workflows,
    resolve_workflow_path,
)


def _write(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _slim_payload() -> dict:
    return {
        "question": "Test question",
        "models": {
            "a": {"provider": "ollama", "model_name": "ollama/a"},
            "b": {"provider": "ollama", "model_name": "ollama/b"},
            "c": {"provider": "ollama", "model_name": "ollama/c"},
            "d": {"provider": "ollama", "model_name": "ollama/d"},
        },
        "workflow": "diamond-tournament",
    }


class TestBuiltinWorkflowRegistry:
    def test_lists_diamond(self) -> None:
        assert "diamond-tournament" in list_builtin_workflows()
        assert "diamond-tournament" in BUILTIN_WORKFLOWS

    def test_resolves_diamond_to_packaged_file(self) -> None:
        path = resolve_workflow_path("diamond-tournament")
        assert path.is_file()
        assert path.name == "diamond-tournament.yml"
        assert "src/certamen/workflows" in str(path)

    def test_resolves_explicit_path(self, tmp_path: Path) -> None:
        wf = tmp_path / "custom.yml"
        wf.write_text("version: '1.0'\nname: custom\nnodes: []", "utf-8")
        assert resolve_workflow_path(str(wf)) == wf.resolve()

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            resolve_workflow_path("does-not-exist")


class TestSlimConfigSchema:
    def test_loads_minimal(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "c.yml", _slim_payload())
        slim = load_slim_config(path)
        assert isinstance(slim, SlimConfig)
        assert slim.question == "Test question"
        assert slim.workflow == "diamond-tournament"
        assert len(slim.models) == 4

    def test_rejects_legacy_keys(self, tmp_path: Path) -> None:
        payload = _slim_payload()
        payload["tournament"] = {"judge_model": "claude"}
        path = _write(tmp_path / "c.yml", payload)
        with pytest.raises(ConfigurationError, match="legacy keys"):
            load_slim_config(path)

    def test_rejects_unknown_top_level_keys(self, tmp_path: Path) -> None:
        payload = _slim_payload()
        payload["mystery"] = {}
        path = _write(tmp_path / "c.yml", payload)
        with pytest.raises(ConfigurationError):
            load_slim_config(path)

    def test_is_slim_detects_workflow_key(self) -> None:
        assert is_slim_config(_slim_payload()) is True

    def test_is_slim_rejects_legacy(self) -> None:
        payload = _slim_payload()
        payload["tournament"] = {}
        assert is_slim_config(payload) is False


class TestMaterializeWorkflow:
    def _build_slim(self, **overrides) -> SlimConfig:
        payload = _slim_payload()
        payload.update(overrides)
        return SlimConfig(**payload)

    def test_injects_question(self) -> None:
        slim = self._build_slim()
        wf = materialize_workflow(slim)
        question_node = next(n for n in wf["nodes"] if n["id"] == "question")
        assert question_node["properties"]["texts"] == ["Test question"]

    def test_injects_models_positionally(self) -> None:
        slim = self._build_slim()
        wf = materialize_workflow(slim)
        llms = [n for n in wf["nodes"] if n["type"] == "simple/llm"]
        assert len(llms) == 4
        names = [n["properties"]["model_name"] for n in llms]
        assert names == [
            "ollama/a",
            "ollama/b",
            "ollama/c",
            "ollama/d",
        ]

    def test_applies_dot_path_overrides(self) -> None:
        slim = self._build_slim(
            overrides={
                "gate.max_rounds": 2,
                "peer_review.criteria": "be terse",
            },
        )
        wf = materialize_workflow(slim)
        gate = next(n for n in wf["nodes"] if n["id"] == "gate")
        peer = next(n for n in wf["nodes"] if n["id"] == "peer_review")
        assert gate["properties"]["max_rounds"] == 2
        assert peer["properties"]["criteria"] == "be terse"

    def test_rejects_override_for_unknown_node(self) -> None:
        slim = self._build_slim(overrides={"phantom.knob": 1})
        with pytest.raises(ConfigurationError, match="phantom"):
            materialize_workflow(slim)

    def test_rejects_model_count_mismatch(self) -> None:
        slim = self._build_slim(
            models={
                "a": {"provider": "ollama", "model_name": "ollama/a"},
            },
        )
        with pytest.raises(ConfigurationError, match="expects"):
            materialize_workflow(slim)


class TestLoadAndMaterialize:
    def test_round_trip_against_packaged_diamond(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "c.yml", _slim_payload())
        slim, wf = load_and_materialize(path)
        assert slim.workflow == "diamond-tournament"
        assert wf["name"].startswith("Diamond")
        # Workflow stays valid for the executor.
        from certamen.infrastructure.serialization import WorkflowLoader

        WorkflowLoader._validate_workflow(wf, "<materialized>")


class TestSlimOutputsDirPriority:
    def test_cli_outputs_dir_overrides_config(self, tmp_path: Path) -> None:
        from certamen.interfaces.cli.main import App

        payload = _slim_payload()
        payload["outputs_dir"] = "./from-config"
        cfg = _write(tmp_path / "c.yml", payload)

        app = App(
            args={
                "config": str(cfg),
                "outputs_dir": "./from-cli",
                "no_secrets": True,
                "verbose": False,
                "debug": False,
                "command": "tournament",
                "question": None,
            }
        )
        slim = load_slim_config(cfg)
        resolved = app.outputs_dir or slim.outputs_dir
        assert resolved == "./from-cli"

    def test_falls_back_to_config_outputs_dir(self, tmp_path: Path) -> None:
        from certamen.interfaces.cli.main import App

        payload = _slim_payload()
        payload["outputs_dir"] = "./from-config"
        cfg = _write(tmp_path / "c.yml", payload)

        app = App(
            args={
                "config": str(cfg),
                "outputs_dir": None,
                "no_secrets": True,
                "verbose": False,
                "debug": False,
                "command": "tournament",
                "question": None,
            }
        )
        slim = load_slim_config(cfg)
        resolved = app.outputs_dir or slim.outputs_dir
        assert resolved == "./from-config"
