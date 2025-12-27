"""End-to-end integration tests for CLI application."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from arbitrium_core.domain.errors import FatalError
from arbitrium_core.interfaces.cli.main import App
from tests.integration.conftest import MockModel


class TestAppInitialization:
    """Test App initialization with real arguments."""

    def test_app_initializes_with_default_args(self) -> None:
        """Test App initializes when no arguments provided."""
        args = {
            "config": "config.example.yml",
            "no_secrets": False,
            "interactive": False,
            "models": None,
            "debug": False,
            "verbose": False,
        }

        app = App(args)

        assert app.args is not None
        assert app.logger is not None
        assert app.arbitrium is None

    def test_app_preserves_custom_outputs_dir(self) -> None:
        """Test App preserves custom outputs directory from arguments."""
        custom_dir = "/custom/output/path"
        args = {
            "config": "config.yml",
            "outputs_dir": custom_dir,
            "no_secrets": False,
            "interactive": False,
            "models": None,
            "debug": False,
            "verbose": False,
        }

        app = App(args)

        assert app.outputs_dir == custom_dir

    def test_app_handles_none_outputs_dir(self) -> None:
        """Test App handles when outputs_dir is None."""
        args = {
            "config": "config.yml",
            "outputs_dir": None,
            "no_secrets": False,
            "interactive": False,
            "models": None,
            "debug": False,
            "verbose": False,
        }

        app = App(args)

        assert app.outputs_dir is None


class TestConfigLoading:
    """Test loading YAML config files."""

    def test_load_config_success(
        self, basic_config: dict, tmp_output_dir: Path
    ) -> None:
        """Test loading valid YAML config file."""
        config_file = tmp_output_dir / "config.yml"
        with open(config_file, "w") as f:
            yaml.dump(basic_config, f)

        args = {
            "config": str(config_file),
            "outputs_dir": None,
            "no_secrets": False,
            "interactive": False,
            "models": None,
            "debug": False,
            "verbose": False,
        }
        app = App(args)

        config_obj = app._load_config(str(config_file))

        assert config_obj is not None
        assert config_obj.config_data is not None
        assert "models" in config_obj.config_data

    def test_load_config_file_not_found(self) -> None:
        """Test loading non-existent config file."""
        args = {
            "config": "nonexistent.yml",
            "outputs_dir": None,
            "no_secrets": False,
            "interactive": False,
            "models": None,
            "debug": False,
            "verbose": False,
        }
        app = App(args)

        with pytest.raises(FatalError, match="Failed to load configuration"):
            app._load_config("nonexistent.yml")

    def test_load_config_invalid_yaml(self, tmp_output_dir: Path) -> None:
        """Test loading invalid YAML file."""
        config_file = tmp_output_dir / "invalid.yml"
        with open(config_file, "w") as f:
            f.write("{ invalid yaml [[[")

        args = {
            "config": str(config_file),
            "outputs_dir": None,
            "no_secrets": False,
            "interactive": False,
            "models": None,
            "debug": False,
            "verbose": False,
        }
        app = App(args)

        with pytest.raises(FatalError, match="Failed to load configuration"):
            app._load_config(str(config_file))


class TestArbitriumCreation:
    """Test creating Arbitrium instance from config."""

    @pytest.mark.asyncio
    async def test_create_arbitrium_from_config(
        self, basic_config: dict, tmp_output_dir: Path
    ) -> None:
        """Test creating Arbitrium from real config."""
        config_file = tmp_output_dir / "config.yml"
        with open(config_file, "w") as f:
            yaml.dump(basic_config, f)

        args = {
            "config": str(config_file),
            "outputs_dir": None,
            "no_secrets": True,
            "interactive": False,
            "models": None,
            "debug": False,
            "verbose": False,
        }
        app = App(args)

        arbitrium = await app._create_arbitrium_from_config(
            str(config_file), skip_secrets=True
        )

        assert arbitrium is not None
        assert len(arbitrium.all_models) > 0

    @pytest.mark.asyncio
    async def test_arbitrium_outputs_dir_override(
        self, basic_config: dict, tmp_output_dir: Path
    ) -> None:
        """Test that CLI outputs_dir overrides config."""
        config_file = tmp_output_dir / "config.yml"
        with open(config_file, "w") as f:
            yaml.dump(basic_config, f)

        cli_output_dir = str(tmp_output_dir / "cli_output")
        args = {
            "config": str(config_file),
            "outputs_dir": cli_output_dir,
            "no_secrets": True,
            "interactive": False,
            "models": None,
            "debug": False,
            "verbose": False,
        }
        app = App(args)

        config_obj = app._load_config(str(config_file))
        arbitrium = await app._try_create_arbitrium_from_config_obj(
            config_obj, skip_secrets=True
        )

        assert arbitrium.config_data["outputs_dir"] == cli_output_dir


class TestModelFiltering:
    """Test filtering models by name."""

    def test_filter_requested_models_success(
        self, basic_config: dict, tmp_output_dir: Path
    ) -> None:
        """Test filtering models when requested models are available."""
        args = {
            "config": "config.yml",
            "outputs_dir": None,
            "models": "model_a,model_b",
            "no_secrets": False,
            "interactive": False,
            "debug": False,
            "verbose": False,
        }
        app = App(args)

        app.arbitrium = AsyncMock()
        app.arbitrium.healthy_models = {
            "model_a": MockModel(model_name="a", display_name="A"),
            "model_b": MockModel(model_name="b", display_name="B"),
            "model_c": MockModel(model_name="c", display_name="C"),
        }

        app._filter_requested_models()

        assert len(app.arbitrium._healthy_models) == 2
        assert "model_a" in app.arbitrium._healthy_models
        assert "model_b" in app.arbitrium._healthy_models

    def test_filter_requested_models_none_available(
        self, basic_config: dict
    ) -> None:
        """Test filtering when none of requested models are available."""
        args = {
            "config": "config.yml",
            "outputs_dir": None,
            "models": "nonexistent1,nonexistent2",
            "no_secrets": False,
            "interactive": False,
            "debug": False,
            "verbose": False,
        }
        app = App(args)

        app.arbitrium = AsyncMock()
        app.arbitrium.healthy_models = {
            "model_a": MockModel(model_name="a", display_name="A"),
            "model_b": MockModel(model_name="b", display_name="B"),
        }

        with pytest.raises(FatalError, match="None of the requested models"):
            app._filter_requested_models()

    def test_filter_requested_models_skipped_when_no_models_arg(
        self, basic_config: dict
    ) -> None:
        """Test filter is skipped when no models argument provided."""
        args = {
            "config": "config.yml",
            "outputs_dir": None,
            "models": None,
            "no_secrets": False,
            "interactive": False,
            "debug": False,
            "verbose": False,
        }
        app = App(args)

        app.arbitrium = AsyncMock()
        app.arbitrium.healthy_models = {
            "model_a": MockModel(model_name="a", display_name="A"),
            "model_b": MockModel(model_name="b", display_name="B"),
        }

        app._filter_requested_models()

        # Should not have modified models
        assert len(app.arbitrium.healthy_models) == 2


class TestArbitriumValidation:
    """Test Arbitrium readiness validation."""

    def test_validate_arbitrium_ready_success(self) -> None:
        """Test validation passes when Arbitrium is ready."""
        args = {
            "config": "config.yml",
            "outputs_dir": None,
            "no_secrets": False,
            "interactive": False,
            "models": None,
            "debug": False,
            "verbose": False,
        }
        app = App(args)

        app.arbitrium = AsyncMock()
        app.arbitrium.is_ready = True

        app._validate_arbitrium_ready()

    def test_validate_arbitrium_not_initialized(self) -> None:
        """Test validation fails when Arbitrium not initialized."""
        args = {
            "config": "config.yml",
            "outputs_dir": None,
            "no_secrets": False,
            "interactive": False,
            "models": None,
            "debug": False,
            "verbose": False,
        }
        app = App(args)

        app.arbitrium = None

        with pytest.raises(FatalError, match="not initialized"):
            app._validate_arbitrium_ready()

    def test_validate_arbitrium_not_ready(self) -> None:
        """Test validation fails when no models passed health check."""
        args = {
            "config": "config.yml",
            "outputs_dir": None,
            "no_secrets": False,
            "interactive": False,
            "models": None,
            "debug": False,
            "verbose": False,
        }
        app = App(args)

        app.arbitrium = AsyncMock()
        app.arbitrium.is_ready = False

        with pytest.raises(FatalError, match="No models passed health check"):
            app._validate_arbitrium_ready()


class TestCliWorkflow:
    """Test complete CLI workflow with real config files."""

    @pytest.mark.asyncio
    async def test_initialize_arbitrium_success(
        self, basic_config: dict, tmp_output_dir: Path
    ) -> None:
        """Test complete Arbitrium initialization."""
        config_file = tmp_output_dir / "config.yml"
        with open(config_file, "w") as f:
            yaml.dump(basic_config, f)

        args = {
            "config": str(config_file),
            "outputs_dir": str(tmp_output_dir),
            "no_secrets": True,
            "interactive": False,
            "models": None,
            "debug": False,
            "verbose": False,
        }
        app = App(args)

        await app._initialize_arbitrium()

        assert app.arbitrium is not None
        assert app.arbitrium.is_ready

    @pytest.mark.asyncio
    async def test_initialize_arbitrium_with_missing_config(self) -> None:
        """Test initialization fails with missing config file."""
        args = {
            "config": "missing_config.yml",
            "outputs_dir": None,
            "no_secrets": False,
            "interactive": False,
            "models": None,
            "debug": False,
            "verbose": False,
        }
        app = App(args)

        with pytest.raises(FatalError, match="Failed to load configuration"):
            await app._initialize_arbitrium()

    @pytest.mark.asyncio
    async def test_full_cli_flow_with_mock_models(
        self, basic_config: dict, tmp_output_dir: Path
    ) -> None:
        """Test full CLI flow: initialize -> get question -> run tournament."""
        config_file = tmp_output_dir / "config.yml"
        basic_config["question"] = "What is 2+2?"
        with open(config_file, "w") as f:
            yaml.dump(basic_config, f)

        args = {
            "config": str(config_file),
            "outputs_dir": str(tmp_output_dir),
            "no_secrets": True,
            "interactive": False,
            "models": None,
            "debug": False,
            "verbose": False,
        }
        app = App(args)

        await app._initialize_arbitrium()
        assert app.arbitrium is not None

        question = await app._get_app_question()
        assert question == "What is 2+2?"

        result, metrics = await app.arbitrium.run_tournament(question)

        assert result is not None
        assert metrics["champion_model"] is not None
        assert len(result) > 0


class TestQuestionHandling:
    """Test question retrieval from config and interactive input."""

    @pytest.mark.asyncio
    async def test_get_question_from_config(
        self, basic_config: dict, tmp_output_dir: Path
    ) -> None:
        """Test getting question from config file."""
        config_file = tmp_output_dir / "config.yml"
        test_question = "What is the meaning of life?"
        basic_config["question"] = test_question
        with open(config_file, "w") as f:
            yaml.dump(basic_config, f)

        args = {
            "config": str(config_file),
            "outputs_dir": None,
            "no_secrets": True,
            "interactive": False,
            "models": None,
            "debug": False,
            "verbose": False,
        }
        app = App(args)

        await app._initialize_arbitrium()
        question = await app._get_app_question()

        assert question == test_question

    @pytest.mark.asyncio
    async def test_get_question_interactive_mode(
        self, basic_config: dict, tmp_output_dir: Path
    ) -> None:
        """Test getting question in interactive mode."""
        config_file = tmp_output_dir / "config.yml"
        with open(config_file, "w") as f:
            yaml.dump(basic_config, f)

        args = {
            "config": str(config_file),
            "outputs_dir": None,
            "no_secrets": True,
            "interactive": True,
            "models": None,
            "debug": False,
            "verbose": False,
        }
        app = App(args)

        await app._initialize_arbitrium()

        with patch(
            "arbitrium_core.interfaces.cli.main.async_input",
            new_callable=AsyncMock,
            return_value="Interactive question?",
        ):
            question = await app._get_app_question()

        assert question == "Interactive question?"

    @pytest.mark.asyncio
    async def test_get_question_fallback_to_interactive(
        self, basic_config: dict, tmp_output_dir: Path
    ) -> None:
        """Test falling back to interactive when no config question."""
        config_file = tmp_output_dir / "config.yml"
        with open(config_file, "w") as f:
            yaml.dump(basic_config, f)

        args = {
            "config": str(config_file),
            "outputs_dir": None,
            "no_secrets": True,
            "interactive": False,
            "models": None,
            "debug": False,
            "verbose": False,
        }
        app = App(args)

        await app._initialize_arbitrium()

        with patch(
            "arbitrium_core.interfaces.cli.main.async_input",
            new_callable=AsyncMock,
            return_value="Fallback question?",
        ):
            question = await app._get_app_question()

        assert question == "Fallback question?"


class TestAppRunMethod:
    """Test App.run() end-to-end execution."""

    @pytest.mark.asyncio
    async def test_run_complete_tournament(
        self, basic_config: dict, tmp_output_dir: Path
    ) -> None:
        """Test running complete tournament from App.run()."""
        config_file = tmp_output_dir / "config.yml"
        basic_config["question"] = "Test question?"
        with open(config_file, "w") as f:
            yaml.dump(basic_config, f)

        args = {
            "config": str(config_file),
            "outputs_dir": str(tmp_output_dir),
            "no_secrets": True,
            "interactive": False,
            "models": None,
            "debug": False,
            "verbose": False,
            "command": "tournament",
        }
        app = App(args)

        await app.run()

        assert app.arbitrium is not None

    @pytest.mark.asyncio
    async def test_run_with_keyboard_interrupt(
        self, basic_config: dict, tmp_output_dir: Path
    ) -> None:
        """Test App.run() handles keyboard interrupt gracefully."""
        config_file = tmp_output_dir / "config.yml"
        basic_config["question"] = "Test question?"
        with open(config_file, "w") as f:
            yaml.dump(basic_config, f)

        args = {
            "config": str(config_file),
            "outputs_dir": str(tmp_output_dir),
            "no_secrets": True,
            "interactive": False,
            "models": None,
            "debug": False,
            "verbose": False,
        }
        app = App(args)

        app.arbitrium = AsyncMock()
        app.arbitrium.run_tournament = AsyncMock(
            side_effect=KeyboardInterrupt()
        )

        with patch(
            "arbitrium_core.interfaces.cli.main.App._initialize_arbitrium"
        ):
            with patch(
                "arbitrium_core.interfaces.cli.main.App._get_app_question",
                new_callable=AsyncMock,
                return_value="Test",
            ):
                await app.run()

    @pytest.mark.asyncio
    async def test_run_with_tournament_error(
        self, basic_config: dict, tmp_output_dir: Path
    ) -> None:
        """Test App.run() handles tournament errors."""
        config_file = tmp_output_dir / "config.yml"
        basic_config["question"] = "Test question?"
        with open(config_file, "w") as f:
            yaml.dump(basic_config, f)

        args = {
            "config": str(config_file),
            "outputs_dir": str(tmp_output_dir),
            "no_secrets": True,
            "interactive": False,
            "models": None,
            "debug": False,
            "verbose": False,
        }
        app = App(args)

        app.arbitrium = AsyncMock()
        app.arbitrium.run_tournament = AsyncMock(
            side_effect=Exception("Tournament error")
        )

        with patch(
            "arbitrium_core.interfaces.cli.main.App._initialize_arbitrium"
        ):
            with patch(
                "arbitrium_core.interfaces.cli.main.App._get_app_question",
                new_callable=AsyncMock,
                return_value="Test",
            ):
                with pytest.raises(
                    FatalError, match="Error during tournament"
                ):
                    await app.run()


class TestRunFromCliFunction:
    """Test run_from_cli() entry point."""

    def test_run_from_cli_success(
        self, basic_config: dict, tmp_output_dir: Path
    ) -> None:
        """Test run_from_cli() successfully runs tournament."""
        from arbitrium_core.interfaces.cli.main import run_from_cli

        config_file = tmp_output_dir / "config.yml"
        basic_config["question"] = "Test question?"
        with open(config_file, "w") as f:
            yaml.dump(basic_config, f)

        args = {
            "config": str(config_file),
            "outputs_dir": str(tmp_output_dir),
            "no_secrets": True,
            "interactive": False,
            "models": None,
            "debug": False,
            "verbose": False,
            "command": "tournament",
        }

        with patch(
            "arbitrium_core.interfaces.cli.main.parse_arguments",
            return_value=args,
        ):
            with patch("arbitrium_core.shared.logging.setup.setup_logging"):
                with patch("arbitrium_core.interfaces.cli.main.colorama.init"):
                    with patch(
                        "arbitrium_core.interfaces.cli.main.asyncio.run"
                    ):
                        run_from_cli()

    def test_run_from_cli_fatal_error_handling(self) -> None:
        """Test run_from_cli() handles FatalError gracefully."""
        from arbitrium_core.interfaces.cli.main import run_from_cli

        args = {
            "config": "config.yml",
            "command": "tournament",
            "debug": False,
            "verbose": False,
        }

        with patch(
            "arbitrium_core.interfaces.cli.main.parse_arguments",
            return_value=args,
        ):
            with patch("arbitrium_core.shared.logging.setup.setup_logging"):
                with patch("arbitrium_core.interfaces.cli.main.colorama.init"):
                    with patch(
                        "arbitrium_core.interfaces.cli.main.asyncio.run",
                        side_effect=FatalError("Fatal error"),
                    ):
                        with patch("sys.exit") as mock_exit:
                            run_from_cli()
                            mock_exit.assert_called_once_with(1)

    def test_run_from_cli_keyboard_interrupt_handling(self) -> None:
        """Test run_from_cli() handles KeyboardInterrupt with exit code 130."""
        from arbitrium_core.interfaces.cli.main import run_from_cli

        args = {
            "config": "config.yml",
            "command": "tournament",
            "debug": False,
            "verbose": False,
        }

        with patch(
            "arbitrium_core.interfaces.cli.main.parse_arguments",
            return_value=args,
        ):
            with patch("arbitrium_core.shared.logging.setup.setup_logging"):
                with patch("arbitrium_core.interfaces.cli.main.colorama.init"):
                    with patch(
                        "arbitrium_core.interfaces.cli.main.asyncio.run",
                        side_effect=KeyboardInterrupt(),
                    ):
                        with patch("sys.exit") as mock_exit:
                            run_from_cli()
                            mock_exit.assert_called_once_with(130)
