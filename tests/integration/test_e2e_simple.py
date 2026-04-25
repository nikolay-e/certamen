"""Simple E2E tests that don't require full tournament execution."""

import pytest

from certamen import Certamen
from tests.integration.conftest import MockModel


class TestCertamenInitialization:
    """Test basic Certamen initialization."""

    @pytest.mark.asyncio
    async def test_certamen_from_settings_succeeds(
        self,
        basic_config: dict,
    ) -> None:
        """Test that Certamen initializes from settings."""
        certamen = await Certamen.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        assert certamen is not None
        assert certamen.config_data is not None
        assert len(certamen.all_models) > 0

    @pytest.mark.asyncio
    async def test_certamen_has_correct_model_count(
        self,
        basic_config: dict,
    ) -> None:
        """Test that Certamen loads correct number of models."""
        certamen = await Certamen.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        assert len(certamen.all_models) == 3
        assert "model_a" in certamen.all_models
        assert "model_b" in certamen.all_models
        assert "model_c" in certamen.all_models

    @pytest.mark.asyncio
    async def test_mock_models_can_generate_responses(
        self,
        basic_config: dict,
    ) -> None:
        """Test that MockModel can generate responses."""
        mock_model = MockModel(
            model_name="test",
            display_name="Test Model",
            response_text="Test response",
        )

        response = await mock_model.generate("Test prompt")

        assert response.is_successful
        assert "Test response" in response.content
        assert not response.is_error()

    @pytest.mark.asyncio
    async def test_certamen_with_mock_models(
        self,
        basic_config: dict,
    ) -> None:
        """Test Certamen with mock models injected."""
        certamen = await Certamen.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        # Inject mock models
        mock_models = {
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
            "model_b": MockModel(model_name="test-b", display_name="Model B"),
        }
        certamen._healthy_models = mock_models  # type: ignore[assignment]

        assert len(certamen.healthy_models) == 2
        assert certamen.is_ready


class TestModelComparison:
    """Test ModelComparison initialization."""

    @pytest.mark.asyncio
    async def test_comparison_creation(
        self,
        basic_config: dict,
    ) -> None:
        """Test that comparison can be created."""
        certamen = await Certamen.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
            "model_b": MockModel(model_name="test-b", display_name="Model B"),
        }
        certamen._healthy_models = mock_models  # type: ignore[assignment]

        comparison = certamen._create_comparison()

        assert comparison is not None
        assert comparison.config is not None
        assert len(comparison.models) == 2

    @pytest.mark.asyncio
    async def test_comparison_has_correct_configuration(
        self,
        basic_config: dict,
    ) -> None:
        """Test that comparison receives correct configuration."""
        certamen = await Certamen.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
        }
        certamen._healthy_models = mock_models  # type: ignore[assignment]

        comparison = certamen._create_comparison()

        assert "prompts" in comparison.config
        assert "initial" in comparison.config["prompts"]
        assert comparison.features is not None


class TestKnowledgeBankBasics:
    """Test basic Knowledge Bank functionality without full tournaments."""

    @pytest.mark.asyncio
    async def test_knowledge_bank_initialized(
        self,
        kb_enabled_config: dict,
        tmp_output_dir,
    ) -> None:
        """Test that KB is initialized when enabled."""
        certamen = await Certamen.from_settings(
            settings=kb_enabled_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
        }
        certamen._healthy_models = mock_models  # type: ignore[assignment]

        comparison = certamen._create_comparison()
        kb = comparison.knowledge_bank

        assert kb is not None
        assert len(kb.insights_db) == 0

    @pytest.mark.asyncio
    async def test_knowledge_bank_can_add_insights(
        self,
        kb_enabled_config: dict,
        tmp_output_dir,
    ) -> None:
        """Test that KB can add insights."""
        certamen = await Certamen.from_settings(
            settings=kb_enabled_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
        }
        certamen._healthy_models = mock_models  # type: ignore[assignment]

        comparison = certamen._create_comparison()
        kb = comparison.knowledge_bank

        # Add insights directly
        claims = [
            "First important insight",
            "Second critical point",
            "Third key consideration",
        ]
        await kb._add_insights_to_db(claims, "TestModel", source_round=1)

        assert len(kb.insights_db) == 3
        all_insights = await kb.get_all_insights()
        assert len(all_insights) == 3


class TestConfigurationValidation:
    """Test configuration validation without full execution."""

    @pytest.mark.asyncio
    async def test_minimal_valid_config_accepted(
        self,
        minimal_config: dict,
        tmp_output_dir,
    ) -> None:
        """Test that minimal valid config is accepted."""
        certamen = await Certamen.from_settings(
            settings=minimal_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        assert certamen is not None
        assert len(certamen.all_models) == 2

    @pytest.mark.asyncio
    async def test_config_with_all_sections(
        self,
        basic_config: dict,
    ) -> None:
        """Test config with all sections."""
        certamen = await Certamen.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        config_data = certamen.config_data

        assert "models" in config_data
        assert "retry" in config_data
        assert "features" in config_data
        assert "prompts" in config_data
        assert "outputs_dir" in config_data


class TestSingleModelExecution:
    """Test single model execution without tournaments."""

    @pytest.mark.asyncio
    async def test_run_single_model(
        self,
        basic_config: dict,
    ) -> None:
        """Test running a single model."""
        certamen = await Certamen.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        # Inject mock models
        mock_models = {
            "model_a": MockModel(
                model_name="test-a",
                display_name="Model A",
                response_text="Single model response",
            ),
        }
        certamen._healthy_models = mock_models  # type: ignore[assignment]

        response = await certamen.run_single_model("model_a", "Test prompt")

        assert response.is_successful
        assert "Single model response" in response.content

    @pytest.mark.asyncio
    async def test_run_single_model_with_error(
        self,
        basic_config: dict,
    ) -> None:
        """Test single model that fails."""
        certamen = await Certamen.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        # Inject failing mock model
        mock_models = {
            "failing": MockModel(
                model_name="test-fail",
                display_name="Failing Model",
                should_fail=True,
            ),
        }
        certamen._healthy_models = mock_models  # type: ignore[assignment]

        response = await certamen.run_single_model("failing", "Test prompt")

        assert response.is_error()
        assert response.error is not None
