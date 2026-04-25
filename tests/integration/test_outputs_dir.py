"""
Integration tests for outputs_dir configuration behavior.

Tests the bug where CLI was overwriting config file's outputs_dir with None,
causing files to go to temp directory instead of the configured location.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from certamen.interfaces.cli.main import App


class TestOutputsDirBehavior:
    """Test outputs_dir priority and override behavior."""

    @pytest.mark.asyncio
    @patch("certamen.interfaces.cli.main.parse_arguments")
    @patch("certamen.interfaces.cli.main.Certamen")
    async def test_config_outputs_dir_not_overridden_when_cli_not_specified(
        self, mock_certamen_class: MagicMock, mock_parse: MagicMock
    ) -> None:
        """
        Test that config file's outputs_dir is preserved when CLI doesn't specify it.

        Regression test for bug where CLI unconditionally overwrote config's
        outputs_dir with None, causing files to go to wrong location.
        """
        # Create a temporary config file with outputs_dir="./test_output"
        config = {
            "models": {
                "test": {
                    "provider": "mock",
                    "model_name": "test-model",
                }
            },
            "outputs_dir": "./test_output",
            "retry": {},
            "features": {},
            "prompts": {},
        }
        fd, config_path = tempfile.mkstemp(suffix=".yml")
        os.close(fd)
        Path(config_path).write_text(yaml.dump(config))

        try:
            # Mock parse_arguments to return no outputs_dir (user didn't specify --outputs-dir)
            mock_parse.return_value = {
                "config": config_path,
                "outputs_dir": None,  # CLI argument not provided
                "no_secrets": True,
                "command": "tournament",
            }

            # Mock Certamen.from_settings to capture what settings it receives
            mock_certamen_instance = AsyncMock()
            mock_certamen_class.from_settings = AsyncMock(
                return_value=mock_certamen_instance
            )

            # Create app and initialize
            app = App()
            await app._initialize_certamen()

            # Verify Certamen.from_settings was called
            assert mock_certamen_class.from_settings.called

            # Get the settings that were passed
            call_args = mock_certamen_class.from_settings.call_args
            settings = call_args.kwargs["settings"]

            # CRITICAL: outputs_dir should be "./test_output" from config, NOT None
            assert (
                settings["outputs_dir"] == "./test_output"
            ), "Config file's outputs_dir should not be overridden when CLI doesn't specify it"

        finally:
            # Cleanup
            Path(config_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    @patch("certamen.interfaces.cli.main.parse_arguments")
    @patch("certamen.interfaces.cli.main.Certamen")
    async def test_cli_outputs_dir_overrides_config(
        self, mock_certamen_class: MagicMock, mock_parse: MagicMock
    ) -> None:
        """
        Test that CLI --outputs-dir flag overrides config file's outputs_dir.
        """
        # Create a temporary config file with outputs_dir="./config_output"
        config = {
            "models": {
                "test": {
                    "provider": "mock",
                    "model_name": "test-model",
                }
            },
            "outputs_dir": "./config_output",
            "retry": {},
            "features": {},
            "prompts": {},
        }
        fd, config_path = tempfile.mkstemp(suffix=".yml")
        os.close(fd)
        Path(config_path).write_text(yaml.dump(config))

        try:
            # Mock parse_arguments with CLI override
            mock_parse.return_value = {
                "config": config_path,
                "outputs_dir": "./cli_output",  # CLI explicitly specified different path
                "no_secrets": True,
                "command": "tournament",
            }

            # Mock Certamen.from_settings
            mock_certamen_instance = AsyncMock()
            mock_certamen_class.from_settings = AsyncMock(
                return_value=mock_certamen_instance
            )

            # Create app and initialize
            app = App()
            await app._initialize_certamen()

            # Get the settings that were passed
            call_args = mock_certamen_class.from_settings.call_args
            settings = call_args.kwargs["settings"]

            # CLI should override config
            assert (
                settings["outputs_dir"] == "./cli_output"
            ), "CLI --outputs-dir should override config file's value"

        finally:
            # Cleanup
            Path(config_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    @patch("certamen.interfaces.cli.main.parse_arguments")
    @patch("certamen.interfaces.cli.main.Certamen")
    async def test_none_outputs_dir_uses_current_directory(
        self, mock_certamen_class: MagicMock, mock_parse: MagicMock
    ) -> None:
        """
        Test that outputs_dir=None uses current directory (not temp).
        """
        # Create a temporary config file with outputs_dir=null
        config = {
            "models": {
                "test": {
                    "provider": "mock",
                    "model_name": "test-model",
                }
            },
            "outputs_dir": None,
            "retry": {},
            "features": {},
            "prompts": {},
        }
        fd, config_path = tempfile.mkstemp(suffix=".yml")
        os.close(fd)
        Path(config_path).write_text(yaml.dump(config))

        try:
            mock_parse.return_value = {
                "config": config_path,
                "outputs_dir": None,
                "no_secrets": True,
                "command": "tournament",
            }

            # Mock Certamen.from_settings
            mock_certamen_instance = AsyncMock()
            mock_certamen_class.from_settings = AsyncMock(
                return_value=mock_certamen_instance
            )

            # Create app and initialize
            app = App()
            await app._initialize_certamen()

            # Get the settings
            call_args = mock_certamen_class.from_settings.call_args
            settings = call_args.kwargs["settings"]

            # None should remain None (Certamen will convert to ".")
            assert settings["outputs_dir"] is None

        finally:
            # Cleanup
            Path(config_path).unlink(missing_ok=True)
