from pathlib import Path
from types import SimpleNamespace

import pytest

from arbitrium_core.domain.errors import FatalError
from arbitrium_core.interfaces.cli.main import App


@pytest.mark.asyncio
async def test_question_file_is_used_over_config(tmp_path: Path) -> None:
    qfile = tmp_path / "q.txt"
    qfile.write_text("  hello from file  \n", encoding="utf-8")

    args = {
        "interactive": False,
        "question": str(qfile),
        "outputs_dir": None,
        "config": "does-not-matter.yml",
        "no_secrets": True,
    }
    app = App(args=args)
    app.arbitrium = SimpleNamespace(
        config_data={"question": "config question"}
    )

    question = await app._get_app_question()
    assert question == "hello from file"


@pytest.mark.asyncio
async def test_question_file_missing_raises_fatal_error(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.txt"

    args = {
        "interactive": False,
        "question": str(missing),
        "outputs_dir": None,
        "config": "does-not-matter.yml",
        "no_secrets": True,
    }
    app = App(args=args)
    app.arbitrium = SimpleNamespace(config_data={})

    with pytest.raises(FatalError):
        await app._get_app_question()
