import arbitrium_core.interfaces.cli.ui as ui


def test_configure_display_disables_color(monkeypatch) -> None:
    monkeypatch.setattr(ui.Display, "_should_use_color", lambda self: True)

    ui.configure_display(use_color=False)
    assert ui._display.use_color is False

    ui.configure_display(use_color=True)
    assert ui._display.use_color is True
