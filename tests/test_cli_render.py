from qitos.render.cli_render import _debug_message_preview


def test_debug_message_preview_is_bounded_and_keeps_both_ends() -> None:
    content = {"message": "debug-head\n" + ("x" * 20_000) + "\ndebug-tail"}

    preview = _debug_message_preview(content)

    assert len(preview) <= 8_000
    assert "debug-head" in preview
    assert "debug-tail" in preview
    assert "debug preview truncated" in preview
