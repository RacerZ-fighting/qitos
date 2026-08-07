"""Model-facing contract tests for the generic child Agent tool."""

from qitos.kit.tool.agent import AgentTool


def test_description_explains_parallel_delegation_boundary() -> None:
    tool = AgentTool(allow_background=False)
    description = tool.spec.description.lower()

    assert "independent multi-step tasks" in description
    assert "same response" in description
    assert "run concurrently" in description
    assert "do not repeat" in description
    assert "dependent steps" in description
    assert "mechanical variants" in description
