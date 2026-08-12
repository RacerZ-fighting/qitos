from qitos import ToolRegistry
from qitos.kit.tool import ThinkingToolSet


def test_thinking_toolset_register_and_flow():
    registry = ToolRegistry()
    registry.register_toolset(ThinkingToolSet())

    assert "thinking.sequential_thinking" in registry.list_tools()
    assert "thinking.get_thoughts" in registry.list_tools()
    assert "thinking.clear_thoughts" in registry.list_tools()

    thinking = registry.get("thinking.sequential_thinking")
    assert thinking is not None
    r1 = thinking.execute(
        {
            "thought": "Analyze the bug surface first.",
            "thought_number": 1,
            "total_thoughts": 3,
            "next_thought_needed": True,
        }
    )
    assert r1["status"] == "success"
    assert r1["thought_history_count"] == 1

    r2 = thinking.execute(
        {
            "thought": "Try an alternative hypothesis.",
            "thought_number": 2,
            "total_thoughts": 3,
            "next_thought_needed": True,
            "branch_from_thought": 1,
            "branch_id": "alt",
        }
    )
    assert r2["status"] == "success"
    assert r2["active_branch_count"] == 1

    r3 = thinking.execute(
        {
            "thought": "Revise first assumption.",
            "thought_number": 3,
            "total_thoughts": 3,
            "next_thought_needed": False,
            "is_revision": True,
            "revises_thought": 1,
        }
    )
    assert r3["status"] == "success"

    get_thoughts = registry.get("thinking.get_thoughts")
    clear_thoughts = registry.get("thinking.clear_thoughts")
    assert get_thoughts is not None
    assert clear_thoughts is not None
    snapshot = get_thoughts.execute({})
    assert snapshot["status"] == "success"
    assert snapshot["history_count"] == 2
    assert snapshot["branch_count"] == 1
    assert "alt" in snapshot["branches"]

    cleared = clear_thoughts.execute({})
    assert cleared["status"] == "success"
    snapshot2 = get_thoughts.execute({})
    assert snapshot2["history_count"] == 0
    assert snapshot2["branch_count"] == 0
