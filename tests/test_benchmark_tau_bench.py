from __future__ import annotations

from examples.benchmarks.tau_bench_eval import run_tau_recipe_task as wrapper_run_tau_recipe_task
from qitos.benchmark import TauBenchAdapter, resolve_builtin_runner
from qitos.recipes.benchmarks.tau_bench import TauActionTool, run_tau_recipe_task


def test_tau_adapter_to_tasks_from_records():
    records = [
        {
            "task_id": "tau_1",
            "instruction": "Help user cancel order.",
            "outputs": ["order canceled"],
            "actions": [{"name": "cancel_pending_order", "kwargs": {"order_id": "1"}}],
            "user_id": "u1",
        },
        {
            "instruction": "Find direct flights.",
            "outputs": ["flight options"],
            "actions": [],
        },
    ]

    adapter = TauBenchAdapter(
        env_name="retail", task_split="test", include_raw_record=False
    )
    tasks = adapter.to_tasks(records, split="test")

    assert len(tasks) == 2
    assert tasks[0].id == "tau_1"
    assert tasks[0].objective == "Help user cancel order."
    assert tasks[0].inputs["benchmark"] == "tau-bench"
    assert tasks[0].metadata["benchmark"] == "tau-bench"
    assert tasks[0].env_spec is not None and tasks[0].env_spec.type == "tau_bench"

    assert tasks[1].id.startswith("tau_retail_test_")
    assert tasks[1].success_criteria


def test_tau_adapter_normalizes_action_models_shape():
    adapter = TauBenchAdapter(env_name="retail", task_split="test")
    rec = {
        "instruction": "x",
        "outputs": ["ok"],
        "actions": [{"name": "respond", "arguments": {"content": "ok"}}],
    }
    task = adapter.to_task(rec, split="test", idx=3)
    assert task.metadata["reference_actions"][0]["name"] == "respond"


def test_tau_adapter_load_records_without_external_dependency():
    adapter = TauBenchAdapter(env_name="retail", task_split="test")
    rows = adapter.load_records()
    assert rows
    assert "instruction" in rows[0]


def test_tau_bench_uses_builtin_runner_and_thin_example_wrapper():
    runner = resolve_builtin_runner(benchmark="tau-bench", strategy="tau_smoke")
    assert callable(runner)
    assert wrapper_run_tau_recipe_task is run_tau_recipe_task


def test_tau_action_tool_uses_canonical_execute_contract():
    calls = []

    def runner(name, args):
        calls.append((name, args))
        return {"ok": True}

    tool = TauActionTool(
        "lookup",
        "Lookup a record",
        {"record_id": {"type": "string"}},
        ["record_id"],
        runner,
    ).impl

    result = tool.execute({"record_id": "42"}, runtime_context={"run_id": "test"})

    assert result == {"ok": True}
    assert calls == [("lookup", {"record_id": "42"})]
