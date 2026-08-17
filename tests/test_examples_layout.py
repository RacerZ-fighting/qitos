from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_canonical_examples_exist_and_parse() -> None:
    canonical = [
        "examples/quickstart/minimal_agent.py",
        "examples/patterns/embedding_vectorstore.py",
        "examples/patterns/function_tool_custom.py",
        "examples/real/desktop_env_smoke.py",
    ]
    for rel in canonical:
        path = ROOT / rel
        assert path.exists(), rel
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_legacy_example_wrappers_are_gone() -> None:
    removed = [
        "examples/coding_agent.py",
        "examples/computer_use_agent.py",
        "examples/epub_reader_tot_agent.py",
        "examples/swe_dynamic_planning_agent.py",
        "examples/real/open_deep_research_gaia_agent.py",
        "examples/real/tau_bench_eval.py",
        "examples/real/cybench_eval.py",
        "examples/common.py",
    ]
    for rel in removed:
        assert not (ROOT / rel).exists(), rel


def test_examples_readme_points_to_canonical_layout() -> None:
    readme = _read("examples/README.md")
    assert "`examples/quickstart/`" in readme
    assert "`examples/patterns/`" in readme
    assert "`examples/real/`" in readme
    assert "benchmark/eval runners remain under `examples/real/`" not in readme
