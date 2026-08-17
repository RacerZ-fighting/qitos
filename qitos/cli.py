"""Top-level qit CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from qitos.benchmark import (
    evaluate_benchmark_results,
    read_benchmark_results,
)
from qitos.qita._cli_app import _cmd_export as qita_export
from qitos.qita._cli_app import _cmd_replay as qita_replay


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--version":
        from qitos import __version__

        print(f"qit {__version__}")
        return 0
    if args and args[0] in {"-h", "--help"}:
        parser = argparse.ArgumentParser(
            prog="qit",
            description="QitOS CLI for benchmarks and developer workflows",
        )
        subparsers = parser.add_subparsers(dest="command")
        subparsers.add_parser("bench", help="Unified benchmark CLI")
        subparsers.add_parser("leaderboard", help="Local benchmark leaderboard")
        subparsers.add_parser("push", help="Push trace artifacts to HF Hub")
        subparsers.add_parser("pull", help="Pull trace artifacts from HF Hub")
        parser.print_help()
        return 0
    if args:
        command = args[0]
        remaining = args[1:]
        if command == "bench":
            return _bench_main(remaining)
        if command == "leaderboard":
            return _leaderboard_main(remaining)
        if command == "push":
            return _push_main(remaining)
        if command == "pull":
            return _pull_main(remaining)
    parser = argparse.ArgumentParser(
        prog="qit",
        description="QitOS CLI for benchmarks and developer workflows",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("bench", help="Unified benchmark CLI")
    subparsers.add_parser("leaderboard", help="Local benchmark leaderboard")
    subparsers.add_parser("push", help="Push trace artifacts to HF Hub")
    subparsers.add_parser("pull", help="Pull trace artifacts from HF Hub")
    parser.print_help()
    return 1


def _bench_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="qit bench", description="QitOS benchmark CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("eval", help="Aggregate benchmark results")
    p_eval.add_argument("--input", required=True)
    p_eval.add_argument("--json", action="store_true")

    p_replay = sub.add_parser("replay", help="Replay one benchmark run")
    p_replay.add_argument("--run", required=True)
    p_replay.add_argument("--host", default="127.0.0.1")
    p_replay.add_argument("--port", type=int, default=8765)
    p_replay.add_argument("--print-url", action="store_true")

    p_export = sub.add_parser("export", help="Export one benchmark run")
    p_export.add_argument("--run", required=True)
    p_export.add_argument("--html", required=True)

    sub.add_parser("presets", help="List available model-family presets")

    args = parser.parse_args(argv)
    if args.command == "eval":
        return _bench_eval(args)
    if args.command == "replay":
        return _bench_replay(args)
    if args.command == "export":
        return _bench_export(args)
    if args.command == "presets":
        return _bench_presets(args)
    return 1


def _bench_eval(args: argparse.Namespace) -> int:
    rows = read_benchmark_results(args.input)
    summary = evaluate_benchmark_results(rows)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"benchmark={summary.get('benchmark')} split={summary.get('split')}")
        print(
            f"total={summary.get('total', 0)} success_rate={summary.get('success_rate', 0.0):.3f} avg_steps={summary.get('avg_steps', 0.0):.2f}"
        )
    return 0


def _bench_replay(args: argparse.Namespace) -> int:
    run_dir = Path(args.run).expanduser().resolve()
    if args.print_url:
        print(f"http://{args.host}:{int(args.port)}/replay/{run_dir.name}")
        return 0
    return qita_replay(run=str(run_dir), host=str(args.host), port=int(args.port))


def _bench_export(args: argparse.Namespace) -> int:
    return qita_export(run=str(args.run), html_path=str(args.html))


def _bench_presets(args: argparse.Namespace) -> int:
    from qitos.harness._presets import known_family_presets

    gold_ids = {"qwen", "kimi", "minimax", "gpt-oss", "gemma-4"}
    for preset in known_family_presets():
        marker = " *" if preset.id in gold_ids else ""
        ctx = preset.context_policy.context_window_hint
        ctx_str = f"{ctx // 1000}k" if ctx else "-"
        models = (
            ", ".join(preset.recommended_models[:2])
            if preset.recommended_models
            else "-"
        )
        print(
            f"  {preset.id:12s}{marker}  {preset.display_name:16s}  "
            f"{preset.default_protocol:26s}  {preset.tool_policy.primary_delivery:18s}  "
            f"ctx={ctx_str:>6s}  {models}"
        )
    print()
    print("  * = gold preset (most thoroughly tested)")
    return 0


# ---------------------------------------------------------------------------
# qit leaderboard
# ---------------------------------------------------------------------------


def _leaderboard_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="qit leaderboard", description="Local benchmark leaderboard"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_submit = sub.add_parser(
        "submit", help="Submit benchmark results to the leaderboard"
    )
    p_submit.add_argument("--results", help="Path to JSONL results file")
    p_submit.add_argument("--run-dir", help="Path to a single run directory")
    p_submit.add_argument(
        "--db", default="./runs/leaderboard.db", help="Leaderboard database path"
    )

    p_show = sub.add_parser("show", help="Show leaderboard entries")
    p_show.add_argument("--benchmark", help="Filter by benchmark name")
    p_show.add_argument("--split", help="Filter by split")
    p_show.add_argument("--model", help="Filter by model name")
    p_show.add_argument(
        "--official", action="store_true", help="Show only official runs"
    )
    p_show.add_argument("--sort-by", default="submitted_at", help="Sort field")
    p_show.add_argument("--limit", type=int, default=50, help="Max rows to show")
    p_show.add_argument(
        "--db", default="./runs/leaderboard.db", help="Leaderboard database path"
    )

    p_summary = sub.add_parser("summary", help="Show aggregated statistics")
    p_summary.add_argument("--benchmark", required=True, help="Benchmark name")
    p_summary.add_argument("--split", required=True, help="Split name")
    p_summary.add_argument("--official", action="store_true", help="Official runs only")
    p_summary.add_argument(
        "--db", default="./runs/leaderboard.db", help="Leaderboard database path"
    )

    args = parser.parse_args(argv)

    from qitos.leaderboard.store import LeaderboardStore

    if args.command == "submit":
        store = LeaderboardStore(args.db)
        try:
            if args.results:
                count = store.submit_results_file(args.results)
                print(f"Submitted {count} results from {args.results}")
            elif args.run_dir:
                sid = store.submit_run_dir(args.run_dir)
                print(f"Submitted run {args.run_dir} as {sid}")
            else:
                print("Error: provide --results or --run-dir", file=sys.stderr)
                return 1
        finally:
            store.close()
        return 0

    if args.command == "show":
        store = LeaderboardStore(args.db)
        try:
            rows = store.query(
                benchmark=args.benchmark,
                split=args.split,
                model_name=args.model,
                is_official=args.official or None,
                sort_by=args.sort_by,
                limit=args.limit,
            )
            if not rows:
                print("No entries found.")
                return 0
            print(
                f"{'model':30s} {'bench':15s} {'split':10s} {'ok':3s} {'steps':6s} {'lat':8s} {'official':3s} {'submitted':20s}"
            )
            for r in rows:
                print(
                    f"{r.model_name:30s} {r.benchmark:15s} {r.split:10s} "
                    f"{'Y' if r.success else 'N':3s} {r.steps:6d} {r.latency_seconds:8.1f} "
                    f"{'Y' if r.is_official else 'N':3s} {r.submitted_at[:19]:20s}"
                )
        finally:
            store.close()
        return 0

    if args.command == "summary":
        store = LeaderboardStore(args.db)
        try:
            s = store.summary(args.benchmark, args.split, is_official=args.official)
            print(json.dumps(s, ensure_ascii=False, indent=2))
        finally:
            store.close()
        return 0

    return 1


# ---------------------------------------------------------------------------
# qit push / qit pull
# ---------------------------------------------------------------------------


def _push_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="qit push", description="Push trace artifacts to HF Hub"
    )
    parser.add_argument("--run", help="Path to a single run directory")
    parser.add_argument("--logdir", help="Push all runs in a logdir")
    parser.add_argument("--repo", required=True, help="HF Hub dataset repo ID")
    parser.add_argument("--token", help="HF Hub API token")
    parser.add_argument("--revision", help="Git revision/branch")
    parser.add_argument(
        "--private",
        action="store_true",
        default=True,
        help="Make repo private (default)",
    )
    parser.add_argument(
        "--public", action="store_false", dest="private", help="Make repo public"
    )

    args = parser.parse_args(argv)

    try:
        from qitos.hf.hub import push_run
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.run:
        try:
            url = push_run(
                args.run,
                args.repo,
                token=args.token,
                revision=args.revision,
                private=args.private,
            )
            print(f"Pushed to {url}")
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.logdir:
        logdir = Path(args.logdir).expanduser().resolve()
        if not logdir.is_dir():
            print(f"Error: {args.logdir} is not a directory", file=sys.stderr)
            return 1
        count = 0
        for run_dir in sorted(logdir.iterdir()):
            if run_dir.is_dir() and (run_dir / "manifest.json").exists():
                try:
                    url = push_run(
                        run_dir,
                        args.repo,
                        token=args.token,
                        revision=args.revision,
                        private=args.private,
                    )
                    print(f"Pushed {run_dir.name} -> {url}")
                    count += 1
                except Exception as exc:
                    print(f"Warning: skipped {run_dir.name}: {exc}", file=sys.stderr)
        print(f"Pushed {count} runs")
        return 0

    print("Error: provide --run or --logdir", file=sys.stderr)
    return 1


def _pull_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="qit pull", description="Pull trace artifacts from HF Hub"
    )
    parser.add_argument("--run-id", required=True, help="Run ID to download")
    parser.add_argument("--repo", required=True, help="HF Hub dataset repo ID")
    parser.add_argument("--output", default="./runs", help="Local output directory")
    parser.add_argument("--token", help="HF Hub API token")
    parser.add_argument("--revision", help="Git revision/branch")

    args = parser.parse_args(argv)

    try:
        from qitos.hf.hub import pull_run
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        local_path = pull_run(
            args.run_id,
            args.repo,
            args.output,
            token=args.token,
            revision=args.revision,
        )
        print(f"Pulled to {local_path}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
