#!/usr/bin/env python3
"""Run a bounded universal Tg experiment queue.

The script is intentionally stdlib-only so it can run as a long-lived server
process without importing project training modules. Each queue item launches the
existing training CLI as a subprocess, captures logs, reads summary.json, and
appends a persistent markdown experiment log.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


BASELINE_SUMMARY = Path("results/universal_single_regressor/exp45_homo_local_nopure/summary.json")
STATE_PATH = Path("results/universal_single_regressor/iteration_queue_state.json")
MARKDOWN_LOG = Path("docs/research/universal-tg-iteration-log.md")
RUN_LOG_DIR = Path("logs/universal_tg_iteration_queue")


DEFAULT_QUEUE: list[dict[str, Any]] = [
    {
        "name": "exp50_homo_local_nopure_nw40",
        "hypothesis": "Lower nucleobase sample weight may reduce over-correction and improve nucleobase group-holdout without hurting PolyInfo.",
        "args": [
            "--table",
            "results/universal_single_regressor/unified_training_table_fixedratio_homo186_nopure.parquet",
            "--output-dir",
            "results/universal_single_regressor/exp50_homo_local_nopure_nw40",
            "--model",
            "physics_homo_local_light",
            "--feature-layer",
            "HYBRID-HOMO186",
            "--max-virtual",
            "0",
            "--virtual-weight",
            "0",
            "--copolymer-weight",
            "10",
            "--nucleobase-weight",
            "40",
            "--group-eval",
        ],
    },
    {
        "name": "exp51_homo_local_nopure_cw5_nw60",
        "hypothesis": "Lower PolyInfo weight may improve balance if copolymer group-holdout is overfitting a few noisy systems.",
        "args": [
            "--table",
            "results/universal_single_regressor/unified_training_table_fixedratio_homo186_nopure.parquet",
            "--output-dir",
            "results/universal_single_regressor/exp51_homo_local_nopure_cw5_nw60",
            "--model",
            "physics_homo_local_light",
            "--feature-layer",
            "HYBRID-HOMO186",
            "--max-virtual",
            "0",
            "--virtual-weight",
            "0",
            "--copolymer-weight",
            "5",
            "--nucleobase-weight",
            "60",
            "--group-eval",
        ],
    },
    {
        "name": "exp52_homo_local_nopure_cw5_nw40",
        "hypothesis": "Jointly lowering PolyInfo and nucleobase weights may improve the minimum R2 by reducing small-sample dominance.",
        "args": [
            "--table",
            "results/universal_single_regressor/unified_training_table_fixedratio_homo186_nopure.parquet",
            "--output-dir",
            "results/universal_single_regressor/exp52_homo_local_nopure_cw5_nw40",
            "--model",
            "physics_homo_local_light",
            "--feature-layer",
            "HYBRID-HOMO186",
            "--max-virtual",
            "0",
            "--virtual-weight",
            "0",
            "--copolymer-weight",
            "5",
            "--nucleobase-weight",
            "40",
            "--group-eval",
        ],
    },
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def metric_block(summary: dict[str, Any]) -> dict[str, dict[str, float]]:
    metrics = summary.get("metrics", {})
    wanted = [
        "holdout_homopolymer_real",
        "group_holdout_polyinfo_real",
        "group_holdout_nucleobase_real",
    ]
    return {key: metrics.get(key, {}) for key in wanted}


def min_primary_r2(summary: dict[str, Any]) -> float | None:
    values = []
    for metrics in metric_block(summary).values():
        r2 = metrics.get("r2")
        if isinstance(r2, (int, float)):
            values.append(float(r2))
    return min(values) if len(values) == 3 else None


def command_for_item(item: dict[str, Any], python_exe: str) -> list[str]:
    if "command" in item:
        command = item["command"]
        if isinstance(command, str):
            return shlex.split(command)
        return [str(part) for part in command]
    args = [str(part) for part in item.get("args", [])]
    return [python_exe, "-u", "scripts/train_universal_tg_single_regressor.py", *args]


def output_dir_for_item(item: dict[str, Any]) -> Path | None:
    args = item.get("args")
    if isinstance(args, list) and "--output-dir" in args:
        idx = args.index("--output-dir")
        if idx + 1 < len(args):
            return Path(str(args[idx + 1]))
    output_dir = item.get("output_dir")
    return Path(str(output_dir)) if output_dir else None


def append_markdown_log(
    item: dict[str, Any],
    command: list[str],
    returncode: int,
    summary: dict[str, Any],
    baseline: dict[str, Any],
    run_log: Path,
) -> None:
    MARKDOWN_LOG.parent.mkdir(parents=True, exist_ok=True)
    current_min = min_primary_r2(summary)
    baseline_min = min_primary_r2(baseline)
    decision = "investigate"
    if returncode != 0:
        decision = "reject"
    elif current_min is not None and baseline_min is not None:
        decision = "keep" if current_min > baseline_min else "reject"

    lines = [
        "",
        f"## {now_iso()} - {item.get('name', 'unnamed')}",
        "",
        f"- hypothesis: {item.get('hypothesis', '')}",
        f"- command: `{' '.join(shlex.quote(part) for part in command)}`",
        f"- returncode: `{returncode}`",
        f"- run log: `{run_log}`",
        f"- decision: `{decision}`",
        "",
        "metrics:",
        "",
        "```json",
        json.dumps(metric_block(summary), indent=2, ensure_ascii=False),
        "```",
        "",
        "baseline metrics:",
        "",
        "```json",
        json.dumps(metric_block(baseline), indent=2, ensure_ascii=False),
        "```",
    ]
    with MARKDOWN_LOG.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def run_command(command: list[str], log_path: Path, timeout_seconds: int | None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"[{now_iso()}] START {' '.join(shlex.quote(part) for part in command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        start = time.monotonic()
        assert process.stdout is not None
        while True:
            line = process.stdout.readline()
            if line:
                log.write(line)
                log.flush()
                print(line, end="")
            if process.poll() is not None:
                break
            if timeout_seconds is not None and time.monotonic() - start > timeout_seconds:
                log.write(f"\n[{now_iso()}] TIMEOUT after {timeout_seconds}s\n")
                process.terminate()
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                return 124
        return int(process.wait())


def load_queue(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return DEFAULT_QUEUE
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("experiments", [])
    if not isinstance(data, list):
        raise ValueError("Queue JSON must be a list or an object with an 'experiments' list.")
    return [dict(item) for item in data]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a bounded queue of universal Tg experiments.")
    parser.add_argument("--queue-json", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-hours", type=float, default=5.0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--baseline-summary", type=Path, default=BASELINE_SUMMARY)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    queue = load_queue(args.queue_json)
    baseline = load_json(args.baseline_summary)
    deadline = time.monotonic() + max(float(args.max_hours), 0.01) * 3600.0
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    state: dict[str, Any] = {
        "run_id": run_id,
        "started_at": now_iso(),
        "max_hours": args.max_hours,
        "baseline_summary": str(args.baseline_summary),
        "experiments": [],
    }

    for index, item in enumerate(queue):
        if index < args.start_index:
            continue
        remaining = int(deadline - time.monotonic())
        if remaining <= 120:
            state["stopped_reason"] = "time_budget_exhausted"
            break
        name = str(item.get("name", f"experiment_{index:03d}"))
        command = command_for_item(item, args.python)
        run_log = RUN_LOG_DIR / f"{run_id}_{index:03d}_{name}.log"
        print(f"[{now_iso()}] queued {name}")
        print(" ".join(shlex.quote(part) for part in command))
        if args.dry_run:
            state["experiments"].append({"name": name, "status": "dry_run", "command": command})
            continue
        returncode = run_command(command, run_log, timeout_seconds=remaining)
        out_dir = output_dir_for_item(item)
        summary_path = out_dir / "summary.json" if out_dir else Path("")
        summary = load_json(summary_path) if summary_path else {}
        append_markdown_log(item, command, returncode, summary, baseline, run_log)
        state["experiments"].append(
            {
                "name": name,
                "finished_at": now_iso(),
                "returncode": returncode,
                "summary_path": str(summary_path),
                "run_log": str(run_log),
                "primary_min_r2": min_primary_r2(summary),
            }
        )
        write_json(STATE_PATH, state)
        if returncode != 0 and item.get("stop_on_failure", False):
            state["stopped_reason"] = f"failed:{name}"
            break

    state.setdefault("stopped_reason", "queue_complete")
    state["finished_at"] = now_iso()
    write_json(STATE_PATH, state)
    print(json.dumps(state, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
