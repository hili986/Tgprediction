#!/usr/bin/env python3
"""Restart local Codex exec in bounded rounds for autonomous Tg iteration.

This is an agent loop harness, not an experiment runner. Each round launches a
fresh `codex exec` session with the current AGENTS.md, task queue, recent log,
and scoreboard. The Codex round must do one bounded iteration and end with one
of:

    TG_CONTINUE
    TG_BLOCKED
    TG_CONVERGED

The harness then decides whether to launch the next round.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENTS_PATH = PROJECT_ROOT / "AGENTS.md"
TASK_QUEUE_PATH = PROJECT_ROOT / "docs/research/universal-tg-task-queue.md"
ITERATION_LOG_PATH = PROJECT_ROOT / "docs/research/universal-tg-iteration-log.md"
SCOREBOARD_PATH = PROJECT_ROOT / "results/universal_single_regressor/scoreboard.json"
STATE_PATH = PROJECT_ROOT / "results/universal_single_regressor/agent_loop_state.json"
EVENTS_PATH = PROJECT_ROOT / "results/universal_single_regressor/agent_loop_events.jsonl"
LOOP_LOG_DIR = PROJECT_ROOT / "logs/codex_universal_tg_agent_loop"
SIGNALS = ("TG_CONTINUE", "TG_BLOCKED", "TG_CONVERGED")
DEFAULT_REMOTE_HOST = "sheng-xiang@100.64.0.4"
DEFAULT_REMOTE_PROJECT_DIR = "~/Tgprediction"
DEFAULT_REMOTE_PYTHON = "/home/sheng-xiang/miniconda3/envs/llm4graphgen/bin/python"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_text(path: Path, default: str = "") -> str:
    if not path.exists():
        return default
    return path.read_text(encoding="utf-8", errors="replace")


def tail_lines(text: str, limit: int) -> str:
    lines = text.splitlines()
    if len(lines) <= limit:
        return text
    return "\n".join(lines[-limit:])


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json_text(path: Path) -> str:
    if not path.exists():
        return "{}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return json.dumps({"error": f"failed to parse {path}: {exc}"}, ensure_ascii=False, indent=2)
    return json.dumps(data, ensure_ascii=False, indent=2)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n")


def write_state(state: dict[str, Any], run_state_path: Path | None = None) -> None:
    write_json(STATE_PATH, state)
    if run_state_path is not None:
        write_json(run_state_path, state)


def compact_text(text: str, limit: int = 2400) -> str:
    if len(text) <= limit:
        return text
    return text[: limit // 2] + "\n...[truncated]...\n" + text[-limit // 2 :]


def build_round_prompt(
    round_index: int,
    max_rounds: int,
    extra_instruction: str = "",
    remote_host: str = DEFAULT_REMOTE_HOST,
    remote_project_dir: str = DEFAULT_REMOTE_PROJECT_DIR,
    remote_python: str = DEFAULT_REMOTE_PYTHON,
) -> tuple[str, str]:
    agents_text = read_text(AGENTS_PATH, "")
    task_queue = read_text(TASK_QUEUE_PATH, "# Missing task queue\n")
    iteration_log = read_text(ITERATION_LOG_PATH, "# No iteration log yet\n")
    scoreboard = load_json_text(SCOREBOARD_PATH)
    loop_state = load_json_text(STATE_PATH)
    loop_events = tail_lines(read_text(EVENTS_PATH, ""), 80)
    agents_hash = sha256_text(agents_text)

    prompt = f"""You are Codex running one bounded autonomous iteration for the Tgprediction project.

This is round {round_index + 1} of at most {max_rounds}.

Hard requirements:
- You are running locally. Treat this local project directory as controller state: {PROJECT_ROOT}
- Heavy experiments should run remotely over SSH, not by requiring Codex on the server.
- Remote host: {remote_host}
- Remote project directory: {remote_project_dir}
- Remote Python: {remote_python}
- Remote safety: every remote command must start with `cd {remote_project_dir}` and must not operate outside that directory.
- Before substantive work, read AGENTS.md from disk and verify it matches this hash: {agents_hash}
- Read docs/research/universal-tg-task-queue.md.
- Read docs/research/universal-tg-iteration-log.md if present.
- Read results/universal_single_regressor/scoreboard.json if present.
- Do not execute the task queue as a simple linear checklist. Start with a global analysis of the scoreboard, prior log, open tasks, failed hypotheses, and current bottleneck.
- Generate 2-4 competing hypotheses or next actions, compare expected value / risk / cost, then choose exactly one bounded hypothesis/task for this round.
- You may re-prioritize the task queue or add a better task if the global analysis shows the listed next task is not the best use of the round.
- Run tests before any long or expensive experiment when code changed.
- Write or update docs/research/universal-tg-iteration-log.md.
- Write or update results/universal_single_regressor/scoreboard.json if metrics changed or were inspected.
- Every iteration-log entry must include: global analysis, competing hypotheses considered, chosen hypothesis, commands/code changes, metrics, keep/reject/investigate decision, and next action.
- Do not ask the user questions unless blocked by missing credentials, missing data, or unsafe ambiguity.
- Your final response must end with exactly one standalone signal line:
  TG_CONTINUE
  TG_BLOCKED
  TG_CONVERGED

Signal rules:
- Use TG_CONTINUE only if another independent iteration is justified.
- Use TG_BLOCKED if user input is needed, the environment is broken, the target is data-limited, or three serious iterations failed to improve the primary objective.
- Use TG_CONVERGED only if all target categories have R2 >= 0.95 under the required metrics.

AGENTS.md SHA256:
{agents_hash}

AGENTS.md content loaded by harness:
```md
{agents_text}
```

Task queue:
```md
{task_queue}
```

Recent iteration log tail:
```md
{tail_lines(iteration_log, 160)}
```

Current scoreboard:
```json
{scoreboard}
```

Current loop state:
```json
{loop_state}
```

Recent loop event tail:
```jsonl
{loop_events or "(none)"}
```

Remote execution reminder:
- Use local Codex for reasoning and code edits.
- Use SSH only for server experiments and metric collection.
- After remote experiments, pull or summarize metrics into local logs/scoreboard.
- The server does not need Codex CLI.
- Prefer SSH keys. If unavailable, use local `scripts/remote_tg_command.py` with `TG_REMOTE_PASSWORD` set in the environment; never write passwords into files.
- Remote helper example: `python scripts/remote_tg_command.py --use-paramiko "git rev-parse --short HEAD"`

Additional launcher instruction:
{extra_instruction or "(none)"}
"""
    return prompt, agents_hash


def build_codex_command(args: argparse.Namespace, last_message_path: Path) -> list[str]:
    codex_cmd = resolve_codex_command(args.codex_cmd)
    command = [
        codex_cmd,
        "exec",
        "--cd",
        str(PROJECT_ROOT),
        "--sandbox",
        args.codex_sandbox,
        "-c",
        f"approval_policy={json.dumps(args.codex_approval)}",
        "-c",
        f"reasoning_effort={json.dumps(args.reasoning_effort)}",
        "--output-last-message",
        str(last_message_path),
    ]
    if args.model:
        command.extend(["--model", args.model])
    if args.profile:
        command.extend(["--profile", args.profile])
    if args.search:
        command.append("--search")
    if args.skip_git_repo_check:
        command.append("--skip-git-repo-check")
    for item in args.codex_arg:
        command.append(item)
    command.append("-")
    return command


def resolve_codex_command(codex_cmd: str) -> str:
    command_path = Path(codex_cmd)
    if command_path.parent != Path("."):
        return str(command_path)
    resolved = shutil.which(codex_cmd)
    if resolved is None:
        return codex_cmd
    resolved_path = Path(resolved)
    if os.name == "nt":
        cmd_peer = resolved_path.with_suffix(".cmd")
        if cmd_peer.exists():
            return str(cmd_peer)
    return str(resolved_path)


def stream_process(
    command: list[str],
    prompt: str,
    stdout_path: Path,
    timeout_seconds: int,
) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    with stdout_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"[{now_iso()}] COMMAND: {' '.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        try:
            process.stdin.write(prompt)
            process.stdin.close()
        except BrokenPipeError:
            message = f"[{now_iso()}] STDIN broken pipe while sending prompt; process likely exited early.\n"
            print(message, end="")
            log.write(message)
            log.flush()
            try:
                process.stdin.close()
            except Exception:
                pass

        started = time.monotonic()
        while True:
            line = process.stdout.readline()
            if line:
                print(line, end="")
                log.write(line)
                log.flush()
            if process.poll() is not None:
                break
            if time.monotonic() - started > timeout_seconds:
                message = f"\n[{now_iso()}] ROUND TIMEOUT after {timeout_seconds}s\n"
                print(message, end="")
                log.write(message)
                log.flush()
                process.terminate()
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                return 124
        return int(process.wait())


def parse_signal(last_message: str) -> str | None:
    for line in reversed(last_message.splitlines()):
        stripped = line.strip()
        if stripped in SIGNALS:
            return stripped
    return None


def preflight(args: argparse.Namespace) -> None:
    if not AGENTS_PATH.exists():
        raise FileNotFoundError(f"Missing {AGENTS_PATH}")
    if not TASK_QUEUE_PATH.exists():
        raise FileNotFoundError(f"Missing {TASK_QUEUE_PATH}")
    if args.dry_run:
        return
    command_path = Path(args.codex_cmd)
    if command_path.parent == Path("."):
        if shutil.which(args.codex_cmd) is None:
            raise FileNotFoundError(f"Codex command not found on PATH: {args.codex_cmd}")
    elif not command_path.exists():
        raise FileNotFoundError(f"Codex command path does not exist: {args.codex_cmd}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Codex in repeated autonomous Tg iteration rounds.")
    parser.add_argument("--codex-cmd", default="codex")
    parser.add_argument("--codex-sandbox", default="danger-full-access")
    parser.add_argument("--codex-approval", default="never")
    parser.add_argument("--codex-arg", action="append", default=[], help="Extra raw argument passed to codex exec.")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--profile", default="")
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--skip-git-repo-check", action="store_true")
    parser.add_argument("--max-hours", type=float, default=5.0)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--round-timeout-minutes", type=float, default=45.0)
    parser.add_argument("--extra-instruction", default="")
    parser.add_argument("--remote-host", default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--remote-project-dir", default=DEFAULT_REMOTE_PROJECT_DIR)
    parser.add_argument("--remote-python", default=DEFAULT_REMOTE_PYTHON)
    parser.add_argument("--continue-on-no-signal", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    LOOP_LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ITERATION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not ITERATION_LOG_PATH.exists():
        ITERATION_LOG_PATH.write_text("# Universal Tg Iteration Log\n", encoding="utf-8")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_state_path = LOOP_LOG_DIR / f"{run_id}.state.json"
    deadline = time.monotonic() + max(args.max_hours, 0.01) * 3600.0
    state: dict[str, Any] = {
        "run_id": run_id,
        "started_at": now_iso(),
        "status": "running",
        "strategy_mode": "global_analysis_then_bounded_hypothesis",
        "max_hours": args.max_hours,
        "max_rounds": args.max_rounds,
        "state_path": str(STATE_PATH.relative_to(PROJECT_ROOT)),
        "run_state_path": str(run_state_path.relative_to(PROJECT_ROOT)),
        "events_path": str(EVENTS_PATH.relative_to(PROJECT_ROOT)),
        "log_dir": str(LOOP_LOG_DIR.relative_to(PROJECT_ROOT)),
        "preflight": "pending",
        "rounds": [],
    }
    write_state(state, run_state_path)
    try:
        preflight(args)
    except Exception as exc:
        state.update(
            {
                "status": "failed",
                "preflight": "failed",
                "stopped_reason": "preflight_failed",
                "exception": repr(exc),
                "traceback": compact_text(traceback.format_exc(), 5000),
                "finished_at": now_iso(),
            }
        )
        write_state(state, run_state_path)
        append_jsonl(
            EVENTS_PATH,
            {
                "type": "preflight_failed",
                "time": state["finished_at"],
                "run_id": run_id,
                "exception": repr(exc),
                "run_state_path": str(run_state_path.relative_to(PROJECT_ROOT)),
            },
        )
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 2

    state["preflight"] = "ok"
    write_state(state, run_state_path)
    append_jsonl(
        EVENTS_PATH,
        {
            "type": "run_started",
            "time": now_iso(),
            "run_id": run_id,
            "max_hours": args.max_hours,
            "max_rounds": args.max_rounds,
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "strategy_mode": state["strategy_mode"],
        },
    )

    for round_index in range(max(args.max_rounds, 0)):
        remaining = int(deadline - time.monotonic())
        if remaining <= 120:
            state["stopped_reason"] = "time_budget_exhausted"
            append_jsonl(
                EVENTS_PATH,
                {
                    "type": "run_stopping",
                    "time": now_iso(),
                    "run_id": run_id,
                    "reason": state["stopped_reason"],
                    "remaining_seconds": remaining,
                },
            )
            break

        prompt, agents_hash = build_round_prompt(
            round_index,
            args.max_rounds,
            args.extra_instruction,
            args.remote_host,
            args.remote_project_dir,
            args.remote_python,
        )
        round_prefix = LOOP_LOG_DIR / f"{run_id}_round_{round_index + 1:03d}"
        prompt_path = round_prefix.with_suffix(".prompt.md")
        stdout_path = round_prefix.with_suffix(".stdout.log")
        last_message_path = round_prefix.with_suffix(".last_message.md")
        prompt_path.write_text(prompt, encoding="utf-8")

        command = build_codex_command(args, last_message_path)
        round_state: dict[str, Any] = {
            "round": round_index + 1,
            "started_at": now_iso(),
            "status": "started",
            "agents_sha256": agents_hash,
            "prompt_path": str(prompt_path.relative_to(PROJECT_ROOT)),
            "stdout_path": str(stdout_path.relative_to(PROJECT_ROOT)),
            "last_message_path": str(last_message_path.relative_to(PROJECT_ROOT)),
            "command": command,
        }
        state["rounds"].append(round_state)
        write_state(state, run_state_path)
        append_jsonl(
            EVENTS_PATH,
            {
                "type": "round_started",
                "time": round_state["started_at"],
                "run_id": run_id,
                "round": round_state["round"],
                "prompt_path": round_state["prompt_path"],
                "stdout_path": round_state["stdout_path"],
                "last_message_path": round_state["last_message_path"],
                "agents_sha256": agents_hash,
            },
        )

        print(f"[{now_iso()}] Starting Codex round {round_index + 1}/{args.max_rounds}")
        if args.dry_run:
            stdout_path.write_text(f"[{now_iso()}] DRY RUN: codex exec not started.\n", encoding="utf-8")
            last_message_path.write_text("DRY RUN\nTG_CONTINUE\n", encoding="utf-8")
            round_state.update({"status": "dry_run", "signal": "TG_CONTINUE"})
            write_state(state, run_state_path)
            append_jsonl(
                EVENTS_PATH,
                {
                    "type": "round_finished",
                    "time": now_iso(),
                    "run_id": run_id,
                    "round": round_state["round"],
                    "status": round_state["status"],
                    "signal": round_state["signal"],
                },
            )
            continue

        timeout = min(int(args.round_timeout_minutes * 60), remaining)
        try:
            returncode = stream_process(command, prompt, stdout_path, timeout)
        except Exception as exc:
            round_state.update(
                {
                    "finished_at": now_iso(),
                    "status": "exception",
                    "returncode": None,
                    "signal": None,
                    "exception": repr(exc),
                    "traceback": compact_text(traceback.format_exc(), 5000),
                }
            )
            state["stopped_reason"] = "harness_exception"
            write_state(state, run_state_path)
            append_jsonl(
                EVENTS_PATH,
                {
                    "type": "round_exception",
                    "time": round_state["finished_at"],
                    "run_id": run_id,
                    "round": round_state["round"],
                    "exception": repr(exc),
                    "stdout_path": round_state["stdout_path"],
                },
            )
            break
        last_message = read_text(last_message_path, "")
        signal = parse_signal(last_message)
        round_state.update(
            {
                "finished_at": now_iso(),
                "status": "finished",
                "returncode": returncode,
                "signal": signal,
                "last_message_excerpt": compact_text(last_message, 2400),
            }
        )
        write_state(state, run_state_path)
        append_jsonl(
            EVENTS_PATH,
            {
                "type": "round_finished",
                "time": round_state["finished_at"],
                "run_id": run_id,
                "round": round_state["round"],
                "status": round_state["status"],
                "returncode": returncode,
                "signal": signal,
                "stdout_path": round_state["stdout_path"],
                "last_message_path": round_state["last_message_path"],
            },
        )

        if returncode != 0:
            state["stopped_reason"] = f"codex_returncode_{returncode}"
            break
        if signal == "TG_CONVERGED":
            state["stopped_reason"] = "converged"
            break
        if signal == "TG_BLOCKED":
            state["stopped_reason"] = "blocked"
            break
        if signal == "TG_CONTINUE":
            continue
        if args.continue_on_no_signal:
            continue
        state["stopped_reason"] = "missing_terminal_signal"
        break

    state.setdefault("stopped_reason", "max_rounds_complete")
    state["status"] = "finished"
    state["finished_at"] = now_iso()
    write_state(state, run_state_path)
    append_jsonl(
        EVENTS_PATH,
        {
            "type": "run_finished",
            "time": state["finished_at"],
            "run_id": run_id,
            "status": state["status"],
            "stopped_reason": state["stopped_reason"],
            "round_count": len(state["rounds"]),
            "run_state_path": str(run_state_path.relative_to(PROJECT_ROOT)),
        },
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
