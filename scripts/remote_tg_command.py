#!/usr/bin/env python3
"""Run a command inside the remote Tgprediction directory.

This helper is for the local Codex loop. It lets a local agent run experiments
on the server without requiring Codex CLI on the server.

Credentials are read from environment variables or SSH keys; do not hard-code
passwords in the repository.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_HOST = "100.64.0.4"
DEFAULT_USER = "sheng-xiang"
DEFAULT_REMOTE_DIR = "~/Tgprediction"
DEFAULT_REMOTE_PYTHON = "/home/sheng-xiang/miniconda3/envs/llm4graphgen/bin/python"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a command under remote ~/Tgprediction.")
    parser.add_argument("command", help="Command to run after cd ~/Tgprediction.")
    parser.add_argument("--host", default=os.environ.get("TG_REMOTE_HOST", DEFAULT_HOST))
    parser.add_argument("--user", default=os.environ.get("TG_REMOTE_USER", DEFAULT_USER))
    parser.add_argument("--remote-dir", default=os.environ.get("TG_REMOTE_DIR", DEFAULT_REMOTE_DIR))
    parser.add_argument("--password-env", default="TG_REMOTE_PASSWORD")
    parser.add_argument("--python", default=os.environ.get("TG_REMOTE_PYTHON", DEFAULT_REMOTE_PYTHON))
    parser.add_argument("--timeout", type=int, default=0, help="Timeout seconds; 0 means no helper-level timeout.")
    parser.add_argument("--use-paramiko", action="store_true", help="Use Paramiko. Useful for password env auth.")
    return parser


def remote_shell(command: str, remote_dir: str) -> str:
    if remote_dir == "~" or remote_dir.startswith("~/"):
        cd_target = "$HOME" + remote_dir[1:]
    else:
        cd_target = shlex.quote(remote_dir)
    return f"cd {cd_target} && {command}"


def run_with_ssh(args: argparse.Namespace) -> int:
    target = f"{args.user}@{args.host}"
    cmd = ["ssh", target, remote_shell(args.command, args.remote_dir)]
    completed = subprocess.run(
        cmd,
        text=True,
        timeout=args.timeout or None,
    )
    return int(completed.returncode)


def run_with_paramiko(args: argparse.Namespace) -> int:
    try:
        import paramiko
    except ImportError as exc:
        raise SystemExit("Paramiko is not installed. Install it or use SSH key auth with plain ssh.") from exc

    password = os.environ.get(args.password_env)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        args.host,
        username=args.user,
        password=password,
        timeout=20,
        look_for_keys=True,
        allow_agent=True,
    )
    try:
        stdin, stdout, stderr = client.exec_command(
            remote_shell(args.command, args.remote_dir),
            timeout=args.timeout or None,
            get_pty=False,
        )
        for line in stdout:
            print(line, end="")
        err = stderr.read().decode("utf-8", errors="replace")
        if err:
            print(err, file=sys.stderr, end="")
        return int(stdout.channel.recv_exit_status())
    finally:
        client.close()


def main() -> int:
    args = build_parser().parse_args()
    if not args.command.strip():
        raise SystemExit("empty command")
    if args.use_paramiko or os.environ.get(args.password_env):
        return run_with_paramiko(args)
    return run_with_ssh(args)


if __name__ == "__main__":
    raise SystemExit(main())
