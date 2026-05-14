#!/usr/bin/env python3
"""Fetch files or directories from the remote Tgprediction server via SFTP.

Companion to ``scripts/remote_tg_command.py``. Credentials are read from the
``TG_REMOTE_PASSWORD`` environment variable (or SSH keys/agent if available);
do not hard-code passwords.

Examples
--------

Fetch a single file::

    python scripts/remote_tg_fetch.py file \
        results/universal_single_regressor/exp56_*/summary.json \
        results/universal_single_regressor/exp56/summary.json

Fetch a directory recursively (model weights excluded by default)::

    python scripts/remote_tg_fetch.py dir \
        results/universal_router_fulltest \
        results/universal_router_fulltest

Fetch with extra exclude patterns::

    python scripts/remote_tg_fetch.py dir \
        results/universal_single_regressor/exp56_homo_local_fox_pred_delta_nonhomo_cal_lowfox_shrink_nopure \
        results/universal_single_regressor/exp56_homo_local_fox_pred_delta_nonhomo_cal_lowfox_shrink_nopure \
        --exclude '*.joblib' --exclude '*.parquet'
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import posixpath
import sys
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_HOST = "100.64.0.4"
DEFAULT_USER = "sheng-xiang"
DEFAULT_REMOTE_DIR = "Tgprediction"  # Resolved against the user's home directory.
DEFAULT_EXCLUDES = ("*.joblib", "__pycache__", "*.pyc", "*.pyo")


def _expand_remote(base_home: str, remote_dir: str, relative: str) -> str:
    """Resolve ``relative`` against ``base_home/remote_dir`` using POSIX paths."""
    if relative.startswith("/"):
        return relative
    return posixpath.join(base_home, remote_dir, relative)


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def _connect(args: argparse.Namespace):
    try:
        import paramiko
    except ImportError as exc:
        raise SystemExit("paramiko is not installed; pip install paramiko") from exc

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
    sftp = client.open_sftp()
    stdin, stdout, stderr = client.exec_command("echo $HOME")
    home = stdout.read().decode("utf-8", "replace").strip() or f"/home/{args.user}"
    return client, sftp, home


def _fetch_file(sftp, remote_path: str, local_path: Path) -> int:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        info = sftp.stat(remote_path)
    except FileNotFoundError:
        print(f"[skip] missing on remote: {remote_path}", file=sys.stderr)
        return 0
    sftp.get(remote_path, str(local_path))
    size = int(info.st_size or 0)
    print(f"[get] {remote_path} -> {local_path} ({size} bytes)")
    return size


def _walk_remote(sftp, remote_dir: str):
    stack = [remote_dir]
    while stack:
        current = stack.pop()
        for entry in sftp.listdir_attr(current):
            full = posixpath.join(current, entry.filename)
            if (entry.st_mode or 0) & 0o040000:
                stack.append(full)
            else:
                yield full, entry.filename, int(entry.st_size or 0)


def fetch_file(args: argparse.Namespace) -> int:
    client, sftp, home = _connect(args)
    try:
        remote = _expand_remote(home, args.remote_dir, args.remote)
        _fetch_file(sftp, remote, Path(args.local))
    finally:
        sftp.close()
        client.close()
    return 0


def fetch_dir(args: argparse.Namespace) -> int:
    client, sftp, home = _connect(args)
    excludes = tuple(args.exclude) if args.exclude else DEFAULT_EXCLUDES
    try:
        remote_root = _expand_remote(home, args.remote_dir, args.remote)
        local_root = Path(args.local)
        total_files = 0
        total_bytes = 0
        for full, name, size in _walk_remote(sftp, remote_root):
            if _matches_any(name, excludes):
                continue
            relative = posixpath.relpath(full, remote_root)
            target = local_root.joinpath(*relative.split(posixpath.sep))
            written = _fetch_file(sftp, full, target)
            total_files += 1 if written else 0
            total_bytes += written
        print(f"[done] {total_files} files, {total_bytes} bytes -> {local_root}")
    finally:
        sftp.close()
        client.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch files from remote Tgprediction via SFTP.")
    parser.add_argument("--host", default=os.environ.get("TG_REMOTE_HOST", DEFAULT_HOST))
    parser.add_argument("--user", default=os.environ.get("TG_REMOTE_USER", DEFAULT_USER))
    parser.add_argument("--remote-dir", default=os.environ.get("TG_REMOTE_DIR", DEFAULT_REMOTE_DIR))
    parser.add_argument("--password-env", default="TG_REMOTE_PASSWORD")
    sub = parser.add_subparsers(dest="mode", required=True)

    file_p = sub.add_parser("file", help="Fetch a single file.")
    file_p.add_argument("remote", help="Path relative to ~/Tgprediction (or absolute).")
    file_p.add_argument("local", help="Local destination path.")
    file_p.set_defaults(func=fetch_file)

    dir_p = sub.add_parser("dir", help="Fetch a directory recursively.")
    dir_p.add_argument("remote", help="Directory path relative to ~/Tgprediction (or absolute).")
    dir_p.add_argument("local", help="Local destination directory.")
    dir_p.add_argument(
        "--exclude",
        action="append",
        help=f"Glob pattern to skip (filename only). Defaults: {', '.join(DEFAULT_EXCLUDES)}",
    )
    dir_p.set_defaults(func=fetch_dir)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
