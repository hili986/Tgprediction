"""
Create a clean PoLyInfo copolymer Tg CSV by removing high-conflict duplicates.

The conflict test needs normalized composition (``w1_used``), so it uses the
details CSV produced by ``evaluate_polyinfo_copolymer_physics.py`` to identify
source rows in ``data/external/polyinfo_copolymer_tg.csv``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _default_path(*parts: str) -> str:
    return str(PROJECT_ROOT.joinpath(*parts))


def _load_conflict_rows(
    details_csv: Path,
    threshold_k: float,
    include_pure_endpoints: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    details = pd.read_csv(details_csv)
    for col in ["source_row_index", "w1_used", "Tg_C"]:
        if col in details.columns:
            details[col] = pd.to_numeric(details[col], errors="coerce")

    evaluated = details[
        details["status"].astype(str).eq("usable")
        & np.isfinite(details["source_row_index"])
        & np.isfinite(details["w1_used"])
        & np.isfinite(details["Tg_C"])
    ].copy()
    if not include_pure_endpoints:
        evaluated = evaluated[
            (evaluated["w1_used"].astype(float) > 1e-9)
            & (evaluated["w1_used"].astype(float) < 1.0 - 1e-9)
        ].copy()

    grouped = (
        evaluated.groupby(["COID", "w1_used"], dropna=False)["Tg_C"]
        .agg(["size", "mean", "std", "min", "max"])
        .reset_index()
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    conflict_groups = grouped[(grouped["size"] > 1) & (grouped["std"] > threshold_k)].copy()

    if conflict_groups.empty:
        empty = evaluated.iloc[0:0].copy()
        return empty, conflict_groups

    conflict_keys = {
        (str(row.COID), float(row.w1_used))
        for row in conflict_groups.itertuples(index=False)
    }
    conflict_rows = evaluated[
        evaluated.apply(lambda row: (str(row["COID"]), float(row["w1_used"])) in conflict_keys, axis=1)
    ].copy()
    conflict_rows = conflict_rows.merge(
        conflict_groups.rename(
            columns={
                "size": "conflict_group_size",
                "mean": "conflict_group_tg_mean_c",
                "std": "conflict_group_tg_std_k",
                "min": "conflict_group_tg_min_c",
                "max": "conflict_group_tg_max_c",
            }
        ),
        on=["COID", "w1_used"],
        how="left",
    )
    return conflict_rows, conflict_groups


def filter_polyinfo_conflicts(
    raw_csv: Path,
    details_csv: Path,
    output_csv: Path,
    removed_csv: Path,
    summary_json: Path,
    threshold_k: float,
    include_pure_endpoints: bool,
) -> dict[str, object]:
    raw = pd.read_csv(raw_csv)
    conflict_rows, conflict_groups = _load_conflict_rows(
        details_csv,
        threshold_k=threshold_k,
        include_pure_endpoints=include_pure_endpoints,
    )

    remove_indices = sorted(
        {
            int(idx)
            for idx in conflict_rows.get("source_row_index", pd.Series(dtype=float)).dropna().tolist()
        }
    )
    missing = [idx for idx in remove_indices if idx < 0 or idx >= len(raw)]
    if missing:
        raise ValueError(f"Conflict source_row_index values outside raw CSV range: {missing[:10]}")

    clean = raw.drop(index=remove_indices).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_csv, index=False, encoding="utf-8-sig")

    removed_csv.parent.mkdir(parents=True, exist_ok=True)
    removed = raw.iloc[remove_indices].copy() if remove_indices else raw.iloc[0:0].copy()
    if not conflict_rows.empty:
        keep_cols = [
            "source_row_index",
            "w1_used",
            "Tg_C",
            "conflict_group_size",
            "conflict_group_tg_mean_c",
            "conflict_group_tg_std_k",
            "conflict_group_tg_min_c",
            "conflict_group_tg_max_c",
        ]
        audit = conflict_rows[[col for col in keep_cols if col in conflict_rows.columns]].copy()
        audit["source_row_index"] = audit["source_row_index"].astype(int)
        removed = removed.reset_index(drop=False).rename(columns={"index": "source_row_index"})
        removed = removed.merge(audit, on="source_row_index", how="left", suffixes=("", "_normalized"))
    removed.to_csv(removed_csv, index=False, encoding="utf-8-sig")

    summary = {
        "raw_csv": str(raw_csv),
        "details_csv": str(details_csv),
        "output_csv": str(output_csv),
        "removed_csv": str(removed_csv),
        "threshold_k": threshold_k,
        "include_pure_endpoints": include_pure_endpoints,
        "n_raw": int(len(raw)),
        "n_clean": int(len(clean)),
        "n_removed_rows": int(len(remove_indices)),
        "removed_source_row_indices": remove_indices,
        "n_conflict_groups": int(len(conflict_groups)),
        "conflict_groups": conflict_groups.to_dict(orient="records"),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remove high-conflict PoLyInfo duplicate rows.")
    parser.add_argument("--raw-csv", default=_default_path("data", "external", "polyinfo_copolymer_tg.csv"))
    parser.add_argument(
        "--details-csv",
        default=_default_path("results", "copolymer_residual_model", "polyinfo_physics_details.csv"),
    )
    parser.add_argument(
        "--output-csv",
        default=_default_path("data", "external", "polyinfo_copolymer_tg_clean.csv"),
    )
    parser.add_argument(
        "--removed-csv",
        default=_default_path("results", "copolymer_residual_model", "polyinfo_conflict_removed_rows.csv"),
    )
    parser.add_argument(
        "--summary-json",
        default=_default_path("results", "copolymer_residual_model", "polyinfo_conflict_filter_summary.json"),
    )
    parser.add_argument("--threshold-k", type=float, default=10.0)
    parser.add_argument("--include-pure-endpoints", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = filter_polyinfo_conflicts(
        raw_csv=Path(args.raw_csv),
        details_csv=Path(args.details_csv),
        output_csv=Path(args.output_csv),
        removed_csv=Path(args.removed_csv),
        summary_json=Path(args.summary_json),
        threshold_k=args.threshold_k,
        include_pure_endpoints=args.include_pure_endpoints,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved clean CSV: {args.output_csv}")
    print(f"Saved removed audit: {args.removed_csv}")
    print(f"Saved summary: {args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
