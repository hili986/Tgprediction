"""
Audit whether copolymer components can reuse precomputed homopolymer features.

The report is intentionally cheap: it reads the unified 7k SMILES table and a
copolymer CSV, then checks exact and RDKit-canonical component matches. It does
not load polyBERT, GNN, or TabPFN.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.predict_tg_tabpfn_186d import _canonical_repeat_unit_key
from src.ml.copolymer_tg_model import CopolymerRecord, read_copolymer_records


def _default_path(*parts: str) -> str:
    return str(PROJECT_ROOT.joinpath(*parts))


def _build_canonical_map(smiles_values: Iterable[str]) -> tuple[dict[str, str], set[str]]:
    canonical_to_smiles: dict[str, str] = {}
    duplicate_keys: set[str] = set()
    for value in smiles_values:
        smiles = str(value).strip()
        if not smiles:
            continue
        key = _canonical_repeat_unit_key(smiles)
        if key is None:
            continue
        if key in canonical_to_smiles and canonical_to_smiles[key] != smiles:
            duplicate_keys.add(key)
            continue
        canonical_to_smiles[key] = smiles
    return canonical_to_smiles, duplicate_keys


def build_reuse_report(
    records: Sequence[CopolymerRecord],
    precomputed_smiles: Iterable[str],
) -> pd.DataFrame:
    exact_smiles = {str(value).strip() for value in precomputed_smiles if str(value).strip()}
    canonical_to_smiles, duplicate_keys = _build_canonical_map(exact_smiles)
    component_counts = Counter(
        str(component).strip()
        for record in records
        for component in record.components
        if str(component).strip()
    )

    rows: List[dict[str, object]] = []
    for component, count in sorted(component_counts.items()):
        canonical_key = _canonical_repeat_unit_key(component)
        mapped_smiles = ""
        match_type = "miss"
        if component in exact_smiles:
            mapped_smiles = component
            match_type = "exact"
        elif canonical_key is None:
            match_type = "invalid"
        elif canonical_key in canonical_to_smiles:
            mapped_smiles = canonical_to_smiles[canonical_key]
            match_type = "canonical"

        rows.append(
            {
                "component_smiles": component,
                "canonical_key": canonical_key or "",
                "mapped_smiles": mapped_smiles,
                "match_type": match_type,
                "is_canonical_duplicate_key": bool(
                    canonical_key is not None and canonical_key in duplicate_keys
                ),
                "record_count": int(count),
            }
        )

    columns = [
        "component_smiles",
        "canonical_key",
        "mapped_smiles",
        "match_type",
        "is_canonical_duplicate_key",
        "record_count",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit exact/canonical reuse of precomputed 7k component features."
    )
    parser.add_argument(
        "--real-csv",
        default=_default_path("data", "external", "polyinfo_copolymer_tg.csv"),
        help="Real copolymer CSV parsed from PoLyInfo/raw manual data.",
    )
    parser.add_argument(
        "--data-path",
        default=_default_path("data", "unified_tg.parquet"),
        help="Unified homopolymer dataset containing the precomputed SMILES universe.",
    )
    parser.add_argument(
        "--out",
        default=_default_path(
            "results",
            "copolymer_residual_model",
            "component_reuse_map.csv",
        ),
        help="Output CSV mapping table.",
    )
    parser.add_argument("--real-target", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    real_csv = Path(args.real_csv)
    data_path = Path(args.data_path)
    out_path = Path(args.out)

    records = read_copolymer_records(real_csv, target_column=args.real_target)
    unified = pd.read_parquet(data_path, columns=["smiles"])
    report = build_reuse_report(records, unified["smiles"].astype(str).tolist())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_path, index=False, encoding="utf-8-sig")

    counts = report["match_type"].value_counts().to_dict()
    print(f"Records: {len(records)}")
    print(f"Unique components: {len(report)}")
    print(
        "Reuse summary: "
        + ", ".join(f"{key}={counts.get(key, 0)}" for key in ["exact", "canonical", "miss", "invalid"])
    )
    if "miss" in counts:
        misses = report.loc[report["match_type"] == "miss", "component_smiles"].tolist()
        print("Misses:")
        for smiles in misses:
            print(f"  {smiles}")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
