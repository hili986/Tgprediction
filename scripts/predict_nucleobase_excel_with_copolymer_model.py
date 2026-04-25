"""
Predict nucleobase-functionalized acrylic Tg rows with the copolymer residual model.

This is a cross-domain evaluation helper for Nucleobase_Tg_Data_Compilation.xlsx.
The source workbook does not contain repeat-unit SMILES, so this script maps only
clear nBA/acrylic A/T/C/G random-copolymer and homopolymer rows to approximate
repeat-unit SMILES. Block copolymers, blends, styrenic analogs, and protected
derivatives are skipped by default.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_virtual_copolymer_dataset import build_inference_paths
from scripts.predict_tg_tabpfn_186d import BestTgPredictor, _require_paths
from src.ml.copolymer_tg_model import (
    CopolymerRecord,
    build_feature_matrix,
    predict_with_residual,
    regression_metrics,
)


NBA_REPEAT_SMILES = "*CC(*)C(=O)OCCCC"
NUCLEOBASE_ACRYLATE_REPEAT_SMILES = {
    "A": "*CC(*)C(=O)OCCn1cnc2c(N)ncnc12",
    "T": "*CC(*)C(=O)OCCn1cc(C)c(=O)[nH]c1=O",
    "C": "*CC(*)C(=O)OCCn1ccc(N)nc1=O",
    "G": "*CC(*)C(=O)OCCn1cnc2c(=O)[nH]c(N)nc12",
}
MONOMER_TO_BASE = {
    "acrylic adenine": "A",
    "acrylic thymine": "T",
    "cya": "C",
    "gua": "G",
}


@dataclass(frozen=True)
class MappedRow:
    record: CopolymerRecord
    status: str
    reason: str
    mapping_note: str


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _safe_float(value: object) -> Optional[float]:
    text = _clean_text(value)
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _target_tg_k(row: pd.Series) -> Optional[float]:
    tg_c = _safe_float(row.get("Tg_C"))
    return None if tg_c is None else tg_c + 273.15


def map_nucleobase_row(row: pd.Series, row_index: int) -> MappedRow:
    architecture = _clean_text(row.get("Architecture")).lower()
    monomer = _clean_text(row.get("Monomer_or_Block")).lower()
    polymer_system = _clean_text(row.get("Polymer_System")).lower()
    target_tg_k = _target_tg_k(row)
    metadata = {str(key): row.get(key) for key in row.index}
    metadata["excel_row_index"] = row_index

    if "styrenic" in architecture or "styrenic" in monomer:
        return MappedRow(None, "skipped", "styrenic analog has no repeat-unit mapping", "")
    if "block" in architecture or "blend" in architecture:
        return MappedRow(None, "skipped", "block/blend rows lack usable composition for this model", "")
    if "ucya" in monomer or "ucya" in polymer_system or "derivative" in _clean_text(row.get("Nucleobase")).lower():
        return MappedRow(None, "skipped", "protected/ureido cytosine derivative is not mapped to natural C", "")

    if monomer == "nba" or "poly(n-butyl acrylate)" in polymer_system:
        record = CopolymerRecord(
            components=(NBA_REPEAT_SMILES,),
            weights=(1.0,),
            architecture="random",
            target_tg_k=target_tg_k,
            metadata=metadata,
        )
        return MappedRow(record, "mapped", "nBA homopolymer baseline", "exact nBA repeat unit")

    base = MONOMER_TO_BASE.get(monomer)
    if base is None:
        return MappedRow(None, "skipped", f"no repeat-unit mapping for monomer '{monomer}'", "")

    base_smiles = NUCLEOBASE_ACRYLATE_REPEAT_SMILES[base]
    if "homopolymer" in architecture:
        record = CopolymerRecord(
            components=(base_smiles,),
            weights=(1.0,),
            architecture="random",
            target_tg_k=target_tg_k,
            metadata=metadata,
        )
        return MappedRow(
            record,
            "mapped",
            f"{base} acrylate homopolymer",
            "approximate 2-carbon acrylate nucleobase linker",
        )

    polymer_mol_pct = _safe_float(row.get("Polymer_mol_pct"))
    if polymer_mol_pct is None:
        return MappedRow(None, "skipped", "missing Polymer_mol_pct for random copolymer", "")
    base_weight = min(max(polymer_mol_pct / 100.0, 0.0), 1.0)
    if base_weight <= 0.0:
        return MappedRow(None, "skipped", "non-positive nucleobase composition", "")

    record = CopolymerRecord(
        components=(NBA_REPEAT_SMILES, base_smiles),
        weights=(1.0 - base_weight, base_weight),
        architecture="random",
        target_tg_k=target_tg_k,
        metadata=metadata,
    )
    return MappedRow(
        record,
        "mapped",
        f"nBA-co-{base} random copolymer",
        "approximate 2-carbon acrylate nucleobase linker",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict nucleobase workbook Tg rows with the trained copolymer model."
    )
    parser.add_argument("--input-xlsx", default=str(PROJECT_ROOT / "Nucleobase_Tg_Data_Compilation.xlsx"))
    parser.add_argument(
        "--model-path",
        default=str(
            PROJECT_ROOT
            / "results"
            / "copolymer_residual_model"
            / "polyinfo_223_from_bicerano_virtual"
            / "copolymer_tg_residual_model.joblib"
        ),
    )
    parser.add_argument(
        "--output-csv",
        default=str(
            PROJECT_ROOT
            / "results"
            / "copolymer_residual_model"
            / "nucleobase_excel_predictions.csv"
        ),
    )
    parser.add_argument(
        "--metrics-json",
        default=str(
            PROJECT_ROOT
            / "results"
            / "copolymer_residual_model"
            / "nucleobase_excel_metrics.json"
        ),
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--chain-physics-confs", type=int, default=10)
    parser.add_argument("--polybert-batch-size", type=int, default=64)
    parser.add_argument("--data-path", type=str, default=str(PROJECT_ROOT / "data" / "unified_tg.parquet"))
    parser.add_argument("--phyc-cache", type=str, default=str(PROJECT_ROOT / "data" / "feature_matrix_PHY-C.parquet"))
    parser.add_argument("--gnn-cache", type=str, default=str(PROJECT_ROOT / "data" / "gnn_embeddings_64d.parquet"))
    parser.add_argument("--pbert-cache", type=str, default=str(PROJECT_ROOT / "data" / "polybert_embeddings.parquet"))
    parser.add_argument(
        "--chain-physics-cache",
        type=str,
        default=str(PROJECT_ROOT / "data" / "chain_physics_features.parquet"),
    )
    parser.add_argument("--polybert-model-dir", type=str, default=str(PROJECT_ROOT / "data" / "polybert_model"))
    parser.add_argument("--gnn-checkpoint", type=str, default=str(PROJECT_ROOT / "checkpoints" / "gnn_pretrained.pt"))
    return parser


def _result_base(row: pd.Series, mapped: Optional[MappedRow]) -> Dict[str, object]:
    return {
        "Data_ID": row.get("Data_ID"),
        "Source_ID": row.get("Source_ID"),
        "Nucleobase": row.get("Nucleobase"),
        "Sample_Name": row.get("Sample_Name"),
        "Architecture": row.get("Architecture"),
        "Polymer_System": row.get("Polymer_System"),
        "Monomer_or_Block": row.get("Monomer_or_Block"),
        "Polymer_mol_pct": row.get("Polymer_mol_pct"),
        "Tg_C_actual": row.get("Tg_C"),
        "status": "" if mapped is None else mapped.status,
        "reason": "" if mapped is None else mapped.reason,
        "mapping_note": "" if mapped is None else mapped.mapping_note,
    }


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    input_xlsx = Path(args.input_xlsx)
    model_path = Path(args.model_path)
    output_csv = Path(args.output_csv)
    metrics_json = Path(args.metrics_json)

    frame = pd.read_excel(input_xlsx, sheet_name="Quantitative_Tg_Data")
    mapped_rows = [map_nucleobase_row(row, int(idx)) for idx, row in frame.iterrows()]
    artifact = joblib.load(model_path)

    paths = build_inference_paths(args)
    _require_paths(paths)
    predictor = BestTgPredictor(
        paths=paths,
        device=args.device,
        chain_physics_confs=args.chain_physics_confs,
        polybert_batch_size=args.polybert_batch_size,
    )
    predictor.fit()

    residual_names = artifact.get("residual_feature_names") or []
    include_teacher = any(str(name).startswith("teacher_") for name in residual_names)
    output_rows = []
    y_true = []
    y_pred = []

    for source_index, (row, mapped) in enumerate(zip(frame.to_dict(orient="records"), mapped_rows)):
        source_row = pd.Series(row)
        result = _result_base(source_row, mapped)
        if mapped.record is None:
            output_rows.append(result)
            continue
        try:
            base_features = build_feature_matrix(
                [mapped.record],
                predictor,
                include_teacher_scalars=False,
            )
            if base_features.feature_names != artifact["base_feature_names"]:
                raise ValueError("base feature names do not match trained artifact")
            base_pred = np.asarray(artifact["base_model"].predict(base_features.X), dtype=float)
            final_pred = base_pred
            if artifact.get("residual_model") is not None:
                residual_features = build_feature_matrix(
                    [mapped.record],
                    predictor,
                    include_teacher_scalars=include_teacher,
                )
                if residual_features.feature_names != residual_names:
                    raise ValueError("residual feature names do not match trained artifact")
                final_pred = predict_with_residual(
                    base_pred,
                    artifact["residual_model"],
                    residual_features.X,
                )

            actual_k = mapped.record.target_tg_k
            pred_k = float(final_pred[0])
            result.update(
                {
                    "status": "predicted",
                    "components_serialized": "|".join(mapped.record.components),
                    "weights_serialized": "|".join(f"{weight:.8g}" for weight in mapped.record.weights),
                    "base_pred_tg_k": float(base_pred[0]),
                    "base_pred_tg_c": float(base_pred[0] - 273.15),
                    "pred_tg_k": pred_k,
                    "pred_tg_c": pred_k - 273.15,
                    "error_k": None if actual_k is None else pred_k - actual_k,
                    "abs_error_k": None if actual_k is None else abs(pred_k - actual_k),
                }
            )
            if actual_k is not None and np.isfinite(actual_k):
                y_true.append(actual_k)
                y_pred.append(pred_k)
        except Exception as exc:
            result["status"] = "error"
            result["reason"] = str(exc)
        output_rows.append(result)

    output = pd.DataFrame(output_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False, encoding="utf-8-sig")

    predicted = output.loc[output["status"] == "predicted"].copy()
    metrics: Dict[str, object] = {
        "input_xlsx": str(input_xlsx),
        "model_path": str(model_path),
        "output_csv": str(output_csv),
        "n_rows": int(len(output)),
        "n_predicted": int(len(predicted)),
        "status_counts": output["status"].value_counts(dropna=False).to_dict(),
        "mapping_note": (
            "Only clear nBA/acrylic A/T/C/G random-copolymer and homopolymer rows "
            "were mapped. Nucleobase repeat units use approximate 2-carbon acrylate linkers."
        ),
    }
    if y_true:
        metrics["overall"] = regression_metrics(np.asarray(y_true), np.asarray(y_pred))
        by_base: Dict[str, object] = {}
        for base, group in predicted.groupby("Nucleobase", dropna=False):
            valid = group[["Tg_C_actual", "pred_tg_c"]].dropna()
            if len(valid) == 0:
                continue
            by_base[str(base)] = regression_metrics(
                valid["Tg_C_actual"].to_numpy(dtype=float) + 273.15,
                valid["pred_tg_c"].to_numpy(dtype=float) + 273.15,
            )
        metrics["by_nucleobase"] = by_base

    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved predictions: {output_csv}")
    print(f"Saved metrics: {metrics_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
