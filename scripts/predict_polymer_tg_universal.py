"""
Universal polymer Tg prediction router.

Routes by task and available evidence:

1. Homopolymer:
   BestTgPredictor / 186d TabPFN, loaded lazily once.
2. Binary random copolymer:
   Prefer same-system local residual calibration when matching clean PoLyInfo
   calibration rows exist. Otherwise use global Linear-Fox calibration. If no
   endpoint data exists, optionally fall back to BestTg descriptor mixing.
3. Block copolymer:
   Report a Fox-style miscible proxy plus component Tg window; true multiple-Tg
   phase behavior is not explicitly modeled.
4. Multi-component copolymer:
   Use endpoint Fox when endpoint Tg values are available; otherwise optionally
   fall back to BestTg weighted descriptor/embedding mixing.

The expensive BestTg model is never loaded unless a route actually needs it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.predict_tg_tabpfn_186d import (
    BestTgPredictor,
    InferencePaths,
    _canonical_repeat_unit_key,
    _parse_component_entries,
    _require_paths,
    _validate_repeat_unit_smiles,
)
from src.ml.copolymer_physics import fox_tg_k


SUPPORTED_ARCHITECTURES = {"random", "block"}


def _default_path(*parts: str) -> str:
    return str(PROJECT_ROOT.joinpath(*parts))


def _finite_float(value: object) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normalize_weights(weights: Sequence[float]) -> np.ndarray:
    arr = np.asarray(weights, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("weights must be a non-empty 1D sequence.")
    if not np.isfinite(arr).all() or np.any(arr < 0):
        raise ValueError("weights must be finite and non-negative.")
    total = float(arr.sum())
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value.")
    return arr / total


def _cell_to_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _normalise_repeat_unit_for_besttg(smiles: str) -> str:
    """Narrow compatibility fix for ethene-like PoLyInfo rows."""
    smi = str(smiles).strip()
    return "*CC*" if smi == "*C*" else smi


@dataclass
class EndpointValue:
    tg_k: float
    source: str


@dataclass
class CalibrationMatch:
    rows: pd.DataFrame
    match_type: str


class UniversalTgRouter:
    def __init__(
        self,
        unified_tg_path: Path,
        calibration_details_csv: Optional[Path],
        besttg_paths: InferencePaths,
        device: str = "cuda",
        chain_physics_confs: int = 50,
        use_besttg_fallback: bool = True,
        normalise_besttg_ethylene: bool = True,
        local_k: int = 3,
        local_power: float = 1.0,
        min_local_points: int = 2,
        local_calibration_mode: str = "row",
    ) -> None:
        self.unified_tg_path = unified_tg_path
        self.calibration_details_csv = calibration_details_csv
        self.besttg_paths = besttg_paths
        self.device = device
        self.chain_physics_confs = chain_physics_confs
        self.use_besttg_fallback = use_besttg_fallback
        self.normalise_besttg_ethylene = normalise_besttg_ethylene
        self.local_k = local_k
        self.local_power = local_power
        self.min_local_points = min_local_points
        self.local_calibration_mode = local_calibration_mode

        self._endpoint_lookup: Optional[Dict[str, EndpointValue]] = None
        self._calibration: Optional[pd.DataFrame] = None
        self._global_linear_model = None
        self._besttg: Optional[BestTgPredictor] = None

    def _load_endpoint_lookup(self) -> Dict[str, EndpointValue]:
        if self._endpoint_lookup is not None:
            return self._endpoint_lookup

        unified = pd.read_parquet(self.unified_tg_path)
        lookup: Dict[str, EndpointValue] = {}
        for _, row in unified.iterrows():
            tg_k = _finite_float(row.get("tg_k"))
            if tg_k is None:
                continue
            key = _canonical_repeat_unit_key(str(row.get("smiles", "")))
            if key is None:
                continue
            lookup[key] = EndpointValue(tg_k=tg_k, source=str(row.get("source", "unified_tg")))
        self._endpoint_lookup = lookup
        return lookup

    def _load_calibration(self) -> pd.DataFrame:
        if self._calibration is not None:
            return self._calibration
        if self.calibration_details_csv is None or not self.calibration_details_csv.exists():
            self._calibration = pd.DataFrame()
            return self._calibration

        frame = pd.read_csv(self.calibration_details_csv)
        for col in [
            "Tg_C",
            "w1_used",
            "endpoint_1_tg_c",
            "endpoint_2_tg_c",
            "fox_endpoint_tg_c",
        ]:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame[
            frame["status"].astype(str).eq("usable")
            & np.isfinite(frame["Tg_C"])
            & np.isfinite(frame["w1_used"])
            & (frame["w1_used"].astype(float) > 1e-9)
            & (frame["w1_used"].astype(float) < 1.0 - 1e-9)
            & frame.get("canonical_1", pd.Series(dtype=str)).astype(str).ne("")
            & frame.get("canonical_2", pd.Series(dtype=str)).astype(str).ne("")
        ].copy()
        self._calibration = frame.reset_index(drop=True)
        return self._calibration

    def _fit_global_linear_model(self):
        if self._global_linear_model is not None:
            return self._global_linear_model
        calibration = self._load_calibration()
        if calibration.empty:
            self._global_linear_model = None
            return None

        valid = calibration[
            np.isfinite(calibration["fox_endpoint_tg_c"])
            & np.isfinite(calibration["Tg_C"])
        ].copy()
        if len(valid) < 3:
            self._global_linear_model = None
            return None

        from sklearn.linear_model import RidgeCV

        model = RidgeCV(alphas=np.logspace(-6, 4, 24))
        model.fit(
            (valid["fox_endpoint_tg_c"].to_numpy(dtype=float) + 273.15).reshape(-1, 1),
            valid["Tg_C"].to_numpy(dtype=float) + 273.15,
        )
        self._global_linear_model = model
        return model

    def _load_besttg(self) -> BestTgPredictor:
        if self._besttg is not None:
            return self._besttg
        _require_paths(self.besttg_paths)
        predictor = BestTgPredictor(
            paths=self.besttg_paths,
            device=self.device,
            chain_physics_confs=self.chain_physics_confs,
        )
        predictor.fit()
        self._besttg = predictor
        return predictor

    def _endpoint_from_unified_or_besttg(self, smiles: str) -> Optional[EndpointValue]:
        key = _canonical_repeat_unit_key(smiles)
        lookup = self._load_endpoint_lookup()
        if key is not None and key in lookup:
            return lookup[key]

        if not self.use_besttg_fallback:
            return None

        query = _normalise_repeat_unit_for_besttg(smiles) if self.normalise_besttg_ethylene else smiles
        result = self._load_besttg().predict_homopolymer(query)
        return EndpointValue(float(result["tg_k_pred"]), "besttg_homopolymer_fallback")

    def _endpoint_values(self, smiles_list: Sequence[str]) -> tuple[list[Optional[EndpointValue]], list[str]]:
        endpoints: list[Optional[EndpointValue]] = []
        errors: list[str] = []
        for smiles in smiles_list:
            try:
                endpoints.append(self._endpoint_from_unified_or_besttg(smiles))
                if endpoints[-1] is None:
                    errors.append(f"missing endpoint Tg for {smiles}")
            except Exception as exc:
                endpoints.append(None)
                errors.append(f"endpoint route failed for {smiles}: {exc}")
        return endpoints, errors

    def _global_linear_fox_c(self, fox_c: float) -> Optional[float]:
        model = self._fit_global_linear_model()
        if model is None or not math.isfinite(fox_c):
            return None
        pred_k = float(model.predict(np.array([[fox_c + 273.15]], dtype=float))[0])
        return pred_k - 273.15

    def _calibration_match(
        self,
        key1: str,
        key2: str,
        system_id: Optional[str],
    ) -> Optional[CalibrationMatch]:
        calibration = self._load_calibration()
        if calibration.empty:
            return None

        ordered = calibration[
            calibration["canonical_1"].astype(str).eq(key1)
            & calibration["canonical_2"].astype(str).eq(key2)
        ].copy()
        ordered["query_w1_used"] = ordered["w1_used"].astype(float)
        ordered["match_orientation"] = "ordered"

        swapped = calibration[
            calibration["canonical_1"].astype(str).eq(key2)
            & calibration["canonical_2"].astype(str).eq(key1)
        ].copy()
        swapped["query_w1_used"] = 1.0 - swapped["w1_used"].astype(float)
        swapped["match_orientation"] = "swapped"

        pair = pd.concat([ordered, swapped], ignore_index=True)
        if system_id:
            exact_system = pair[pair["COID"].astype(str).eq(str(system_id))].copy()
            if len(exact_system) >= self.min_local_points:
                return CalibrationMatch(exact_system, "system_id_component_pair")
        if len(pair) >= self.min_local_points:
            return CalibrationMatch(pair, "component_pair")
        return None

    def _local_residual_correction(
        self,
        match: CalibrationMatch,
        query_w1: float,
        query_base_c: float,
    ) -> Optional[dict[str, object]]:
        rows = match.rows.copy()
        if "query_w1_used" not in rows.columns:
            rows["query_w1_used"] = rows["w1_used"].astype(float)

        global_model = self._fit_global_linear_model()
        if global_model is None:
            return None

        rows = rows[np.isfinite(rows["fox_endpoint_tg_c"]) & np.isfinite(rows["Tg_C"])].copy()
        if self.local_calibration_mode == "leave-composition":
            rows = rows[np.abs(rows["query_w1_used"].astype(float) - query_w1) > 1e-6].copy()
        if len(rows) < self.min_local_points:
            return None

        base_train_k = global_model.predict(
            (rows["fox_endpoint_tg_c"].to_numpy(dtype=float) + 273.15).reshape(-1, 1)
        )
        residual_c = rows["Tg_C"].to_numpy(dtype=float) - (base_train_k - 273.15)
        w_train = rows["query_w1_used"].to_numpy(dtype=float)
        mask = np.isfinite(w_train) & np.isfinite(residual_c)
        w_train = w_train[mask]
        residual_c = residual_c[mask]
        if len(residual_c) == 0:
            return None

        distance = np.abs(w_train - query_w1)
        exact = distance < 1e-12
        if bool(np.any(exact)):
            correction_c = float(np.mean(residual_c[exact]))
            used = int(np.sum(exact))
        else:
            order = np.argsort(distance)[: min(self.local_k, len(distance))]
            weights = 1.0 / (np.power(distance[order], self.local_power) + 1e-9)
            correction_c = float(np.sum(weights * residual_c[order]) / np.sum(weights))
            used = int(len(order))

        return {
            "tg_c_pred": float(query_base_c + correction_c),
            "correction_c": correction_c,
            "local_rows_total": int(len(match.rows)),
            "local_rows_used": used,
            "local_match_type": match.match_type,
            "local_calibration_mode": self.local_calibration_mode,
        }

    def predict_homopolymer(self, smiles: str) -> dict[str, object]:
        clean = _validate_repeat_unit_smiles(smiles)
        result = self._load_besttg().predict_homopolymer(clean)
        return {
            **result,
            "router_route": "homopolymer_besttg",
            "primary_method": "BestTgPredictor_186d_TabPFN",
            "confidence": "validated_homopolymer_model",
        }

    def predict_multicomponent(
        self,
        smiles_list: Sequence[str],
        weights: Sequence[float],
        architecture: str = "random",
        system_id: Optional[str] = None,
    ) -> dict[str, object]:
        if architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(f"Unsupported architecture '{architecture}'.")
        clean_smiles = [_validate_repeat_unit_smiles(s) for s in smiles_list]
        norm_weights = _normalize_weights(weights)
        if len(clean_smiles) != len(norm_weights):
            raise ValueError("number of SMILES and weights must match.")
        if len(clean_smiles) == 1:
            return self.predict_homopolymer(clean_smiles[0])

        endpoints, endpoint_errors = self._endpoint_values(clean_smiles)
        endpoint_tg_k = [ep.tg_k for ep in endpoints if ep is not None]
        endpoint_sources = [None if ep is None else ep.source for ep in endpoints]
        has_all_endpoints = len(endpoint_tg_k) == len(clean_smiles)

        fox_c = None
        component_window_c = None
        if has_all_endpoints:
            fox_k = fox_tg_k(endpoint_tg_k, norm_weights)
            fox_c = None if not math.isfinite(fox_k) else fox_k - 273.15
            component_window_c = [float(np.min(endpoint_tg_k) - 273.15), float(np.max(endpoint_tg_k) - 273.15)]

        result: dict[str, object] = {
            "mode": "binary_copolymer" if len(clean_smiles) == 2 else "multicomponent_copolymer",
            "architecture": architecture,
            "n_components": len(clean_smiles),
            "components": clean_smiles,
            "weights_normalized": [float(w) for w in norm_weights],
            "endpoint_sources": endpoint_sources,
            "endpoint_tg_c": [None if ep is None else float(ep.tg_k - 273.15) for ep in endpoints],
            "fox_endpoint_tg_c": None if fox_c is None else float(fox_c),
            "component_tg_window_c": component_window_c,
            "system_id": system_id or "",
        }

        if architecture == "block":
            if fox_c is not None:
                result.update(
                    {
                        "tg_c_pred": float(fox_c),
                        "tg_k_pred": float(fox_c + 273.15),
                        "router_route": "block_fox_proxy",
                        "primary_method": "fox_proxy_from_endpoint_tg",
                        "confidence": "proxy_block_or_phase_separated",
                        "warning": (
                            "Block copolymer route is a miscible-equivalent proxy; real samples may show "
                            "multiple Tg values or phase separation."
                        ),
                    }
                )
                return result
            return self._besttg_multicomponent_fallback(clean_smiles, norm_weights, architecture, result, endpoint_errors)

        if len(clean_smiles) == 2 and fox_c is not None:
            global_linear_c = self._global_linear_fox_c(float(fox_c))
            result["global_linear_fox_tg_c"] = global_linear_c
            query_base_c = float(global_linear_c if global_linear_c is not None else fox_c)

            key1 = _canonical_repeat_unit_key(clean_smiles[0])
            key2 = _canonical_repeat_unit_key(clean_smiles[1])
            if key1 is not None and key2 is not None:
                match = self._calibration_match(key1, key2, system_id)
                if match is not None:
                    local = self._local_residual_correction(match, float(norm_weights[0]), query_base_c)
                    if local is not None:
                        result.update(local)
                        result.update(
                            {
                                "tg_k_pred": float(result["tg_c_pred"] + 273.15),
                                "router_route": "binary_known_system_local_residual_idw",
                                "primary_method": "global_linear_fox_plus_same_system_residual_idw",
                                "confidence": "known_system_calibrated",
                            }
                        )
                        return result

            result.update(
                {
                    "tg_c_pred": query_base_c,
                    "tg_k_pred": query_base_c + 273.15,
                    "router_route": "binary_global_linear_fox" if global_linear_c is not None else "binary_fox_endpoint",
                    "primary_method": "global_linear_fox_calibration" if global_linear_c is not None else "fox_endpoint_tg",
                    "confidence": "new_system_extrapolation",
                }
            )
            return result

        if fox_c is not None:
            result.update(
                {
                    "tg_c_pred": float(fox_c),
                    "tg_k_pred": float(fox_c + 273.15),
                    "router_route": "multicomponent_fox_endpoint",
                    "primary_method": "fox_endpoint_tg",
                    "confidence": "multicomponent_proxy",
                }
            )
            return result

        return self._besttg_multicomponent_fallback(clean_smiles, norm_weights, architecture, result, endpoint_errors)

    def _besttg_multicomponent_fallback(
        self,
        clean_smiles: Sequence[str],
        norm_weights: Sequence[float],
        architecture: str,
        partial_result: dict[str, object],
        endpoint_errors: Sequence[str],
    ) -> dict[str, object]:
        if not self.use_besttg_fallback:
            raise ValueError("; ".join(endpoint_errors) or "no endpoint or BestTg route available.")

        query_smiles = [
            _normalise_repeat_unit_for_besttg(s) if self.normalise_besttg_ethylene else s
            for s in clean_smiles
        ]
        besttg = self._load_besttg().predict_multicomponent(query_smiles, norm_weights, architecture=architecture)
        return {
            **partial_result,
            "tg_c_pred": float(besttg["tg_c_pred"]),
            "tg_k_pred": float(besttg["tg_k_pred"]),
            "besttg_descriptor_mix_tg_c": besttg.get("descriptor_mix_tg_c"),
            "besttg_fox_reference_tg_c": besttg.get("fox_reference_tg_c"),
            "router_route": "besttg_multicomponent_fallback",
            "primary_method": besttg.get("primary_method", "weighted_descriptor_embedding_mix"),
            "confidence": "exploratory_besttg_copolymer_approximation",
            "warning": besttg.get("warning", ""),
            "endpoint_errors": list(endpoint_errors),
        }

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for idx, row in frame.iterrows():
            try:
                result = self._predict_row(row)
                for col in frame.columns:
                    value = row.get(col)
                    if not pd.isna(value):
                        result[f"input_{col}"] = value
                expected = _finite_float(row.get("expected_tg_c", row.get("Tg_C", None)))
                if expected is not None and "tg_c_pred" in result:
                    result["expected_tg_c"] = expected
                    result["error_c"] = float(result["tg_c_pred"]) - expected
                    result["abs_error_c"] = abs(float(result["error_c"]))
                result["row_index"] = int(idx)
                rows.append(_flatten_result(result))
            except Exception as exc:
                out = {f"input_{col}": row.get(col) for col in frame.columns}
                out.update({"row_index": int(idx), "mode": "error", "error": str(exc)})
                rows.append(out)
        return pd.DataFrame(rows)

    def _predict_row(self, row: pd.Series) -> dict[str, object]:
        architecture = _cell_to_text(row.get("architecture", "random")).lower() or "random"
        system_id = _cell_to_text(row.get("system_id", row.get("COID", ""))) or None

        smiles = _cell_to_text(row.get("smiles", ""))
        if smiles:
            return self.predict_homopolymer(smiles)

        components = _cell_to_text(row.get("components", ""))
        if components:
            specs = [part.strip() for part in components.split("|") if part.strip()]
            smiles_list, weights = _parse_component_entries(specs)
            return self.predict_multicomponent(smiles_list, weights, architecture=architecture, system_id=system_id)

        indexed_smiles: Dict[int, str] = {}
        indexed_weights: Dict[int, float] = {}
        for col in row.index:
            col_name = str(col)
            lower = col_name.lower()
            if lower.startswith("smiles_"):
                suffix = lower.rsplit("_", 1)[-1]
            elif lower.startswith("smiles") and lower[6:].isdigit():
                suffix = lower[6:]
            elif lower.startswith("component_"):
                suffix = lower.rsplit("_", 1)[-1]
            elif lower.startswith("component") and lower[9:].isdigit():
                suffix = lower[9:]
            else:
                suffix = ""
            if suffix.isdigit() and ("smiles" in lower or "component" in lower):
                value = _cell_to_text(row.get(col, ""))
                if value:
                    indexed_smiles[int(suffix)] = value
                continue

            if lower.startswith("w_"):
                weight_suffix = lower.rsplit("_", 1)[-1]
            elif lower.startswith("weight_"):
                weight_suffix = lower.rsplit("_", 1)[-1]
            elif lower.startswith("w") and lower[1:].isdigit():
                weight_suffix = lower[1:]
            elif lower.startswith("weight") and lower[6:].isdigit():
                weight_suffix = lower[6:]
            else:
                weight_suffix = ""
            if weight_suffix.isdigit():
                value = row.get(col)
                if not pd.isna(value) and str(value).strip() != "":
                    indexed_weights[int(weight_suffix)] = float(value)

        if indexed_smiles:
            order = sorted(indexed_smiles)
            smiles_list = [indexed_smiles[i] for i in order]
            weights = _weights_for_order(order, indexed_weights)
            return self.predict_multicomponent(smiles_list, weights, architecture=architecture, system_id=system_id)

        smiles1 = _cell_to_text(row.get("smiles1", row.get("SMILES_1", "")))
        smiles2 = _cell_to_text(row.get("smiles2", row.get("SMILES_2", "")))
        if smiles1 and smiles2:
            w1 = _finite_float(row.get("w1", row.get("w1_used", row.get("weight1", 0.5))))
            if w1 is None:
                raise ValueError("missing w1 / w1_used.")
            return self.predict_multicomponent([smiles1, smiles2], [w1, 1.0 - w1], architecture=architecture, system_id=system_id)

        raise ValueError("row needs smiles, components, or smiles1/w1/smiles2 columns.")


def _weights_for_order(order: Sequence[int], indexed_weights: Dict[int, float]) -> list[float]:
    has_any = any(i in indexed_weights for i in order)
    has_all = all(i in indexed_weights for i in order)
    if not has_any:
        return [1.0] * len(order)
    if has_all:
        return [indexed_weights[i] for i in order]
    if len(order) == 2 and len(indexed_weights) == 1:
        provided_index = next(iter(indexed_weights))
        if provided_index == order[0]:
            return [indexed_weights[provided_index], 1.0 - indexed_weights[provided_index]]
        if provided_index == order[1]:
            return [1.0 - indexed_weights[provided_index], indexed_weights[provided_index]]
    raise ValueError("partial component weights. Provide all weights or one binary weight.")


def _flatten_result(result: dict[str, object]) -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, value in result.items():
        if isinstance(value, (list, dict)):
            flat[key] = json.dumps(value, ensure_ascii=False)
        else:
            flat[key] = value
    return flat


def _make_besttg_paths(args: argparse.Namespace) -> InferencePaths:
    return InferencePaths(
        data_path=Path(args.data_path),
        phyc_cache=Path(args.phyc_cache),
        gnn_cache=Path(args.gnn_cache),
        pbert_cache=Path(args.pbert_cache),
        chain_physics_cache=Path(args.chain_physics_cache),
        polybert_model_dir=Path(args.polybert_model_dir),
        gnn_checkpoint=Path(args.gnn_checkpoint),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Universal polymer Tg prediction router.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smiles", type=str, help="Repeat-unit SMILES for homopolymer prediction.")
    mode.add_argument("--component", action="append", default=None, help="Repeatable 'SMILES::weight'.")
    mode.add_argument("--input-csv", type=str, help="Batch CSV input.")
    mode.add_argument("--smiles1", type=str, help="Component 1 repeat-unit SMILES.")

    parser.add_argument("--smiles2", type=str, default=None)
    parser.add_argument("--w1", type=float, default=0.5)
    parser.add_argument("--architecture", choices=sorted(SUPPORTED_ARCHITECTURES), default="random")
    parser.add_argument("--system-id", "--coid", dest="system_id", default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--chain-physics-confs", type=int, default=50)
    parser.add_argument("--no-besttg-fallback", action="store_true")
    parser.add_argument("--no-besttg-ethylene-normalization", action="store_true")
    parser.add_argument("--local-k", type=int, default=3)
    parser.add_argument("--local-power", type=float, default=1.0)
    parser.add_argument("--min-local-points", type=int, default=2)
    parser.add_argument(
        "--local-calibration-mode",
        choices=["row", "leave-composition"],
        default="row",
        help="Known-system local correction mode. 'row' can use duplicate same-composition calibration rows.",
    )

    parser.add_argument("--unified-tg", default=_default_path("data", "unified_tg.parquet"))
    parser.add_argument(
        "--calibration-details",
        default=_default_path("results", "copolymer_residual_model", "polyinfo_physics_details_clean.csv"),
    )
    parser.add_argument("--data-path", default=_default_path("data", "unified_tg.parquet"))
    parser.add_argument("--phyc-cache", default=_default_path("data", "feature_matrix_PHY-C.parquet"))
    parser.add_argument("--gnn-cache", default=_default_path("data", "gnn_embeddings_64d.parquet"))
    parser.add_argument("--pbert-cache", default=_default_path("data", "polybert_embeddings.parquet"))
    parser.add_argument(
        "--chain-physics-cache",
        default=_default_path("data", "chain_physics_features.parquet"),
    )
    parser.add_argument("--polybert-model-dir", default=_default_path("data", "polybert_model"))
    parser.add_argument("--gnn-checkpoint", default=_default_path("checkpoints", "gnn_pretrained.pt"))
    return parser


def _build_router(args: argparse.Namespace) -> UniversalTgRouter:
    calibration = Path(args.calibration_details) if args.calibration_details else None
    return UniversalTgRouter(
        unified_tg_path=Path(args.unified_tg),
        calibration_details_csv=calibration,
        besttg_paths=_make_besttg_paths(args),
        device=args.device,
        chain_physics_confs=args.chain_physics_confs,
        use_besttg_fallback=not args.no_besttg_fallback,
        normalise_besttg_ethylene=not args.no_besttg_ethylene_normalization,
        local_k=args.local_k,
        local_power=args.local_power,
        min_local_points=args.min_local_points,
        local_calibration_mode=args.local_calibration_mode,
    )


def _write_output(payload: object, output: Optional[str]) -> None:
    if not output:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, pd.DataFrame):
        payload.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {path}")


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smiles1 and not args.smiles2:
        raise SystemExit("--smiles1 requires --smiles2.")

    router = _build_router(args)

    if args.smiles:
        _write_output(router.predict_homopolymer(args.smiles), args.output)
        return 0

    if args.component:
        smiles_list, weights = _parse_component_entries(args.component)
        result = router.predict_multicomponent(
            smiles_list,
            weights,
            architecture=args.architecture,
            system_id=args.system_id,
        )
        _write_output(result, args.output)
        return 0

    if args.smiles1:
        result = router.predict_multicomponent(
            [args.smiles1, args.smiles2],
            [args.w1, 1.0 - args.w1],
            architecture=args.architecture,
            system_id=args.system_id,
        )
        _write_output(result, args.output)
        return 0

    frame = pd.read_csv(args.input_csv)
    out = router.predict_frame(frame)
    _write_output(out, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
