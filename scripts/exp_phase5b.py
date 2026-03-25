"""
Phase 5B: 方案A 收尾实验
Phase 5B: Remaining experiments for Plan A completion

Experiments:
    E25: TabPFN + Stacking — add TabPFN to Stacking ensemble
    E26: Nucleotide Tg prediction with M2M-V features + multiple models
         Config A: Bicerano(304) + bridge(205) + M2M-V → compare with Phase 3
         Config B: Unified(7486) + bridge(205) + M2M-V + CatBoost
         Config C: Unified(7486) + bridge(205) + M2M-V + TabPFN

Usage:
    python scripts/exp_phase5b.py                  # Run all
    python scripts/exp_phase5b.py --exp E25         # Single experiment
    python scripts/exp_phase5b.py --exp E26         # Nucleotide only
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Silence warnings
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore", message=".*wmic.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*PYTORCH_CUDA_ALLOC_CONF.*")

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.bridge_polymers import load_all_bridge_data
from src.data.external_datasets import load_unified_dataset
from src.features.feature_pipeline import compute_features, get_feature_names
from src.ml.evaluation import save_result, simple_cv
from src.ml.sklearn_models import (
    build_preprocessing,
    build_stacking_v2,
    get_estimator,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_LAYER = "M2M-V"  # 46-dim
RESULT_DIR = PROJECT_ROOT / "results" / "phase5"
DATA_PATH = PROJECT_ROOT / "data" / "unified_tg.parquet"

# Nucleic acid targets (same as Phase 3)
NUCLEIC_ACID_TARGETS = [
    {
        "name": "DNA dry film (poly-AT)",
        "smiles": "c1nc(N)c2ncn(C3OC(COP(=O)(O)O)C(O)C3O)c2n1",
        "tg_expected": 448,
        "tg_range": (443, 453),
        "source": "Simperler 2006",
    },
    {
        "name": "ATP",
        "smiles": "c1nc(N)c2ncn(C3OC(COP(=O)(O)OP(=O)(O)OP(=O)(O)O)C(O)C3O)c2n1",
        "tg_expected": 246,
        "tg_range": (220, 270),
        "source": "Simperler 2006",
    },
    {
        "name": "ADP",
        "smiles": "c1nc(N)c2ncn(C3OC(COP(=O)(O)OP(=O)(O)O)C(O)C3O)c2n1",
        "tg_expected": 244,
        "tg_range": (220, 270),
        "source": "Simperler 2006",
    },
    {
        "name": "AMP",
        "smiles": "c1nc(N)c2ncn(C3OC(COP(=O)(O)O)C(O)C3O)c2n1",
        "tg_expected": 249,
        "tg_range": (220, 270),
        "source": "Simperler 2006",
    },
    # Additional nucleotides for broader coverage
    {
        "name": "GMP",
        "smiles": "c1nc2c(nc(N)nc2[nH]1)[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]1O",
        "tg_expected": 260,
        "tg_range": (230, 290),
        "source": "estimated",
    },
    {
        "name": "UMP",
        "smiles": "O=c1ccn([C@@H]2O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]2O)c(=O)[nH]1",
        "tg_expected": 255,
        "tg_range": (225, 285),
        "source": "estimated",
    },
    {
        "name": "CMP",
        "smiles": "Nc1ccn([C@@H]2O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]2O)c(=O)n1",
        "tg_expected": 252,
        "tg_range": (222, 282),
        "source": "estimated",
    },
]


def _safe_print(msg: str):
    """Print with encoding fallback for Windows terminals."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("gbk", errors="replace").decode("gbk"))


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _compute_features_safe(smiles: str, layer: str = FEATURE_LAYER) -> Optional[np.ndarray]:
    """Compute features with NaN check."""
    try:
        x = compute_features(smiles, None, layer)
        if np.any(np.isnan(x)):
            return None
        return x
    except Exception:
        return None


def load_unified_features(
    layer: str = FEATURE_LAYER,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load unified dataset and compute features."""
    if verbose:
        _safe_print(f"\n  Loading unified dataset: {DATA_PATH}")

    df_train = load_unified_dataset(str(DATA_PATH), split="train")
    df_test = load_unified_dataset(str(DATA_PATH), split="test")
    df = load_unified_dataset(str(DATA_PATH))  # All data

    if verbose:
        _safe_print(f"  Total: {len(df)}, Train: {len(df_train)}, Test: {len(df_test)}")

    X_list, y_list = [], []
    skipped = 0
    t0 = time.time()

    for _, row in df.iterrows():
        x = _compute_features_safe(row["smiles"], layer)
        if x is not None:
            X_list.append(x)
            y_list.append(float(row["tg_k"]))
        else:
            skipped += 1

    elapsed = time.time() - t0
    feature_names = get_feature_names(layer)

    if verbose:
        _safe_print(
            f"  Features: {len(X_list)} samples, {len(feature_names)} dims "
            f"(skipped {skipped}, {elapsed:.1f}s)"
        )

    return np.array(X_list), np.array(y_list), feature_names


def load_bridge_features(
    layer: str = FEATURE_LAYER,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load bridge polymer data with M2M-V features."""
    data = load_all_bridge_data()

    X_list, y_list, names = [], [], []
    skipped = 0

    for entry in data:
        x = _compute_features_safe(entry["smiles"], layer)
        if x is not None:
            X_list.append(x)
            y_list.append(float(entry["tg_k"]))
            names.append(entry["name"])
        else:
            skipped += 1

    if verbose:
        _safe_print(f"  Bridge polymers: {len(X_list)} (skipped {skipped})")

    return np.array(X_list), np.array(y_list), names


def load_bicerano_features(
    layer: str = FEATURE_LAYER,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load Bicerano dataset with M2M-V features."""
    from src.features.feature_pipeline import build_dataset_v2

    X, y, _, _, _ = build_dataset_v2(layer=layer, verbose=verbose)
    return X, y


# ---------------------------------------------------------------------------
# E25: TabPFN + Stacking
# ---------------------------------------------------------------------------

def run_e25_tabpfn_stacking(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """E25: Stacking with TabPFN as additional base estimator.

    TabPFN + CatBoost + LightGBM + ExtraTrees + XGBoost -> Ridge
    """
    _safe_print(f"\n{'='*60}")
    _safe_print("  E25: TabPFN + Stacking (5 base learners -> Ridge)")
    _safe_print(f"{'='*60}")

    try:
        from tabpfn import TabPFNRegressor
    except ImportError:
        _safe_print("  [!] TabPFN not installed. Skipping E25.")
        return None

    try:
        from sklearn.ensemble import StackingRegressor
        from sklearn.linear_model import Ridge
        from catboost import CatBoostRegressor
        from lightgbm import LGBMRegressor
        from xgboost import XGBRegressor
        from sklearn.ensemble import ExtraTreesRegressor

        estimators = [
            ("tabpfn", TabPFNRegressor()),
            ("catboost", CatBoostRegressor(
                iterations=1000, learning_rate=0.05, depth=6,
                l2_leaf_reg=3.0, random_seed=42, verbose=0,
            )),
            ("lightgbm", LGBMRegressor(
                n_estimators=1000, learning_rate=0.05, num_leaves=31,
                min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbose=-1,
            )),
            ("extratrees", ExtraTreesRegressor(
                n_estimators=500, max_features="sqrt",
                min_samples_leaf=2, random_state=42, n_jobs=-1,
            )),
            ("xgboost", XGBRegressor(
                n_estimators=1000, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0,
            )),
        ]

        stacking = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=5,
            passthrough=False,
        )

        # Simple CV on training set
        _safe_print("  Running CV...")
        cv_result = simple_cv(
            X_train, y_train, stacking,
            n_splits=5, n_repeats=3,
        )

        cv_metrics = cv_result.get("metrics", {})
        _safe_print(
            f"  CV R2={cv_metrics.get('R2_mean', 0):.4f} +/- "
            f"{cv_metrics.get('R2_std', 0):.4f}, "
            f"MAE={cv_metrics.get('MAE_mean', 0):.1f}K"
        )

        # Holdout evaluation
        _safe_print("  Training on full train set for holdout...")
        stacking.fit(X_train, y_train)
        y_pred = stacking.predict(X_test)

        from sklearn.metrics import r2_score, mean_absolute_error
        holdout_r2 = r2_score(y_test, y_pred)
        holdout_mae = mean_absolute_error(y_test, y_pred)
        _safe_print(f"  Holdout R2={holdout_r2:.4f}, MAE={holdout_mae:.1f}K")

        result = {
            "experiment": "E25",
            "model": "Stacking_v3_with_TabPFN",
            "base_models": ["TabPFN", "CatBoost", "LightGBM", "ExtraTrees", "XGBoost"],
            "meta_learner": "Ridge",
            "layer": FEATURE_LAYER,
            "n_features": X_train.shape[1],
            "cv": cv_result,
            "holdout": {
                "test_metrics": {
                    "R2": round(holdout_r2, 4),
                    "MAE": round(holdout_mae, 2),
                }
            },
        }
        save_result(result, str(RESULT_DIR / "exp_E25_tabpfn_stacking.json"))
        return result

    except Exception as e:
        _safe_print(f"  [X] E25 failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# E26: Nucleotide Tg Prediction
# ---------------------------------------------------------------------------

def _predict_nucleotides(model, scaler, layer: str = FEATURE_LAYER) -> List[Dict]:
    """Predict Tg for all nucleotide targets."""
    results = []
    for target in NUCLEIC_ACID_TARGETS:
        smiles = target["smiles"]
        x = _compute_features_safe(smiles, layer)
        if x is None:
            _safe_print(f"  {target['name']:25s}: FAILED (feature extraction)")
            results.append({
                "name": target["name"],
                "tg_expected": target["tg_expected"],
                "error": "FAILED",
            })
            continue

        x_scaled = scaler.transform(x.reshape(1, -1))
        tg_pred = float(model.predict(x_scaled)[0])
        error = abs(tg_pred - target["tg_expected"])
        in_range = target["tg_range"][0] <= tg_pred <= target["tg_range"][1]

        results.append({
            "name": target["name"],
            "smiles": smiles,
            "tg_expected": target["tg_expected"],
            "tg_predicted": round(tg_pred, 1),
            "error": round(error, 1),
            "in_range": in_range,
            "source": target["source"],
        })
        status = "OK" if in_range else "MISS"
        _safe_print(
            f"  {target['name']:25s}: pred={tg_pred:7.1f}K, "
            f"exp={target['tg_expected']}K, err={error:6.1f}K  {status}"
        )

    return results


def _make_scaler():
    """Build preprocessing pipeline."""
    return build_preprocessing()


def _train_weighted_model(
    X_main: np.ndarray,
    y_main: np.ndarray,
    X_bridge: np.ndarray,
    y_bridge: np.ndarray,
    model_fn,
    bridge_weight: float = 0.8,
) -> Tuple[Any, Any]:
    """Train model on combined data with sample weights.

    Returns:
        (trained_model, fitted_scaler)
    """
    X_all = np.vstack([X_main, X_bridge])
    y_all = np.concatenate([y_main, y_bridge])
    w_all = np.concatenate([
        np.ones(len(y_main)),
        np.full(len(y_bridge), bridge_weight),
    ])

    scaler = _make_scaler()
    X_all_s = scaler.fit_transform(X_all)

    model = model_fn()
    # Check if model supports sample_weight
    import inspect
    fit_sig = inspect.signature(model.fit)
    if "sample_weight" in fit_sig.parameters:
        model.fit(X_all_s, y_all, sample_weight=w_all)
    else:
        # TabPFN doesn't support weights — use all data equally
        model.fit(X_all_s, y_all)

    return model, scaler


def run_e26_nucleotide(
    X_unified: np.ndarray,
    y_unified: np.ndarray,
    X_bridge: np.ndarray,
    y_bridge: np.ndarray,
    X_bicerano: np.ndarray,
    y_bicerano: np.ndarray,
) -> Dict[str, Any]:
    """E26: Nucleotide Tg prediction with M2M-V features.

    Three configurations:
        A: Bicerano(304) + Bridge(205), GBR → compare Phase 3
        B: Unified(7486) + Bridge(205), CatBoost → best tree model
        C: Unified(7486) + Bridge(205), TabPFN → zero-tuning
    """
    _safe_print(f"\n{'='*60}")
    _safe_print("  E26: Nucleotide Tg Prediction (M2M-V features)")
    _safe_print(f"{'='*60}")

    all_configs = {}

    # ---- Config A: Bicerano + Bridge, GBR ----
    _safe_print("\n  --- Config A: Bicerano(304) + Bridge, GBR ---")
    _safe_print(f"  Training data: {len(y_bicerano)} + {len(y_bridge)} = {len(y_bicerano)+len(y_bridge)}")

    def make_gbr():
        return get_estimator("GBR")

    model_a, scaler_a = _train_weighted_model(
        X_bicerano, y_bicerano, X_bridge, y_bridge,
        make_gbr, bridge_weight=0.8,
    )
    preds_a = _predict_nucleotides(model_a, scaler_a)
    all_configs["config_A_bicerano_gbr"] = {
        "training_data": f"Bicerano({len(y_bicerano)}) + Bridge({len(y_bridge)})",
        "model": "GBR",
        "bridge_weight": 0.8,
        "predictions": preds_a,
    }

    # ---- Config B: Unified + Bridge, CatBoost ----
    _safe_print("\n  --- Config B: Unified(7486) + Bridge, CatBoost ---")
    _safe_print(f"  Training data: {len(y_unified)} + {len(y_bridge)} = {len(y_unified)+len(y_bridge)}")

    def make_catboost():
        return get_estimator("CatBoost")

    model_b, scaler_b = _train_weighted_model(
        X_unified, y_unified, X_bridge, y_bridge,
        make_catboost, bridge_weight=0.8,
    )
    preds_b = _predict_nucleotides(model_b, scaler_b)
    all_configs["config_B_unified_catboost"] = {
        "training_data": f"Unified({len(y_unified)}) + Bridge({len(y_bridge)})",
        "model": "CatBoost",
        "bridge_weight": 0.8,
        "predictions": preds_b,
    }

    # ---- Config C: Unified + Bridge, TabPFN ----
    try:
        from tabpfn import TabPFNRegressor

        _safe_print("\n  --- Config C: Unified(7486) + Bridge, TabPFN ---")
        _safe_print(f"  Training data: {len(y_unified)} + {len(y_bridge)} = {len(y_unified)+len(y_bridge)}")

        def make_tabpfn():
            return TabPFNRegressor()

        model_c, scaler_c = _train_weighted_model(
            X_unified, y_unified, X_bridge, y_bridge,
            make_tabpfn, bridge_weight=0.8,
        )
        preds_c = _predict_nucleotides(model_c, scaler_c)
        all_configs["config_C_unified_tabpfn"] = {
            "training_data": f"Unified({len(y_unified)}) + Bridge({len(y_bridge)})",
            "model": "TabPFN",
            "bridge_weight": "N/A (no weight support)",
            "predictions": preds_c,
        }
    except ImportError:
        _safe_print("  [!] TabPFN not installed. Skipping Config C.")
    except Exception as e:
        _safe_print(f"  [!] Config C failed: {e}")

    # ---- Config D: Unified + Bridge, ExtraTrees ----
    _safe_print("\n  --- Config D: Unified(7486) + Bridge, ExtraTrees ---")

    def make_et():
        return get_estimator("ExtraTrees")

    model_d, scaler_d = _train_weighted_model(
        X_unified, y_unified, X_bridge, y_bridge,
        make_et, bridge_weight=0.8,
    )
    preds_d = _predict_nucleotides(model_d, scaler_d)
    all_configs["config_D_unified_extratrees"] = {
        "training_data": f"Unified({len(y_unified)}) + Bridge({len(y_bridge)})",
        "model": "ExtraTrees",
        "bridge_weight": 0.8,
        "predictions": preds_d,
    }

    # ---- Config E: Bicerano + Bridge, GBR, bridge_weight sweep ----
    _safe_print("\n  --- Config E: Weight sweep (Bicerano + Bridge, GBR) ---")
    weight_results = {}
    for bw in [0.2, 0.4, 0.6, 0.8, 1.0]:
        model_e, scaler_e = _train_weighted_model(
            X_bicerano, y_bicerano, X_bridge, y_bridge,
            make_gbr, bridge_weight=bw,
        )
        preds_e = _predict_nucleotides(model_e, scaler_e)
        avg_error = np.mean([
            p["error"] for p in preds_e
            if isinstance(p.get("error"), (int, float)) and p["name"] != "DNA dry film (poly-AT)"
        ])
        weight_results[f"bw_{bw}"] = {
            "bridge_weight": bw,
            "avg_nucleotide_error": round(avg_error, 1),
            "predictions": preds_e,
        }
        _safe_print(f"  bridge_weight={bw}: avg nucleotide error = {avg_error:.1f}K")

    all_configs["config_E_weight_sweep"] = weight_results

    # ---- Comparison summary ----
    _safe_print(f"\n{'='*60}")
    _safe_print("  Nucleotide Prediction Summary")
    _safe_print(f"{'='*60}")
    _safe_print(f"  {'Config':<35s} {'ATP':>8s} {'ADP':>8s} {'AMP':>8s} {'Avg':>8s}")
    _safe_print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    # Phase 3 reference
    _safe_print(f"  {'Phase3 L1H+GBR+bw0.8 (ref)':35s} {'1.0':>8s} {'1.3':>8s} {'9.5':>8s} {'3.9':>8s}")

    for config_name, config_data in all_configs.items():
        if config_name.startswith("config_E"):
            continue  # weight sweep shown separately
        preds = config_data.get("predictions", [])
        errors = {}
        for p in preds:
            if isinstance(p.get("error"), (int, float)):
                errors[p["name"]] = p["error"]

        atp = errors.get("ATP", float("nan"))
        adp = errors.get("ADP", float("nan"))
        amp = errors.get("AMP", float("nan"))
        nucleotide_errors = [v for k, v in errors.items() if k != "DNA dry film (poly-AT)"]
        avg = np.mean(nucleotide_errors) if nucleotide_errors else float("nan")

        label = f"{config_name} ({config_data.get('model', '?')})"
        _safe_print(f"  {label:35s} {atp:8.1f} {adp:8.1f} {amp:8.1f} {avg:8.1f}")

    result = {
        "experiment": "E26",
        "description": "Nucleotide Tg prediction with M2M-V features",
        "layer": FEATURE_LAYER,
        "n_features": 46,
        "phase3_reference": {
            "features": "L1H (34-dim)",
            "model": "GBR + bridge_weight=0.8",
            "ATP_error": 1.0,
            "ADP_error": 1.3,
            "AMP_error": 9.5,
            "avg_error": 3.9,
        },
        "configs": all_configs,
    }
    save_result(result, str(RESULT_DIR / "exp_E26_nucleotide.json"))
    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def generate_summary(results: Dict[str, Any]) -> str:
    """Generate Phase 5B summary markdown."""
    lines = [
        "# Phase 5B: Plan A Completion Experiments",
        "",
        f"Feature layer: {FEATURE_LAYER} (46-dim)",
        "",
    ]

    # E25 summary
    if "E25" in results:
        e25 = results["E25"]
        cv = e25.get("cv", {}).get("metrics", {})
        ho = e25.get("holdout", {}).get("test_metrics", {})
        lines.extend([
            "## E25: TabPFN + Stacking",
            "",
            f"- CV R2: {cv.get('R2_mean', 0):.4f} +/- {cv.get('R2_std', 0):.4f}",
            f"- CV MAE: {cv.get('MAE_mean', 0):.1f}K",
            f"- Holdout R2: {ho.get('R2', 0):.4f}",
            f"- Holdout MAE: {ho.get('MAE', 0):.1f}K",
            f"- Base models: {e25.get('base_models', [])}",
            "",
            "Comparison with E21 (Stacking v2 without TabPFN):",
            "- E21 CV R2: 0.8750, E25 CV R2: see above",
            "",
        ])

    # E26 summary
    if "E26" in results:
        e26 = results["E26"]
        lines.extend([
            "## E26: Nucleotide Tg Prediction",
            "",
            "### Config comparison (error in K)",
            "",
            "| Config | Model | Data | ATP | ADP | AMP | Avg |",
            "|--------|-------|------|-----|-----|-----|-----|",
            "| Phase 3 ref | GBR | Bic(304)+Br, L1H | 1.0 | 1.3 | 9.5 | 3.9 |",
        ])

        for config_name, config_data in e26.get("configs", {}).items():
            if config_name.startswith("config_E"):
                continue
            preds = config_data.get("predictions", [])
            errors = {p["name"]: p.get("error", "?") for p in preds}
            atp = errors.get("ATP", "?")
            adp = errors.get("ADP", "?")
            amp = errors.get("AMP", "?")
            nucleotide_errors = [
                v for k, v in errors.items()
                if k != "DNA dry film (poly-AT)" and isinstance(v, (int, float))
            ]
            avg = f"{np.mean(nucleotide_errors):.1f}" if nucleotide_errors else "?"
            model = config_data.get("model", "?")
            data = config_data.get("training_data", "?")
            lines.append(f"| {config_name} | {model} | {data}, M2M-V | {atp} | {adp} | {amp} | {avg} |")

        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 5B experiments")
    parser.add_argument(
        "--exp", nargs="*", default=None,
        help="Experiment IDs (E25, E26). Default: all.",
    )
    args = parser.parse_args()

    run_ids = [e.upper() for e in args.exp] if args.exp else ["E25", "E26"]

    _safe_print(f"\n{'#'*60}")
    _safe_print(f"  Phase 5B: Plan A Completion Experiments")
    _safe_print(f"  Experiments: {run_ids}")
    _safe_print(f"{'#'*60}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    all_results: Dict[str, Any] = {}

    # ---- Load data ----
    _safe_print("\n=== Loading data ===")

    feature_names = get_feature_names(FEATURE_LAYER)
    _safe_print(f"  Feature layer: {FEATURE_LAYER} ({len(feature_names)} dims)")

    # Always load unified + bridge for E26; E25 needs train/test split
    X_unified, y_unified, _ = load_unified_features(verbose=True)
    X_bridge, y_bridge, bridge_names = load_bridge_features(verbose=True)
    X_bicerano, y_bicerano = load_bicerano_features(verbose=True)

    # For E25: need train/test split
    if "E25" in run_ids:
        df_train = load_unified_dataset(str(DATA_PATH), split="train")
        df_test = load_unified_dataset(str(DATA_PATH), split="test")

        X_train_list, y_train_list = [], []
        X_test_list, y_test_list = [], []

        for _, row in df_train.iterrows():
            x = _compute_features_safe(row["smiles"])
            if x is not None:
                X_train_list.append(x)
                y_train_list.append(float(row["tg_k"]))

        for _, row in df_test.iterrows():
            x = _compute_features_safe(row["smiles"])
            if x is not None:
                X_test_list.append(x)
                y_test_list.append(float(row["tg_k"]))

        X_train = np.array(X_train_list)
        y_train = np.array(y_train_list)
        X_test = np.array(X_test_list)
        y_test = np.array(y_test_list)

        _safe_print(f"  Train/Test split: {len(X_train)}/{len(X_test)}")

        # Preprocess
        pp = build_preprocessing()
        X_train_pp = pp.fit_transform(X_train)
        X_test_pp = pp.transform(X_test)

    # ---- Run experiments ----
    if "E25" in run_ids:
        result = run_e25_tabpfn_stacking(X_train_pp, y_train, X_test_pp, y_test)
        if result:
            all_results["E25"] = result

    if "E26" in run_ids:
        result = run_e26_nucleotide(
            X_unified, y_unified,
            X_bridge, y_bridge,
            X_bicerano, y_bicerano,
        )
        all_results["E26"] = result

    # ---- Summary ----
    t_total = time.time() - t_start
    _safe_print(f"\n{'='*60}")
    _safe_print(f"  Phase 5B complete! Total time: {t_total:.0f}s ({t_total/60:.1f}min)")
    _safe_print(f"{'='*60}")

    if all_results:
        summary = generate_summary(all_results)
        summary_path = RESULT_DIR / "phase5b_summary.md"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        _safe_print(f"\n  Summary saved: {summary_path}")
        _safe_print(summary)


if __name__ == "__main__":
    main()
