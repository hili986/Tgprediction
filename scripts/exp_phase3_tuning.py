"""
Phase 3 实验续：超参数调优 + Stacking + 核酸验证优化

目标：
1. Nested CV 调参（10-20 组超参数）
2. Stacking 集成 (ET + GBR + SVR → Ridge)
3. 核酸预测优化

Usage:
    python scripts/exp_phase3_tuning.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.bridge_polymers import build_bridge_dataset
from src.features.feature_pipeline import build_dataset_v2, compute_features
from src.features.hbond_features import compute_hbond_features
from src.ml.evaluation import compute_metrics


def _safe_print(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


def _make_scaler():
    return Pipeline([
        ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
        ("scale", MinMaxScaler()),
    ])


# ---------------------------------------------------------------------------
# Nucleic acid targets (fixed: DNA uses dinucleotide repeat)
# ---------------------------------------------------------------------------

NUCLEIC_ACID_TARGETS = [
    # DNA dry film — use poly(A) backbone unit (deoxyadenosine phosphate repeat)
    {"name": "DNA backbone unit",
     "smiles": "c1nc(N)c2ncn(C3OC(COP(=O)(O)O)CC3O)c2n1",
     "tg_expected": 448, "tg_range": (400, 500), "source": "Simperler 2006"},
    {"name": "ATP",
     "smiles": "c1nc(N)c2ncn(C3OC(COP(=O)(O)OP(=O)(O)OP(=O)(O)O)C(O)C3O)c2n1",
     "tg_expected": 246, "tg_range": (200, 290), "source": "Simperler 2006"},
    {"name": "ADP",
     "smiles": "c1nc(N)c2ncn(C3OC(COP(=O)(O)OP(=O)(O)O)C(O)C3O)c2n1",
     "tg_expected": 244, "tg_range": (200, 290), "source": "Simperler 2006"},
    {"name": "AMP",
     "smiles": "c1nc(N)c2ncn(C3OC(COP(=O)(O)O)C(O)C3O)c2n1",
     "tg_expected": 249, "tg_range": (200, 290), "source": "Simperler 2006"},
    # Guanosine monophosphate
    {"name": "GMP",
     "smiles": "c1nc2c(nc(N)nc2[nH]1)OC1OC(COP(=O)(O)O)C(O)C1O",
     "tg_expected": 260, "tg_range": (210, 310), "source": "estimated"},
]


# ---------------------------------------------------------------------------
# Nested CV with hyperparameter tuning
# ---------------------------------------------------------------------------

def nested_cv_tuned(
    X_exp, y_exp, estimator, param_space,
    X_bridge=None, y_bridge=None, bridge_weight=0.4,
    outer_splits=5, outer_repeats=3, inner_splits=3,
    n_iter=15, random_state=42, label="",
):
    """Nested CV with inner hyperparameter tuning + bridge data."""
    use_bridge = X_bridge is not None and bridge_weight > 0

    outer_cv = RepeatedKFold(n_splits=outer_splits, n_repeats=outer_repeats,
                             random_state=random_state)
    inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=random_state)

    fold_results = []
    y_true_all, y_pred_all = [], []
    best_params_list = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_exp, y_exp)):
        X_train, X_test = X_exp[train_idx], X_exp[test_idx]
        y_train, y_test = y_exp[train_idx], y_exp[test_idx]

        # Combine with bridge data for training
        if use_bridge:
            X_combined = np.vstack([X_train, X_bridge])
            y_combined = np.concatenate([y_train, y_bridge])
            w_combined = np.concatenate([
                np.ones(len(y_train)),
                np.full(len(y_bridge), bridge_weight),
            ])
        else:
            X_combined = X_train
            y_combined = y_train
            w_combined = np.ones(len(y_train))

        # Preprocess
        scaler = _make_scaler()
        scaler.fit(X_combined)
        X_combined_s = scaler.transform(X_combined)
        X_test_s = scaler.transform(X_test)

        # Inner CV: tune on combined data (with sample weights)
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_space,
            n_iter=n_iter,
            cv=inner_cv,
            scoring="r2",
            random_state=random_state + fold_idx,
            n_jobs=-1,
        )
        search.fit(X_combined_s, y_combined, sample_weight=w_combined)

        y_pred = search.predict(X_test_s)
        fold_results.append(compute_metrics(y_test, y_pred))
        best_params_list.append(search.best_params_)
        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())

    agg = {}
    for key in fold_results[0]:
        vals = [f[key] for f in fold_results]
        agg[f"{key}_mean"] = round(float(np.mean(vals)), 4)
        agg[f"{key}_std"] = round(float(np.std(vals)), 4)
    overall = compute_metrics(y_true_all, y_pred_all)

    _safe_print(
        f"  {label:30s}: R2={agg['R2_mean']:.4f}+/-{agg['R2_std']:.4f}, "
        f"MAE={agg['MAE_mean']:.1f}K"
    )

    return {
        "R2_mean": agg["R2_mean"], "R2_std": agg["R2_std"],
        "MAE_mean": agg["MAE_mean"], "MAE_std": agg["MAE_std"],
        "overall_R2": round(overall["R2"], 4),
        "overall_MAE": round(overall["MAE"], 2),
        "best_params_sample": _serialize_params(best_params_list[0]),
    }


def _serialize_params(params):
    """Convert numpy types in params dict."""
    return {k: int(v) if isinstance(v, (np.integer,)) else
               float(v) if isinstance(v, (np.floating,)) else v
            for k, v in params.items()}


# ---------------------------------------------------------------------------
# Stacking ensemble
# ---------------------------------------------------------------------------

def stacking_cv(
    X_exp, y_exp,
    X_bridge=None, y_bridge=None, bridge_weight=0.4,
    n_splits=5, n_repeats=3, random_state=42,
):
    """Stacking ensemble: ET + GBR + SVR → Ridge."""
    use_bridge = X_bridge is not None and bridge_weight > 0

    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    fold_results = []
    y_true_all, y_pred_all = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_exp, y_exp)):
        X_train, X_test = X_exp[train_idx], X_exp[test_idx]
        y_train, y_test = y_exp[train_idx], y_exp[test_idx]

        if use_bridge:
            X_combined = np.vstack([X_train, X_bridge])
            y_combined = np.concatenate([y_train, y_bridge])
            w_combined = np.concatenate([
                np.ones(len(y_train)),
                np.full(len(y_bridge), bridge_weight),
            ])
        else:
            X_combined = X_train
            y_combined = y_train
            w_combined = np.ones(len(y_train))

        scaler = _make_scaler()
        scaler.fit(X_combined)
        X_combined_s = scaler.transform(X_combined)
        X_test_s = scaler.transform(X_test)

        # Build stacking ensemble
        stacking = StackingRegressor(
            estimators=[
                ("et", ExtraTreesRegressor(
                    n_estimators=500, max_features="sqrt",
                    min_samples_leaf=2, random_state=random_state + fold_idx, n_jobs=-1)),
                ("gbr", GradientBoostingRegressor(
                    n_estimators=300, learning_rate=0.05, max_depth=4,
                    min_samples_leaf=5, subsample=0.8,
                    random_state=random_state + fold_idx)),
                ("svr", SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1)),
            ],
            final_estimator=Ridge(alpha=1.0),
            cv=3,
            n_jobs=-1,
        )

        stacking.fit(X_combined_s, y_combined, sample_weight=w_combined)
        y_pred = stacking.predict(X_test_s)

        fold_results.append(compute_metrics(y_test, y_pred))
        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())

    agg = {}
    for key in fold_results[0]:
        vals = [f[key] for f in fold_results]
        agg[f"{key}_mean"] = round(float(np.mean(vals)), 4)
        agg[f"{key}_std"] = round(float(np.std(vals)), 4)
    overall = compute_metrics(y_true_all, y_pred_all)

    _safe_print(
        f"  Stacking(ET+GBR+SVR→Ridge): R2={agg['R2_mean']:.4f}+/-{agg['R2_std']:.4f}, "
        f"MAE={agg['MAE_mean']:.1f}K"
    )

    return {
        "R2_mean": agg["R2_mean"], "R2_std": agg["R2_std"],
        "MAE_mean": agg["MAE_mean"], "MAE_std": agg["MAE_std"],
        "overall_R2": round(overall["R2"], 4),
        "overall_MAE": round(overall["MAE"], 2),
    }


# ---------------------------------------------------------------------------
# Nucleic acid prediction
# ---------------------------------------------------------------------------

def predict_nucleic_acids(model, scaler, layer="L1H"):
    """Predict Tg for nucleic acid molecules."""
    results = []
    for target in NUCLEIC_ACID_TARGETS:
        smiles = target["smiles"]
        try:
            x = compute_features(smiles, None, layer)
            if not layer.endswith("H"):
                x_hb = compute_hbond_features(smiles)
                x = np.concatenate([x, x_hb])

            x_scaled = scaler.transform(x.reshape(1, -1))
            tg_pred = model.predict(x_scaled)[0]

            error = abs(tg_pred - target["tg_expected"])
            in_range = target["tg_range"][0] <= tg_pred <= target["tg_range"][1]

            results.append({
                "name": target["name"],
                "tg_expected": target["tg_expected"],
                "tg_predicted": round(float(tg_pred), 1),
                "error": round(float(error), 1),
                "in_range": in_range,
            })
            _safe_print(
                f"  {target['name']:25s}: pred={tg_pred:.1f}K, "
                f"exp={target['tg_expected']}K, err={error:.1f}K "
                f"{'OK' if in_range else 'MISS'}"
            )
        except (ValueError, KeyError) as e:
            _safe_print(f"  {target['name']:25s}: ERROR - {e}")
            results.append({"name": target["name"], "error": "FAILED"})

    return results


def train_full_model(X_exp, y_exp, X_bridge, y_bridge, model_fn,
                     bridge_weight=0.4):
    """Train on all data, return model + scaler."""
    X_parts = [X_exp]
    y_parts = [y_exp]
    w_parts = [np.ones(len(y_exp))]

    if X_bridge is not None:
        X_parts.append(X_bridge)
        y_parts.append(y_bridge)
        w_parts.append(np.full(len(y_bridge), bridge_weight))

    X_all = np.vstack(X_parts)
    y_all = np.concatenate(y_parts)
    w_all = np.concatenate(w_parts)

    scaler = _make_scaler()
    scaler.fit(X_all)
    X_all_s = scaler.transform(X_all)

    model = model_fn(42)
    model.fit(X_all_s, y_all, sample_weight=w_all)
    return model, scaler


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    result_dir = Path("results/phase3")
    result_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    t0 = time.time()

    # ---- Build data ----
    _safe_print("=== Building data ===")
    X_exp, y_exp, _, _, _ = build_dataset_v2(layer="L1H", verbose=True)
    X_bridge, y_bridge, _, _ = build_bridge_dataset(
        layer="L1", include_hbond=True, verbose=True
    )

    # ---- 1. Nested CV: ExtraTrees tuning ----
    _safe_print("\n=== Nested CV: ExtraTrees hyperparameter tuning ===")
    et_param_space = {
        "n_estimators": [300, 500, 800, 1000],
        "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0],
        "min_samples_leaf": [1, 2, 3, 5],
        "min_samples_split": [2, 3, 5, 8],
        "max_depth": [None, 10, 20, 30, 50],
    }

    # ET without bridge
    results["ET_tuned_no_bridge"] = nested_cv_tuned(
        X_exp, y_exp,
        ExtraTreesRegressor(n_jobs=-1, random_state=42),
        et_param_space,
        n_iter=20, label="ET tuned (no bridge)",
    )

    # ET with bridge
    for bw in [0.2, 0.4, 0.6, 0.8]:
        results[f"ET_tuned_bridge_{bw}"] = nested_cv_tuned(
            X_exp, y_exp,
            ExtraTreesRegressor(n_jobs=-1, random_state=42),
            et_param_space,
            X_bridge=X_bridge, y_bridge=y_bridge, bridge_weight=bw,
            n_iter=20, label=f"ET tuned (bridge={bw})",
        )

    # ---- 2. Nested CV: GBR tuning ----
    _safe_print("\n=== Nested CV: GBR hyperparameter tuning ===")
    gbr_param_space = {
        "n_estimators": [200, 300, 500, 800],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "max_depth": [3, 4, 5, 6],
        "min_samples_leaf": [3, 5, 8, 10],
        "subsample": [0.7, 0.8, 0.9, 1.0],
    }

    results["GBR_tuned_no_bridge"] = nested_cv_tuned(
        X_exp, y_exp,
        GradientBoostingRegressor(random_state=42),
        gbr_param_space,
        n_iter=20, label="GBR tuned (no bridge)",
    )

    for bw in [0.4, 0.8]:
        results[f"GBR_tuned_bridge_{bw}"] = nested_cv_tuned(
            X_exp, y_exp,
            GradientBoostingRegressor(random_state=42),
            gbr_param_space,
            X_bridge=X_bridge, y_bridge=y_bridge, bridge_weight=bw,
            n_iter=20, label=f"GBR tuned (bridge={bw})",
        )

    # ---- 3. Stacking ensemble ----
    _safe_print("\n=== Stacking ensemble ===")
    results["stacking_no_bridge"] = stacking_cv(X_exp, y_exp)
    results["stacking_bridge_0.4"] = stacking_cv(
        X_exp, y_exp, X_bridge, y_bridge, bridge_weight=0.4
    )
    results["stacking_bridge_0.8"] = stacking_cv(
        X_exp, y_exp, X_bridge, y_bridge, bridge_weight=0.8
    )

    # ---- 4. Nucleic acid validation with best configs ----
    _safe_print("\n=== Nucleic acid validation ===")

    def make_et_tuned(rs):
        return ExtraTreesRegressor(
            n_estimators=800, max_features="sqrt", min_samples_leaf=2,
            random_state=rs, n_jobs=-1,
        )

    def make_gbr_tuned(rs):
        return GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=4,
            min_samples_leaf=5, subsample=0.8, random_state=rs,
        )

    _safe_print("\n--- ET tuned + bridge(0.4) ---")
    m1, s1 = train_full_model(X_exp, y_exp, X_bridge, y_bridge, make_et_tuned, 0.4)
    results["nucleic_et_tuned"] = predict_nucleic_acids(m1, s1, "L1H")

    _safe_print("\n--- GBR tuned + bridge(0.4) ---")
    m2, s2 = train_full_model(X_exp, y_exp, X_bridge, y_bridge, make_gbr_tuned, 0.4)
    results["nucleic_gbr_tuned"] = predict_nucleic_acids(m2, s2, "L1H")

    _safe_print("\n--- GBR tuned + bridge(0.8) ---")
    m3, s3 = train_full_model(X_exp, y_exp, X_bridge, y_bridge, make_gbr_tuned, 0.8)
    results["nucleic_gbr_bridge08"] = predict_nucleic_acids(m3, s3, "L1H")

    # ---- Summary ----
    elapsed = time.time() - t0
    _safe_print(f"\nTotal time: {elapsed:.0f}s")

    # Find best R2
    best_key = ""
    best_r2 = 0
    for k, v in results.items():
        if isinstance(v, dict) and "R2_mean" in v:
            if v["R2_mean"] > best_r2:
                best_r2 = v["R2_mean"]
                best_key = k
    _safe_print(f"\nBest config: {best_key} (R2={best_r2:.4f})")

    # Nucleic acid summary
    for config_name in ["nucleic_et_tuned", "nucleic_gbr_tuned", "nucleic_gbr_bridge08"]:
        if config_name in results and isinstance(results[config_name], list):
            errors = [r["error"] for r in results[config_name]
                      if isinstance(r.get("error"), (int, float))]
            if errors:
                _safe_print(f"  {config_name}: avg_error={np.mean(errors):.1f}K, "
                            f"max_error={max(errors):.1f}K")

    # ---- Save ----
    def _ser(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, dict): return {k: _ser(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_ser(v) for v in obj]
        return obj

    outfile = result_dir / "exp-3.4-tuning-stacking.json"
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(_ser(results), f, indent=2, ensure_ascii=False)
    _safe_print(f"\nSaved: {outfile}")


if __name__ == "__main__":
    main()
