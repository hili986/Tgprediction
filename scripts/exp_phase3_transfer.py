"""
Phase 3 实验：桥梁聚合物迁移学习 + 核酸验证

Step 3.3: 加权合并训练 (exp=1.0, copolymer=0.1, bridge=0.8)
Step 3.4: 核酸 Tg 验证 (DNA ~448K, ATP ~246K, ADP ~244K, AMP ~249K)
Step 3.5: SHAP 分析氢键特征重要性

Usage:
    python scripts/exp_phase3_transfer.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.bridge_polymers import load_all_bridge_data, build_bridge_dataset
from src.features.feature_pipeline import compute_features, build_dataset_v2, get_feature_names
from src.features.hbond_features import compute_hbond_features, hbond_feature_names
from src.ml.evaluation import compute_metrics


def _safe_print(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


# ---------------------------------------------------------------------------
# Nucleic acid validation targets
# ---------------------------------------------------------------------------

NUCLEIC_ACID_TARGETS = [
    {"name": "DNA dry film (poly-AT)", "smiles": "c1nc(N)c2ncn(C3OC(COP(=O)(O)O)C(O)C3O)c2n1",
     "tg_expected": 448, "tg_range": (443, 453), "source": "Simperler 2006"},
    {"name": "ATP", "smiles": "c1nc(N)c2ncn(C3OC(COP(=O)(O)OP(=O)(O)OP(=O)(O)O)C(O)C3O)c2n1",
     "tg_expected": 246, "tg_range": (220, 270), "source": "Simperler 2006"},
    {"name": "ADP", "smiles": "c1nc(N)c2ncn(C3OC(COP(=O)(O)OP(=O)(O)O)C(O)C3O)c2n1",
     "tg_expected": 244, "tg_range": (220, 270), "source": "Simperler 2006"},
    {"name": "AMP", "smiles": "c1nc(N)c2ncn(C3OC(COP(=O)(O)O)C(O)C3O)c2n1",
     "tg_expected": 249, "tg_range": (220, 270), "source": "Simperler 2006"},
]


# ---------------------------------------------------------------------------
# Build preprocessing pipeline
# ---------------------------------------------------------------------------

def _make_scaler():
    return Pipeline([
        ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
        ("scale", MinMaxScaler()),
    ])


# ---------------------------------------------------------------------------
# Weighted CV with multi-source data
# ---------------------------------------------------------------------------

def weighted_cv(
    X_exp, y_exp,
    model_fn,
    model_name,
    X_bridge=None, y_bridge=None, bridge_weight=0.0,
    X_virtual=None, y_virtual=None, virtual_weight=0.0,
    n_splits=5, n_repeats=3, random_state=42,
):
    """Run CV with weighted multi-source data.

    Args:
        X_exp, y_exp: Experimental data (weight=1.0).
        X_bridge, y_bridge: Bridge polymer data (weight=bridge_weight).
        X_virtual, y_virtual: Virtual copolymer data (weight=virtual_weight).
    """
    use_bridge = X_bridge is not None and bridge_weight > 0
    use_virtual = X_virtual is not None and virtual_weight > 0

    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    fold_results = []
    y_true_all, y_pred_all = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_exp, y_exp)):
        X_train, X_test = X_exp[train_idx], X_exp[test_idx]
        y_train, y_test = y_exp[train_idx], y_exp[test_idx]

        # Combine data sources
        X_parts = [X_train]
        y_parts = [y_train]
        w_parts = [np.ones(len(y_train))]

        if use_bridge:
            X_parts.append(X_bridge)
            y_parts.append(y_bridge)
            w_parts.append(np.full(len(y_bridge), bridge_weight))

        if use_virtual:
            X_parts.append(X_virtual)
            y_parts.append(y_virtual)
            w_parts.append(np.full(len(y_virtual), virtual_weight))

        X_all = np.vstack(X_parts)
        y_all = np.concatenate(y_parts)
        w_all = np.concatenate(w_parts)

        # Preprocess
        scaler = _make_scaler()
        scaler.fit(X_all)
        X_train_s = scaler.transform(X_all)
        X_test_s = scaler.transform(X_test)

        # Train
        model = model_fn(random_state + fold_idx)
        model.fit(X_train_s, y_all, sample_weight=w_all)

        y_pred = model.predict(X_test_s)
        fold_results.append(compute_metrics(y_test, y_pred))
        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())

    agg = {}
    for key in fold_results[0]:
        vals = [f[key] for f in fold_results]
        agg[f"{key}_mean"] = round(float(np.mean(vals)), 4)
        agg[f"{key}_std"] = round(float(np.std(vals)), 4)
    overall = compute_metrics(y_true_all, y_pred_all)

    tag_parts = []
    if use_bridge:
        tag_parts.append(f"bridge={bridge_weight}")
    if use_virtual:
        tag_parts.append(f"virt={virtual_weight}")
    tag = ", ".join(tag_parts) if tag_parts else "baseline"

    _safe_print(
        f"  {model_name:15s} ({tag}): "
        f"R2={agg['R2_mean']:.4f}+/-{agg['R2_std']:.4f}, "
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
            x_base = compute_features(smiles, None, layer)
            x_hb = compute_hbond_features(smiles)
            # If layer already includes hbond, don't double-add
            if layer.endswith("H"):
                x = x_base
            else:
                x = np.concatenate([x_base, x_hb])

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
        except Exception as e:
            _safe_print(f"  {target['name']:25s}: ERROR - {e}")
            results.append({
                "name": target["name"],
                "tg_expected": target["tg_expected"],
                "error": "FAILED",
            })

    return results


# ---------------------------------------------------------------------------
# Train final model for nucleic acid prediction
# ---------------------------------------------------------------------------

def train_final_model(X_exp, y_exp, X_bridge, y_bridge, model_fn,
                      bridge_weight=0.8, X_virtual=None, y_virtual=None,
                      virtual_weight=0.1):
    """Train final model on ALL data for nucleic acid prediction."""
    X_parts = [X_exp]
    y_parts = [y_exp]
    w_parts = [np.ones(len(y_exp))]

    if X_bridge is not None and bridge_weight > 0:
        X_parts.append(X_bridge)
        y_parts.append(y_bridge)
        w_parts.append(np.full(len(y_bridge), bridge_weight))

    if X_virtual is not None and virtual_weight > 0:
        X_parts.append(X_virtual)
        y_parts.append(y_virtual)
        w_parts.append(np.full(len(y_virtual), virtual_weight))

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
# Main experiment
# ---------------------------------------------------------------------------

def main():
    result_dir = Path("results/phase3")
    result_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    t0 = time.time()

    # ---- 1. Build experimental data (L1H = L1 + hbond) ----
    _safe_print("=== Building experimental data (L1H) ===")
    X_exp, y_exp, names_exp, feat_names, _ = build_dataset_v2(layer="L1H", verbose=True)

    # Also build L1 baseline for comparison
    _safe_print("\n=== Building experimental data (L1, no hbond) ===")
    X_exp_l1, y_exp_l1, _, _, _ = build_dataset_v2(layer="L1", verbose=True)

    # ---- 2. Build bridge polymer data ----
    _safe_print("\n=== Building bridge polymer data (L1H) ===")
    X_bridge, y_bridge, names_bridge, _ = build_bridge_dataset(
        layer="L1", include_hbond=True, verbose=True
    )

    _safe_print(f"  Bridge Tg range: {y_bridge.min():.0f} - {y_bridge.max():.0f} K")

    # ---- 3. Model factories ----
    def make_et(rs):
        return ExtraTreesRegressor(
            n_estimators=500, max_features="sqrt", min_samples_leaf=2,
            random_state=rs, n_jobs=-1,
        )

    def make_gbr(rs):
        return GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            min_samples_leaf=5, subsample=0.8, random_state=rs,
        )

    # ---- 4. Experiment 3.3a: L1 baselines (no hbond) ----
    _safe_print("\n=== Exp 3.3a: Baselines (L1, no hbond) ===")
    results["ET_L1_baseline"] = weighted_cv(X_exp_l1, y_exp_l1, make_et, "ExtraTrees")
    results["GBR_L1_baseline"] = weighted_cv(X_exp_l1, y_exp_l1, make_gbr, "GBR")

    # ---- 5. Experiment 3.3b: L1H baselines (with hbond) ----
    _safe_print("\n=== Exp 3.3b: Baselines (L1H, with hbond) ===")
    results["ET_L1H_baseline"] = weighted_cv(X_exp, y_exp, make_et, "ExtraTrees")
    results["GBR_L1H_baseline"] = weighted_cv(X_exp, y_exp, make_gbr, "GBR")

    # ---- 6. Experiment 3.3c: Bridge transfer (L1H) ----
    _safe_print("\n=== Exp 3.3c: Bridge transfer (L1H, weight scan) ===")
    for bw in [0.2, 0.4, 0.6, 0.8, 1.0]:
        results[f"ET_bridge_{bw}"] = weighted_cv(
            X_exp, y_exp, make_et, "ExtraTrees",
            X_bridge=X_bridge, y_bridge=y_bridge, bridge_weight=bw,
        )

    _safe_print("\n=== Exp 3.3c: GBR bridge transfer ===")
    for bw in [0.4, 0.8]:
        results[f"GBR_bridge_{bw}"] = weighted_cv(
            X_exp, y_exp, make_gbr, "GBR",
            X_bridge=X_bridge, y_bridge=y_bridge, bridge_weight=bw,
        )

    # ---- 7. Find best bridge weight ----
    best_bw = 0.8
    best_r2 = 0
    for bw in [0.2, 0.4, 0.6, 0.8, 1.0]:
        key = f"ET_bridge_{bw}"
        if results[key]["R2_mean"] > best_r2:
            best_r2 = results[key]["R2_mean"]
            best_bw = bw
    _safe_print(f"\n  Best bridge weight: {best_bw} (R2={best_r2:.4f})")

    # ---- 8. Experiment 3.4: Nucleic acid validation ----
    _safe_print("\n=== Exp 3.4: Nucleic acid Tg validation ===")

    # Train final models with different configs
    _safe_print("\n--- Config A: ET + exp only (L1H) ---")
    model_a, scaler_a = train_final_model(
        X_exp, y_exp, None, None, make_et, bridge_weight=0
    )
    results["nucleic_baseline"] = predict_nucleic_acids(model_a, scaler_a, layer="L1H")

    _safe_print(f"\n--- Config B: ET + bridge (L1H, bw={best_bw}) ---")
    model_b, scaler_b = train_final_model(
        X_exp, y_exp, X_bridge, y_bridge, make_et, bridge_weight=best_bw
    )
    results["nucleic_bridge"] = predict_nucleic_acids(model_b, scaler_b, layer="L1H")

    _safe_print(f"\n--- Config C: GBR + bridge (L1H, bw={best_bw}) ---")
    model_c, scaler_c = train_final_model(
        X_exp, y_exp, X_bridge, y_bridge, make_gbr, bridge_weight=best_bw
    )
    results["nucleic_bridge_gbr"] = predict_nucleic_acids(model_c, scaler_c, layer="L1H")

    # ---- 9. Summary ----
    elapsed = time.time() - t0
    _safe_print(f"\nTotal time: {elapsed:.0f}s")

    # Check acceptance criteria
    best_l1h = results["ET_L1H_baseline"]["R2_mean"]
    best_bridge_r2 = best_r2
    _safe_print(f"\n=== Acceptance criteria ===")
    _safe_print(f"  Bridge data: {len(y_bridge)} entries (target >300: {'PASS' if len(y_bridge) >= 300 else f'PARTIAL ({len(y_bridge)})'}")
    _safe_print(f"  L1H baseline R2: {best_l1h:.4f}")
    _safe_print(f"  Bridge transfer R2: {best_bridge_r2:.4f} (target >0.88: {'PASS' if best_bridge_r2 > 0.88 else 'IN PROGRESS'})")

    # Check nucleic acid errors
    if isinstance(results.get("nucleic_bridge"), list):
        errors = [r["error"] for r in results["nucleic_bridge"] if isinstance(r.get("error"), (int, float))]
        avg_err = np.mean(errors) if errors else float("inf")
        _safe_print(f"  Nucleic acid avg error: {avg_err:.1f}K (target <50K: {'PASS' if avg_err < 50 else 'IN PROGRESS'})")

    # ---- 10. Save ----
    outfile = result_dir / "exp-3.3-transfer-learning.json"
    # Convert numpy types
    def _serialize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialize(v) for v in obj]
        return obj

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(_serialize(results), f, indent=2, ensure_ascii=False)
    _safe_print(f"\nSaved: {outfile}")


if __name__ == "__main__":
    main()
