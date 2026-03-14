"""
Phase 3 Step 3.5: SHAP 特征重要性分析

分析 H-bond 特征对 Tg 预测的贡献，验证桥梁特征的可解释性。

Usage:
    python scripts/exp_phase3_shap.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import shap

from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.bridge_polymers import build_bridge_dataset
from src.features.feature_pipeline import build_dataset_v2
from src.features.hbond_features import hbond_feature_names


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


def shap_analysis(X_exp, y_exp, X_bridge, y_bridge, feature_names,
                  bridge_weight=0.4):
    """Run SHAP analysis on GBR and ET models with bridge data."""
    # Combine data
    X_all = np.vstack([X_exp, X_bridge])
    y_all = np.concatenate([y_exp, y_bridge])
    w_all = np.concatenate([
        np.ones(len(y_exp)),
        np.full(len(y_bridge), bridge_weight),
    ])

    scaler = _make_scaler()
    scaler.fit(X_all)
    X_all_s = scaler.transform(X_all)
    X_exp_s = scaler.transform(X_exp)

    results = {}

    # --- GBR SHAP (TreeExplainer) ---
    _safe_print("\n=== GBR SHAP analysis ===")
    gbr = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        min_samples_leaf=5, subsample=0.8, random_state=42,
    )
    gbr.fit(X_all_s, y_all, sample_weight=w_all)

    explainer_gbr = shap.TreeExplainer(gbr)
    shap_values_gbr = explainer_gbr.shap_values(X_exp_s)

    mean_abs_shap_gbr = np.mean(np.abs(shap_values_gbr), axis=0)
    ranked_gbr = sorted(
        zip(feature_names, mean_abs_shap_gbr.tolist()),
        key=lambda x: x[1], reverse=True,
    )

    _safe_print("\nGBR Top-15 features by mean |SHAP|:")
    for i, (name, val) in enumerate(ranked_gbr[:15], 1):
        _safe_print(f"  {i:2d}. {name:30s} {val:.2f}")

    results["gbr_shap_top15"] = [
        {"feature": name, "mean_abs_shap": round(val, 4)}
        for name, val in ranked_gbr[:15]
    ]
    results["gbr_shap_all"] = [
        {"feature": name, "mean_abs_shap": round(val, 4)}
        for name, val in ranked_gbr
    ]

    # --- ET SHAP (TreeExplainer) ---
    _safe_print("\n=== ET SHAP analysis ===")
    et = ExtraTreesRegressor(
        n_estimators=500, max_features="sqrt",
        min_samples_leaf=2, random_state=42, n_jobs=-1,
    )
    et.fit(X_all_s, y_all, sample_weight=w_all)

    explainer_et = shap.TreeExplainer(et)
    shap_values_et = explainer_et.shap_values(X_exp_s)

    mean_abs_shap_et = np.mean(np.abs(shap_values_et), axis=0)
    ranked_et = sorted(
        zip(feature_names, mean_abs_shap_et.tolist()),
        key=lambda x: x[1], reverse=True,
    )

    _safe_print("\nET Top-15 features by mean |SHAP|:")
    for i, (name, val) in enumerate(ranked_et[:15], 1):
        _safe_print(f"  {i:2d}. {name:30s} {val:.2f}")

    results["et_shap_top15"] = [
        {"feature": name, "mean_abs_shap": round(val, 4)}
        for name, val in ranked_et[:15]
    ]
    results["et_shap_all"] = [
        {"feature": name, "mean_abs_shap": round(val, 4)}
        for name, val in ranked_et
    ]

    # --- H-bond feature group analysis ---
    _safe_print("\n=== H-bond feature group importance ===")
    _hbond_set = set(hbond_feature_names())
    hbond_names = [n for n in feature_names if n in _hbond_set]
    l0_names = [n for n in feature_names if n.startswith("L0_")]
    l1_names = [n for n in feature_names if n.startswith("L1_")]

    groups = {
        "L0_afsordeh": l0_names,
        "L1_rdkit_2d": l1_names,
        "hbond": hbond_names,
    }

    for model_name, ranked in [("GBR", ranked_gbr), ("ET", ranked_et)]:
        shap_dict = {name: val for name, val in ranked}
        _safe_print(f"\n  {model_name} group-level SHAP:")
        group_results = []
        for group_name, names in groups.items():
            group_shap = sum(shap_dict.get(n, 0) for n in names)
            avg_shap = group_shap / len(names) if names else 0
            _safe_print(
                f"    {group_name:15s}: total={group_shap:.2f}, "
                f"avg={avg_shap:.2f}, n_features={len(names)}"
            )
            group_results.append({
                "group": group_name,
                "total_shap": round(group_shap, 4),
                "avg_shap": round(avg_shap, 4),
                "n_features": len(names),
            })
        results[f"{model_name.lower()}_group_shap"] = group_results

    # --- Feature importance comparison: L1 vs L1H ---
    _safe_print("\n=== L1 vs L1H comparison (GBR built-in importance) ===")
    importances = gbr.feature_importances_
    ranked_imp = sorted(
        zip(feature_names, importances.tolist()),
        key=lambda x: x[1], reverse=True,
    )
    _safe_print("\nGBR feature_importances_ Top-15:")
    for i, (name, val) in enumerate(ranked_imp[:15], 1):
        _safe_print(f"  {i:2d}. {name:30s} {val:.4f}")

    results["gbr_importance_top15"] = [
        {"feature": name, "importance": round(val, 6)}
        for name, val in ranked_imp[:15]
    ]

    # H-bond features in top-15
    hbond_in_top15 = [r for r in ranked_imp[:15] if r[0] in hbond_names]
    _safe_print(f"\n  H-bond features in GBR top-15: {len(hbond_in_top15)}")
    for name, val in hbond_in_top15:
        _safe_print(f"    {name}: {val:.4f}")

    results["hbond_in_gbr_top15"] = len(hbond_in_top15)

    return results


def main():
    result_dir = Path("results/phase3")
    result_dir.mkdir(parents=True, exist_ok=True)

    _safe_print("=== Building data ===")
    X_exp, y_exp, _, feat_names, _ = build_dataset_v2(layer="L1H", verbose=True)
    X_bridge, y_bridge, _, _ = build_bridge_dataset(
        layer="L1", include_hbond=True, verbose=True,
    )

    _safe_print(f"\n  Feature names ({len(feat_names)}):")
    for i, name in enumerate(feat_names):
        _safe_print(f"    [{i:2d}] {name}")

    results = shap_analysis(
        X_exp, y_exp, X_bridge, y_bridge,
        feat_names, bridge_weight=0.4,
    )

    out_path = result_dir / "exp-3.5-shap-analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    _safe_print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
