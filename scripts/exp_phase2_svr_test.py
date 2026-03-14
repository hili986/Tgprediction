"""
Phase 2 补充实验：SVR/GBR + 加权虚拟数据 (leakage-free)

SVR 是 Phase 1 最佳模型之一，需要测试与虚拟数据配合的效果。
使用 1K 虚拟样本（SVR 对样本量敏感，O(n²)）。
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.fox_copolymer_generator import generate_copolymer_data
from src.features.feature_pipeline import build_dataset_v2, compute_features
from src.ml.evaluation import compute_metrics
from src.ml.two_stage_training import _filter_virtual_for_fold


def _safe_print(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


def run_cv(
    X_exp, y_exp, model_fn, model_name,
    X_virtual=None, y_virtual=None, virtual_weight=0.0,
    exp_smiles=None, virt_smiles_pairs=None,
    n_splits=5, n_repeats=3, random_state=42,
):
    """Run leakage-free CV. If X_virtual is None, run baseline."""
    use_virtual = X_virtual is not None and virtual_weight > 0
    leakage_free = use_virtual and exp_smiles is not None and virt_smiles_pairs is not None

    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    fold_results = []
    y_true_all, y_pred_all = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_exp, y_exp)):
        X_train, X_test = X_exp[train_idx], X_exp[test_idx]
        y_train, y_test = y_exp[train_idx], y_exp[test_idx]

        if use_virtual:
            if leakage_free:
                test_set = {exp_smiles[i] for i in test_idx}
                mask = _filter_virtual_for_fold(virt_smiles_pairs, test_set)
                X_vf, y_vf = X_virtual[mask], y_virtual[mask]
            else:
                X_vf, y_vf = X_virtual, y_virtual

            scaler = Pipeline([
                ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
                ("scale", MinMaxScaler()),
            ])
            scaler.fit(np.vstack([X_train, X_vf]))
            X_train_s = scaler.transform(X_train)
            X_test_s = scaler.transform(X_test)
            X_vf_s = scaler.transform(X_vf)

            X_all = np.vstack([X_train_s, X_vf_s])
            y_all = np.concatenate([y_train, y_vf])
            w = np.concatenate([np.ones(len(y_train)), np.full(len(y_vf), virtual_weight)])
            model = model_fn(random_state + fold_idx)
            model.fit(X_all, y_all, sample_weight=w)
        else:
            scaler = Pipeline([
                ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
                ("scale", MinMaxScaler()),
            ])
            scaler.fit(X_train)
            X_train_s = scaler.transform(X_train)
            X_test_s = scaler.transform(X_test)
            model = model_fn(random_state + fold_idx)
            model.fit(X_train_s, y_train)

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

    tag = f"w={virtual_weight}" if use_virtual else "baseline"
    _safe_print(
        f"  {model_name:12s} ({tag}): R2={agg['R2_mean']:.4f}+/-{agg['R2_std']:.4f}, MAE={agg['MAE_mean']:.1f}K"
    )

    return {
        "R2_mean": agg["R2_mean"], "R2_std": agg["R2_std"],
        "MAE_mean": agg["MAE_mean"], "MAE_std": agg["MAE_std"],
        "overall_R2": round(overall["R2"], 4),
        "overall_MAE": round(overall["MAE"], 2),
    }


def main():
    result_dir = Path("results/phase2")
    result_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build experimental data with SMILES ----
    _safe_print("=== Building experimental data ===")
    X_exp, y_exp, names_exp, feat_names, _ = build_dataset_v2(layer="L1", verbose=True)
    _safe_print(f"  Experimental: {X_exp.shape}")

    # Get exp SMILES (same order as build_dataset_v2)
    from src.data.bicerano_tg_dataset import BICERANO_DATA
    exp_smiles = []
    for name, smiles, bigsmiles, tg_k in BICERANO_DATA:
        try:
            feat = compute_features(smiles, bigsmiles, "L1")
            if not np.any(np.isnan(feat)):
                exp_smiles.append(smiles)
        except Exception:
            pass

    # ---- Generate 1K virtual copolymer data (L1 features) ----
    _safe_print("\n=== Building virtual data (1K, L1 features) ===")
    from src.data.fox_copolymer_generator import compute_copolymer_features
    copolymer_data = generate_copolymer_data(max_samples=1000, fidelity="F0")

    X_virt_list, y_virt_list, virt_pairs = [], [], []
    for entry in copolymer_data:
        try:
            x = compute_copolymer_features(
                entry["smiles1"], entry["smiles2"], entry["w1"],
                entry.get("bigsmiles1", ""), entry.get("bigsmiles2", ""),
                layer="L1",
            )
            if not np.any(np.isnan(x)):
                X_virt_list.append(x)
                y_virt_list.append(entry["tg_virtual"])
                virt_pairs.append((entry["smiles1"], entry["smiles2"]))
        except Exception:
            pass

    X_virt = np.array(X_virt_list)
    y_virt = np.array(y_virt_list)
    _safe_print(f"  Virtual: {X_virt.shape}")

    # ---- Model factories ----
    def make_et(rs):
        return ExtraTreesRegressor(
            n_estimators=500, max_features="sqrt", min_samples_leaf=2,
            random_state=rs, n_jobs=-1,
        )

    def make_svr(_rs):
        return SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1)

    def make_gbr(rs):
        return GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            min_samples_leaf=5, subsample=0.8, random_state=rs,
        )

    # ---- Experiments ----
    results = {}
    t0 = time.time()

    _safe_print("\n=== Baselines (L1, exp only) ===")
    results["ET_baseline"] = run_cv(X_exp, y_exp, make_et, "ExtraTrees")
    results["SVR_baseline"] = run_cv(X_exp, y_exp, make_svr, "SVR")
    results["GBR_baseline"] = run_cv(X_exp, y_exp, make_gbr, "GBR")

    _safe_print("\n=== With virtual data (w=0.05, 1K Fox, leakage-free) ===")
    common = dict(X_virtual=X_virt, y_virtual=y_virt, exp_smiles=exp_smiles, virt_smiles_pairs=virt_pairs)
    results["ET_virt_0.05"] = run_cv(X_exp, y_exp, make_et, "ExtraTrees", virtual_weight=0.05, **common)
    results["SVR_virt_0.05"] = run_cv(X_exp, y_exp, make_svr, "SVR", virtual_weight=0.05, **common)
    results["GBR_virt_0.05"] = run_cv(X_exp, y_exp, make_gbr, "GBR", virtual_weight=0.05, **common)

    _safe_print("\n=== SVR weight scan (leakage-free) ===")
    for w in [0.01, 0.1, 0.2]:
        results[f"SVR_virt_{w}"] = run_cv(X_exp, y_exp, make_svr, "SVR", virtual_weight=w, **common)

    elapsed = time.time() - t0
    _safe_print(f"\nTotal time: {elapsed:.0f}s")

    # ---- Save ----
    outfile = result_dir / "exp-2.4-multi-model.json"
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    _safe_print(f"Saved: {outfile}")


if __name__ == "__main__":
    main()
