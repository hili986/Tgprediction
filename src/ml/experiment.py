"""
Tg Prediction Experiment Framework — Multi-Model Comparison & Feature Ablation
Tg 预测实验框架 — 多模型比较和特征消融

Builds on:
    - bicerano_tg_dataset.py: 304 polymers with Tg(K)
    - bigsmiles_fingerprint.py: Morgan, fragments, descriptors
    - ml_models.py: 7 regression models + cross-validation

Experiments:
    1. Multi-model benchmark: all 7 models on combined features
    2. Feature ablation: Morgan-only, fragments-only, descriptors-only, combinations
    3. Morgan hyperparameter sweep: bits & radius
    4. Summary table with rankings

Public API / 公共 API:
    build_dataset(...)          — load Bicerano data + extract features
    run_model_comparison(...)   — benchmark all models
    run_feature_ablation(...)   — compare feature subsets
    run_morgan_sweep(...)       — sweep Morgan hyperparameters
    run_all_experiments(...)    — run everything, return full report
"""

import time
from typing import List, Dict, Any, Optional, Tuple

from src.ml.models import (
    available_models, cross_validate, train_test_split,
    r2_score, mae_score, rmse_score, get_model,
)


# ---------------------------------------------------------------------------
# 1. Dataset construction / 构建数据集
# ---------------------------------------------------------------------------

def build_dataset(
    use_morgan: bool = True,
    use_fragments: bool = True,
    use_descriptors: bool = True,
    morgan_radius: int = 2,
    morgan_bits: int = 256,
) -> Tuple[List[List[float]], List[float], List[str], List[str]]:
    """Build feature matrix and target vector from Bicerano dataset.
    从 Bicerano 数据集构建特征矩阵和目标向量。

    Args:
        use_morgan: Include Morgan fingerprint bits.
        use_fragments: Include fragment counts.
        use_descriptors: Include polymer descriptors.
        morgan_radius: Morgan fingerprint radius.
        morgan_bits: Number of Morgan fingerprint bits.

    Returns:
        (X, y, names, feature_names) where X is feature matrix, y is Tg(K),
        names is polymer names, feature_names is feature column labels.
    """
    from src.data.bicerano_tg_dataset import BICERANO_DATA
    from src.bigsmiles.fingerprint import (
        morgan_fingerprint, fragment_vector, fragment_names,
        descriptor_vector, descriptor_names,
    )

    X_rows: List[List[float]] = []
    y_vals: List[float] = []
    poly_names: List[str] = []
    skipped = 0

    for name, smiles, bigsmiles, tg_k in BICERANO_DATA:
        try:
            features: List[float] = []
            if use_morgan:
                fp = morgan_fingerprint(smiles, radius=morgan_radius, n_bits=morgan_bits)
                features.extend(float(x) for x in fp)
            if use_fragments:
                features.extend(float(x) for x in fragment_vector(smiles))
            if use_descriptors:
                features.extend(descriptor_vector(smiles, bigsmiles))

            if not features:
                skipped += 1
                continue

            X_rows.append(features)
            y_vals.append(float(tg_k))
            poly_names.append(name)
        except Exception:
            skipped += 1

    # Build feature name list
    feat_names: List[str] = []
    if use_morgan:
        feat_names.extend(f"morgan_{i}" for i in range(morgan_bits))
    if use_fragments:
        feat_names.extend(f"frag_{n}" for n in fragment_names())
    if use_descriptors:
        feat_names.extend(descriptor_names())

    if skipped > 0:
        print(f"  [info] Skipped {skipped} polymers due to feature extraction errors")
    print(f"  [info] Dataset: {len(X_rows)} samples, {len(feat_names)} features")

    return X_rows, y_vals, poly_names, feat_names


# ---------------------------------------------------------------------------
# 2. Model comparison / 模型比较
# ---------------------------------------------------------------------------

# Default hyperparameters per model (tuned for ~300 samples, ~300 features)
_DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "ridge":     {"alpha": 0.1, "lr": 0.01, "n_iter": 2000},
    "lasso":     {"alpha": 0.1, "lr": 0.01, "n_iter": 2000},
    "elasticnet": {"alpha": 0.1, "l1_ratio": 0.5, "lr": 0.01, "n_iter": 2000},
    "knn":       {"n_neighbors": 5},
    "tree":      {"max_depth": 6, "min_samples_leaf": 5, "seed": 42},
    "rf":        {"n_trees": 30, "max_depth": 8, "min_samples_leaf": 3, "seed": 42},
    "gbr":       {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3, "seed": 42},
}


def _get_model_params(model_name: str,
                      overrides: Optional[Dict[str, Dict[str, Any]]] = None,
                      ) -> Dict[str, Any]:
    """Return model hyperparameters with ``seed`` stripped out.

    ``seed`` lives in ``_DEFAULT_PARAMS`` only to document the default value;
    it is passed explicitly via ``cross_validate`` / ``train_test_split`` so
    must be excluded when unpacking into ``get_model(**kwargs)``.
    """
    source = overrides or _DEFAULT_PARAMS
    return {k: v for k, v in source.get(model_name, {}).items() if k != "seed"}


def run_model_comparison(
    X: List[List[float]], y: List[float],
    k: int = 5, seed: int = 42,
    model_params: Optional[Dict[str, Dict[str, Any]]] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Benchmark all models via k-fold cross-validation.
    通过 K 折交叉验证对所有模型进行基准测试。

    Returns:
        List of result dicts sorted by R2 descending.
    """
    params = model_params or _DEFAULT_PARAMS
    results = []

    for name in available_models():
        kwargs = _get_model_params(name, overrides=params)
        if verbose:
            print(f"  Running {name}...", end="", flush=True)
        t0 = time.time()
        cv_result = cross_validate(name, X, y, k=k, seed=seed, **kwargs)
        elapsed = time.time() - t0
        cv_result["time_sec"] = round(elapsed, 2)
        results.append(cv_result)
        if verbose:
            print(f" R2={cv_result['r2_mean']:.4f} +/- {cv_result['r2_std']:.4f}, "
                  f"MAE={cv_result['mae_mean']:.1f}K, time={elapsed:.1f}s")

    results.sort(key=lambda r: r["r2_mean"], reverse=True)

    if verbose:
        print("\n  === Model Ranking (by R2) ===")
        for i, r in enumerate(results):
            print(f"  #{i+1} {r['model']:12s}  "
                  f"R2={r['r2_mean']:.4f}  MAE={r['mae_mean']:.1f}K  "
                  f"RMSE={r['rmse_mean']:.1f}K")

    return results


# ---------------------------------------------------------------------------
# 3. Feature ablation / 特征消融
# ---------------------------------------------------------------------------

# Feature subset configurations for ablation
_ABLATION_CONFIGS = [
    {"name": "morgan_only",       "use_morgan": True,  "use_fragments": False, "use_descriptors": False},
    {"name": "fragments_only",    "use_morgan": False, "use_fragments": True,  "use_descriptors": False},
    {"name": "descriptors_only",  "use_morgan": False, "use_fragments": False, "use_descriptors": True},
    {"name": "morgan+fragments",  "use_morgan": True,  "use_fragments": True,  "use_descriptors": False},
    {"name": "morgan+descriptors", "use_morgan": True, "use_fragments": False, "use_descriptors": True},
    {"name": "fragments+descriptors", "use_morgan": False, "use_fragments": True, "use_descriptors": True},
    {"name": "all_features",      "use_morgan": True,  "use_fragments": True,  "use_descriptors": True},
]


def run_feature_ablation(
    model_name: str = "ridge",
    morgan_bits: int = 256,
    k: int = 5, seed: int = 42,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Run feature ablation: compare different feature subsets.
    特征消融: 比较不同特征子集的效果。

    Args:
        model_name: Which model to use for ablation.
        morgan_bits: Number of Morgan bits.
        k: Number of cross-validation folds.
        seed: Random seed.

    Returns:
        List of result dicts sorted by R2 descending.
    """
    model_params = _get_model_params(model_name)
    results = []

    for config in _ABLATION_CONFIGS:
        cfg_name = config["name"]
        if verbose:
            print(f"  [{cfg_name}] Building dataset...", end="", flush=True)

        X, y, _, feat_names = build_dataset(
            use_morgan=config["use_morgan"],
            use_fragments=config["use_fragments"],
            use_descriptors=config["use_descriptors"],
            morgan_bits=morgan_bits,
        )

        if verbose:
            print(f" CV({model_name})...", end="", flush=True)

        cv_result = cross_validate(model_name, X, y, k=k, seed=seed, **model_params)
        cv_result["config"] = cfg_name
        cv_result["num_features"] = len(feat_names)
        results.append(cv_result)

        if verbose:
            print(f" R2={cv_result['r2_mean']:.4f}, "
                  f"MAE={cv_result['mae_mean']:.1f}K, "
                  f"features={len(feat_names)}")

    results.sort(key=lambda r: r["r2_mean"], reverse=True)

    if verbose:
        print(f"\n  === Feature Ablation ({model_name}) ===")
        for i, r in enumerate(results):
            print(f"  #{i+1} {r['config']:25s}  "
                  f"R2={r['r2_mean']:.4f}  MAE={r['mae_mean']:.1f}K  "
                  f"features={r['num_features']}")

    return results


# ---------------------------------------------------------------------------
# 4. Morgan hyperparameter sweep / Morgan 超参数扫描
# ---------------------------------------------------------------------------

def run_morgan_sweep(
    model_name: str = "ridge",
    bits_list: Optional[List[int]] = None,
    radius_list: Optional[List[int]] = None,
    k: int = 5, seed: int = 42,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Sweep Morgan fingerprint hyperparameters (bits, radius).
    扫描 Morgan 指纹超参数（位数、半径）。

    Returns:
        List of result dicts sorted by R2 descending.
    """
    if bits_list is None:
        bits_list = [64, 128, 256, 512, 1024]
    if radius_list is None:
        radius_list = [1, 2, 3]

    model_params = _get_model_params(model_name)
    results = []

    for radius in radius_list:
        for bits in bits_list:
            if verbose:
                print(f"  radius={radius}, bits={bits}...", end="", flush=True)

            X, y, _, feat_names = build_dataset(
                use_morgan=True, use_fragments=True, use_descriptors=True,
                morgan_radius=radius, morgan_bits=bits,
            )

            cv_result = cross_validate(model_name, X, y, k=k, seed=seed, **model_params)
            cv_result["morgan_radius"] = radius
            cv_result["morgan_bits"] = bits
            cv_result["num_features"] = len(feat_names)
            results.append(cv_result)

            if verbose:
                print(f" R2={cv_result['r2_mean']:.4f}, "
                      f"features={len(feat_names)}")

    results.sort(key=lambda r: r["r2_mean"], reverse=True)

    if verbose:
        print(f"\n  === Morgan Sweep ({model_name}) ===")
        for i, r in enumerate(results):
            print(f"  #{i+1} radius={r['morgan_radius']}, bits={r['morgan_bits']:4d}  "
                  f"R2={r['r2_mean']:.4f}  MAE={r['mae_mean']:.1f}K  "
                  f"features={r['num_features']}")

    return results


# ---------------------------------------------------------------------------
# 5. Hold-out evaluation / 留出法评估
# ---------------------------------------------------------------------------

def run_holdout_evaluation(
    X: List[List[float]], y: List[float],
    model_name: str = "ridge",
    test_ratio: float = 0.2, seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train/test split evaluation for a single model.
    留出法评估单个模型。

    Returns:
        Dict with r2, mae, rmse, y_true, y_pred.
    """
    model_params = _get_model_params(model_name)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_ratio=test_ratio, seed=seed)

    model = get_model(model_name, **model_params)
    y_pred = model.fit_predict(X_tr, y_tr, X_te)

    r2 = r2_score(y_te, y_pred)
    mae = mae_score(y_te, y_pred)
    rmse = rmse_score(y_te, y_pred)

    result = {
        "model": model_name,
        "r2": round(r2, 4),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "num_train": len(X_tr),
        "num_test": len(X_te),
        "y_true": y_te,
        "y_pred": [round(p, 2) for p in y_pred],
    }

    if verbose:
        print(f"  Hold-out ({model_name}): R2={r2:.4f}, MAE={mae:.1f}K, RMSE={rmse:.1f}K")

    return result


# ---------------------------------------------------------------------------
# 6. Full experiment suite / 完整实验套件
# ---------------------------------------------------------------------------

def run_all_experiments(
    k: int = 5, seed: int = 42,
    morgan_bits: int = 256,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run all experiments and return comprehensive report.
    运行所有实验并返回综合报告。

    Returns:
        Dict with model_comparison, feature_ablation, morgan_sweep, best_holdout.
    """
    report: Dict[str, Any] = {"seed": seed, "k_folds": k}

    # 1. Build default dataset
    print("\n=== Experiment 1: Building dataset ===")
    X, y, names, feat_names = build_dataset(morgan_bits=morgan_bits)
    report["dataset"] = {
        "num_samples": len(X),
        "num_features": len(feat_names),
        "tg_min": round(min(y), 1),
        "tg_max": round(max(y), 1),
        "tg_mean": round(sum(y) / len(y), 1),
    }

    # 2. Model comparison
    print("\n=== Experiment 2: Model comparison ===")
    model_results = run_model_comparison(X, y, k=k, seed=seed, verbose=verbose)
    report["model_comparison"] = model_results

    # 3. Feature ablation (using best linear model = ridge)
    print("\n=== Experiment 3: Feature ablation (ridge) ===")
    ablation_results = run_feature_ablation(
        model_name="ridge", morgan_bits=morgan_bits, k=k, seed=seed, verbose=verbose,
    )
    report["feature_ablation"] = ablation_results

    # 4. Morgan sweep (using ridge)
    print("\n=== Experiment 4: Morgan hyperparameter sweep ===")
    sweep_results = run_morgan_sweep(
        model_name="ridge",
        bits_list=[64, 128, 256, 512],
        radius_list=[1, 2, 3],
        k=k, seed=seed, verbose=verbose,
    )
    report["morgan_sweep"] = sweep_results

    # 5. Hold-out with best model
    best_model = model_results[0]["model"]
    print(f"\n=== Experiment 5: Hold-out evaluation ({best_model}) ===")
    holdout = run_holdout_evaluation(
        X, y, model_name=best_model, seed=seed, verbose=verbose,
    )
    report["best_holdout"] = holdout

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Dataset: {report['dataset']['num_samples']} polymers, "
          f"{report['dataset']['num_features']} features")
    print(f"Tg range: {report['dataset']['tg_min']}K - {report['dataset']['tg_max']}K "
          f"(mean={report['dataset']['tg_mean']}K)")
    print(f"\nBest model (CV): {model_results[0]['model']} "
          f"(R2={model_results[0]['r2_mean']:.4f})")
    print(f"Best feature set: {ablation_results[0]['config']} "
          f"(R2={ablation_results[0]['r2_mean']:.4f})")
    print(f"Best Morgan config: radius={sweep_results[0].get('morgan_radius','?')}, "
          f"bits={sweep_results[0].get('morgan_bits','?')} "
          f"(R2={sweep_results[0]['r2_mean']:.4f})")
    print(f"Hold-out ({best_model}): R2={holdout['r2']}, MAE={holdout['mae']}K")

    return report


# ---------------------------------------------------------------------------
# 7. Report export / 报告导出
# ---------------------------------------------------------------------------

def export_report_csv(report: Dict[str, Any], filepath: str = "experiment_report.csv"):
    """Export model comparison results to CSV.
    导出模型比较结果为 CSV。
    """
    lines = ["experiment,model,r2_mean,r2_std,mae_mean,mae_std,rmse_mean,rmse_std,extra"]

    for r in report.get("model_comparison", []):
        lines.append(
            f"model_comparison,{r['model']},"
            f"{r['r2_mean']},{r['r2_std']},"
            f"{r['mae_mean']},{r['mae_std']},"
            f"{r['rmse_mean']},{r['rmse_std']},"
            f"time={r.get('time_sec', '')}"
        )

    for r in report.get("feature_ablation", []):
        lines.append(
            f"feature_ablation,{r.get('config', r['model'])},"
            f"{r['r2_mean']},{r['r2_std']},"
            f"{r['mae_mean']},{r['mae_std']},"
            f"{r['rmse_mean']},{r['rmse_std']},"
            f"features={r.get('num_features', '')}"
        )

    for r in report.get("morgan_sweep", []):
        lines.append(
            f"morgan_sweep,radius={r.get('morgan_radius')}_bits={r.get('morgan_bits')},"
            f"{r['r2_mean']},{r['r2_std']},"
            f"{r['mae_mean']},{r['mae_std']},"
            f"{r['rmse_mean']},{r['rmse_std']},"
            f"features={r.get('num_features', '')}"
        )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Report exported to: {filepath}")


# ---------------------------------------------------------------------------
# CLI entry / 命令行入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if "--help" in sys.argv:
        print("Usage: python ml_experiment.py [options]")
        print("  --all          Run all experiments (default)")
        print("  --models       Run model comparison only")
        print("  --ablation     Run feature ablation only")
        print("  --sweep        Run Morgan sweep only")
        print("  --csv FILE     Export report to CSV")
        sys.exit(0)

    if "--models" in sys.argv:
        X, y, _, _ = build_dataset()
        run_model_comparison(X, y)
    elif "--ablation" in sys.argv:
        run_feature_ablation()
    elif "--sweep" in sys.argv:
        run_morgan_sweep()
    else:
        report = run_all_experiments()
        csv_arg = None
        for i, arg in enumerate(sys.argv):
            if arg == "--csv" and i + 1 < len(sys.argv):
                csv_arg = sys.argv[i + 1]
        if csv_arg:
            export_report_csv(report, csv_arg)
