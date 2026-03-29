"""
Phase E 剩余实验:
  1. GC 残差学习测试 (策略 A: GC_Tg 作为特征 vs 策略 B: 学残差)
  2. 消融实验 (去掉各尺度特征, 看 R² 下降)
  3. 误差地图 (|error| > 50K 的分子分析)
  4. 核苷酸迁移测试 (L1H 特征集)

Usage:
    python scripts/exp_phase_e_remaining.py
    python scripts/exp_phase_e_remaining.py --step ablation
    python scripts/exp_phase_e_remaining.py --step gc_residual
    python scripts/exp_phase_e_remaining.py --step error_map
    python scripts/exp_phase_e_remaining.py --step nucleotide
"""
import argparse
import json
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedKFold

warnings.filterwarnings("ignore")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_pipeline import get_feature_names, compute_features
from src.ml.sklearn_models import build_preprocessing
from src.ml.evaluation import nested_cv_no_tuning

PHYC_CACHE = PROJECT_ROOT / "data" / "feature_matrix_PHY-C.parquet"
GNN_CACHE = PROJECT_ROOT / "data" / "gnn_embeddings_64d.parquet"
PBERT_CACHE = PROJECT_ROOT / "data" / "polybert_embeddings.parquet"
DATA_PATH = PROJECT_ROOT / "data" / "unified_tg.parquet"
RESULT_DIR = PROJECT_ROOT / "results" / "phase_e"

DROP = [
    "CP_oligomer_level", "CP_Cn_proxy",
    "L1_RingCount", "L1_HeavyAtomCount", "L1_MolWt",
    "IC_hydrophilic_ratio",
]

POLYBERT_PCA_DIM = 64


def load_186d():
    """Load full 186d feature matrix (PHY-C 58d + GNN 64d + polyBERT PCA 64d)."""
    from src.gnn.polybert_embedder import polybert_pca

    df_phyc = pd.read_parquet(PHYC_CACHE)
    all_names = get_feature_names("PHY-C")
    keep_names = [f for f in all_names if f not in DROP]
    X_phyc = df_phyc[keep_names].values
    y = df_phyc["tg_k"].values
    smiles = df_phyc["smiles"].tolist() if "smiles" in df_phyc.columns else None

    df_gnn = pd.read_parquet(GNN_CACHE)
    gnn_cols = sorted([c for c in df_gnn.columns if c.startswith("GNN_")])
    X_gnn = df_gnn[gnn_cols].values

    df_pbert = pd.read_parquet(PBERT_CACHE)
    pbert_cols = sorted([c for c in df_pbert.columns if c.startswith("pBERT_")])
    X_pbert_raw = df_pbert[pbert_cols].values
    X_pbert_pca = polybert_pca(X_pbert_raw, target_dim=POLYBERT_PCA_DIM)

    X_full = np.hstack([X_phyc, X_gnn, X_pbert_pca])
    gnn_names = [f"GNN_{i}" for i in range(64)]
    pbert_names = [f"pBERT_PCA_{i}" for i in range(POLYBERT_PCA_DIM)]
    full_names = keep_names + gnn_names + pbert_names

    if smiles is None:
        df_orig = pd.read_parquet(DATA_PATH)
        smiles = df_orig["smiles"].tolist()

    print(f"Loaded 186d: {X_full.shape}, samples={len(y)}")
    return X_full, y, full_names, keep_names, gnn_names, pbert_names, smiles


def run_tabpfn_cv(X, y, label="", outer_splits=5, outer_repeats=3):
    """Quick TabPFN Nested CV."""
    from tabpfn import TabPFNRegressor

    pp = build_preprocessing()
    X_pp = pp.fit_transform(X)

    estimator = TabPFNRegressor()
    print(f"\n  {label}: {X_pp.shape[1]}d, {len(y)} samples")
    result = nested_cv_no_tuning(X_pp, y, estimator,
                                  outer_splits=outer_splits,
                                  outer_repeats=outer_repeats, verbose=True)
    m = result["metrics"]
    print(f"  → R²={m['R2_mean']:.4f}+-{m['R2_std']:.4f}, MAE={m['MAE_mean']:.1f}K")
    return m, result


# ── 1. GC 残差学习 ──────────────────────────────────────────

def run_gc_residual(X_full, y, full_names):
    """GC residual learning: Strategy A vs B."""
    from tabpfn import TabPFNRegressor

    print(f"\n{'='*65}")
    print("  1. GC 残差学习测试")
    print(f"{'='*65}")

    gc_idx = full_names.index("GC_Tg")
    gc_tg = X_full[:, gc_idx].copy()
    gc_valid = ~np.isnan(gc_tg)
    gc_tg_filled = np.where(np.isnan(gc_tg), np.nanmedian(gc_tg), gc_tg)

    # Strategy A: GC_Tg as feature, target = absolute Tg (already in 186d baseline)
    print("\n  策略 A: GC_Tg 作为特征, 目标 = 绝对 Tg (即 186d 基线)")
    m_a, _ = run_tabpfn_cv(X_full, y, "Strategy A (baseline 186d)")

    # Strategy B: target = Tg - GC_Tg, predict residual
    print("\n  策略 B: 目标 = Tg - GC_Tg, 预测残差")
    y_residual = y - gc_tg_filled
    m_b_res, result_b = run_tabpfn_cv(X_full, y_residual, "Strategy B (residual)")

    # Reconstruct: prediction = GC_Tg + predicted_residual
    # Need to compute actual R² on original Tg scale
    y_true_all = np.array(result_b["y_true_all"])
    y_pred_res = np.array(result_b["y_pred_all"])

    # Collect fold-level reconstruction (approximate using global GC_Tg)
    # For fair comparison, report the residual model metrics
    print(f"\n  注意: 策略 B 的 R² 是对残差的, 不直接可比")
    print(f"  残差 std = {np.std(y_residual):.1f}K (vs 原始 Tg std = {np.std(y):.1f}K)")

    results = {
        "strategy_A": m_a,
        "strategy_B_residual": m_b_res,
        "residual_std": float(np.std(y_residual)),
        "original_std": float(np.std(y)),
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULT_DIR / "gc_residual.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {RESULT_DIR / 'gc_residual.json'}")
    return results


# ── 2. 消融实验 ─────────────────────────────────────────────

def run_ablation(X_full, y, full_names, phyc_names, gnn_names, pbert_names):
    """Ablation: remove each feature group, measure R² drop."""
    print(f"\n{'='*65}")
    print("  2. 消融实验 (去掉各尺度特征)")
    print(f"{'='*65}")

    # Define feature groups to ablate
    cp_features = [f for f in phyc_names if f.startswith("CP_")]
    gc_features = ["GC_Tg", "GC_coverage"]
    ic_features = [f for f in phyc_names if f.startswith("IC_")]
    vpd_features = [f for f in phyc_names if f.startswith("VPD_")]
    ppf_features = [f for f in phyc_names if f.startswith("PPF_")]

    ablation_groups = {
        "full_186d": None,  # Baseline, no removal
        "no_chain_physics": cp_features,
        "no_GNN": gnn_names,
        "no_polyBERT": pbert_names,
        "no_GC": gc_features,
        "no_interchain": ic_features,
        "no_VPD": vpd_features,
        "no_PPF": ppf_features,
        "no_GNN_no_polyBERT": gnn_names + pbert_names,  # Only physics features
    }

    results = {}
    baseline_r2 = None

    for name, remove_feats in ablation_groups.items():
        if remove_feats is None:
            # Full baseline
            m, _ = run_tabpfn_cv(X_full, y, f"Ablation: {name}")
            baseline_r2 = m["R2_mean"]
        else:
            remove_idx = [full_names.index(f) for f in remove_feats if f in full_names]
            keep_idx = [i for i in range(len(full_names)) if i not in remove_idx]
            X_ablated = X_full[:, keep_idx]
            m, _ = run_tabpfn_cv(X_ablated, y,
                                  f"Ablation: {name} (-{len(remove_idx)}d → {len(keep_idx)}d)")

        delta = m["R2_mean"] - baseline_r2 if baseline_r2 is not None else 0
        results[name] = {
            "R2_mean": m["R2_mean"], "R2_std": m["R2_std"],
            "MAE_mean": m["MAE_mean"], "MAE_std": m["MAE_std"],
            "n_removed": len(remove_feats) if remove_feats else 0,
            "n_remaining": len(full_names) - (len(remove_feats) if remove_feats else 0),
            "delta_R2": delta,
        }

    # Print summary
    print(f"\n{'='*75}")
    print("  Ablation Summary")
    print(f"{'='*75}")
    print(f"  {'Config':<25s} {'Dim':>5s} {'R2':>12s} {'MAE':>10s} {'ΔR2':>10s}")
    print(f"  {'-'*65}")
    for name, r in results.items():
        dim = r["n_remaining"]
        delta = f"{r['delta_R2']:+.4f}" if name != "full_186d" else "baseline"
        print(f"  {name:<25s} {dim:>5d} {r['R2_mean']:.4f}+-{r['R2_std']:.4f} "
              f"{r['MAE_mean']:.1f}K {delta:>10s}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULT_DIR / "ablation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {RESULT_DIR / 'ablation.json'}")
    return results


# ── 3. 误差地图 ─────────────────────────────────────────────

def run_error_map(X_full, y, full_names, smiles_list):
    """Error map: analyze molecules with |error| > 50K."""
    from tabpfn import TabPFNRegressor
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    print(f"\n{'='*65}")
    print("  3. 误差地图 (|error| > 50K)")
    print(f"{'='*65}")

    # Collect OOF predictions from full CV
    pp = build_preprocessing()
    X_pp = pp.fit_transform(X_full)

    _, result = run_tabpfn_cv(X_full, y, "Error map (full 186d)")

    y_true = np.array(result["y_true_all"])
    y_pred = np.array(result["y_pred_all"])
    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    # High error analysis
    threshold = 50.0
    high_err_mask = abs_errors > threshold
    n_high = high_err_mask.sum()
    print(f"\n  |error| > {threshold}K: {n_high} samples ({100*n_high/len(y_true):.1f}%)")

    if n_high == 0:
        print("  No high-error samples found!")
        return {}

    # Analyze high-error molecules
    high_idx = np.where(high_err_mask)[0]
    analysis = {
        "threshold_K": threshold,
        "n_high_error": int(n_high),
        "pct_high_error": float(100 * n_high / len(y_true)),
        "mean_abs_error_high": float(abs_errors[high_err_mask].mean()),
        "mean_abs_error_low": float(abs_errors[~high_err_mask].mean()),
    }

    # Structural analysis using RDKit
    print(f"\n  分析高误差分子的结构特征...")
    ring_counts, aromatic_counts, mw_list, tg_list = [], [], [], []
    has_metal, has_halogen, has_si = 0, 0, 0

    for idx in high_idx:
        smi = smiles_list[idx] if idx < len(smiles_list) else None
        tg_list.append(float(y_true[idx]))

        if smi is None:
            continue
        try:
            mol = Chem.MolFromSmiles(smi.replace("[*]", "[H]"))
            if mol is None:
                continue
            ring_counts.append(rdMolDescriptors.CalcNumRings(mol))
            aromatic_counts.append(rdMolDescriptors.CalcNumAromaticRings(mol))
            mw_list.append(Descriptors.MolWt(mol))

            atoms = {a.GetSymbol() for a in mol.GetAtoms()}
            if atoms & {"Zn", "Cu", "Fe", "Pb", "Cd", "Ca", "Mn"}:
                has_metal += 1
            if atoms & {"F", "Cl", "Br", "I"}:
                has_halogen += 1
            if "Si" in atoms:
                has_si += 1
        except Exception:
            pass

    analysis["high_error_molecules"] = {
        "avg_ring_count": float(np.mean(ring_counts)) if ring_counts else None,
        "avg_aromatic_rings": float(np.mean(aromatic_counts)) if aromatic_counts else None,
        "avg_mol_weight": float(np.mean(mw_list)) if mw_list else None,
        "avg_tg_K": float(np.mean(tg_list)),
        "tg_range": [float(np.min(tg_list)), float(np.max(tg_list))],
        "pct_metal": float(100 * has_metal / n_high),
        "pct_halogen": float(100 * has_halogen / n_high),
        "pct_silicon": float(100 * has_si / n_high),
    }

    # Tg range distribution
    tg_bins = [(100, 200), (200, 300), (300, 400), (400, 500), (500, 600), (600, 900)]
    tg_dist_all = Counter()
    tg_dist_high = Counter()
    for t in y_true:
        for lo, hi in tg_bins:
            if lo <= t < hi:
                tg_dist_all[f"{lo}-{hi}K"] += 1
    for t in tg_list:
        for lo, hi in tg_bins:
            if lo <= t < hi:
                tg_dist_high[f"{lo}-{hi}K"] += 1

    print(f"\n  Tg 范围分布 (高误差 vs 全部):")
    print(f"  {'Range':<12s} {'High Err':>10s} {'Total':>10s} {'Enrichment':>12s}")
    print(f"  {'-'*46}")
    for lo, hi in tg_bins:
        key = f"{lo}-{hi}K"
        h = tg_dist_high.get(key, 0)
        a = tg_dist_all.get(key, 0)
        enrich = (h / n_high) / (a / len(y_true)) if a > 0 else 0
        print(f"  {key:<12s} {h:>10d} {a:>10d} {enrich:>10.2f}x")

    analysis["tg_distribution"] = {
        "high_error": dict(tg_dist_high),
        "all": dict(tg_dist_all),
    }

    # Error statistics
    print(f"\n  误差统计:")
    print(f"  全部样本: MAE={abs_errors.mean():.1f}K, RMSE={np.sqrt((errors**2).mean()):.1f}K")
    print(f"  高误差:   MAE={abs_errors[high_err_mask].mean():.1f}K (n={n_high})")
    print(f"  低误差:   MAE={abs_errors[~high_err_mask].mean():.1f}K (n={len(y_true)-n_high})")

    ha = analysis["high_error_molecules"]
    print(f"\n  高误差分子特征:")
    print(f"  平均 Tg: {ha['avg_tg_K']:.0f}K, 范围: {ha['tg_range'][0]:.0f}-{ha['tg_range'][1]:.0f}K")
    if ha['avg_ring_count'] is not None:
        print(f"  平均环数: {ha['avg_ring_count']:.1f}, 芳环: {ha['avg_aromatic_rings']:.1f}")
        print(f"  平均分子量: {ha['avg_mol_weight']:.0f}")
    print(f"  含金属: {ha['pct_metal']:.0f}%, 含卤素: {ha['pct_halogen']:.0f}%, 含硅: {ha['pct_silicon']:.0f}%")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULT_DIR / "error_map.json", "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n  Saved: {RESULT_DIR / 'error_map.json'}")
    return analysis


# ── 4. 核苷酸迁移测试 ──────────────────────────────────────

def run_nucleotide_transfer():
    """Nucleotide Tg prediction using L1H features (independent of new features)."""
    print(f"\n{'='*65}")
    print("  4. 核苷酸迁移测试 (L1H 特征集)")
    print(f"{'='*65}")

    # This reuses the existing nucleotide prediction pipeline
    # Load unified data with bridge samples
    from src.data.external_datasets import load_unified_dataset

    df = load_unified_dataset(str(DATA_PATH), split=None)
    print(f"  Unified dataset: {len(df)} samples")

    # Compute L1H features for all
    feat_names = get_feature_names("L1H")
    X_list, y_list, smi_list = [], [], []
    for _, row in df.iterrows():
        try:
            x = compute_features(row["smiles"], layer="L1H")
            X_list.append(x)
            y_list.append(float(row["tg_k"]))
            smi_list.append(row["smiles"])
        except Exception:
            pass

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"  L1H features: {X.shape[1]}d, {len(y)} samples")

    # Run TabPFN on L1H (baseline for nucleotide transfer)
    print("\n  TabPFN on L1H (全量统一数据集):")
    m_l1h, _ = run_tabpfn_cv(X, y, "L1H TabPFN")

    # Nucleotide targets
    nucleotides = {
        "ATP": {"smiles": "[*]OP(=O)(O)OP(=O)(O)OP(=O)(O)OC1OC(n2cnc3c(N)ncnc32)C(O)C1O[*]", "tg_exp": 378},
        "ADP": {"smiles": "[*]OP(=O)(O)OP(=O)(O)OC1OC(n2cnc3c(N)ncnc32)C(O)C1O[*]", "tg_exp": 441},
        "AMP": {"smiles": "[*]OP(=O)(O)OC1OC(n2cnc3c(N)ncnc32)C(O)C1O[*]", "tg_exp": 436},
    }

    # Train on full unified, predict nucleotides
    from tabpfn import TabPFNRegressor

    pp = build_preprocessing()
    X_pp = pp.fit_transform(X)

    model = TabPFNRegressor()
    model.fit(X_pp, y)

    print(f"\n  核苷酸预测结果:")
    print(f"  {'Name':<6s} {'Tg_exp (K)':>12s} {'Tg_pred (K)':>13s} {'Error (K)':>11s}")
    print(f"  {'-'*45}")

    nuc_results = {}
    for name, info in nucleotides.items():
        try:
            x_nuc = compute_features(info["smiles"], layer="L1H")
            x_nuc_pp = pp.transform(x_nuc.reshape(1, -1))
            tg_pred = model.predict(x_nuc_pp)[0]
            error = abs(tg_pred - info["tg_exp"])
            print(f"  {name:<6s} {info['tg_exp']:>12.0f} {tg_pred:>13.1f} {error:>11.1f}")
            nuc_results[name] = {
                "tg_exp": info["tg_exp"],
                "tg_pred": float(tg_pred),
                "error_K": float(error),
            }
        except Exception as e:
            print(f"  {name:<6s} Failed: {e}")
            nuc_results[name] = {"error": str(e)}

    avg_error = np.mean([v["error_K"] for v in nuc_results.values() if "error_K" in v])
    print(f"\n  平均误差: {avg_error:.1f}K")

    results = {
        "L1H_baseline": m_l1h,
        "nucleotide_predictions": nuc_results,
        "avg_nucleotide_error_K": float(avg_error),
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULT_DIR / "nucleotide_transfer.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {RESULT_DIR / 'nucleotide_transfer.json'}")
    return results


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["gc_residual", "ablation", "error_map", "nucleotide", "all"],
                        default="all")
    args = parser.parse_args()

    steps = [args.step] if args.step != "all" else ["gc_residual", "ablation", "error_map", "nucleotide"]

    # Load data once (except nucleotide which uses L1H)
    if any(s in steps for s in ["gc_residual", "ablation", "error_map"]):
        X_full, y, full_names, phyc_names, gnn_names, pbert_names, smiles = load_186d()

    if "gc_residual" in steps:
        run_gc_residual(X_full, y, full_names)

    if "ablation" in steps:
        run_ablation(X_full, y, full_names, phyc_names, gnn_names, pbert_names)

    if "error_map" in steps:
        run_error_map(X_full, y, full_names, smiles)

    if "nucleotide" in steps:
        run_nucleotide_transfer()

    print(f"\n{'='*65}")
    print("  Phase E 剩余实验全部完成!")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
