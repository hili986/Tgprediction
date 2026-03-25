"""
Compute chain segment physics features for the entire unified dataset.
全量计算链段物理特征 (N_eff + Cn_proxy, 3-mer 构象采样)

Usage:
    python scripts/compute_chain_physics.py                # full (50 confs, ~4h)
    python scripts/compute_chain_physics.py --n-confs 30   # faster (~2.5h)
    python scripts/compute_chain_physics.py --n-jobs 16    # limit cores
    python scripts/compute_chain_physics.py --test 100     # test on 100 samples
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.chain_physics import compute_3mer_physics, chain_physics_feature_names


DATA_PATH = PROJECT_ROOT / "data" / "unified_tg.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "chain_physics_features.parquet"


def compute_one(args):
    """Compute chain physics for a single molecule (for joblib)."""
    idx, smiles, n_confs = args
    try:
        result = compute_3mer_physics(smiles, n_confs=n_confs)
        return idx, result
    except Exception:
        nan_result = {name: float("nan") for name in chain_physics_feature_names()}
        nan_result["oligomer_level"] = 0.0
        return idx, nan_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-confs", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Number of parallel jobs (1=sequential for GPU, -1=all cores for CPU)")
    parser.add_argument("--test", type=int, default=0,
                        help="Test on N samples only (0 = full dataset)")
    args = parser.parse_args()

    from src.data.external_datasets import load_unified_dataset
    df = load_unified_dataset(str(DATA_PATH), split=None)

    if args.test > 0:
        df = df.head(args.test)

    n = len(df)
    print(f"Computing chain physics for {n} molecules")
    print(f"  n_confs={args.n_confs}, n_jobs={args.n_jobs}")
    print(f"  Output: {OUTPUT_PATH}")

    # Prepare args for parallel computation
    tasks = [(i, str(row["smiles"]), args.n_confs)
             for i, (_, row) in enumerate(df.iterrows())]

    t0 = time.time()

    if args.n_jobs == 1:
        import sys
        results = []
        print("  Starting...", flush=True)
        for i, task in enumerate(tasks):
            results.append(compute_one(task))
            if (i + 1) % 10 == 0 or (i + 1) == 1:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (n - i - 1) / rate
                print(f"  {i+1}/{n} ({100*(i+1)/n:.0f}%) "
                      f"{rate:.2f} mol/s, ETA {eta/60:.0f}min", flush=True)
    else:
        from joblib import Parallel, delayed
        n_jobs = args.n_jobs if args.n_jobs > 0 else os.cpu_count()
        print(f"  Using {n_jobs} cores")

        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(compute_one)(task) for task in tasks
        )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Assemble results into DataFrame
    feat_names = chain_physics_feature_names()
    feat_data = {name: np.full(n, np.nan) for name in feat_names}

    for idx, result in results:
        for name in feat_names:
            feat_data[name][idx] = result.get(name, np.nan)

    feat_df = pd.DataFrame(feat_data)

    # Add smiles and tg_k for reference
    feat_df["smiles"] = df["smiles"].values[:n]
    feat_df["tg_k"] = df["tg_k"].values[:n]

    # Stats
    valid = feat_df[feat_df["oligomer_level"] > 0]
    print(f"\nValid: {len(valid)}/{n} ({100*len(valid)/n:.1f}%)")
    print(f"NaN rate: {100*(1-len(valid)/n):.1f}%")

    if len(valid) > 10:
        from scipy.stats import spearmanr
        for col in ["Neff_300K", "Cn_proxy", "curl_ratio"]:
            vals = valid[[col, "tg_k"]].dropna()
            if len(vals) > 5:
                rho, p = spearmanr(vals[col], vals["tg_k"])
                print(f"  Spearman({col}, Tg) = {rho:.4f} (p={p:.2e})")

    # Save
    feat_df.to_parquet(str(OUTPUT_PATH), index=False)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
