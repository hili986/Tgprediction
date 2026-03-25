"""
Compute chain segment physics features for the entire unified dataset.
全量计算链段物理特征 (N_eff + Cn_proxy, 3-mer 构象采样)

Supports checkpoint/resume: saves progress every 100 molecules.
Restart the same command to resume from last checkpoint.

Usage:
    python scripts/compute_chain_physics.py --n-confs 50 --n-jobs 16
    nohup python scripts/compute_chain_physics.py --n-confs 50 --n-jobs 16 > chain_physics.log 2>&1 &
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.chain_physics import compute_3mer_physics, chain_physics_feature_names


DATA_PATH = PROJECT_ROOT / "data" / "unified_tg.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "chain_physics_features.parquet"
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "chain_physics_checkpoint.json"
CHECKPOINT_INTERVAL = 20   # save every N molecules
MOLECULE_TIMEOUT = 120     # seconds per molecule, NaN if exceeded

_NAN = None  # initialized in main


def _make_nan():
    r = {name: float("nan") for name in chain_physics_feature_names()}
    r["oligomer_level"] = 0.0
    return r


def compute_one(args):
    """Compute chain physics for a single molecule."""
    idx, smiles, n_confs = args
    try:
        result = compute_3mer_physics(smiles, n_confs=n_confs)
        return idx, result
    except Exception:
        return idx, _make_nan()


def load_checkpoint():
    """Load checkpoint if exists. Returns dict {idx: result} or empty dict."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r") as f:
            data = json.load(f)
        print(f"  Loaded checkpoint: {len(data)} molecules already computed", flush=True)
        # Convert string keys back to int
        return {int(k): v for k, v in data.items()}
    return {}


def save_checkpoint(completed):
    """Save completed results to checkpoint file."""
    # Convert int keys to string for JSON
    data = {str(k): v for k, v in completed.items()}
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(data, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-confs", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Number of parallel jobs (16 recommended for this server)")
    parser.add_argument("--test", type=int, default=0,
                        help="Test on N samples only (0 = full dataset)")
    parser.add_argument("--reset", action="store_true",
                        help="Ignore checkpoint, start from scratch")
    args = parser.parse_args()

    from src.data.external_datasets import load_unified_dataset
    df = load_unified_dataset(str(DATA_PATH), split=None)

    if args.test > 0:
        df = df.head(args.test)

    n = len(df)
    print(f"Computing chain physics for {n} molecules", flush=True)
    print(f"  n_confs={args.n_confs}, n_jobs={args.n_jobs}", flush=True)
    print(f"  Output: {OUTPUT_PATH}", flush=True)

    # Load checkpoint
    completed = {} if args.reset else load_checkpoint()
    remaining_indices = [i for i in range(n) if i not in completed]
    print(f"  Remaining: {len(remaining_indices)} molecules", flush=True)

    if not remaining_indices:
        print("  All molecules already computed!", flush=True)
    else:
        # Build tasks for remaining molecules
        all_smiles = df["smiles"].astype(str).tolist()
        tasks = [(i, all_smiles[i], args.n_confs) for i in remaining_indices]
        print(f"  Tasks built, starting computation...", flush=True)

        t0 = time.time()

        n_jobs = args.n_jobs if args.n_jobs > 1 else 1
        nan_result = _make_nan()

        if n_jobs == 1:
            for i, task in enumerate(tasks):
                idx, result = compute_one(task)
                completed[idx] = result
                if (i + 1) % CHECKPOINT_INTERVAL == 0 or i == len(tasks) - 1:
                    save_checkpoint(completed)
                    done = len(completed)
                    elapsed = time.time() - t0
                    rate = (i + 1) / max(elapsed, 1)
                    eta = (len(tasks) - i - 1) / max(rate, 0.001)
                    print(f"  {done}/{n} ({100*done/n:.0f}%) "
                          f"{rate:.2f} mol/s, ETA {eta/60:.0f}min",
                          flush=True)
        else:
            print(f"  Using {n_jobs} cores, timeout={MOLECULE_TIMEOUT}s/mol",
                  flush=True)
            pool = Pool(n_jobs, maxtasksperchild=50)
            timeout_count = 0

            for batch_start in range(0, len(tasks), CHECKPOINT_INTERVAL):
                batch = tasks[batch_start:batch_start + CHECKPOINT_INTERVAL]
                async_results = [
                    (task[0], pool.apply_async(compute_one, (task,)))
                    for task in batch
                ]

                for idx, ar in async_results:
                    try:
                        _, result = ar.get(timeout=MOLECULE_TIMEOUT)
                        completed[idx] = result
                    except Exception:
                        completed[idx] = dict(nan_result)
                        timeout_count += 1

                save_checkpoint(completed)
                done = len(completed)
                elapsed = time.time() - t0
                rate = (batch_start + len(batch)) / max(elapsed, 1)
                eta = (len(tasks) - batch_start - len(batch)) / max(rate, 0.001)
                print(f"  {done}/{n} ({100*done/n:.1f}%) "
                      f"{rate:.2f} mol/s, ETA {eta/60:.0f}min "
                      f"[timeout={timeout_count}]",
                      flush=True)

            pool.terminate()
            pool.join()

        elapsed = time.time() - t0
        print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)

    # Assemble results into DataFrame
    feat_names = chain_physics_feature_names()
    feat_data = {name: np.full(n, np.nan) for name in feat_names}

    for idx, result in completed.items():
        for name in feat_names:
            feat_data[name][int(idx)] = result.get(name, float("nan"))

    feat_df = pd.DataFrame(feat_data)
    feat_df["smiles"] = df["smiles"].values[:n]
    feat_df["tg_k"] = df["tg_k"].values[:n]

    # Stats
    valid = feat_df[feat_df["oligomer_level"] > 0]
    print(f"\nValid: {len(valid)}/{n} ({100*len(valid)/n:.1f}%)", flush=True)

    if len(valid) > 10:
        from scipy.stats import spearmanr
        for col in ["Neff_300K", "Cn_proxy", "curl_ratio"]:
            vals = valid[[col, "tg_k"]].dropna()
            if len(vals) > 5:
                rho, p = spearmanr(vals[col], vals["tg_k"])
                print(f"  Spearman({col}, Tg) = {rho:.4f} (p={p:.2e})", flush=True)

    # Save final output
    feat_df.to_parquet(str(OUTPUT_PATH), index=False)
    print(f"\nSaved to {OUTPUT_PATH}", flush=True)

    # Clean up checkpoint after successful completion
    if CHECKPOINT_PATH.exists() and len(completed) == n:
        CHECKPOINT_PATH.unlink()
        print("  Checkpoint removed (computation complete)", flush=True)


if __name__ == "__main__":
    main()
