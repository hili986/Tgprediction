"""
Phase D Track A: GNN Pretrain → Finetune → Extract 64d Embeddings

Step 1: Pretrain PhysicsGAT on external data (~14K, not unified_tg)
Step 2: For each Nested CV outer fold: finetune on train → extract 64d for all
Step 3: Concatenate with PHY-C-light 58d → TabPFN verify

Usage:
    python scripts/phase_d_gnn.py --step pretrain     # Step 1 only (~2-4h)
    python scripts/phase_d_gnn.py --step embed         # Step 2-3 (needs pretrained weights)
    python scripts/phase_d_gnn.py                      # All steps
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.gnn.graph_builder import smiles_to_graph
from src.gnn.tandem_m2m import TandemM2M
from src.gnn.pretrainer import TgPretrainer
from src.data.external_datasets import load_all_external

DATA_PATH = PROJECT_ROOT / "data" / "unified_tg.parquet"
PHYC_CACHE = PROJECT_ROOT / "data" / "feature_matrix_PHY-C.parquet"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULT_DIR = PROJECT_ROOT / "results" / "phase_d"
GNN_EMBED_CACHE = PROJECT_ROOT / "data" / "gnn_embeddings_64d.parquet"


# ── Step 1: Pretrain ──────────────────────────────────────────────

def build_graph_dataset(smiles_list, tg_list, desc=""):
    """Convert SMILES + Tg to list of PyG Data objects."""
    graphs = []
    failed = 0
    for i, (smi, tg) in enumerate(zip(smiles_list, tg_list)):
        g = smiles_to_graph(smi, n_repeat=3, physics_features=True)
        if g is not None:
            g.y = torch.tensor([tg], dtype=torch.float)
            graphs.append(g)
        else:
            failed += 1
        if (i + 1) % 2000 == 0:
            print(f"  {desc} [{i+1}/{len(smiles_list)}] graphs={len(graphs)}, failed={failed}")
    print(f"  {desc}: {len(graphs)} graphs, {failed} failed ({100*failed/len(smiles_list):.1f}%)")
    return graphs


def run_pretrain(epochs: int = 100, batch_size: int = 64):
    """Step 1: Pretrain on external data."""
    print("=" * 60)
    print("  Step 1: GNN Pretrain on External Data")
    print("=" * 60)

    # Load external data (NOT unified_tg — zero leakage)
    print("\nLoading external datasets...")
    entries = load_all_external(verbose=True)
    smiles = [e["smiles"] for e in entries]
    tgs = [e["tg_k"] for e in entries]
    print(f"  Total external entries: {len(entries)}")

    # Convert to graphs
    print("\nBuilding graphs...")
    graphs = build_graph_dataset(smiles, tgs, desc="External")
    if len(graphs) < 100:
        print("ERROR: Too few valid graphs for pretraining!")
        return

    # Train/val split (90/10)
    n_val = max(int(len(graphs) * 0.1), 100)
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(graphs))
    val_graphs = [graphs[i] for i in indices[:n_val]]
    train_graphs = [graphs[i] for i in indices[n_val:]]

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size)
    print(f"  Train: {len(train_graphs)}, Val: {len(val_graphs)}")

    # Model: TandemM2M with tabular_dim=0 (pretrain without tabular features)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TandemM2M(
        in_dim=25, tabular_dim=0,  # No tabular features for pretraining
        gnn_hidden=128, gnn_out=64, gnn_heads=4,
        dropout=0.1, edge_dim=6,
        use_baseline=False,
    )
    print(f"\n  Model on {device}, params: {sum(p.numel() for p in model.parameters()):,}")

    trainer = TgPretrainer(model, device=device, tabular_dim=0)
    print(f"\n  Pretraining for {epochs} epochs...")
    t0 = time.time()
    result = trainer.pretrain(train_loader, val_loader, epochs=epochs)
    elapsed = time.time() - t0

    print(f"\n  Pretrain done in {elapsed:.0f}s")
    print(f"  Best val loss: {result['best_val_loss']:.4f}")

    # Save checkpoint
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(CHECKPOINT_DIR / "gnn_pretrained.pt")
    trainer.save_checkpoint(ckpt_path)
    return ckpt_path


# ── Step 2: Extract Embeddings ────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, graphs, device, batch_size=128):
    """Extract 64d graph embeddings from a trained model."""
    model.eval()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    embeddings = []
    for batch in loader:
        batch = batch.to(device)
        emb = model.get_embedding(batch)  # [B, 64]
        embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def run_embed_extraction():
    """Step 2: Extract 64d GNN embeddings for unified_tg."""
    print("=" * 60)
    print("  Step 2: Extract GNN Embeddings")
    print("=" * 60)

    ckpt_path = CHECKPOINT_DIR / "gnn_pretrained.pt"
    if not ckpt_path.exists():
        print(f"ERROR: No pretrained checkpoint at {ckpt_path}")
        print("Run: python scripts/phase_d_gnn.py --step pretrain")
        return None

    # Load unified data
    df = pd.read_parquet(DATA_PATH)
    smiles_list = df["smiles"].tolist()
    tg_list = df["tg_k"].tolist()

    # Convert to graphs
    print("\nBuilding graphs for unified_tg...")
    graphs = []
    graph_idx = []  # Track which samples have valid graphs
    for i, (smi, tg) in enumerate(zip(smiles_list, tg_list)):
        g = smiles_to_graph(smi, n_repeat=3, physics_features=True)
        if g is not None:
            g.y = torch.tensor([tg], dtype=torch.float)
            graphs.append(g)
            graph_idx.append(i)
        if (i + 1) % 2000 == 0:
            print(f"  [{i+1}/{len(smiles_list)}] valid={len(graphs)}")

    print(f"  Valid graphs: {len(graphs)}/{len(smiles_list)} ({100*len(graphs)/len(smiles_list):.1f}%)")

    # Load pretrained model and extract embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TandemM2M(
        in_dim=25, tabular_dim=0,
        gnn_hidden=128, gnn_out=64, gnn_heads=4,
        dropout=0.0, edge_dim=6,
        use_baseline=False,
    )
    trainer = TgPretrainer(model, device=device, tabular_dim=0)
    trainer.load_checkpoint(str(ckpt_path))

    print("\nExtracting 64d embeddings...")
    t0 = time.time()
    embeddings = extract_embeddings(model, graphs, device)
    print(f"  Extracted: {embeddings.shape} in {time.time()-t0:.1f}s")

    # Fill full matrix (NaN for failed graphs)
    full_embeddings = np.full((len(smiles_list), 64), np.nan)
    for idx, emb in zip(graph_idx, embeddings):
        full_embeddings[idx] = emb

    valid_rate = (~np.isnan(full_embeddings[:, 0])).sum() / len(smiles_list)
    print(f"  Coverage: {valid_rate:.1%}")

    # Save cache
    col_names = [f"GNN_{i}" for i in range(64)]
    df_embed = pd.DataFrame(full_embeddings, columns=col_names)
    df_embed["smiles"] = smiles_list
    df_embed.to_parquet(GNN_EMBED_CACHE, index=False)
    print(f"  Saved: {GNN_EMBED_CACHE}")

    return full_embeddings


# ── Step 3: TabPFN Verification ───────────────────────────────────

def run_tabpfn_verify(gnn_embeddings: np.ndarray):
    """Step 3: PHY-C-light 58d + GNN 64d → TabPFN."""
    from src.features.feature_pipeline import get_feature_names
    from src.ml.sklearn_models import build_preprocessing
    from src.ml.evaluation import nested_cv_no_tuning

    DROP = [
        "CP_oligomer_level", "CP_Cn_proxy",
        "L1_RingCount", "L1_HeavyAtomCount", "L1_MolWt",
        "IC_hydrophilic_ratio",
    ]

    print("\n" + "=" * 60)
    print("  Step 3: TabPFN Verification (PHY-C 58d + GNN 64d)")
    print("=" * 60)

    # Load PHY-C-light
    df_phyc = pd.read_parquet(PHYC_CACHE)
    all_names = get_feature_names("PHY-C")
    keep_names = [f for f in all_names if f not in DROP]
    X_phyc = df_phyc[keep_names].values
    y = df_phyc["tg_k"].values

    # Concatenate
    gnn_names = [f"GNN_{i}" for i in range(64)]
    X_combined = np.hstack([X_phyc, gnn_embeddings])
    combined_names = keep_names + gnn_names
    print(f"  Combined: {X_combined.shape[1]}d (PHY-C 58d + GNN 64d)")

    # Drop all-NaN rows
    valid = ~np.all(np.isnan(X_combined), axis=1)
    X_combined = X_combined[valid]
    y_valid = y[valid]
    print(f"  Valid samples: {len(y_valid)}")

    # Preprocessing
    pp = build_preprocessing()
    X_pp = pp.fit_transform(X_combined)

    # TabPFN
    try:
        from tabpfn import TabPFNRegressor
    except ImportError:
        print("TabPFN not installed. Skipping.")
        return

    estimator = TabPFNRegressor()
    print(f"\n  Running TabPFN Nested CV (5x3 folds) on {X_pp.shape[1]}d...")
    result = nested_cv_no_tuning(X_pp, y_valid, estimator, outer_splits=5, outer_repeats=3, verbose=True)

    metrics = result["metrics"]
    print(f"\n{'='*65}")
    print(f"  TabPFN + PHY-C-light + GNN ({X_pp.shape[1]}d) Results:")
    print(f"  R2  = {metrics['R2_mean']:.4f} +/- {metrics['R2_std']:.4f}")
    print(f"  MAE = {metrics['MAE_mean']:.1f}K +/- {metrics['MAE_std']:.1f}K")
    print(f"{'='*65}")

    # Compare with Phase C baseline
    print(f"\n  Comparison:")
    ref = RESULT_DIR.parent / "phase_c" / "tabpfn_PHY-C-light.json"
    if ref.exists():
        with open(ref) as f:
            m = json.load(f)["metrics"]
        print(f"  TabPFN PHY-C-light 58d:      R2={m['R2_mean']:.4f}, MAE={m['MAE_mean']:.1f}K")
    print(f"  TabPFN PHY-C-light+GNN {X_pp.shape[1]}d: R2={metrics['R2_mean']:.4f}, MAE={metrics['MAE_mean']:.1f}K")

    # Save
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / "tabpfn_phyc_gnn.json"
    with open(out, "w") as f:
        json.dump({
            "experiment": "TabPFN_PHY-C-light_GNN",
            "model": "TabPFN_v2",
            "n_features": X_pp.shape[1],
            "n_samples": len(y_valid),
            "feature_composition": {"PHY-C-light": 58, "GNN_embed": 64},
            "metrics": metrics,
            "outer_cv": "5x3",
        }, f, indent=2)
    print(f"  Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="Phase D GNN Track A")
    parser.add_argument("--step", choices=["pretrain", "embed", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if args.step in ("pretrain", "all"):
        run_pretrain(epochs=args.epochs, batch_size=args.batch_size)

    if args.step in ("embed", "all"):
        if GNN_EMBED_CACHE.exists():
            print(f"\nLoading cached GNN embeddings: {GNN_EMBED_CACHE}")
            df = pd.read_parquet(GNN_EMBED_CACHE)
            gnn_cols = [c for c in df.columns if c.startswith("GNN_")]
            embeddings = df[gnn_cols].values
        else:
            embeddings = run_embed_extraction()
            if embeddings is None:
                return

        run_tabpfn_verify(embeddings)


if __name__ == "__main__":
    main()
