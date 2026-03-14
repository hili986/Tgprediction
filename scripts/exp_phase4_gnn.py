"""
Phase 4 GNN Experiments (E9-E15) — Run on A800 GPU
Phase 4 GNN 实验 — 在 A800 GPU 上运行

Experiments:
    E9:  Tandem-M2M, no pretrain (GNN baseline)
    E10: + pretrain on ~59K data
    E11: VPD-Deep (GNN 3-mer embedding)
    E12: PPF+VPD+GNN(64d) -> GBR fusion
    E13: + multitask (Tg+density+sol)
    E14: Deep Ensemble x5 + Conformal
    E15: M2M-Deep full framework

Usage:
    python scripts/exp_phase4_gnn.py [--exp E9] [--device cuda]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _ensure_results_dir():
    os.makedirs("results/phase4", exist_ok=True)


def _save_result(exp_id: str, result: dict):
    _ensure_results_dir()
    path = f"results/phase4/exp_{exp_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"Result saved to {path}")


def run_e9(device="cuda"):
    """E9: Tandem-M2M, no pretrain — GNN baseline."""
    import torch
    from src.data.bicerano_tg_dataset import BICERANO_TG_DATA
    from src.features.feature_pipeline import build_dataset_v2
    from src.ml.gnn_evaluation import nested_cv_gnn

    print("=" * 60)
    print("E9: Tandem-M2M GNN baseline (no pretrain)")
    print("=" * 60)

    X, y, feat_names, smiles_list = build_dataset_v2(layer="M2M")

    result = nested_cv_gnn(
        smiles_list=smiles_list,
        y=y,
        tabular=X,
        feature_names=feat_names,
        n_splits=5,
        n_repeats=3,
        pretrain_data=None,
        finetune_epochs=50,
        patience=10,
        device=device,
        gnn_config={"in_dim": 25, "gnn_hidden": 128, "gnn_out": 64},
    )

    result["experiment"] = "E9"
    result["description"] = "Tandem-M2M GNN, no pretrain"
    result["features"] = "M2M 56d + graph"
    _save_result("E9", result)
    print(f"E9 Result: R2={result['R2_mean']:.4f} +/- {result['R2_std']:.4f}")
    return result


def run_e10(device="cuda"):
    """E10: Tandem-M2M + pretrain on ~59K data."""
    import torch
    from src.features.feature_pipeline import build_dataset_v2
    from src.data.external_datasets import build_extended_dataset
    from src.ml.gnn_evaluation import nested_cv_gnn

    print("=" * 60)
    print("E10: Tandem-M2M + pretrain (~59K)")
    print("=" * 60)

    X, y, feat_names, smiles_list = build_dataset_v2(layer="M2M")

    # Build pretrain data from external sources
    try:
        ext_X, ext_y, _, ext_smiles = build_extended_dataset(layer="M2M")
        pretrain_data = {
            "smiles": ext_smiles,
            "y": ext_y.tolist(),
            "tabular": ext_X,
        }
        print(f"Pretrain data: {len(ext_smiles)} polymers")
    except Exception as e:
        print(f"Warning: Cannot load pretrain data: {e}")
        print("Running without pretrain (same as E9)")
        pretrain_data = None

    result = nested_cv_gnn(
        smiles_list=smiles_list,
        y=y,
        tabular=X,
        feature_names=feat_names,
        n_splits=5,
        n_repeats=3,
        pretrain_data=pretrain_data,
        pretrain_epochs=100,
        finetune_epochs=50,
        patience=10,
        device=device,
    )

    result["experiment"] = "E10"
    result["description"] = "Tandem-M2M + pretrain"
    _save_result("E10", result)
    print(f"E10 Result: R2={result['R2_mean']:.4f} +/- {result['R2_std']:.4f}")
    return result


def run_e11(device="cuda"):
    """E11: VPD-Deep — GNN as 3-mer embedding extractor."""
    import torch
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import RepeatedKFold
    from sklearn.metrics import r2_score, mean_absolute_error
    from src.features.feature_pipeline import build_dataset_v2
    from src.gnn.graph_builder import batch_smiles_to_graphs
    from src.gnn.tandem_m2m import TandemM2M
    from src.gnn.pretrainer import TgPretrainer
    from torch_geometric.loader import DataLoader

    print("=" * 60)
    print("E11: VPD-Deep (GNN 3-mer embedding)")
    print("=" * 60)

    X, y, feat_names, smiles_list = build_dataset_v2(layer="M2M")

    # Build graphs
    graphs = batch_smiles_to_graphs(smiles_list, y_list=y.tolist())
    valid_idx = list(range(len(graphs)))

    if len(graphs) < len(smiles_list):
        print(f"Warning: {len(smiles_list) - len(graphs)} SMILES failed to convert")

    # Train a GNN to get embeddings, then use GBR on embeddings
    model = TandemM2M(in_dim=25, tabular_dim=X.shape[1])
    model.to(device)

    # Quick train to get meaningful embeddings
    for g in graphs:
        g.tabular = torch.tensor(X[valid_idx[graphs.index(g)]], dtype=torch.float).unsqueeze(0)

    loader = DataLoader(graphs, batch_size=32, shuffle=True)
    trainer = TgPretrainer(model, device=device, lr_pretrain=1e-3)
    trainer.pretrain(loader, epochs=30)

    # Extract embeddings
    model.eval()
    embeddings = []
    with torch.no_grad():
        for g in graphs:
            g_dev = g.to(device)
            g_dev.batch = torch.zeros(g_dev.x.size(0), dtype=torch.long, device=device)
            emb = model.get_embedding(g_dev).cpu().numpy()
            embeddings.append(emb.squeeze())

    embeddings = np.array(embeddings)  # [N, 64]

    # Combine with tabular: [N, 56+64=120]
    X_combined = np.hstack([X[:len(embeddings)], embeddings])
    y_valid = y[:len(embeddings)]

    # Nested CV with GBR on combined features
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    r2_scores, mae_scores = [], []

    for train_idx, test_idx in cv.split(X_combined):
        gbr = GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            random_state=42,
        )
        gbr.fit(X_combined[train_idx], y_valid[train_idx])
        y_pred = gbr.predict(X_combined[test_idx])
        r2_scores.append(r2_score(y_valid[test_idx], y_pred))
        mae_scores.append(mean_absolute_error(y_valid[test_idx], y_pred))

    result = {
        "experiment": "E11",
        "description": "VPD-Deep: GNN embedding + tabular -> GBR",
        "R2_mean": float(np.mean(r2_scores)),
        "R2_std": float(np.std(r2_scores)),
        "MAE_mean": float(np.mean(mae_scores)),
        "MAE_std": float(np.std(mae_scores)),
        "features": f"M2M 56d + GNN 64d = {X_combined.shape[1]}d",
        "model": "GBR on combined features",
    }
    _save_result("E11", result)
    print(f"E11 Result: R2={result['R2_mean']:.4f} +/- {result['R2_std']:.4f}")
    return result


def run_e12(device="cuda"):
    """E12: PPF+VPD+GNN(64d) -> GBR fusion (cross-paradigm)."""
    import torch
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import RepeatedKFold
    from sklearn.metrics import r2_score, mean_absolute_error
    from src.features.feature_pipeline import build_dataset_v2
    from src.gnn.graph_builder import batch_smiles_to_graphs
    from src.gnn.tandem_m2m import TandemM2M
    from src.gnn.pretrainer import TgPretrainer
    from torch_geometric.loader import DataLoader

    print("=" * 60)
    print("E12: PPF+VPD+GNN(64d) -> GBR fusion")
    print("=" * 60)

    X, y, feat_names, smiles_list = build_dataset_v2(layer="M2M")

    # Build and pretrain GNN
    graphs = batch_smiles_to_graphs(smiles_list, y_list=y.tolist())

    model = TandemM2M(in_dim=25, tabular_dim=X.shape[1])
    model.to(device)

    for i, g in enumerate(graphs):
        g.tabular = torch.tensor(X[i], dtype=torch.float).unsqueeze(0)

    loader = DataLoader(graphs, batch_size=32, shuffle=True)
    trainer = TgPretrainer(model, device=device)
    trainer.pretrain(loader, epochs=50)

    # Try loading pretrain data for better embeddings
    try:
        from src.data.external_datasets import build_extended_dataset
        ext_X, ext_y, _, ext_smiles = build_extended_dataset(layer="M2M")
        ext_graphs = batch_smiles_to_graphs(ext_smiles, y_list=ext_y.tolist())
        for i, g in enumerate(ext_graphs):
            g.tabular = torch.tensor(ext_X[i], dtype=torch.float).unsqueeze(0)
        ext_loader = DataLoader(ext_graphs, batch_size=256, shuffle=True)
        trainer.pretrain(ext_loader, epochs=50)
        print(f"Pretrained on {len(ext_graphs)} external polymers")
    except Exception as e:
        print(f"Pretrain data unavailable: {e}")

    # Finetune
    trainer.finetune(loader, epochs=30, patience=10)

    # Extract embeddings
    model.eval()
    embeddings = []
    with torch.no_grad():
        for g in graphs:
            g_dev = g.to(device)
            g_dev.batch = torch.zeros(g_dev.x.size(0), dtype=torch.long, device=device)
            emb = model.get_embedding(g_dev).cpu().numpy()
            embeddings.append(emb.squeeze())

    embeddings = np.array(embeddings)
    X_fused = np.hstack([X[:len(embeddings)], embeddings])
    y_valid = y[:len(embeddings)]

    # Nested CV with GBR
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    r2_scores, mae_scores = [], []

    for train_idx, test_idx in cv.split(X_fused):
        gbr = GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            random_state=42,
        )
        gbr.fit(X_fused[train_idx], y_valid[train_idx])
        y_pred = gbr.predict(X_fused[test_idx])
        r2_scores.append(r2_score(y_valid[test_idx], y_pred))
        mae_scores.append(mean_absolute_error(y_valid[test_idx], y_pred))

    result = {
        "experiment": "E12",
        "description": "PPF+VPD+GNN(64d) -> GBR cross-paradigm fusion",
        "R2_mean": float(np.mean(r2_scores)),
        "R2_std": float(np.std(r2_scores)),
        "MAE_mean": float(np.mean(mae_scores)),
        "MAE_std": float(np.std(mae_scores)),
        "features": f"M2M 56d + GNN 64d = {X_fused.shape[1]}d",
        "model": "GBR on fused features",
    }
    _save_result("E12", result)
    print(f"E12 Result: R2={result['R2_mean']:.4f} +/- {result['R2_std']:.4f}")
    return result


def run_e13(device="cuda"):
    """E13: Multitask (Tg + density + solubility param)."""
    import torch
    from src.features.feature_pipeline import build_dataset_v2
    from src.gnn.graph_builder import batch_smiles_to_graphs
    from src.gnn.multitask import MultiTaskTgModel
    from torch_geometric.loader import DataLoader
    from sklearn.model_selection import RepeatedKFold
    from sklearn.metrics import r2_score, mean_absolute_error

    print("=" * 60)
    print("E13: Multitask GNN (Tg + auxiliary tasks)")
    print("=" * 60)

    X, y, feat_names, smiles_list = build_dataset_v2(layer="M2M")
    graphs = batch_smiles_to_graphs(smiles_list, y_list=y.tolist())

    # Nested CV
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    r2_scores, mae_scores = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(y)):
        train_graphs = [graphs[i] for i in train_idx if i < len(graphs)]
        test_graphs = [graphs[i] for i in test_idx if i < len(graphs)]

        if len(train_graphs) == 0 or len(test_graphs) == 0:
            continue

        model = MultiTaskTgModel(in_dim=25).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

        # Train
        model.train()
        for epoch in range(50):
            for batch in train_loader:
                batch = batch.to(device)
                preds = model(batch)
                targets = {"tg": batch.y.squeeze()}
                loss = model.compute_loss(preds, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                preds = model(batch)
                pred = preds["tg"].squeeze().cpu().numpy()
                true = batch.y.squeeze().cpu().numpy()
                if pred.ndim == 0:
                    pred = np.array([pred.item()])
                    true = np.array([true.item()])
                all_preds.extend(pred.tolist())
                all_true.extend(true.tolist())

        if len(all_true) > 1:
            r2_scores.append(r2_score(all_true, all_preds))
            mae_scores.append(mean_absolute_error(all_true, all_preds))

    result = {
        "experiment": "E13",
        "description": "Multitask GNN (Tg + density + sol_param)",
        "R2_mean": float(np.mean(r2_scores)) if r2_scores else 0,
        "R2_std": float(np.std(r2_scores)) if r2_scores else 0,
        "MAE_mean": float(np.mean(mae_scores)) if mae_scores else 0,
        "MAE_std": float(np.std(mae_scores)) if mae_scores else 0,
        "model": "MultiTaskTgModel",
    }
    _save_result("E13", result)
    print(f"E13 Result: R2={result['R2_mean']:.4f} +/- {result['R2_std']:.4f}")
    return result


def run_e14(device="cuda"):
    """E14: Deep Ensemble x5 + Conformal Prediction."""
    import torch
    from src.features.feature_pipeline import build_dataset_v2
    from src.gnn.graph_builder import batch_smiles_to_graphs
    from src.gnn.tandem_m2m import TandemM2M
    from src.gnn.ensemble import DeepEnsembleTg
    from torch_geometric.loader import DataLoader
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error

    print("=" * 60)
    print("E14: Deep Ensemble x5 + Conformal")
    print("=" * 60)

    X, y, feat_names, smiles_list = build_dataset_v2(layer="M2M")
    graphs = batch_smiles_to_graphs(smiles_list, y_list=y.tolist())

    for i, g in enumerate(graphs):
        g.tabular = torch.tensor(X[i], dtype=torch.float).unsqueeze(0)

    # Split: train 60%, cal 20%, test 20%
    n = len(graphs)
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)
    train_idx, cal_idx = train_test_split(train_idx, test_size=0.25, random_state=42)

    train_graphs = [graphs[i] for i in train_idx]
    cal_graphs = [graphs[i] for i in cal_idx]
    test_graphs = [graphs[i] for i in test_idx]

    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    cal_loader = DataLoader(cal_graphs, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

    tabular_dim = X.shape[1]

    def model_fn():
        return TandemM2M(in_dim=25, tabular_dim=tabular_dim)

    ensemble = DeepEnsembleTg(model_fn, n_models=5, device=device)

    # Train ensemble
    ensemble.fit(
        finetune_loader=train_loader,
        finetune_epochs=50,
        patience=10,
    )

    # Calibrate
    ensemble.calibrate(cal_loader, confidence=0.9)

    # Evaluate on test set
    all_preds, all_true, all_lower, all_upper = [], [], [], []
    for batch in test_loader:
        batch = batch.to(device)
        tab = batch.tabular if hasattr(batch, "tabular") else torch.zeros(
            batch.num_graphs, tabular_dim, device=device
        )
        pred, lower, upper = ensemble.predict_interval(batch, tab)
        true = batch.y.squeeze().cpu().numpy()
        all_preds.extend(pred.tolist())
        all_true.extend(true.tolist() if true.ndim > 0 else [true.item()])
        all_lower.extend(lower.tolist())
        all_upper.extend(upper.tolist())

    y_true = np.array(all_true)
    y_pred = np.array(all_preds)
    y_lower = np.array(all_lower)
    y_upper = np.array(all_upper)

    coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))
    avg_width = np.mean(y_upper - y_lower)

    result = {
        "experiment": "E14",
        "description": "Deep Ensemble x5 + Conformal Prediction",
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "coverage_90": float(coverage),
        "avg_interval_width": float(avg_width),
        "n_test": len(y_true),
        "model": "DeepEnsemble(TandemM2M) x5",
    }
    _save_result("E14", result)
    print(f"E14 Result: R2={result['R2']:.4f}, Coverage={coverage:.3f}, Width={avg_width:.1f}K")
    return result


def run_e15(device="cuda"):
    """E15: M2M-Deep full framework."""
    import torch
    from src.features.feature_pipeline import build_dataset_v2
    from src.gnn.graph_builder import batch_smiles_to_graphs
    from src.gnn.tandem_m2m import TandemM2M
    from src.gnn.ensemble import DeepEnsembleTg
    from src.gnn.pretrainer import TgPretrainer
    from torch_geometric.loader import DataLoader
    from sklearn.model_selection import RepeatedKFold
    from sklearn.metrics import r2_score, mean_absolute_error

    print("=" * 60)
    print("E15: M2M-Deep Full Framework")
    print("=" * 60)

    X, y, feat_names, smiles_list = build_dataset_v2(layer="M2M")

    # Try to load pretrain data
    pretrain_loader = None
    try:
        from src.data.external_datasets import build_extended_dataset
        ext_X, ext_y, _, ext_smiles = build_extended_dataset(layer="M2M")
        ext_graphs = batch_smiles_to_graphs(ext_smiles, y_list=ext_y.tolist())
        for i, g in enumerate(ext_graphs):
            g.tabular = torch.tensor(ext_X[i], dtype=torch.float).unsqueeze(0)
        pretrain_loader = DataLoader(ext_graphs, batch_size=256, shuffle=True)
        print(f"Pretrain data: {len(ext_graphs)} polymers")
    except Exception as e:
        print(f"Pretrain data unavailable: {e}")

    # Build Bicerano graphs
    graphs = batch_smiles_to_graphs(smiles_list, y_list=y.tolist())
    for i, g in enumerate(graphs):
        g.tabular = torch.tensor(X[i], dtype=torch.float).unsqueeze(0)

    tabular_dim = X.shape[1]

    # Nested CV
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    r2_scores, mae_scores = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(y)):
        train_graphs = [graphs[i] for i in train_idx if i < len(graphs)]
        test_graphs = [graphs[i] for i in test_idx if i < len(graphs)]

        train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

        # Create model
        model = TandemM2M(in_dim=25, tabular_dim=tabular_dim)
        trainer = TgPretrainer(model, device=device)

        # Pretrain if available
        if pretrain_loader is not None:
            trainer.pretrain(pretrain_loader, epochs=50)

        # Finetune
        trainer.finetune(train_loader, epochs=50, patience=10)

        # Evaluate
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                tab = batch.tabular if hasattr(batch, "tabular") else torch.zeros(
                    batch.num_graphs, tabular_dim, device=device
                )
                pred = model(batch, tab).squeeze().cpu().numpy()
                true = batch.y.squeeze().cpu().numpy()
                if pred.ndim == 0:
                    pred = np.array([pred.item()])
                    true = np.array([true.item()])
                all_preds.extend(pred.tolist())
                all_true.extend(true.tolist())

        if len(all_true) > 1:
            r2_scores.append(r2_score(all_true, all_preds))
            mae_scores.append(mean_absolute_error(all_true, all_preds))

        print(f"Fold {fold_idx+1}/15: R2={r2_scores[-1]:.4f}")

    result = {
        "experiment": "E15",
        "description": "M2M-Deep full framework (pretrain + finetune + Nested CV)",
        "R2_mean": float(np.mean(r2_scores)),
        "R2_std": float(np.std(r2_scores)),
        "MAE_mean": float(np.mean(mae_scores)),
        "MAE_std": float(np.std(mae_scores)),
        "n_folds": len(r2_scores),
        "model": "TandemM2M (pretrained)",
        "features": f"M2M {tabular_dim}d + graph 25d atoms + 6d edges",
    }
    _save_result("E15", result)
    print(f"\nE15 Result: R2={result['R2_mean']:.4f} +/- {result['R2_std']:.4f}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    "E9": run_e9,
    "E10": run_e10,
    "E11": run_e11,
    "E12": run_e12,
    "E13": run_e13,
    "E14": run_e14,
    "E15": run_e15,
}


def main():
    parser = argparse.ArgumentParser(description="Phase 4 GNN Experiments")
    parser.add_argument(
        "--exp", type=str, default=None,
        help="Run specific experiment (E9-E15). Default: run all.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Compute device (default: cuda)",
    )
    args = parser.parse_args()

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    print(f"Device: {args.device}")
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.exp:
        exps = [e.strip() for e in args.exp.split(",")]
    else:
        exps = list(EXPERIMENTS.keys())

    results = {}
    for exp_id in exps:
        if exp_id not in EXPERIMENTS:
            print(f"Unknown experiment: {exp_id}")
            continue
        start = time.time()
        try:
            result = EXPERIMENTS[exp_id](device=args.device)
            result["runtime_seconds"] = time.time() - start
            results[exp_id] = result
        except Exception as e:
            print(f"ERROR in {exp_id}: {e}")
            import traceback
            traceback.print_exc()
            results[exp_id] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for exp_id, r in results.items():
        if "error" in r:
            print(f"  {exp_id}: ERROR - {r['error']}")
        elif "R2_mean" in r:
            print(f"  {exp_id}: R2={r['R2_mean']:.4f} +/- {r['R2_std']:.4f}")
        elif "R2" in r:
            print(f"  {exp_id}: R2={r['R2']:.4f}")


if __name__ == "__main__":
    main()
