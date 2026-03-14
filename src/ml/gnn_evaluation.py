"""
GNN Evaluation Module — Adapts GNN to Nested CV Framework
GNN 评估模块 — 适配 GNN 到 Nested CV 框架

Provides nested cross-validation for GNN models, with each fold doing:
    pretrain -> finetune -> evaluate

Output format is compatible with evaluation.nested_cv() results.

Public API:
    nested_cv_gnn(smiles, y, tabular, n_splits=5, n_repeats=3, ...) -> dict
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def nested_cv_gnn(
    smiles_list: List[str],
    y: np.ndarray,
    tabular: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
    pretrain_data: Optional[dict] = None,
    pretrain_epochs: int = 100,
    finetune_epochs: int = 50,
    patience: int = 10,
    batch_size_pretrain: int = 256,
    batch_size_finetune: int = 32,
    device: str = "cuda",
    gnn_config: Optional[dict] = None,
) -> Dict:
    """Nested CV for GNN models.

    Each fold:
      1. Build graphs for train/test SMILES
      2. (Optional) Pretrain on external data
      3. Finetune on train fold
      4. Evaluate on test fold

    Args:
        smiles_list: List of polymer SMILES.
        y: Target Tg values [N].
        tabular: Optional tabular features [N, D].
        feature_names: Feature names for tabular.
        n_splits: CV splits (default: 5).
        n_repeats: CV repeats (default: 3).
        random_state: Random seed.
        pretrain_data: Dict with "smiles", "y", "tabular" for pretraining.
        pretrain_epochs: Pretraining epochs.
        finetune_epochs: Finetuning epochs.
        patience: Early stopping patience.
        batch_size_pretrain: Pretrain batch size.
        batch_size_finetune: Finetune batch size.
        device: Compute device.
        gnn_config: Optional GNN hyperparameters.

    Returns:
        Dict compatible with evaluation.nested_cv():
            R2_mean, R2_std, MAE_mean, MAE_std, RMSE_mean, RMSE_std,
            fold_results, model_name
    """
    try:
        import torch
        from torch_geometric.loader import DataLoader
        from src.gnn.graph_builder import smiles_to_graph, batch_smiles_to_graphs
        from src.gnn.tandem_m2m import TandemM2M
        from src.gnn.pretrainer import TgPretrainer
    except ImportError as e:
        raise ImportError(
            f"PyTorch/PyG required for GNN evaluation: {e}. "
            "Install: pip install torch torch-geometric"
        )

    config = gnn_config or {}
    tabular_dim = tabular.shape[1] if tabular is not None else 1

    # Strategy B: pre-filter invalid SMILES, then split on valid subset
    # This ensures consistent fold assignments across all experiments (E9-E15)
    all_graphs, valid_idx = batch_smiles_to_graphs(
        smiles_list, y_list=y.tolist(),
    )
    valid_idx = np.array(valid_idx)
    y_valid = y[valid_idx]
    tabular_valid = tabular[valid_idx] if tabular is not None else None

    # Attach tabular features once during pre-build (avoids per-fold mutation)
    if tabular_valid is not None:
        for i, g in enumerate(all_graphs):
            g.tabular = torch.tensor(
                tabular_valid[i], dtype=torch.float
            ).unsqueeze(0)

    n_filtered = len(smiles_list) - len(valid_idx)
    if n_filtered > 0:
        print(f"Pre-filtered {n_filtered} invalid SMILES, {len(valid_idx)} remain")

    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    fold_r2, fold_mae, fold_rmse = [], [], []
    fold_results = []

    # Build pretrain graphs once if provided
    pretrain_loader = None
    if pretrain_data is not None:
        pretrain_graphs, pretrain_valid_idx = batch_smiles_to_graphs(
            pretrain_data["smiles"],
            y_list=pretrain_data["y"],
        )
        if pretrain_data.get("tabular") is not None:
            for i, g in enumerate(pretrain_graphs):
                orig_i = pretrain_valid_idx[i]
                g.tabular = torch.tensor(
                    pretrain_data["tabular"][orig_i], dtype=torch.float
                ).unsqueeze(0)
        pretrain_loader = DataLoader(
            pretrain_graphs, batch_size=batch_size_pretrain, shuffle=True,
        )

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(y_valid)):
        # Graphs already built — select by fold index
        train_graphs = [all_graphs[i] for i in train_idx]
        test_graphs = [all_graphs[i] for i in test_idx]

        train_loader = DataLoader(
            train_graphs, batch_size=batch_size_finetune, shuffle=True,
        )
        test_loader = DataLoader(
            test_graphs, batch_size=batch_size_finetune, shuffle=False,
        )

        # Create fresh model for each fold
        model = TandemM2M(
            in_dim=config.get("in_dim", 25),
            tabular_dim=tabular_dim,
            gnn_hidden=config.get("gnn_hidden", 128),
            gnn_out=config.get("gnn_out", 64),
            dropout=config.get("dropout", 0.1),
            edge_dim=config.get("edge_dim", 6),
        )

        trainer = TgPretrainer(model, device=device)

        # Stage 1: Pretrain
        if pretrain_loader is not None:
            trainer.pretrain(pretrain_loader, epochs=pretrain_epochs)

        # Stage 2: Finetune
        trainer.finetune(train_loader, epochs=finetune_epochs, patience=patience)

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
                all_preds.extend(pred.tolist() if pred.ndim > 0 else [pred.item()])
                all_true.extend(true.tolist() if true.ndim > 0 else [true.item()])

        y_pred = np.array(all_preds)
        y_true = np.array(all_true)

        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        fold_r2.append(r2)
        fold_mae.append(mae)
        fold_rmse.append(rmse)

        fold_results.append({
            "fold": fold_idx,
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse,
            "n_train": len(train_graphs),
            "n_test": len(test_graphs),
        })

        print(
            f"Fold {fold_idx+1}/{n_splits*n_repeats}: "
            f"R2={r2:.4f}, MAE={mae:.1f}K, RMSE={rmse:.1f}K"
        )

    return {
        "R2_mean": float(np.mean(fold_r2)),
        "R2_std": float(np.std(fold_r2)),
        "MAE_mean": float(np.mean(fold_mae)),
        "MAE_std": float(np.std(fold_mae)),
        "RMSE_mean": float(np.mean(fold_rmse)),
        "RMSE_std": float(np.std(fold_rmse)),
        "fold_results": fold_results,
        "model_name": "TandemM2M-GNN",
        "n_folds": n_splits * n_repeats,
    }
