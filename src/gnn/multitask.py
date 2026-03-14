"""
Multi-Task GNN with Shared PhysicsGAT Backbone
多任务 GNN — 共享 PhysicsGAT 骨干网络

Predicts Tg + auxiliary properties (density, solubility parameter) to
regularize the graph representation. Auxiliary signals come from
PolyMetriX which has 112 property columns.

Architecture:
    PhysicsGAT (shared) -> 64-dim embedding
    -> Head_Tg: Linear(64->32->1)    (primary)
    -> Head_density: Linear(64->32->1) (auxiliary)
    -> Head_sol: Linear(64->32->1)     (auxiliary)

Loss = loss_tg + lambda_aux * (loss_density + loss_sol)
Only backprop auxiliary losses when auxiliary labels exist.

Public API:
    MultiTaskTgModel(in_dim, hidden, out_dim, n_aux_tasks, ...) -> Module
"""

import torch
import torch.nn as nn

from src.gnn.physics_gat import PhysicsGAT


class _PredictionHead(nn.Module):
    """Simple 2-layer MLP prediction head."""

    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskTgModel(nn.Module):
    """Multi-task GNN: shared backbone + task-specific heads.

    Args:
        in_dim: Atom feature dimension (default: 25).
        gnn_hidden: GNN hidden dimension (default: 128).
        gnn_out: GNN embedding dimension (default: 64).
        heads: GAT attention heads (default: 4).
        dropout: Dropout rate (default: 0.1).
        edge_dim: Edge feature dimension (default: 6).
        aux_task_names: Names of auxiliary tasks (default: ["density", "sol_param"]).
        lambda_aux: Weight for auxiliary losses (default: 0.1).
    """

    def __init__(
        self,
        in_dim: int = 25,
        gnn_hidden: int = 128,
        gnn_out: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
        edge_dim: int = 6,
        aux_task_names: list = None,
        lambda_aux: float = 0.1,
    ):
        super().__init__()

        self.aux_task_names = aux_task_names or ["density", "sol_param"]
        self.lambda_aux = lambda_aux

        # Shared GNN backbone
        self.gnn = PhysicsGAT(
            in_dim=in_dim,
            hidden_dim=gnn_hidden,
            out_dim=gnn_out,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
        )

        # Primary head: Tg
        self.head_tg = _PredictionHead(gnn_out)

        # Auxiliary heads
        self.aux_heads = nn.ModuleDict({
            name: _PredictionHead(gnn_out) for name in self.aux_task_names
        })

    def forward(self, data) -> dict:
        """Forward pass returning all task predictions.

        Args:
            data: PyG Data/Batch with graph features.

        Returns:
            Dict with "tg" and auxiliary task predictions, each [B, 1].
        """
        embedding = self.gnn(data)  # [B, gnn_out]

        preds = {"tg": self.head_tg(embedding)}
        for name, head in self.aux_heads.items():
            preds[name] = head(embedding)

        return preds

    def compute_loss(
        self,
        preds: dict,
        targets: dict,
        criterion: nn.Module = None,
    ) -> torch.Tensor:
        """Compute weighted multi-task loss.

        Only includes auxiliary losses when targets are available
        (non-NaN values).

        Args:
            preds: Dict of predictions from forward().
            targets: Dict of target tensors {"tg": ..., "density": ..., ...}.
            criterion: Loss function (default: MSELoss).

        Returns:
            Total weighted loss.
        """
        if criterion is None:
            criterion = nn.MSELoss()

        # Primary Tg loss (always required)
        loss = criterion(preds["tg"].squeeze(), targets["tg"].squeeze())

        # Auxiliary losses (only when labels exist)
        for name in self.aux_task_names:
            if name in targets and targets[name] is not None:
                aux_target = targets[name].squeeze()
                # Mask NaN values
                valid_mask = ~torch.isnan(aux_target)
                if valid_mask.any():
                    aux_pred = preds[name].squeeze()[valid_mask]
                    aux_true = aux_target[valid_mask]
                    loss = loss + self.lambda_aux * criterion(aux_pred, aux_true)

        return loss

    def get_embedding(self, data) -> torch.Tensor:
        """Get shared graph embedding."""
        return self.gnn(data)
