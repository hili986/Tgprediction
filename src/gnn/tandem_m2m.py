"""
Tandem M2M Architecture: Physics-Guided GNN + Tabular Fusion
串联 M2M 架构: 物理引导 GNN + 表格特征融合

Core idea:
    Tg = Tg_baseline + alpha * GNN_residual(graph, tabular)

The GNN learns a residual correction on top of a group-contribution (GC)
baseline, weighted by a learnable alpha parameter. This ensures the model
respects physics while learning complex non-linear corrections.

Architecture:
    1. PhysicsGAT(25 -> 64) extracts graph embedding
    2. MLP fuses graph embedding + tabular features -> Tg residual
    3. alpha * residual added to GC baseline (or direct prediction if no baseline)

Public API:
    TandemM2M(in_dim=25, tabular_dim=56, hidden=64, out_dim=1, ...) -> Module
"""

import torch
import torch.nn as nn

from src.gnn.physics_gat import PhysicsGAT


class TandemM2M(nn.Module):
    """Tandem M2M: GNN + tabular fusion with learnable physics weight.

    Args:
        in_dim: Atom feature dimension (default: 25).
        tabular_dim: Tabular feature dimension (default: 56, M2M features).
        gnn_hidden: GNN hidden dimension (default: 128).
        gnn_out: GNN output/embedding dimension (default: 64).
        gnn_heads: GAT attention heads (default: 4).
        dropout: Dropout rate (default: 0.1).
        pool_strategy: GRIN pooling strategy (default: "max").
        edge_dim: Edge feature dimension (default: 6).
        alpha_init: Initial value for learnable alpha (default: 0.5).
        use_baseline: Whether to add predictions to a GC baseline (default: False).
    """

    def __init__(
        self,
        in_dim: int = 25,
        tabular_dim: int = 56,
        gnn_hidden: int = 128,
        gnn_out: int = 64,
        gnn_heads: int = 4,
        dropout: float = 0.1,
        pool_strategy: str = "max",
        edge_dim: int = 6,
        alpha_init: float = 0.5,
        use_baseline: bool = False,
    ):
        super().__init__()

        self.use_baseline = use_baseline

        # GNN backbone
        self.gnn = PhysicsGAT(
            in_dim=in_dim,
            hidden_dim=gnn_hidden,
            out_dim=gnn_out,
            heads=gnn_heads,
            dropout=dropout,
            pool_strategy=pool_strategy,
            edge_dim=edge_dim,
        )

        # Fusion MLP: GNN embedding + tabular -> prediction
        fusion_in = gnn_out + tabular_dim
        self.mlp = nn.Sequential(
            nn.Linear(fusion_in, 64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # Learnable physics/learning weight
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(
        self,
        data,
        tabular: torch.Tensor,
        baseline: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            data: PyG Data/Batch object with graph features.
            tabular: Tabular features [B, tabular_dim].
            baseline: Optional GC baseline predictions [B, 1].
                Only used when use_baseline=True.

        Returns:
            Predicted Tg values [B, 1].
        """
        # GNN embedding
        graph_embed = self.gnn(data)  # [B, gnn_out]

        # Fuse graph + tabular
        fused = torch.cat([graph_embed, tabular], dim=1)  # [B, gnn_out + tabular_dim]
        residual = self.mlp(fused)  # [B, 1]

        if self.use_baseline and baseline is not None:
            return baseline + self.alpha * residual
        else:
            return self.alpha * residual

    def get_embedding(self, data) -> torch.Tensor:
        """Get 64-dim graph embedding (for sklearn fusion in E12).

        Args:
            data: PyG Data/Batch object.

        Returns:
            Graph embeddings [B, gnn_out].
        """
        return self.gnn(data)

    def freeze_gnn_layers(self, n_layers: int = 2):
        """Freeze first n GAT layers for fine-tuning.

        During fine-tuning on small datasets (e.g., Bicerano 304),
        freeze early layers to prevent overfitting while allowing
        the final GAT layer + MLP + alpha to adapt.

        Args:
            n_layers: Number of GAT layers to freeze (default: 2).
        """
        layers_to_freeze = []
        if n_layers >= 1:
            layers_to_freeze.extend([self.gnn.conv1, self.gnn.bn1])
        if n_layers >= 2:
            layers_to_freeze.extend([self.gnn.conv2, self.gnn.bn2])

        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def count_parameters(self) -> dict:
        """Count trainable and total parameters.

        Returns:
            Dict with 'total', 'trainable', 'frozen' counts.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }
