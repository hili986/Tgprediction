"""
Physics-Embedded GAT with GRIN Repeat-Invariant Pooling
物理增强图注意力网络 + GRIN 重复不变池化

Architecture:
    GATConv(25->128, 4heads) -> BN -> ELU
    GATConv(128->128, 4heads) -> BN -> ELU
    GATConv(128->64, 1head) -> ELU
    GRIN Pool (middle repeat unit) -> [batch, 64]

GRIN (Graph Repeat-unit Invariant Neighbor) pooling ensures the graph
embedding is invariant to the number of repeat units by only pooling
over the middle unit (mask==1), using left/right units as context.

Public API:
    RepeatInvariantPooling(strategy="max") -> Module
    PhysicsGAT(in_dim=25, hidden=128, out_dim=64, heads=4, dropout=0.1) -> Module
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, BatchNorm
from torch_geometric.utils import softmax

# Prefer PyG built-in scatter (2.4+), fallback to torch_scatter
try:
    from torch_geometric.utils import scatter
except ImportError:
    from torch_scatter import scatter


class RepeatInvariantPooling(nn.Module):
    """GRIN-style pooling: aggregate only middle repeat-unit atoms.

    For a 3-RU oligomer (mask values 0, 1, 2), we pool over atoms with
    mask==1 (the middle unit). The left (0) and right (2) units serve as
    message-passing context during GATConv layers.

    This ensures the graph-level embedding is invariant to the number of
    repeat units and focuses on the representative local environment.

    Args:
        strategy: Pooling strategy. "max" (default), "mean", or "attention".
        hidden_dim: Hidden dimension (only used when strategy="attention").
    """

    def __init__(self, strategy: str = "max", hidden_dim: int = 64):
        super().__init__()
        self.strategy = strategy
        if strategy == "attention":
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        repeat_unit_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool node features to graph-level embeddings.

        Args:
            x: Node features [N, D].
            batch: Batch assignment [N].
            repeat_unit_mask: Repeat unit assignment [N], values in {0,1,...}.

        Returns:
            Graph-level embeddings [B, D].
        """
        # Select middle unit atoms (mask == 1)
        middle_mask = repeat_unit_mask == 1
        x_mid = x[middle_mask]
        batch_mid = batch[middle_mask]

        if x_mid.shape[0] == 0:
            # Fallback: if no middle atoms, pool all atoms
            x_mid = x
            batch_mid = batch

        num_graphs = batch.max().item() + 1

        if self.strategy == "max":
            out = scatter(x_mid, batch_mid, dim=0, dim_size=num_graphs, reduce="max")
        elif self.strategy == "mean":
            out = scatter(x_mid, batch_mid, dim=0, dim_size=num_graphs, reduce="mean")
        elif self.strategy == "attention":
            gate_scores = self.gate(x_mid)  # [N_mid, 1]
            gate_scores = softmax(gate_scores, batch_mid, num_nodes=x_mid.shape[0])
            out = scatter(
                x_mid * gate_scores, batch_mid, dim=0, dim_size=num_graphs, reduce="sum"
            )
        else:
            raise ValueError(f"Unknown pooling strategy: {self.strategy}")

        return out


class PhysicsGAT(nn.Module):
    """3-layer GAT with physics-enhanced features and GRIN pooling.

    Processes molecular graphs with 25-dim atom features (15 standard +
    10 physics-enhanced) through 3 GAT layers, then pools the middle
    repeat unit to produce a 64-dim graph embedding.

    Args:
        in_dim: Input atom feature dimension (default: 25).
        hidden_dim: Hidden layer dimension (default: 128).
        out_dim: Output embedding dimension (default: 64).
        heads: Number of attention heads for layers 1-2 (default: 4).
        dropout: Dropout rate (default: 0.1).
        pool_strategy: GRIN pooling strategy (default: "max").
        edge_dim: Edge feature dimension (default: 6). Set to None to
            disable edge features in attention.
    """

    def __init__(
        self,
        in_dim: int = 25,
        hidden_dim: int = 128,
        out_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
        pool_strategy: str = "max",
        edge_dim: int = 6,
    ):
        super().__init__()

        # Layer 1: in_dim -> hidden_dim (multi-head)
        self.conv1 = GATConv(
            in_dim, hidden_dim // heads, heads=heads,
            dropout=dropout, edge_dim=edge_dim,
        )
        self.bn1 = BatchNorm(hidden_dim)

        # Layer 2: hidden_dim -> hidden_dim (multi-head)
        self.conv2 = GATConv(
            hidden_dim, hidden_dim // heads, heads=heads,
            dropout=dropout, edge_dim=edge_dim,
        )
        self.bn2 = BatchNorm(hidden_dim)

        # Layer 3: hidden_dim -> out_dim (single head)
        self.conv3 = GATConv(
            hidden_dim, out_dim, heads=1,
            dropout=dropout, edge_dim=edge_dim,
        )

        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)

        # GRIN pooling
        self.pool = RepeatInvariantPooling(
            strategy=pool_strategy, hidden_dim=out_dim,
        )

    def forward(self, data) -> torch.Tensor:
        """Forward pass: graph -> 64-dim embedding.

        Args:
            data: PyG Data object with x, edge_index, edge_attr,
                  repeat_unit_mask, and batch.

        Returns:
            Graph-level embeddings [B, out_dim].
        """
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        batch = getattr(data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        repeat_unit_mask = data.repeat_unit_mask

        # Layer 1
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)

        # Layer 2
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = self.act(x)
        x = self.dropout(x)

        # Layer 3
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        x = self.act(x)

        # GRIN pooling -> graph-level
        out = self.pool(x, batch, repeat_unit_mask)

        return out

    def get_node_embeddings(self, data) -> torch.Tensor:
        """Get node-level embeddings before pooling.

        Useful for visualization and analysis.

        Args:
            data: PyG Data object.

        Returns:
            Node embeddings [N, out_dim].
        """
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, "edge_attr", None)

        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = self.act(x)

        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        x = self.act(x)

        return x
