"""
Two-Stage GNN Pretrainer for Tg Prediction
两阶段 GNN 预训练器

Stage 1 (Pretrain): Large external dataset (~59K) with standard LR and dropout.
Stage 2 (Finetune): Bicerano 304 with lower LR, higher dropout, frozen early layers.

Usage:
    trainer = TgPretrainer(model, device="cuda")
    trainer.pretrain(pretrain_loader, val_loader, epochs=100)
    trainer.finetune(finetune_loader, val_loader, epochs=50)
    trainer.save_checkpoint("best_model.pt")

Public API:
    TgPretrainer(model, device, lr_pretrain, lr_finetune, ...)
    pretrain(train_loader, val_loader, epochs) -> dict
    finetune(train_loader, val_loader, epochs, patience) -> dict
    save_checkpoint(path) / load_checkpoint(path)
"""

import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.gnn.tandem_m2m import TandemM2M


class TgPretrainer:
    """Two-stage training manager for TandemM2M.

    Args:
        model: TandemM2M model instance.
        device: Device string ("cuda" or "cpu").
        lr_pretrain: Learning rate for pretraining (default: 1e-3).
        lr_finetune: Learning rate for finetuning (default: 1e-4).
        weight_decay: L2 regularization (default: 1e-4).
        freeze_layers: Number of GAT layers to freeze during finetuning.
    """

    def __init__(
        self,
        model: TandemM2M,
        device: str = "cuda",
        lr_pretrain: float = 1e-3,
        lr_finetune: float = 1e-4,
        weight_decay: float = 1e-4,
        freeze_layers: int = 2,
    ):
        self.model = model.to(device)
        self.device = device
        self.lr_pretrain = lr_pretrain
        self.lr_finetune = lr_finetune
        self.weight_decay = weight_decay
        self.freeze_layers = freeze_layers
        self.criterion = nn.MSELoss()
        self.history = {"pretrain": [], "finetune": []}

    def pretrain(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 100,
    ) -> Dict:
        """Stage 1: Pretrain on large external dataset.

        Args:
            train_loader: DataLoader for pretraining data.
            val_loader: Optional validation DataLoader.
            epochs: Number of pretraining epochs.

        Returns:
            Training history dict.
        """
        self.model.unfreeze_all()
        optimizer = Adam(
            self.model.parameters(),
            lr=self.lr_pretrain,
            weight_decay=self.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss = float("inf")
        best_state = None
        history = []

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader, optimizer)
            scheduler.step()

            record = {"epoch": epoch, "train_loss": train_loss}

            if val_loader is not None:
                val_loss = self._eval_epoch(val_loader)
                record["val_loss"] = val_loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {
                        k: v.clone() for k, v in self.model.state_dict().items()
                    }
                    record["best"] = True
            else:
                # Track train loss when no val set
                if train_loss < best_val_loss:
                    best_val_loss = train_loss
                    best_state = {
                        k: v.clone() for k, v in self.model.state_dict().items()
                    }

            history.append(record)

            if epoch % 10 == 0:
                msg = f"[Pretrain] Epoch {epoch}/{epochs} | Train: {train_loss:.4f}"
                if val_loader is not None:
                    msg += f" | Val: {record.get('val_loss', 0):.4f}"
                print(msg)

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.history["pretrain"] = history
        return {"history": history, "best_val_loss": best_val_loss}

    def finetune(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 50,
        patience: int = 10,
    ) -> Dict:
        """Stage 2: Finetune on Bicerano 304 with frozen early layers.

        Args:
            train_loader: DataLoader for finetuning data.
            val_loader: Optional validation DataLoader.
            epochs: Maximum finetuning epochs.
            patience: Early stopping patience.

        Returns:
            Training history dict.
        """
        # Freeze early GAT layers
        self.model.freeze_gnn_layers(self.freeze_layers)

        # Only optimize trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = Adam(
            trainable_params,
            lr=self.lr_finetune,
            weight_decay=self.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0
        history = []

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader, optimizer)
            scheduler.step()

            record = {"epoch": epoch, "train_loss": train_loss}

            if val_loader is not None:
                val_loss = self._eval_epoch(val_loader)
                record["val_loss"] = val_loss

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {
                        k: v.clone() for k, v in self.model.state_dict().items()
                    }
                    epochs_without_improvement = 0
                    record["best"] = True
                else:
                    epochs_without_improvement += 1
            else:
                # Without val, track train loss for early stopping
                if train_loss < best_val_loss:
                    best_val_loss = train_loss
                    best_state = {
                        k: v.clone() for k, v in self.model.state_dict().items()
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            history.append(record)

            if epoch % 10 == 0:
                params = self.model.count_parameters()
                msg = (
                    f"[Finetune] Epoch {epoch}/{epochs} | "
                    f"Train: {train_loss:.4f} | "
                    f"Trainable params: {params['trainable']:,}"
                )
                if val_loader is not None:
                    msg += f" | Val: {record.get('val_loss', 0):.4f}"
                print(msg)

            if epochs_without_improvement >= patience:
                print(
                    f"[Finetune] Early stopping at epoch {epoch} "
                    f"(no improvement for {patience} epochs)"
                )
                break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.history["finetune"] = history
        return {"history": history, "best_val_loss": best_val_loss}

    def _train_epoch(self, loader, optimizer) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        for batch in loader:
            batch = batch.to(self.device)
            tabular = batch.tabular if hasattr(batch, "tabular") else torch.zeros(
                batch.num_graphs, 1, device=self.device
            )
            baseline = batch.baseline if hasattr(batch, "baseline") else None

            optimizer.zero_grad()
            y_pred = self.model(batch, tabular, baseline).squeeze()
            loss = self.criterion(y_pred, batch.y.squeeze())
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            n_samples += batch.num_graphs

        return total_loss / max(n_samples, 1)

    @torch.no_grad()
    def _eval_epoch(self, loader) -> float:
        """Evaluate on a dataset."""
        self.model.eval()
        total_loss = 0.0
        n_samples = 0

        for batch in loader:
            batch = batch.to(self.device)
            tabular = batch.tabular if hasattr(batch, "tabular") else torch.zeros(
                batch.num_graphs, 1, device=self.device
            )
            baseline = batch.baseline if hasattr(batch, "baseline") else None

            y_pred = self.model(batch, tabular, baseline).squeeze()
            loss = self.criterion(y_pred, batch.y.squeeze())

            total_loss += loss.item() * batch.num_graphs
            n_samples += batch.num_graphs

        return total_loss / max(n_samples, 1)

    def save_checkpoint(self, path: str):
        """Save model checkpoint.

        Args:
            path: File path (e.g., "checkpoints/best_model.pt").
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "history": self.history,
                "model_config": {
                    "freeze_layers": self.freeze_layers,
                    "lr_pretrain": self.lr_pretrain,
                    "lr_finetune": self.lr_finetune,
                },
            },
            path,
        )
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint.

        Args:
            path: File path to checkpoint.
        """
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            # PyTorch < 2.0 doesn't support weights_only
            checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.history = checkpoint.get("history", {"pretrain": [], "finetune": []})
        print(f"Checkpoint loaded from {path}")
