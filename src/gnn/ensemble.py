"""
Deep Ensemble + Conformal Prediction for GNN Uncertainty
深度集成 + 共形预测的 GNN 不确定性量化

Strategy:
    1. Train 5 independent TandemM2M models (different random seeds)
    2. Mean prediction = point estimate
    3. Std across ensemble = epistemic uncertainty
    4. MAPIE conformal calibration = guaranteed coverage intervals

Public API:
    DeepEnsembleTg(model_fn, n_models=5) -> ensemble manager
    fit(train_loader, val_loader, ...) -> train all models
    predict(data, tabular) -> (mean, std, individual_preds)
    calibrate(cal_loader) -> fit conformal on calibration set
    predict_interval(data, tabular, confidence=0.9) -> (pred, lower, upper)
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.gnn.tandem_m2m import TandemM2M
from src.gnn.pretrainer import TgPretrainer


class DeepEnsembleTg:
    """Deep ensemble of TandemM2M models with conformal calibration.

    Args:
        model_fn: Factory function () -> TandemM2M. Called n_models times.
        n_models: Number of ensemble members (default: 5).
        device: Compute device (default: "cuda").
    """

    def __init__(
        self,
        model_fn: Callable[[], TandemM2M],
        n_models: int = 5,
        device: str = "cuda",
        tabular_dim: int = 56,
    ):
        self.n_models = n_models
        self.device = device
        self.tabular_dim = tabular_dim
        self.models = []
        for i in range(n_models):
            torch.manual_seed(42 + i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42 + i)
            self.models.append(model_fn().to(device))
        self.trainers = [
            TgPretrainer(model, device=device, tabular_dim=tabular_dim)
            for model in self.models
        ]
        self.conformal_scores = None  # Calibrated after training

    def fit(
        self,
        pretrain_loader=None,
        finetune_loader=None,
        val_loader=None,
        pretrain_epochs: int = 100,
        finetune_epochs: int = 50,
        patience: int = 10,
    ) -> List[Dict]:
        """Train all ensemble members.

        Args:
            pretrain_loader: Large dataset DataLoader (Stage 1).
            finetune_loader: Bicerano DataLoader (Stage 2).
            val_loader: Validation DataLoader.
            pretrain_epochs: Pretraining epochs per model.
            finetune_epochs: Finetuning epochs per model.
            patience: Early stopping patience.

        Returns:
            List of training histories for each model.
        """
        histories = []
        for i, trainer in enumerate(self.trainers):
            print(f"\n{'='*60}")
            print(f"Training ensemble member {i+1}/{self.n_models}")
            print(f"{'='*60}")

            history = {}
            if pretrain_loader is not None:
                h = trainer.pretrain(pretrain_loader, val_loader, pretrain_epochs)
                history["pretrain"] = h

            if finetune_loader is not None:
                h = trainer.finetune(
                    finetune_loader, val_loader, finetune_epochs, patience
                )
                history["finetune"] = h

            histories.append(history)

        return histories

    @torch.no_grad()
    def predict(
        self,
        data,
        tabular: torch.Tensor,
        baseline: torch.Tensor = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with uncertainty estimation.

        Args:
            data: PyG Data/Batch.
            tabular: Tabular features [B, D].
            baseline: Optional GC baseline [B, 1].

        Returns:
            Tuple of (mean_pred [B], std_pred [B], all_preds [n_models, B]).
        """
        all_preds = []
        for model in self.models:
            model.eval()
            pred = model(data.to(self.device), tabular.to(self.device),
                        baseline.to(self.device) if baseline is not None else None)
            all_preds.append(np.atleast_1d(pred.squeeze().cpu().numpy()))

        all_preds = np.array(all_preds)  # [n_models, B]
        mean_pred = all_preds.mean(axis=0)
        std_pred = all_preds.std(axis=0)

        return mean_pred, std_pred, all_preds

    def calibrate(self, cal_loader, confidence: float = 0.9):
        """Calibrate conformal prediction on calibration set.

        Uses split conformal: compute nonconformity scores on calibration
        data, then use quantile for prediction intervals.

        Args:
            cal_loader: Calibration DataLoader.
            confidence: Target coverage level (default: 0.9).
        """
        residuals = []

        for batch in cal_loader:
            batch = batch.to(self.device)
            tabular = batch.tabular if hasattr(batch, "tabular") else torch.zeros(
                batch.num_graphs, self.tabular_dim, device=self.device
            )
            baseline = batch.baseline if hasattr(batch, "baseline") else None

            mean_pred, _, _ = self.predict(batch, tabular, baseline)
            y_true = np.atleast_1d(batch.y.squeeze().cpu().numpy())
            residuals.extend(np.atleast_1d(np.abs(y_true - mean_pred)).tolist())

        residuals = np.array(residuals)
        n = len(residuals)
        # Conformal quantile with finite-sample correction
        q = np.ceil((n + 1) * confidence) / n
        q = min(q, 1.0)
        self.conformal_scores = np.quantile(residuals, q)
        print(f"Conformal calibration: q={q:.3f}, score={self.conformal_scores:.2f}K")

    @torch.no_grad()
    def predict_interval(
        self,
        data,
        tabular: torch.Tensor,
        baseline: torch.Tensor = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with conformal prediction intervals.

        Must call calibrate() first.

        Args:
            data: PyG Data/Batch.
            tabular: Tabular features [B, D].
            baseline: Optional GC baseline [B, 1].

        Returns:
            Tuple of (pred [B], lower [B], upper [B]).
        """
        if self.conformal_scores is None:
            raise RuntimeError("Call calibrate() before predict_interval()")

        mean_pred, std_pred, _ = self.predict(data, tabular, baseline)
        lower = mean_pred - self.conformal_scores
        upper = mean_pred + self.conformal_scores

        return mean_pred, lower, upper

    def save_all(self, directory: str):
        """Save all ensemble checkpoints."""
        import os
        os.makedirs(directory, exist_ok=True)
        for i, trainer in enumerate(self.trainers):
            path = os.path.join(directory, f"ensemble_member_{i}.pt")
            trainer.save_checkpoint(path)
        if self.conformal_scores is not None:
            score_path = os.path.join(directory, "conformal_scores.npy")
            np.save(score_path, self.conformal_scores)
        print(f"Ensemble saved to {directory}/")

    def load_all(self, directory: str):
        """Load all ensemble checkpoints."""
        import os
        for i, trainer in enumerate(self.trainers):
            path = os.path.join(directory, f"ensemble_member_{i}.pt")
            trainer.load_checkpoint(path)
        score_path = os.path.join(directory, "conformal_scores.npy")
        if os.path.exists(score_path):
            self.conformal_scores = float(np.load(score_path))
        print(f"Ensemble loaded from {directory}/")
