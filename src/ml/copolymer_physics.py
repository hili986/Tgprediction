from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class BinaryCopolymerPoint:
    tg1_k: float
    tg2_k: float
    w1: float
    target_tg_k: float


def _normalize_weights(weights: Sequence[float]) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("weights must be a non-empty 1D sequence.")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("weights must be finite and non-negative.")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("weights must sum to a positive value.")
    return values / total


def fox_tg_k(tg_values_k: Sequence[float], weights: Sequence[float]) -> float:
    tg = np.asarray(tg_values_k, dtype=float)
    w = _normalize_weights(weights)
    if tg.shape != w.shape:
        raise ValueError("tg_values_k and weights must have the same length.")
    if not np.isfinite(tg).all() or np.any(tg <= 0):
        return float("nan")
    denom = float(np.sum(w / tg))
    return float("nan") if denom <= 0 else 1.0 / denom


def gordon_taylor_binary_tg_k(tg1_k: float, tg2_k: float, w1: float, k: float = 1.0) -> float:
    w1 = float(w1)
    w2 = 1.0 - w1
    denom = w1 + k * w2
    if denom <= 0 or not np.isfinite([tg1_k, tg2_k, w1, k]).all():
        return float("nan")
    return float((w1 * tg1_k + k * w2 * tg2_k) / denom)


def kwei_binary_tg_k(tg1_k: float, tg2_k: float, w1: float, k: float, q: float) -> float:
    gt = gordon_taylor_binary_tg_k(tg1_k, tg2_k, w1, k)
    if not np.isfinite(gt):
        return float("nan")
    return float(gt + q * w1 * (1.0 - w1))


def estimate_gordon_taylor_k(tg1_k: float, tg2_k: float) -> float:
    if tg2_k <= 0 or not np.isfinite([tg1_k, tg2_k]).all():
        return 1.0
    return float(np.clip(tg1_k / tg2_k, 0.3, 3.0))


def _points_array(points: Iterable[BinaryCopolymerPoint]) -> list[BinaryCopolymerPoint]:
    clean = []
    for point in points:
        values = [point.tg1_k, point.tg2_k, point.w1, point.target_tg_k]
        if np.isfinite(values).all() and 0.0 <= point.w1 <= 1.0:
            clean.append(point)
    if not clean:
        raise ValueError("no finite binary copolymer points.")
    return clean


def fit_gordon_taylor_k(
    points: Iterable[BinaryCopolymerPoint],
    k_grid: Sequence[float] | None = None,
) -> float:
    clean = _points_array(points)
    if k_grid is None:
        k_grid = np.linspace(0.2, 5.0, 481)
    best_k = 1.0
    best_mse = float("inf")
    for k in k_grid:
        pred = np.array([
            gordon_taylor_binary_tg_k(point.tg1_k, point.tg2_k, point.w1, float(k))
            for point in clean
        ])
        mse = float(np.mean((pred - np.array([point.target_tg_k for point in clean])) ** 2))
        if mse < best_mse:
            best_k = float(k)
            best_mse = mse
    return best_k


def fit_kwei_k_q(
    points: Iterable[BinaryCopolymerPoint],
    k_grid: Sequence[float] | None = None,
) -> tuple[float, float]:
    clean = _points_array(points)
    if k_grid is None:
        k_grid = np.linspace(0.2, 5.0, 481)
    y = np.array([point.target_tg_k for point in clean], dtype=float)
    interaction = np.array([point.w1 * (1.0 - point.w1) for point in clean], dtype=float)
    best_k = 1.0
    best_q = 0.0
    best_mse = float("inf")
    for k in k_grid:
        gt = np.array([
            gordon_taylor_binary_tg_k(point.tg1_k, point.tg2_k, point.w1, float(k))
            for point in clean
        ])
        denom = float(np.dot(interaction, interaction))
        q = 0.0 if denom <= 1e-12 else float(np.dot(interaction, y - gt) / denom)
        pred = gt + q * interaction
        mse = float(np.mean((pred - y) ** 2))
        if mse < best_mse:
            best_k = float(k)
            best_q = q
            best_mse = mse
    return best_k, best_q
