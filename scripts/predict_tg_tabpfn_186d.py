"""
Best Tg inference script: TabPFN v2 on 186d multiscale features.

Validated training recipe in this repo:
    PHY-C-light 58d + GNN 64d + polyBERT PCA 64d = 186d
    TabPFN v2, R2 ~= 0.9167 on unified_tg

This script is intended for server-side inference on new repeat-unit SMILES.

Supported modes:
    1. Homopolymer prediction from one repeat-unit SMILES
    2. Binary copolymer prediction from two repeat-unit SMILES + composition w1
    3. Multi-component copolymer prediction from arbitrary components + weights
    4. Batch CSV prediction

Important:
    - Input should be repeat-unit SMILES with two attachment points (* or [*]).
    - Multi-component copolymer prediction is an engineering approximation:
      weighted mixing of component descriptors / embeddings.
      It is useful for exploratory comparison, but is not a separately
      benchmarked copolymer model.
    - `architecture=block` is only a proxy mode:
      the script reports a Fox-style miscible-equivalent Tg and a component
      Tg window, because true block sequencing is not explicitly modeled.

Required server artifacts:
    data/unified_tg.parquet
    data/feature_matrix_PHY-C.parquet
    data/gnn_embeddings_64d.parquet
    data/polybert_embeddings.parquet
    data/polybert_model/
    checkpoints/gnn_pretrained.pt

Examples:
    python scripts/predict_tg_tabpfn_186d.py --smiles "*CC(*)"
    python scripts/predict_tg_tabpfn_186d.py --smiles1 "*CC(*)" --smiles2 "*CO(*)" --w1 0.5
    python scripts/predict_tg_tabpfn_186d.py --architecture random \
      --component "*CC(*)::0.6" --component "*CO(*)::0.3" --component "*CN(*)::0.1"
    python scripts/predict_tg_tabpfn_186d.py --input-csv data/query.csv --output results/pred.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_pipeline import compute_features, get_feature_names
from src.features.chain_physics import (
    chain_physics_feature_names,
    compute_3mer_physics,
)
from src.features.chain_physics_cache import load_chain_physics_cache


DROP_FEATURES = [
    "CP_oligomer_level",
    "CP_Cn_proxy",
    "L1_RingCount",
    "L1_HeavyAtomCount",
    "L1_MolWt",
    "IC_hydrophilic_ratio",
]

PHY_C_DIM = 64
PHY_C_LIGHT_DIM = 58
GNN_DIM = 64
PBERT_PCA_DIM = 64
FULL_DIM = PHY_C_LIGHT_DIM + GNN_DIM + PBERT_PCA_DIM
SUPPORTED_ARCHITECTURES = {"random", "block"}
COMPONENT_COL_RE = re.compile(r"^(?:smiles|component)_?(\d+)$", re.IGNORECASE)
WEIGHT_COL_RE = re.compile(r"^(?:w|weight)_?(\d+)$", re.IGNORECASE)


@dataclass
class InferencePaths:
    data_path: Path
    phyc_cache: Path
    gnn_cache: Path
    pbert_cache: Path
    chain_physics_cache: Path
    polybert_model_dir: Path
    gnn_checkpoint: Path


def _numeric_suffix(name: str) -> int:
    return int(name.rsplit("_", 1)[1])


def _validate_repeat_unit_smiles(smiles: str) -> str:
    smi = (smiles or "").strip()
    if not smi:
        raise ValueError("SMILES is empty.")
    if smi.count("*") < 2:
        raise ValueError(
            f"SMILES '{smi}' does not look like a repeat-unit SMILES with two attachment points."
        )
    return smi


def _normalize_weights(weights: Sequence[float]) -> np.ndarray:
    arr = np.asarray(weights, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("Weights must be a non-empty 1D sequence.")
    if np.any(~np.isfinite(arr)):
        raise ValueError("Weights contain NaN/inf.")
    if np.any(arr < 0):
        raise ValueError("Weights must be non-negative.")
    total = float(arr.sum())
    if total <= 0:
        raise ValueError("Weights must sum to a positive value.")
    return arr / total


def _fox_blend_tg(tg_values_k: Sequence[float], weights: Sequence[float]) -> float:
    tg = np.asarray(tg_values_k, dtype=float)
    w = _normalize_weights(weights)
    if np.any(~np.isfinite(tg)) or np.any(tg <= 0):
        return float("nan")
    denom = float(np.sum(w / tg))
    if denom <= 0:
        return float("nan")
    return 1.0 / denom


def _parse_component_entry(spec: str) -> Tuple[str, Optional[float]]:
    text = (spec or "").strip()
    if not text:
        raise ValueError("Empty component spec.")
    if "::" in text:
        smiles_part, weight_part = text.rsplit("::", 1)
        smiles = _validate_repeat_unit_smiles(smiles_part.strip())
        try:
            weight = float(weight_part.strip())
        except ValueError as exc:
            raise ValueError(
                f"Invalid component weight in '{text}'. Expected format 'SMILES::weight'."
            ) from exc
        return smiles, weight
    return _validate_repeat_unit_smiles(text), None


def _parse_component_entries(specs: Sequence[str]) -> Tuple[List[str], List[float]]:
    if not specs:
        raise ValueError("No component entries provided.")
    smiles_list: List[str] = []
    weights: List[Optional[float]] = []
    for spec in specs:
        smiles, weight = _parse_component_entry(spec)
        smiles_list.append(smiles)
        weights.append(weight)

    present = [w is not None for w in weights]
    if any(present) and not all(present):
        raise ValueError("Either provide weights for all components or for none of them.")

    final_weights = [float(w) for w in weights] if all(present) else [1.0] * len(smiles_list)
    return smiles_list, final_weights


def _resolve_device(requested: str) -> str:
    try:
        import torch
    except ImportError:
        return "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable, fallback to CPU.")
        return "cpu"
    return requested


def _require_paths(paths: InferencePaths) -> None:
    required = {
        "unified dataset": paths.data_path,
        "PHY-C cache": paths.phyc_cache,
        "GNN cache": paths.gnn_cache,
        "polyBERT cache": paths.pbert_cache,
        "polyBERT model dir": paths.polybert_model_dir,
        "GNN checkpoint": paths.gnn_checkpoint,
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inference artifacts:\n" + "\n".join(missing))


def _align_block(
    base_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    value_cols: Sequence[str],
    label: str,
) -> np.ndarray:
    # Match the original Phase D/E training scripts:
    # when cache lengths are equal, trust row order instead of merging on SMILES.
    if len(base_df) == len(feat_df):
        if "smiles" in base_df.columns and "smiles" in feat_df.columns:
            mismatch = int((base_df["smiles"].astype(str) != feat_df["smiles"].astype(str)).sum())
            if mismatch > 0:
                print(
                    f"[warn] {label}: {mismatch} SMILES rows differ between caches; "
                    "using row-order alignment to match original training scripts."
                )
        mat = feat_df[list(value_cols)].to_numpy(dtype=float)
    elif "smiles" in base_df.columns and "smiles" in feat_df.columns:
        merged = base_df[["smiles"]].merge(
            feat_df[["smiles", *value_cols]],
            on="smiles",
            how="left",
            sort=False,
            validate="one_to_one",
        )
        mat = merged[list(value_cols)].to_numpy(dtype=float)
    else:
        raise ValueError(
            f"{label} cache length mismatch: base={len(base_df)}, feature={len(feat_df)}."
        )
    return mat


def _load_training_blocks(paths: InferencePaths) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_df = pd.read_parquet(paths.data_path)
    y = base_df["tg_k"].to_numpy(dtype=float)

    all_phyc_names = get_feature_names("PHY-C")
    keep_phyc_names = [name for name in all_phyc_names if name not in DROP_FEATURES]
    if len(keep_phyc_names) != PHY_C_LIGHT_DIM:
        raise ValueError(
            f"Unexpected PHY-C-light dimension: got {len(keep_phyc_names)}, expected {PHY_C_LIGHT_DIM}."
        )

    df_phyc = pd.read_parquet(paths.phyc_cache)
    for col in keep_phyc_names:
        if col not in df_phyc.columns:
            raise KeyError(f"Missing column in PHY-C cache: {col}")
    X_phyc = _align_block(base_df, df_phyc, keep_phyc_names, "PHY-C-light")

    df_gnn = pd.read_parquet(paths.gnn_cache)
    gnn_cols = sorted([c for c in df_gnn.columns if c.startswith("GNN_")], key=_numeric_suffix)
    if len(gnn_cols) != GNN_DIM:
        raise ValueError(f"Unexpected GNN dimension: got {len(gnn_cols)}, expected {GNN_DIM}.")
    X_gnn = _align_block(base_df, df_gnn, gnn_cols, "GNN embeddings")

    df_pbert = pd.read_parquet(paths.pbert_cache)
    pbert_cols = sorted([c for c in df_pbert.columns if c.startswith("pBERT_")], key=_numeric_suffix)
    if len(pbert_cols) < PBERT_PCA_DIM:
        raise ValueError(
            f"polyBERT raw dimension too small: got {len(pbert_cols)}, need at least {PBERT_PCA_DIM}."
        )
    X_pbert = _align_block(base_df, df_pbert, pbert_cols, "polyBERT embeddings")

    return X_phyc, X_gnn, X_pbert, y


class BestTgPredictor:
    def __init__(
        self,
        paths: InferencePaths,
        device: str = "cuda",
        chain_physics_confs: int = 50,
        polybert_batch_size: int = 64,
    ) -> None:
        self.paths = paths
        self.device = _resolve_device(device)
        self.chain_physics_confs = chain_physics_confs
        self.polybert_batch_size = polybert_batch_size

        self.phyc_keep_names = [name for name in get_feature_names("PHY-C") if name not in DROP_FEATURES]
        self.phyc_keep_indices = [
            idx for idx, name in enumerate(get_feature_names("PHY-C")) if name not in DROP_FEATURES
        ]
        self.chain_names = chain_physics_feature_names()

        self.model = None
        self.preprocess = None
        self.pca = None
        self.chain_cache = load_chain_physics_cache(str(self.paths.chain_physics_cache))

        self._component_cache: Dict[str, Dict[str, np.ndarray | str]] = {}
        self._gnn_model = None

    def fit(self) -> None:
        if self.model is not None:
            return

        print("Loading training artifacts...")
        X_phyc, X_gnn, X_pbert_raw, y = _load_training_blocks(self.paths)

        from sklearn.decomposition import PCA
        from tabpfn import TabPFNRegressor

        print("Fitting PCA on polyBERT raw embeddings...")
        valid_pbert = ~np.isnan(X_pbert_raw).any(axis=1)
        if valid_pbert.sum() <= PBERT_PCA_DIM:
            raise ValueError(
                f"Not enough valid polyBERT rows for PCA-{PBERT_PCA_DIM}: {valid_pbert.sum()}."
            )
        self.pca = PCA(n_components=PBERT_PCA_DIM, random_state=42)
        self.pca.fit(X_pbert_raw[valid_pbert])
        X_pbert = np.full((len(X_pbert_raw), PBERT_PCA_DIM), np.nan)
        X_pbert[valid_pbert] = self.pca.transform(X_pbert_raw[valid_pbert])

        X_full = np.hstack([X_phyc, X_gnn, X_pbert])
        if X_full.shape[1] != FULL_DIM:
            raise ValueError(f"Unexpected full feature dim: {X_full.shape[1]} != {FULL_DIM}")

        valid_rows = ~np.all(np.isnan(X_full), axis=1)
        dropped = int((~valid_rows).sum())
        if dropped > 0:
            print(f"[warn] dropping {dropped} all-NaN training rows to match original phase scripts.")
        X_full = X_full[valid_rows]
        y = y[valid_rows]

        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import MinMaxScaler, PowerTransformer
        print(f"Training TabPFN on full unified set: n={len(y)}, d={X_full.shape[1]}")
        self.preprocess = Pipeline(
            [
                ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
                ("scale", MinMaxScaler()),
            ]
        )
        X_train = self.preprocess.fit_transform(X_full)

        self.model = TabPFNRegressor()
        self.model.fit(X_train, y)
        print("Model fitted.")

    def _load_gnn_model(self):
        if self._gnn_model is not None:
            return self._gnn_model

        import torch
        from src.gnn.pretrainer import TgPretrainer
        from src.gnn.tandem_m2m import TandemM2M

        model = TandemM2M(
            in_dim=25,
            tabular_dim=0,
            gnn_hidden=128,
            gnn_out=64,
            gnn_heads=4,
            dropout=0.0,
            edge_dim=6,
            use_baseline=False,
        )
        trainer = TgPretrainer(model, device=self.device, tabular_dim=0)
        trainer.load_checkpoint(str(self.paths.gnn_checkpoint))
        trainer.model.eval()
        trainer.model.to(self.device)
        self._gnn_model = trainer.model
        return self._gnn_model

    def _compute_chain_physics_vector(self, smiles: str) -> Tuple[np.ndarray, str]:
        cached = self.chain_cache.get(smiles)
        if cached is not None:
            vec = np.array([cached.get(name, np.nan) for name in self.chain_names], dtype=float)
            if np.isfinite(vec).all():
                return vec, "cache"

        result = compute_3mer_physics(smiles, n_confs=self.chain_physics_confs)
        vec = np.array([result[name] for name in self.chain_names], dtype=float)
        if not np.isfinite(vec).all():
            raise ValueError(
                "Chain-physics computation failed. "
                "Try a cleaner repeat-unit SMILES or inspect 3-mer generation."
            )
        return vec, "computed"

    def _compute_phyc_light(self, smiles: str) -> Tuple[np.ndarray, str]:
        base = compute_features(smiles, layer="PHY-B2")
        if not np.isfinite(base).all():
            raise ValueError("PHY-B2 feature extraction failed.")

        cp_vec, cp_source = self._compute_chain_physics_vector(smiles)
        phyc64 = np.concatenate([base, cp_vec])
        if len(phyc64) != PHY_C_DIM:
            raise ValueError(f"Unexpected PHY-C dimension: {len(phyc64)} != {PHY_C_DIM}")

        phyc58 = phyc64[self.phyc_keep_indices]
        if not np.isfinite(phyc58).all():
            raise ValueError("PHY-C-light feature vector contains NaN/inf.")
        return phyc58, cp_source

    def _compute_gnn_embedding(self, smiles: str) -> np.ndarray:
        import torch
        from torch_geometric.data import Batch
        from src.gnn.graph_builder import smiles_to_graph

        model = self._load_gnn_model()
        graph = smiles_to_graph(smiles, n_repeat=3, physics_features=True)
        if graph is None:
            raise ValueError("GNN graph construction failed.")

        batch = Batch.from_data_list([graph]).to(self.device)
        with torch.no_grad():
            emb = model.get_embedding(batch).squeeze(0).detach().cpu().numpy().astype(float)

        if emb.shape[0] != GNN_DIM or not np.isfinite(emb).all():
            raise ValueError("Invalid GNN embedding.")
        return emb

    def _compute_polybert_pca(self, smiles: str) -> np.ndarray:
        from src.gnn.polybert_embedder import extract_polybert_embeddings

        raw = extract_polybert_embeddings(
            [smiles],
            model_path=str(self.paths.polybert_model_dir),
            batch_size=1,
            device=self.device,
        )[0]

        if not np.isfinite(raw).all():
            raise ValueError("polyBERT embedding extraction failed.")

        vec = self.pca.transform(raw.reshape(1, -1))[0]
        if vec.shape[0] != PBERT_PCA_DIM or not np.isfinite(vec).all():
            raise ValueError("Invalid polyBERT PCA vector.")
        return vec.astype(float)

    def featurize_component(self, smiles: str) -> Dict[str, np.ndarray | str]:
        smi = _validate_repeat_unit_smiles(smiles)
        if smi in self._component_cache:
            return self._component_cache[smi]

        phyc, cp_source = self._compute_phyc_light(smi)
        gnn = self._compute_gnn_embedding(smi)
        pbert = self._compute_polybert_pca(smi)

        out = {
            "smiles": smi,
            "phyc": phyc,
            "gnn": gnn,
            "pbert": pbert,
            "chain_physics_source": cp_source,
        }
        self._component_cache[smi] = out
        return out

    def _predict_from_full_vector(self, x_full: np.ndarray) -> float:
        if self.model is None or self.preprocess is None or self.pca is None:
            self.fit()

        x_full = np.asarray(x_full, dtype=float).reshape(1, -1)
        if x_full.shape[1] != FULL_DIM:
            raise ValueError(f"Expected {FULL_DIM} features, got {x_full.shape[1]}.")
        if not np.isfinite(x_full).all():
            raise ValueError("Query feature vector contains NaN/inf.")

        x_pp = self.preprocess.transform(x_full)
        pred = float(self.model.predict(x_pp)[0])
        return pred

    def _predict_component_homopolymer_k(self, component: Dict[str, np.ndarray | str]) -> float:
        x_full = np.hstack([component["phyc"], component["gnn"], component["pbert"]])
        return self._predict_from_full_vector(x_full)

    def predict_homopolymer(self, smiles: str) -> Dict[str, object]:
        comp = self.featurize_component(smiles)
        tg_k = self._predict_component_homopolymer_k(comp)
        return {
            "mode": "homopolymer",
            "smiles": comp["smiles"],
            "tg_k_pred": tg_k,
            "tg_c_pred": tg_k - 273.15,
            "chain_physics_source": comp["chain_physics_source"],
            "model": "TabPFN_v2_on_186d",
        }

    def predict_multicomponent(
        self,
        smiles_list: Sequence[str],
        weights: Sequence[float],
        architecture: str = "random",
    ) -> Dict[str, object]:
        if architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Unsupported architecture '{architecture}'. Expected one of: "
                f"{sorted(SUPPORTED_ARCHITECTURES)}."
            )

        clean_smiles = [_validate_repeat_unit_smiles(s) for s in smiles_list]
        norm_weights = _normalize_weights(weights)
        if len(clean_smiles) != len(norm_weights):
            raise ValueError("Number of SMILES and weights must match.")

        if len(clean_smiles) == 1:
            result = self.predict_homopolymer(clean_smiles[0])
            result["architecture"] = "homopolymer"
            return result

        components = [self.featurize_component(smi) for smi in clean_smiles]
        phyc_mix = np.sum(
            [w * np.asarray(comp["phyc"], dtype=float) for w, comp in zip(norm_weights, components)],
            axis=0,
        )
        gnn_mix = np.sum(
            [w * np.asarray(comp["gnn"], dtype=float) for w, comp in zip(norm_weights, components)],
            axis=0,
        )
        pbert_mix = np.sum(
            [w * np.asarray(comp["pbert"], dtype=float) for w, comp in zip(norm_weights, components)],
            axis=0,
        )
        descriptor_mix_tg_k = self._predict_from_full_vector(np.hstack([phyc_mix, gnn_mix, pbert_mix]))

        component_tg_k = [self._predict_component_homopolymer_k(comp) for comp in components]
        fox_tg_k = _fox_blend_tg(component_tg_k, norm_weights)

        primary_tg_k = descriptor_mix_tg_k
        primary_method = "weighted_descriptor_embedding_mix"
        warning = (
            "Random copolymer output uses weighted mixing of component descriptors/embeddings. "
            "Treat as exploratory inference."
        )
        if architecture == "block":
            if np.isfinite(fox_tg_k):
                primary_tg_k = fox_tg_k
                primary_method = "fox_proxy_from_component_homopolymer_predictions"
            warning = (
                "Block architecture is approximated only. Primary Tg uses a Fox-style proxy "
                "when available, and real block copolymers may exhibit multiple transitions."
            )

        component_rows = []
        for idx, (comp, w, tg_k) in enumerate(zip(components, norm_weights, component_tg_k), start=1):
            component_rows.append(
                {
                    "index": idx,
                    "smiles": comp["smiles"],
                    "weight": float(w),
                    "chain_physics_source": comp["chain_physics_source"],
                    "homopolymer_tg_k_pred": float(tg_k),
                    "homopolymer_tg_c_pred": float(tg_k - 273.15),
                }
            )

        result = {
            "mode": "binary_copolymer" if len(clean_smiles) == 2 else "multicomponent_copolymer",
            "architecture": architecture,
            "n_components": len(clean_smiles),
            "components": component_rows,
            "weights_normalized": [float(w) for w in norm_weights],
            "tg_k_pred": float(primary_tg_k),
            "tg_c_pred": float(primary_tg_k - 273.15),
            "primary_method": primary_method,
            "descriptor_mix_tg_k": float(descriptor_mix_tg_k),
            "descriptor_mix_tg_c": float(descriptor_mix_tg_k - 273.15),
            "fox_reference_tg_k": None if not np.isfinite(fox_tg_k) else float(fox_tg_k),
            "fox_reference_tg_c": None if not np.isfinite(fox_tg_k) else float(fox_tg_k - 273.15),
            "component_tg_window_k": [float(np.min(component_tg_k)), float(np.max(component_tg_k))],
            "component_tg_window_c": [
                float(np.min(component_tg_k) - 273.15),
                float(np.max(component_tg_k) - 273.15),
            ],
            "model": "TabPFN_v2_on_186d",
            "warning": warning,
        }

        if len(clean_smiles) == 2:
            result["smiles1"] = clean_smiles[0]
            result["smiles2"] = clean_smiles[1]
            result["w1"] = float(norm_weights[0])
            result["w2"] = float(norm_weights[1])
            result["chain_physics_source_1"] = components[0]["chain_physics_source"]
            result["chain_physics_source_2"] = components[1]["chain_physics_source"]

        return result

    def predict_binary(self, smiles1: str, smiles2: str, w1: float) -> Dict[str, object]:
        if not (0.0 <= w1 <= 1.0):
            raise ValueError("w1 must be in [0, 1].")
        return self.predict_multicomponent([smiles1, smiles2], [w1, 1.0 - w1], architecture="random")

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []

        for idx, row in df.iterrows():
            try:
                smiles = _cell_to_text(row.get("smiles", ""))
                smiles1 = _cell_to_text(row.get("smiles1", ""))
                smiles2 = _cell_to_text(row.get("smiles2", ""))
                architecture = _cell_to_text(row.get("architecture", "random")).lower() or "random"

                if smiles:
                    result = self.predict_homopolymer(smiles)
                else:
                    specs_text = _cell_to_text(row.get("components", ""))
                    if specs_text:
                        specs = [part.strip() for part in specs_text.split("|") if part.strip()]
                        smiles_list, weights = _parse_component_entries(specs)
                        result = self.predict_multicomponent(smiles_list, weights, architecture=architecture)
                    else:
                        indexed_smiles: Dict[int, str] = {}
                        indexed_weights: Dict[int, float] = {}
                        for col in row.index:
                            col_name = str(col)
                            match_s = COMPONENT_COL_RE.match(col_name)
                            if match_s:
                                value = _cell_to_text(row.get(col, ""))
                                if value:
                                    indexed_smiles[int(match_s.group(1))] = value
                                continue
                            match_w = WEIGHT_COL_RE.match(col_name)
                            if match_w:
                                value = row.get(col)
                                if not pd.isna(value) and str(value).strip() != "":
                                    indexed_weights[int(match_w.group(1))] = float(value)

                        if indexed_smiles:
                            ordered = sorted(indexed_smiles)
                            smiles_list = [indexed_smiles[i] for i in ordered]
                            has_any_weight = any(i in indexed_weights for i in ordered)
                            has_all_weight = all(i in indexed_weights for i in ordered)
                            if has_any_weight and not has_all_weight:
                                if len(ordered) == 2 and set(indexed_weights) in ({ordered[0]}, {ordered[1]}):
                                    if ordered[0] in indexed_weights:
                                        w_first = float(indexed_weights[ordered[0]])
                                        weights = [w_first, 1.0 - w_first]
                                    else:
                                        w_second = float(indexed_weights[ordered[1]])
                                        weights = [1.0 - w_second, w_second]
                                    result = self.predict_multicomponent(
                                        smiles_list,
                                        weights,
                                        architecture=architecture,
                                    )
                                    result["row_index"] = idx
                                    rows.append(result)
                                    continue
                                raise ValueError(
                                    "Multi-component CSV row has partial weights. Provide all or none."
                                )
                            weights = [indexed_weights[i] for i in ordered] if has_all_weight else [1.0] * len(ordered)
                            result = self.predict_multicomponent(smiles_list, weights, architecture=architecture)
                        elif smiles1 and smiles2:
                            w1 = float(row.get("w1", row.get("weight1", 0.5)))
                            result = self.predict_multicomponent(
                                [smiles1, smiles2],
                                [w1, 1.0 - w1],
                                architecture=architecture,
                            )
                        else:
                            raise ValueError(
                                "Each row needs either 'smiles', 'components', "
                                "or smiles/component columns like smiles1,w1,smiles2,w2,..."
                            )

                result["row_index"] = idx
                rows.append(result)
            except Exception as exc:
                rows.append(
                    {
                        "row_index": idx,
                        "mode": "error",
                        "error": str(exc),
                    }
                )

        return pd.DataFrame(rows)


def _make_paths(args: argparse.Namespace) -> InferencePaths:
    return InferencePaths(
        data_path=Path(args.data_path),
        phyc_cache=Path(args.phyc_cache),
        gnn_cache=Path(args.gnn_cache),
        pbert_cache=Path(args.pbert_cache),
        chain_physics_cache=Path(args.chain_physics_cache),
        polybert_model_dir=Path(args.polybert_model_dir),
        gnn_checkpoint=Path(args.gnn_checkpoint),
    )


def _default_path(*parts: str) -> str:
    return str(PROJECT_ROOT.joinpath(*parts))


def _cell_to_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict Tg with the best validated 186d + TabPFN model."
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smiles", type=str, help="Repeat-unit SMILES for homopolymer prediction.")
    mode.add_argument(
        "--component",
        action="append",
        default=None,
        help="Repeatable multi-component spec. Format: 'SMILES::weight'. If all weights are omitted, equal weights are assumed.",
    )
    mode.add_argument(
        "--input-csv",
        type=str,
        help="Batch input CSV. Columns: smiles, or components, or smiles1/w1/smiles2/w2/..., plus optional architecture.",
    )
    mode.add_argument(
        "--smiles1",
        type=str,
        help="Component 1 repeat-unit SMILES for binary copolymer prediction.",
    )

    parser.add_argument("--smiles2", type=str, default=None, help="Component 2 repeat-unit SMILES.")
    parser.add_argument(
        "--w1",
        type=float,
        default=0.5,
        help="Composition weight for component 1, used as linear mixing coefficient.",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="random",
        choices=sorted(SUPPORTED_ARCHITECTURES),
        help="Approximate architecture tag. 'random' uses descriptor mixing; 'block' uses a Fox-style proxy as primary output.",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional output path (.json or .csv).")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--chain-physics-confs",
        type=int,
        default=50,
        help="Number of conformers when chain-physics cache miss triggers on-the-fly computation.",
    )
    parser.add_argument(
        "--polybert-batch-size",
        type=int,
        default=64,
        help="Reserved for future extension. Single-query polyBERT currently runs one-by-one.",
    )

    parser.add_argument("--data-path", type=str, default=_default_path("data", "unified_tg.parquet"))
    parser.add_argument(
        "--phyc-cache",
        type=str,
        default=_default_path("data", "feature_matrix_PHY-C.parquet"),
    )
    parser.add_argument(
        "--gnn-cache",
        type=str,
        default=_default_path("data", "gnn_embeddings_64d.parquet"),
    )
    parser.add_argument(
        "--pbert-cache",
        type=str,
        default=_default_path("data", "polybert_embeddings.parquet"),
    )
    parser.add_argument(
        "--chain-physics-cache",
        type=str,
        default=_default_path("data", "chain_physics_features.parquet"),
    )
    parser.add_argument(
        "--polybert-model-dir",
        type=str,
        default=_default_path("data", "polybert_model"),
    )
    parser.add_argument(
        "--gnn-checkpoint",
        type=str,
        default=_default_path("checkpoints", "gnn_pretrained.pt"),
    )

    return parser


def _dump_json(data: Dict[str, object], path: Optional[str]) -> None:
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    if path:
        Path(path).write_text(payload, encoding="utf-8")
        print(f"Saved: {path}")
    else:
        print(payload)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.smiles1 and not args.smiles2:
        parser.error("--smiles1 requires --smiles2.")

    paths = _make_paths(args)
    _require_paths(paths)

    predictor = BestTgPredictor(
        paths=paths,
        device=args.device,
        chain_physics_confs=args.chain_physics_confs,
        polybert_batch_size=args.polybert_batch_size,
    )
    predictor.fit()

    if args.smiles:
        result = predictor.predict_homopolymer(args.smiles)
        _dump_json(result, args.output)
        return

    if args.component:
        smiles_list, weights = _parse_component_entries(args.component)
        result = predictor.predict_multicomponent(
            smiles_list,
            weights,
            architecture=args.architecture,
        )
        _dump_json(result, args.output)
        return

    if args.smiles1:
        result = predictor.predict_multicomponent(
            [args.smiles1, args.smiles2],
            [args.w1, 1.0 - args.w1],
            architecture=args.architecture,
        )
        _dump_json(result, args.output)
        return

    df = pd.read_csv(args.input_csv)
    pred_df = predictor.predict_batch(df)

    if args.output:
        out_path = Path(args.output)
        if out_path.suffix.lower() == ".json":
            payload = json.dumps(pred_df.to_dict(orient="records"), ensure_ascii=False, indent=2)
            out_path.write_text(payload, encoding="utf-8")
        else:
            pred_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Saved: {out_path}")
    else:
        print(pred_df.to_string(index=False))


if __name__ == "__main__":
    main()
