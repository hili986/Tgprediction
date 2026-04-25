from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_pipeline import compute_features
from src.ml.universal_tg_features import (
    ComponentRecord,
    PolymerRecord,
    normalize_weights,
    numeric_feature_columns,
    polymer_record_to_features,
)
from src.ml.universal_tg_model import PhysicsResidualKernelRegressor


NUCLEOBASE_ENDPOINT_TG_C = {
    "none": -47.0,
    "A": 65.0,
    "T": 43.0,
    "C": 90.0,
    "G": 130.0,
}


@dataclass
class BuildReport:
    missing_sources: list[str]
    source_counts: dict[str, int]
    skipped_counts: dict[str, int]


class ComponentFeatureFactory:
    def __init__(self, layer: str = "M2M-V", morgan_bits: int = 256) -> None:
        self.layer = layer
        self.morgan_bits = morgan_bits
        self._cache: dict[str, np.ndarray] = {}
        self._error_cache: dict[str, str] = {}
        self._hybrid_lookup: Optional[dict[str, dict[str, np.ndarray | str]]] = None
        self._hybrid_canonical_lookup: Optional[dict[str, dict[str, np.ndarray | str]]] = None

    def vector(self, smiles: str) -> np.ndarray:
        key = str(smiles).strip()
        if not key:
            raise ValueError("empty component SMILES")
        if key in self._cache:
            return self._cache[key].copy()
        if key in self._error_cache:
            raise ValueError(self._error_cache[key])
        try:
            vector = self._compute_vector(key)
            vector = np.asarray(vector, dtype=float).reshape(-1)
        except Exception as exc:
            self._error_cache[key] = f"feature extraction failed for {key}: {exc}"
            raise ValueError(self._error_cache[key]) from exc
        self._cache[key] = vector
        return vector.copy()

    def _compute_vector(self, smiles: str) -> np.ndarray:
        if self.layer.upper() not in {"HYBRID-186", "HYBRID-HOMO186"}:
            return compute_features(smiles, layer=self.layer, morgan_bits=self.morgan_bits)
        base = compute_features(smiles, layer="M2M-V", morgan_bits=self.morgan_bits)
        cached = self._cached_186_component(smiles)
        if cached is None:
            besttg = np.full(186, np.nan, dtype=float)
            hit = 0.0
        else:
            besttg = np.hstack(
                [
                    np.asarray(cached["phyc"], dtype=float),
                    np.asarray(cached["gnn"], dtype=float),
                    np.asarray(cached["pbert"], dtype=float),
                ]
            )
            hit = 1.0
        return np.hstack([base, besttg, np.array([hit], dtype=float)])

    def _cached_186_component(self, smiles: str) -> Optional[dict[str, np.ndarray | str]]:
        if self._hybrid_lookup is None or self._hybrid_canonical_lookup is None:
            self._load_hybrid_186_cache()
        assert self._hybrid_lookup is not None
        assert self._hybrid_canonical_lookup is not None
        if smiles in self._hybrid_lookup:
            return self._hybrid_lookup[smiles]
        from scripts.predict_tg_tabpfn_186d import _canonical_repeat_unit_key

        canonical = _canonical_repeat_unit_key(smiles)
        if canonical is None:
            return None
        return self._hybrid_canonical_lookup.get(canonical)

    def _load_hybrid_186_cache(self) -> None:
        from sklearn.decomposition import PCA

        from scripts.predict_tg_tabpfn_186d import (
            InferencePaths,
            PBERT_PCA_DIM,
            _build_canonical_component_lookup,
            _build_precomputed_component_lookup,
            _load_training_blocks,
        )

        paths = InferencePaths(
            data_path=PROJECT_ROOT / "data" / "unified_tg.parquet",
            phyc_cache=PROJECT_ROOT / "data" / "feature_matrix_PHY-C.parquet",
            gnn_cache=PROJECT_ROOT / "data" / "gnn_embeddings_64d.parquet",
            pbert_cache=PROJECT_ROOT / "data" / "polybert_embeddings.parquet",
            chain_physics_cache=PROJECT_ROOT / "data" / "chain_physics_cache.parquet",
            polybert_model_dir=PROJECT_ROOT / "data" / "polybert_model",
            gnn_checkpoint=PROJECT_ROOT / "checkpoints" / "gnn_pretrained.pt",
        )
        smiles, x_phyc, x_gnn, x_pbert_raw, _ = _load_training_blocks(paths)
        valid = np.isfinite(x_pbert_raw).all(axis=1)
        pca = PCA(n_components=PBERT_PCA_DIM, random_state=42)
        pca.fit(x_pbert_raw[valid])
        x_pbert = np.full((len(x_pbert_raw), PBERT_PCA_DIM), np.nan, dtype=float)
        x_pbert[valid] = pca.transform(x_pbert_raw[valid])
        self._hybrid_lookup = _build_precomputed_component_lookup(smiles, x_phyc, x_gnn, x_pbert)
        self._hybrid_canonical_lookup = _build_canonical_component_lookup(self._hybrid_lookup)


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    yt = y_true[mask]
    yp = y_pred[mask]
    return {
        "n": int(mask.sum()),
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(math.sqrt(mean_squared_error(yt, yp))),
        "r2": float(r2_score(yt, yp)) if len(yt) >= 2 else float("nan"),
    }


def choose_model(name: str, random_state: int = 42):
    key = str(name).lower()
    if key == "catboost":
        try:
            from catboost import CatBoostRegressor

            return CatBoostRegressor(
                iterations=1600,
                depth=6,
                learning_rate=0.03,
                loss_function="RMSE",
                random_seed=random_state,
                verbose=False,
                allow_writing_files=False,
            )
        except Exception:
            key = "extratrees"
    if key == "physics_kernel":
        return PhysicsResidualKernelRegressor(
            n_landmarks=1200,
            prior_lambda=0.5,
            residual_lambda=2.0,
            random_state=random_state,
        )
    if key == "physics_multikernel":
        return PhysicsResidualKernelRegressor(
            n_landmarks=700,
            prior_lambda=0.3,
            residual_lambda=1.0,
            random_state=random_state,
            kernel_scales=(0.25, 1.0, 4.0),
        )
    if key == "physics_local":
        return PhysicsResidualKernelRegressor(
            n_landmarks=1200,
            prior_lambda=0.5,
            residual_lambda=2.0,
            random_state=random_state,
            local_k=12,
            local_weight=0.55,
        )
    if key == "physics_local_light":
        return PhysicsResidualKernelRegressor(
            n_landmarks=1200,
            prior_lambda=0.5,
            residual_lambda=2.0,
            random_state=random_state,
            local_k=8,
            local_weight=0.15,
        )
    if key == "physics_hybrid_balanced":
        return PhysicsResidualKernelRegressor(
            n_landmarks=1200,
            prior_lambda=0.5,
            residual_lambda=2.0,
            random_state=random_state,
            high_dim_start=46,
            high_dim_end=233,
            high_dim_kernel_weight=0.25,
        )
    if key == "physics_homo_correction":
        return PhysicsResidualKernelRegressor(
            n_landmarks=1200,
            prior_lambda=0.5,
            residual_lambda=2.0,
            random_state=random_state,
            high_dim_start=46,
            high_dim_end=233,
            high_dim_kernel_weight=0.0,
            homo_correction=True,
            homo_correction_lambda=0.5,
            homo_correction_landmarks=900,
        )
    if key == "physics_additive_kernel":
        return PhysicsResidualKernelRegressor(
            n_landmarks=700,
            prior_lambda=0.5,
            residual_lambda=2.0,
            random_state=random_state,
            additive_kernel_groups=(
                (
                    "endpoint_tg_",
                    "endpoint_missing",
                    "w_",
                    "n_components",
                    "is_homopolymer",
                    "is_random",
                    "is_block",
                    "is_multicomponent",
                ),
                (
                    "emb_mean_",
                    "emb_std_",
                    "emb_contrast_",
                ),
            ),
        )
    if key == "histgradient":
        return HistGradientBoostingRegressor(
            max_iter=800,
            learning_rate=0.035,
            l2_regularization=0.02,
            random_state=random_state,
        )
    if key == "xgboost":
        try:
            from xgboost import XGBRegressor

            return XGBRegressor(
                n_estimators=1200,
                max_depth=5,
                learning_rate=0.025,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=2.0,
                objective="reg:squarederror",
                n_jobs=-1,
                random_state=random_state,
            )
        except Exception:
            key = "extratrees"
    if key == "lightgbm":
        try:
            from lightgbm import LGBMRegressor

            return LGBMRegressor(
                n_estimators=1400,
                num_leaves=31,
                learning_rate=0.025,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=2.0,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            )
        except Exception:
            key = "extratrees"
    if key == "extratrees":
        return ExtraTreesRegressor(
            n_estimators=600,
            min_samples_leaf=1,
            max_features=0.65,
            n_jobs=-1,
            random_state=random_state,
        )
    raise ValueError(f"unsupported model: {name}")


def make_sample_weights(
    frame: pd.DataFrame,
    homopolymer_weight: float,
    virtual_weight: float,
    copolymer_weight: float,
    nucleobase_weight: float,
) -> np.ndarray:
    source = frame["source"].astype(str).str.lower()
    weights = np.ones(len(frame), dtype=float)
    weights[source.str.contains("homopolymer", regex=False)] = homopolymer_weight
    weights[source.str.contains("virtual", regex=False)] = virtual_weight
    weights[source.str.contains("polyinfo", regex=False) | source.str.contains("copolymer_real", regex=False)] = copolymer_weight
    weights[source.str.contains("nucleobase", regex=False)] = nucleobase_weight
    return weights


def _finite_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _split_serialized(text: object) -> list[str]:
    if text is None or pd.isna(text):
        return []
    raw = str(text).strip()
    if not raw:
        return []
    sep = "|" if "|" in raw else ";"
    return [part.strip() for part in raw.split(sep) if part.strip()]


def _parse_components_and_weights(row: pd.Series) -> tuple[list[str], list[float]]:
    components = _split_serialized(row.get("components_serialized"))
    weights = [_finite_float(part) for part in _split_serialized(row.get("weights_serialized"))]
    if components and len(weights) == len(components) and all(w is not None for w in weights):
        return components, [float(w) for w in weights if w is not None]

    component_cols = sorted(
        [col for col in row.index if str(col).lower().startswith(("smiles", "component"))],
        key=str,
    )
    weight_cols = sorted(
        [col for col in row.index if str(col).lower().startswith(("w", "weight"))],
        key=str,
    )
    components = [str(row[col]).strip() for col in component_cols if str(row[col]).strip()]
    weights = [_finite_float(row[col]) for col in weight_cols[: len(components)]]
    if components and len(weights) == len(components) and all(w is not None for w in weights):
        return components, [float(w) for w in weights if w is not None]
    if components:
        return components, [1.0] * len(components)
    return [], []


def build_table_from_records(records: Sequence[PolymerRecord]) -> pd.DataFrame:
    rows = [polymer_record_to_features(record) for record in records]
    frame = pd.DataFrame(rows)
    for column in frame.columns:
        if column in {"sample_id", "source", "architecture", "split_group", "component_smiles", "weights_serialized"}:
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def mask_hybrid186_for_nonhomopolymer(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    non_homo = out["is_homopolymer"].astype(float).lt(0.5)
    for column in out.columns:
        if not column.startswith(("emb_mean_", "emb_std_", "emb_min_", "emb_max_", "emb_contrast_")):
            continue
        try:
            dim = int(column.rsplit("_", 1)[1])
        except ValueError:
            continue
        if 46 <= dim <= 232:
            out.loc[non_homo, column] = np.nan
    return out


def _component(factory: ComponentFeatureFactory, smiles: str, endpoint_tg_c: Optional[float], endpoint_source: str) -> ComponentRecord:
    return ComponentRecord(
        smiles=str(smiles),
        vector=factory.vector(str(smiles)),
        endpoint_tg_c=endpoint_tg_c,
        endpoint_source=endpoint_source,
    )


def load_homopolymer_records(path: Path, factory: ComponentFeatureFactory, limit: Optional[int] = None) -> tuple[list[PolymerRecord], int]:
    if limit == 0:
        return [], 0
    if not path.exists():
        return [], 0
    frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    if limit is not None:
        frame = frame.head(limit)
    records: list[PolymerRecord] = []
    skipped = 0
    for idx, row in frame.iterrows():
        smiles = str(row.get("smiles", "")).strip()
        tg_k = _finite_float(row.get("tg_k"))
        tg_c = _finite_float(row.get("tg_c"))
        if tg_c is None and tg_k is not None:
            tg_c = tg_k - 273.15
        if not smiles or tg_c is None:
            skipped += 1
            continue
        try:
            records.append(
                PolymerRecord(
                    sample_id=f"homopolymer:{idx}",
                    source="homopolymer_real",
                    architecture="homo",
                    components=[_component(factory, smiles, None, "missing")],
                    weights=[1.0],
                    target_tg_c=tg_c,
                    split_group=smiles,
                )
            )
        except ValueError:
            skipped += 1
    return records, skipped


def load_polyinfo_records(path: Path, factory: ComponentFeatureFactory, limit: Optional[int] = None) -> tuple[list[PolymerRecord], int]:
    if limit == 0:
        return [], 0
    if not path.exists():
        return [], 0
    frame = pd.read_csv(path)
    if "status" in frame.columns:
        frame = frame[frame["status"].astype(str).eq("usable")].copy()
    if limit is not None:
        frame = frame.head(limit)
    records: list[PolymerRecord] = []
    skipped = 0
    for idx, row in frame.iterrows():
        target = _finite_float(row.get("Tg_C"))
        w1 = _finite_float(row.get("w1_used"))
        w2 = _finite_float(row.get("w2_used"))
        smiles1 = str(row.get("canonical_1") or row.get("SMILES_1") or "").strip()
        smiles2 = str(row.get("canonical_2") or row.get("SMILES_2") or "").strip()
        endpoint1 = _finite_float(row.get("endpoint_1_tg_c"))
        endpoint2 = _finite_float(row.get("endpoint_2_tg_c"))
        if target is None or w1 is None or w2 is None or not smiles1 or not smiles2:
            skipped += 1
            continue
        try:
            records.append(
                PolymerRecord(
                    sample_id=f"polyinfo:{row.get('sample_id', idx)}",
                    source="polyinfo_real",
                    architecture="random",
                    components=[
                        _component(factory, smiles1, endpoint1, "measured" if endpoint1 is not None else "missing"),
                        _component(factory, smiles2, endpoint2, "measured" if endpoint2 is not None else "missing"),
                    ],
                    weights=[w1, w2],
                    target_tg_c=target,
                    split_group=str(row.get("COID", "")) or None,
                    metadata={"polyinfo_w1": w1, "polyinfo_w2": w2},
                )
            )
        except ValueError:
            skipped += 1
    return records, skipped


def _nucleobase_endpoint_for_component(index: int, base: str) -> tuple[Optional[float], str]:
    key = "none" if index == 0 else str(base).strip().upper()
    if key in NUCLEOBASE_ENDPOINT_TG_C:
        return NUCLEOBASE_ENDPOINT_TG_C[key], "measured"
    return None, "missing"


def load_nucleobase_records(path: Path, factory: ComponentFeatureFactory, limit: Optional[int] = None) -> tuple[list[PolymerRecord], int]:
    if limit == 0:
        return [], 0
    if not path.exists():
        return [], 0
    frame = pd.read_csv(path)
    if "status" in frame.columns:
        frame = frame[frame["status"].astype(str).eq("predicted")].copy()
    if "Architecture" in frame.columns:
        frame = frame[frame["Architecture"].astype(str).str.contains("random", case=False, na=False)].copy()
    if limit is not None:
        frame = frame.head(limit)
    records: list[PolymerRecord] = []
    skipped = 0
    for idx, row in frame.iterrows():
        target = _finite_float(row.get("Tg_C_actual"))
        components, weights = _parse_components_and_weights(row)
        if target is None or len(components) < 2 or len(components) != len(weights):
            skipped += 1
            continue
        base = str(row.get("Nucleobase", "")).strip().upper()
        try:
            comp_records = []
            for comp_idx, smiles in enumerate(components):
                endpoint, source = _nucleobase_endpoint_for_component(comp_idx, base)
                comp_records.append(_component(factory, smiles, endpoint, source))
            records.append(
                PolymerRecord(
                    sample_id=f"nucleobase:{row.get('Data_ID', idx)}",
                    source="nucleobase_real",
                    architecture="random",
                    components=comp_records,
                    weights=weights,
                    target_tg_c=target,
                    split_group=base or None,
                    metadata={"nucleobase_mol_pct": _finite_float(row.get("Polymer_mol_pct")) or float("nan")},
                )
            )
        except ValueError:
            skipped += 1
    return records, skipped


def _target_from_virtual_row(row: pd.Series) -> Optional[float]:
    for column in ["tg_c_pred", "tg_c", "Tg_C", "target_tg_c", "pred_tg_c"]:
        value = _finite_float(row.get(column))
        if value is not None:
            return value
    tg_k = _finite_float(row.get("tg_k_pred"))
    return tg_k - 273.15 if tg_k is not None else None


def load_virtual_records(path: Path, factory: ComponentFeatureFactory, limit: Optional[int] = None) -> tuple[list[PolymerRecord], int]:
    if limit == 0:
        return [], 0
    if not path.exists():
        return [], 0
    frame = pd.read_csv(path)
    if "status" in frame.columns:
        frame = frame[~frame["status"].astype(str).str.contains("error", case=False, na=False)].copy()
    if limit is not None:
        frame = frame.head(limit)
    records: list[PolymerRecord] = []
    skipped = 0
    for idx, row in frame.iterrows():
        target = _target_from_virtual_row(row)
        components, weights = _parse_components_and_weights(row)
        architecture = str(row.get("architecture", "random")).strip().lower() or "random"
        if target is None or len(components) < 2 or len(components) != len(weights):
            skipped += 1
            continue
        if architecture not in {"random", "block", "multicomponent"}:
            architecture = "random" if len(components) == 2 else "multicomponent"
        try:
            records.append(
                PolymerRecord(
                    sample_id=f"virtual:{row.get('recipe_id', idx)}",
                    source="virtual_copolymer",
                    architecture=architecture,
                    components=[_component(factory, smiles, None, "missing") for smiles in components],
                    weights=weights,
                    target_tg_c=target,
                    split_group=str(row.get("recipe_id", idx)),
                )
            )
        except ValueError:
            skipped += 1
    return records, skipped


def build_unified_training_table(args: argparse.Namespace) -> tuple[pd.DataFrame, BuildReport]:
    factory = ComponentFeatureFactory(layer=args.feature_layer, morgan_bits=args.morgan_bits)
    sources = [
        ("homopolymer", Path(args.homopolymer_data), load_homopolymer_records, args.max_homopolymer),
        ("polyinfo", Path(args.polyinfo_details), load_polyinfo_records, args.max_polyinfo),
        ("nucleobase", Path(args.nucleobase_details), load_nucleobase_records, args.max_nucleobase),
        ("virtual", Path(args.virtual_csv), load_virtual_records, args.max_virtual),
    ]
    records: list[PolymerRecord] = []
    missing_sources: list[str] = []
    source_counts: dict[str, int] = {}
    skipped_counts: dict[str, int] = {}
    for name, path, loader, limit in sources:
        if not path.exists():
            missing_sources.append(f"{name}:{path}")
            source_counts[name] = 0
            skipped_counts[name] = 0
            continue
        loaded, skipped = loader(path, factory, limit)
        records.extend(loaded)
        source_counts[name] = len(loaded)
        skipped_counts[name] = skipped
    table = build_table_from_records(records)
    if str(args.feature_layer).upper() == "HYBRID-HOMO186":
        table = mask_hybrid186_for_nonhomopolymer(table)
    return table, BuildReport(missing_sources, source_counts, skipped_counts)


def load_unified_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def make_estimator(model_name: str, random_state: int) -> Pipeline:
    if str(model_name).lower() == "physics_kernel":
        return choose_model(model_name, random_state=random_state)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", choose_model(model_name, random_state=random_state)),
        ]
    )


def _fit_with_optional_weights(estimator: Pipeline, x: pd.DataFrame, y: pd.Series, weights: np.ndarray) -> None:
    if not isinstance(estimator, Pipeline):
        estimator.fit(x, y, sample_weight=weights)
        return
    try:
        estimator.fit(x, y, model__sample_weight=weights)
    except TypeError:
        estimator.fit(x, y)


def evaluate_holdout(frame: pd.DataFrame, feature_columns: list[str], args: argparse.Namespace) -> tuple[Pipeline, dict, pd.DataFrame]:
    stratify = None
    if frame["source"].nunique() > 1 and frame["source"].value_counts().min() >= 2:
        stratify = frame["source"]
    train, test = train_test_split(
        frame,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )
    estimator = make_estimator(args.model, args.random_state)
    weights = make_sample_weights(
        train,
        args.homopolymer_weight,
        args.virtual_weight,
        args.copolymer_weight,
        args.nucleobase_weight,
    )
    _fit_with_optional_weights(estimator, train[feature_columns], train["target_tg_c"], weights)
    pred = estimator.predict(test[feature_columns])
    details_cols = [
        col
        for col in ["sample_id", "source", "architecture", "split_group", "target_tg_c"]
        if col in test.columns
    ]
    details = test[details_cols].copy()
    details["pred_tg_c"] = pred
    details["split"] = "holdout"
    summary = {"overall_holdout": compute_metrics(details["target_tg_c"], details["pred_tg_c"])}
    for source, group in details.groupby("source"):
        summary[f"holdout_{source}"] = compute_metrics(group["target_tg_c"], group["pred_tg_c"])
    return estimator, summary, details


def evaluate_group_holdout(
    frame: pd.DataFrame,
    feature_columns: list[str],
    args: argparse.Namespace,
    source_name: str,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    if "split_group" not in frame.columns:
        return {}, pd.DataFrame()
    subset = frame[
        frame["source"].astype(str).eq(source_name)
        & frame["split_group"].notna()
        & frame["split_group"].astype(str).ne("")
    ].copy()
    groups = sorted(subset["split_group"].astype(str).unique())
    if len(groups) < 2:
        return {}, pd.DataFrame()
    if args.max_group_eval_folds > 0:
        groups = groups[: args.max_group_eval_folds]

    details: list[pd.DataFrame] = []
    for group in groups:
        test_index = subset.index[subset["split_group"].astype(str).eq(group)]
        train = frame.drop(index=test_index)
        test = frame.loc[test_index]
        if train.empty or test.empty:
            continue
        estimator = make_estimator(args.model, args.random_state)
        weights = make_sample_weights(
            train,
            args.homopolymer_weight,
            args.virtual_weight,
            args.copolymer_weight,
            args.nucleobase_weight,
        )
        _fit_with_optional_weights(estimator, train[feature_columns], train["target_tg_c"], weights)
        fold = test[["sample_id", "source", "architecture", "split_group", "target_tg_c"]].copy()
        fold["pred_tg_c"] = estimator.predict(test[feature_columns])
        fold["split"] = f"group_holdout:{source_name}"
        fold["heldout_group"] = group
        details.append(fold)

    if not details:
        return {}, pd.DataFrame()
    all_details = pd.concat(details, ignore_index=True)
    metric_name = f"group_holdout_{source_name}"
    return {
        metric_name: compute_metrics(all_details["target_tg_c"], all_details["pred_tg_c"])
    }, all_details


def train_final_model(frame: pd.DataFrame, feature_columns: list[str], args: argparse.Namespace) -> Pipeline:
    estimator = make_estimator(args.model, args.random_state)
    weights = make_sample_weights(
        frame,
        args.homopolymer_weight,
        args.virtual_weight,
        args.copolymer_weight,
        args.nucleobase_weight,
    )
    _fit_with_optional_weights(estimator, frame[feature_columns], frame["target_tg_c"], weights)
    return estimator


def _write_experiment_log(path: Path, summary: dict) -> None:
    lines = [
        "# Universal Single-Regressor Tg Experiment",
        "",
        "## Guess",
        "",
        "A single regressor can use intrinsic component features, composition statistics, and endpoint priors to cover homopolymers, general copolymers, and nucleobase copolymers.",
        "",
        "## Experiment",
        "",
        f"- model: `{summary.get('model')}`",
        f"- feature layer: `{summary.get('feature_layer')}`",
        f"- n rows: `{summary.get('n_rows')}`",
        f"- n features: `{summary.get('n_features')}`",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary.get("metrics", {}), indent=2),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one universal Tg single-regressor model.")
    parser.add_argument("--build-table", action="store_true", help="Build unified feature table before training.")
    parser.add_argument("--table", default="results/universal_single_regressor/unified_training_table.parquet")
    parser.add_argument("--output-dir", default="results/universal_single_regressor/exp_default")
    parser.add_argument("--model", default="catboost", choices=["catboost", "extratrees", "histgradient", "xgboost", "lightgbm", "physics_kernel", "physics_multikernel", "physics_local", "physics_local_light", "physics_hybrid_balanced", "physics_homo_correction", "physics_additive_kernel"])
    parser.add_argument("--feature-layer", default="M2M-V")
    parser.add_argument("--morgan-bits", type=int, default=256)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--homopolymer-weight", type=float, default=1.0)
    parser.add_argument("--virtual-weight", type=float, default=0.2)
    parser.add_argument("--copolymer-weight", type=float, default=10.0)
    parser.add_argument("--nucleobase-weight", type=float, default=20.0)
    parser.add_argument("--homopolymer-data", default="data/unified_tg.parquet")
    parser.add_argument("--polyinfo-details", default="results/copolymer_residual_model/polyinfo_physics_details_clean.csv")
    parser.add_argument("--nucleobase-details", default="results/copolymer_residual_model/nucleobase_strategy_details.csv")
    parser.add_argument("--virtual-csv", default="results/virtual_data/bicerano_binary_random_100k_reuse.csv")
    parser.add_argument("--max-homopolymer", type=int, default=-1)
    parser.add_argument("--max-polyinfo", type=int, default=-1)
    parser.add_argument("--max-nucleobase", type=int, default=-1)
    parser.add_argument("--max-virtual", type=int, default=20000)
    parser.add_argument("--group-eval", action="store_true", help="Run expensive source-specific group holdout evaluation.")
    parser.add_argument("--max-group-eval-folds", type=int, default=20)
    return parser


def _normalise_limit(value: int) -> Optional[int]:
    return value if value >= 0 else None


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.max_homopolymer = _normalise_limit(args.max_homopolymer)
    args.max_polyinfo = _normalise_limit(args.max_polyinfo)
    args.max_nucleobase = _normalise_limit(args.max_nucleobase)
    args.max_virtual = _normalise_limit(args.max_virtual)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = Path(args.table)

    build_report = BuildReport([], {}, {})
    if args.build_table or not table_path.exists():
        table, build_report = build_unified_training_table(args)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(table_path, index=False)
        table.to_parquet(out_dir / "unified_training_table.parquet", index=False)
    else:
        table = load_unified_table(table_path)

    table["target_tg_c"] = pd.to_numeric(table["target_tg_c"], errors="coerce")
    table = table[np.isfinite(table["target_tg_c"])].copy()
    if len(table) < 5:
        raise ValueError(f"Need at least 5 training rows, got {len(table)}.")

    feature_columns = numeric_feature_columns(table)
    if not feature_columns:
        raise ValueError("No numeric feature columns found.")

    holdout_model, metrics, predictions = evaluate_holdout(table, feature_columns, args)
    prediction_frames = [predictions]
    if args.group_eval:
        for source_name in ["polyinfo_real", "nucleobase_real"]:
            group_metrics, group_predictions = evaluate_group_holdout(
                table,
                feature_columns,
                args,
                source_name,
            )
            metrics.update(group_metrics)
            if not group_predictions.empty:
                prediction_frames.append(group_predictions)
    final_model = train_final_model(table, feature_columns, args)

    summary = {
        "model": args.model,
        "feature_layer": args.feature_layer,
        "morgan_bits": args.morgan_bits,
        "n_rows": int(len(table)),
        "n_features": int(len(feature_columns)),
        "source_counts": table["source"].value_counts().to_dict(),
        "build_source_counts": build_report.source_counts,
        "skipped_counts": build_report.skipped_counts,
        "missing_sources": build_report.missing_sources,
        "weights": {
            "homopolymer": args.homopolymer_weight,
            "virtual": args.virtual_weight,
            "copolymer": args.copolymer_weight,
            "nucleobase": args.nucleobase_weight,
        },
        "metrics": metrics,
    }

    joblib.dump(final_model, out_dir / "model.joblib")
    joblib.dump(holdout_model, out_dir / "holdout_model.joblib")
    (out_dir / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.concat(prediction_frames, ignore_index=True, sort=False).to_csv(
        out_dir / "predictions_by_split.csv",
        index=False,
    )
    _write_experiment_log(out_dir / "experiment_log.md", summary)
    print(json.dumps(summary["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
