"""
Phase 5 GNN Experiment Script: Tandem-M2M Embedding Fusion (E25-E31)
Phase 5 GNN 实验脚本：Tandem-M2M 嵌入融合实验 (E25-E31)

Independent from Phase 5 tabular experiments (E16-E24 in exp_phase5.py).
Uses same unified_tg.parquet and Nested CV framework.

Experiment Matrix:
    E25: GNN 64d embedding → CatBoost (no pretrain)
    E26: GNN 64d embedding → CatBoost (pretrain ~59K)
    E27: M2M-V 46d + GNN 64d = 110d → CatBoost
    E28: 110d → Stacking v2 (with TabPFN v2)  — CORE EXPERIMENT
    E29: VPD-Deep 64d replaces VPD 12d → Stacking v2
    E30: E28 + MultiTask (Tg+density+SOL)
    E31: Deep Ensemble ×5 + Conformal (E28 config)

Usage:
    python scripts/exp_phase5_gnn.py                     # 运行全部实验
    python scripts/exp_phase5_gnn.py --exp E25            # 运行单个实验
    python scripts/exp_phase5_gnn.py --exp E25 E26 E27    # 运行多个实验
    python scripts/exp_phase5_gnn.py --layer3             # 启用 Kuenneth 共聚物预训练
    python scripts/exp_phase5_gnn.py --device cpu         # 使用 CPU
    python scripts/exp_phase5_gnn.py --pretrain-epochs 50 # 自定义预训练轮数
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Silence joblib/loky wmic deprecation warning on Windows 11
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore", message=".*wmic.*", category=UserWarning)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_LAYER = "M2M-V"  # 46-dim tabular features
RESULT_DIR = PROJECT_ROOT / "results" / "phase5"
DATA_PATH = PROJECT_ROOT / "data" / "unified_tg.parquet"

# GNN defaults
GNN_EMBEDDING_DIM = 64
TABULAR_DIM = 46
FUSED_DIM = TABULAR_DIM + GNN_EMBEDDING_DIM  # 110

# Training defaults
PRETRAIN_EPOCHS = 100
FINETUNE_EPOCHS = 50
PATIENCE = 10
BATCH_SIZE_PRETRAIN = 256
BATCH_SIZE_FINETUNE = 32

# CV defaults
OUTER_SPLITS = 5
OUTER_REPEATS = 3


# ---------------------------------------------------------------------------
# Lazy imports — only import GNN/torch when actually needed
# ---------------------------------------------------------------------------

def _build_stacking_b2():
    """Build Stacking per 方案B.2: CatBoost + TabPFN v2 + LightGBM + ExtraTrees → Ridge.

    Per B.8, src/ml/ files are not modified. This builds the correct stacking
    with TabPFN v2 as specified in 方案B lines 194-198.

    Falls back to build_stacking_v2() (XGBoost) if TabPFN is not installed.

    Returns:
        Tuple of (StackingRegressor, base_model_names: list[str]).
    """
    try:
        from tabpfn import TabPFNRegressor
        use_tabpfn = True
    except ImportError:
        use_tabpfn = False

    if not use_tabpfn:
        print("  [!] TabPFN 未安装，回退到 XGBoost (build_stacking_v2)")
        from src.ml.sklearn_models import build_stacking_v2
        stacking = build_stacking_v2()
        return stacking, ["CatBoost", "LightGBM", "ExtraTrees", "XGBoost"]

    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import ExtraTreesRegressor, StackingRegressor
    from sklearn.linear_model import Ridge

    cb = CatBoostRegressor(
        iterations=1000, learning_rate=0.05, depth=6,
        l2_leaf_reg=3.0, random_seed=42, verbose=0,
    )
    tabpfn = TabPFNRegressor()
    lgbm = LGBMRegressor(
        n_estimators=1000, learning_rate=0.05, num_leaves=31,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1,
    )
    et = ExtraTreesRegressor(
        n_estimators=500, max_features="sqrt",
        min_samples_leaf=2, random_state=42, n_jobs=-1,
    )

    stacking = StackingRegressor(
        estimators=[
            ("catboost", cb),
            ("tabpfn", tabpfn),
            ("lgbm", lgbm),
            ("et", et),
        ],
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1,
    )
    return stacking, ["CatBoost", "TabPFN v2", "LightGBM", "ExtraTrees"]


def _check_torch():
    """Check that PyTorch and PyG are available."""
    try:
        import torch
        import torch_geometric
        return True
    except ImportError as e:
        print(f"[!] PyTorch/PyG 未安装: {e}")
        print("  安装: pip install torch torch-geometric")
        return False


def _get_device(requested: str = "cuda") -> str:
    """Resolve device string."""
    import torch
    if requested == "cuda" and not torch.cuda.is_available():
        print("  [!] CUDA 不可用，回退到 CPU")
        return "cpu"
    return requested


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_unified_data(verbose: bool = True) -> Tuple[list, np.ndarray, list, np.ndarray]:
    """Load unified dataset, return (smiles_train, y_train, smiles_test, y_test).

    Returns:
        Tuple of (smiles_train, y_train, smiles_test, y_test).
    """
    from src.data.external_datasets import load_unified_dataset

    if verbose:
        print(f"\n{'='*60}")
        print(f"  加载统一数据集: {DATA_PATH}")
        print(f"{'='*60}")

    df_train = load_unified_dataset(str(DATA_PATH), split="train")
    df_test = load_unified_dataset(str(DATA_PATH), split="test")

    smiles_train = df_train["smiles"].tolist()
    y_train = df_train["tg_k"].values.astype(np.float32)
    smiles_test = df_test["smiles"].tolist()
    y_test = df_test["tg_k"].values.astype(np.float32)

    if verbose:
        print(f"  训练集: {len(smiles_train)} 条")
        print(f"  测试集: {len(smiles_test)} 条")
        print(f"  Tg 范围: [{y_train.min():.0f}, {y_train.max():.0f}] K (train)")

    return smiles_train, y_train, smiles_test, y_test


def load_tabular_features(
    smiles_list: List[str],
    layer: str = FEATURE_LAYER,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """Compute M2M-V 46d tabular features for a list of SMILES.

    Returns:
        (X, feature_names, valid_indices) — skips SMILES that fail.
    """
    from src.features.feature_pipeline import compute_features, get_feature_names

    feature_names = get_feature_names(layer)
    X_list = []
    valid_idx = []

    t0 = time.time()
    for i, smi in enumerate(smiles_list):
        try:
            x = compute_features(smi, layer=layer)
            if not np.any(np.isnan(x)):
                X_list.append(x)
                valid_idx.append(i)
        except Exception:
            pass

    elapsed = time.time() - t0
    if verbose:
        print(f"  表格特征: {len(X_list)}/{len(smiles_list)} 成功 "
              f"({layer}, {len(feature_names)}d, {elapsed:.1f}s)")

    return np.array(X_list), feature_names, valid_idx


def build_pretrain_data(
    include_layer3: bool = False,
    exclude_smiles: Optional[set] = None,
    verbose: bool = True,
) -> Tuple[List[str], List[float]]:
    """Build pretraining dataset (Layer 1 + Layer 2 + optional Layer 3).

    Layer 1: ~13,400 experimental homopolymer Tg (PolyMetriX + NeurIPS)
    Layer 2: ~46,000 Fox virtual copolymer Tg
    Layer 3: ~18,445 Kuenneth copolymer Tg (OPTIONAL, default OFF)

    Args:
        include_layer3: Enable Kuenneth copolymer data (default: False).
        exclude_smiles: Set of canonical SMILES to exclude (anti-leak).
        verbose: Print statistics.

    Returns:
        (smiles_list, tg_list) for pretraining.
    """
    from src.data.external_datasets import load_all_external
    from src.data.fox_copolymer_generator import generate_copolymer_data

    smiles_all = []
    tg_all = []

    # Layer 1: Experimental homopolymers (~13,400)
    if verbose:
        print("\n  --- 预训练数据构建 ---")
        print("  Layer 1: 实验均聚物...")
    external = load_all_external(verbose=verbose)
    for entry in external:
        smiles_all.append(entry["smiles"])
        tg_all.append(entry["tg_k"])
    if verbose:
        print(f"  Layer 1: {len(external)} 条")

    # Layer 2: Fox virtual copolymers (~46,000)
    if verbose:
        print("  Layer 2: Fox 虚拟共聚物...")
    fox_data = generate_copolymer_data(max_samples=50000)
    for entry in fox_data:
        smiles_all.append(entry["smiles"])
        tg_all.append(entry["tg"])
    if verbose:
        print(f"  Layer 2: {len(fox_data)} 条")

    # Layer 3: Kuenneth copolymers (OPTIONAL, default OFF)
    if include_layer3:
        if verbose:
            print("  Layer 3: Kuenneth 共聚物 (可选)...")
        from src.data.external_datasets import _load_kuenneth_copolymer
        kuenneth = _load_kuenneth_copolymer()
        for entry in kuenneth:
            smiles_all.append(entry["smiles"])
            tg_all.append(entry["tg_k"])
        if verbose:
            print(f"  Layer 3: {len(kuenneth)} 条")
    elif verbose:
        print("  Layer 3: Kuenneth 共聚物 — 跳过 (使用 --layer3 启用)")

    # Anti-leak: exclude target dataset SMILES
    if exclude_smiles:
        before = len(smiles_all)
        filtered = [
            (s, t) for s, t in zip(smiles_all, tg_all)
            if s not in exclude_smiles
        ]
        smiles_all = [s for s, _ in filtered]
        tg_all = [t for _, t in filtered]
        if verbose:
            print(f"  防泄漏: 排除 {before - len(smiles_all)} 条重叠 SMILES")

    if verbose:
        print(f"  预训练总计: {len(smiles_all)} 条")

    return smiles_all, tg_all


# ---------------------------------------------------------------------------
# GNN training utilities
# ---------------------------------------------------------------------------

def train_gnn_with_pretrain(
    smiles_train: List[str],
    y_train: np.ndarray,
    tabular_train: Optional[np.ndarray],
    pretrain_smiles: Optional[List[str]],
    pretrain_tg: Optional[List[float]],
    device: str = "cuda",
    pretrain_epochs: int = PRETRAIN_EPOCHS,
    finetune_epochs: int = FINETUNE_EPOCHS,
    patience: int = PATIENCE,
    gnn_config: Optional[dict] = None,
    tab_valid_indices: Optional[List[int]] = None,
) -> "TgPretrainer":
    """Train a TandemM2M model with optional pretraining.

    Args:
        tab_valid_indices: Original SMILES indices for each row in
            tabular_train (from load_tabular_features). Required when
            tabular_train is a pre-filtered array.

    Returns:
        Trained TgPretrainer instance.
    """
    import torch
    from torch_geometric.loader import DataLoader
    from src.gnn.graph_builder import batch_smiles_to_graphs
    from src.gnn.tandem_m2m import TandemM2M
    from src.gnn.pretrainer import TgPretrainer

    config = gnn_config or {}
    tabular_dim = tabular_train.shape[1] if tabular_train is not None else 1

    # Build finetune graphs
    print("  构建微调图...")
    finetune_graphs, valid_idx = batch_smiles_to_graphs(
        smiles_train, y_list=y_train.tolist(),
    )

    # Attach tabular features (index mapping: valid_idx → tab_valid_indices)
    if tabular_train is not None:
        tab_idx_map = (
            {orig: pos for pos, orig in enumerate(tab_valid_indices)}
            if tab_valid_indices is not None
            else None
        )
        for i, g in enumerate(finetune_graphs):
            orig_i = valid_idx[i]
            if tab_idx_map is not None and orig_i in tab_idx_map:
                g.tabular = torch.tensor(
                    tabular_train[tab_idx_map[orig_i]], dtype=torch.float
                ).unsqueeze(0)
            elif tab_idx_map is None:
                g.tabular = torch.tensor(
                    tabular_train[orig_i], dtype=torch.float
                ).unsqueeze(0)
            else:
                g.tabular = torch.zeros(
                    1, tabular_train.shape[1], dtype=torch.float
                )

    finetune_loader = DataLoader(
        finetune_graphs, batch_size=BATCH_SIZE_FINETUNE, shuffle=True,
    )

    # Build pretrain graphs if provided
    pretrain_loader = None
    if pretrain_smiles is not None and len(pretrain_smiles) > 0:
        print(f"  构建预训练图 ({len(pretrain_smiles)} 条)...")
        pretrain_graphs, _ = batch_smiles_to_graphs(
            pretrain_smiles, y_list=pretrain_tg,
        )
        pretrain_loader = DataLoader(
            pretrain_graphs, batch_size=BATCH_SIZE_PRETRAIN, shuffle=True,
        )
        print(f"  预训练图: {len(pretrain_graphs)} 条成功")

    # Create model
    model = TandemM2M(
        in_dim=config.get("in_dim", 25),
        tabular_dim=tabular_dim,
        gnn_hidden=config.get("gnn_hidden", 128),
        gnn_out=config.get("gnn_out", GNN_EMBEDDING_DIM),
        dropout=config.get("dropout", 0.1),
        edge_dim=config.get("edge_dim", 6),
    )

    trainer = TgPretrainer(model, device=device, tabular_dim=tabular_dim)

    # Stage 1: Pretrain
    if pretrain_loader is not None:
        print(f"  Stage 1: 预训练 ({pretrain_epochs} epochs)...")
        trainer.pretrain(pretrain_loader, epochs=pretrain_epochs)

    # Stage 2: Finetune
    print(f"  Stage 2: 微调 ({finetune_epochs} epochs, patience={patience})...")
    trainer.finetune(finetune_loader, epochs=finetune_epochs, patience=patience)

    return trainer


def extract_gnn_embeddings(
    trainer: "TgPretrainer",
    smiles_list: List[str],
    device: str = "cuda",
) -> Tuple[np.ndarray, List[int]]:
    """Extract 64d GNN embeddings for a list of SMILES.

    Returns:
        (embeddings [N, 64], valid_indices).
    """
    import torch
    from src.gnn.graph_builder import smiles_to_graph

    model = trainer.model
    model.eval()

    embeddings = []
    valid_idx = []

    with torch.no_grad():
        for i, smi in enumerate(smiles_list):
            graph = smiles_to_graph(smi)
            if graph is None:
                continue
            graph = graph.to(device)
            emb = model.get_embedding(graph)  # [1, 64]
            embeddings.append(emb.squeeze(0).cpu().numpy())
            valid_idx.append(i)

    return np.array(embeddings), valid_idx


def fuse_features(
    tabular: np.ndarray,
    gnn_emb: np.ndarray,
    tab_valid: List[int],
    gnn_valid: List[int],
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Fuse tabular (46d) + GNN embedding (64d) → 110d.

    Only keeps samples that succeeded in BOTH pipelines.

    Returns:
        (X_fused [N, 110], y_aligned [N], common_indices).
    """
    tab_set = set(tab_valid)
    gnn_set = set(gnn_valid)
    common = sorted(tab_set & gnn_set)

    tab_map = {idx: pos for pos, idx in enumerate(tab_valid)}
    gnn_map = {idx: pos for pos, idx in enumerate(gnn_valid)}

    X_fused = np.column_stack([
        tabular[[tab_map[i] for i in common]],
        gnn_emb[[gnn_map[i] for i in common]],
    ])
    y_aligned = y[common]

    return X_fused, y_aligned, common


# ---------------------------------------------------------------------------
# Individual experiment functions
# ---------------------------------------------------------------------------

def run_e25_gnn_baseline(
    smiles_train: List[str], y_train: np.ndarray,
    smiles_test: List[str], y_test: np.ndarray,
    device: str = "cuda",
    finetune_epochs: int = FINETUNE_EPOCHS,
) -> Dict[str, Any]:
    """E25: GNN 64d embedding → CatBoost (no pretrain)."""
    print(f"\n{'='*60}")
    print("  E25: GNN 64d 嵌入 → CatBoost (无预训练)")
    print(f"{'='*60}")

    from src.ml.evaluation import nested_cv_no_tuning, holdout_evaluate
    from src.ml.sklearn_models import get_estimator

    # Train GNN without pretraining
    trainer = train_gnn_with_pretrain(
        smiles_train, y_train,
        tabular_train=None,
        pretrain_smiles=None,
        pretrain_tg=None,
        device=device,
        finetune_epochs=finetune_epochs,
    )

    # Extract embeddings
    print("  提取训练集嵌入...")
    emb_train, valid_train = extract_gnn_embeddings(trainer, smiles_train, device)
    print("  提取测试集嵌入...")
    emb_test, valid_test = extract_gnn_embeddings(trainer, smiles_test, device)

    y_train_valid = y_train[valid_train]
    y_test_valid = y_test[valid_test]

    print(f"  训练嵌入: {emb_train.shape}, 测试嵌入: {emb_test.shape}")

    # CatBoost on GNN embeddings
    estimator = get_estimator("CatBoost")
    cv_result = nested_cv_no_tuning(emb_train, y_train_valid, estimator)
    holdout_result = holdout_evaluate(
        emb_train, y_train_valid, emb_test, y_test_valid, estimator
    )

    result = {
        "experiment": "E25",
        "model": "CatBoost",
        "input": "GNN_64d",
        "pretrain": False,
        "n_features": GNN_EMBEDDING_DIM,
        "n_train": len(y_train_valid),
        "n_test": len(y_test_valid),
        "cv": cv_result,
        "holdout": holdout_result,
    }
    _save_result(result, "exp_E25_gnn_baseline.json")
    return result


def run_e26_gnn_pretrain(
    smiles_train: List[str], y_train: np.ndarray,
    smiles_test: List[str], y_test: np.ndarray,
    pretrain_smiles: List[str], pretrain_tg: List[float],
    device: str = "cuda",
    pretrain_epochs: int = PRETRAIN_EPOCHS,
    finetune_epochs: int = FINETUNE_EPOCHS,
) -> Dict[str, Any]:
    """E26: GNN 64d embedding → CatBoost (pretrain ~59K)."""
    print(f"\n{'='*60}")
    print("  E26: GNN 64d 嵌入 → CatBoost (预训练 ~59K)")
    print(f"{'='*60}")

    from src.ml.evaluation import nested_cv_no_tuning, holdout_evaluate
    from src.ml.sklearn_models import get_estimator

    # Train GNN with pretraining
    trainer = train_gnn_with_pretrain(
        smiles_train, y_train,
        tabular_train=None,
        pretrain_smiles=pretrain_smiles,
        pretrain_tg=pretrain_tg,
        device=device,
        pretrain_epochs=pretrain_epochs,
        finetune_epochs=finetune_epochs,
    )

    # Extract embeddings
    print("  提取训练集嵌入...")
    emb_train, valid_train = extract_gnn_embeddings(trainer, smiles_train, device)
    print("  提取测试集嵌入...")
    emb_test, valid_test = extract_gnn_embeddings(trainer, smiles_test, device)

    y_train_valid = y_train[valid_train]
    y_test_valid = y_test[valid_test]

    print(f"  训练嵌入: {emb_train.shape}, 测试嵌入: {emb_test.shape}")

    # CatBoost on GNN embeddings
    estimator = get_estimator("CatBoost")
    cv_result = nested_cv_no_tuning(emb_train, y_train_valid, estimator)
    holdout_result = holdout_evaluate(
        emb_train, y_train_valid, emb_test, y_test_valid, estimator
    )

    result = {
        "experiment": "E26",
        "model": "CatBoost",
        "input": "GNN_64d",
        "pretrain": True,
        "pretrain_size": len(pretrain_smiles),
        "n_features": GNN_EMBEDDING_DIM,
        "n_train": len(y_train_valid),
        "n_test": len(y_test_valid),
        "cv": cv_result,
        "holdout": holdout_result,
    }
    _save_result(result, "exp_E26_gnn_pretrain.json")
    return result


def run_e27_fusion_catboost(
    smiles_train: List[str], y_train: np.ndarray,
    smiles_test: List[str], y_test: np.ndarray,
    pretrain_smiles: List[str], pretrain_tg: List[float],
    device: str = "cuda",
    pretrain_epochs: int = PRETRAIN_EPOCHS,
    finetune_epochs: int = FINETUNE_EPOCHS,
) -> Dict[str, Any]:
    """E27: M2M-V 46d + GNN 64d = 110d → CatBoost."""
    print(f"\n{'='*60}")
    print("  E27: M2M-V 46d + GNN 64d = 110d → CatBoost")
    print(f"{'='*60}")

    from src.ml.evaluation import nested_cv_no_tuning, holdout_evaluate
    from src.ml.sklearn_models import get_estimator, build_preprocessing

    # Compute tabular features
    print("  计算表格特征 (M2M-V 46d)...")
    tab_train, feat_names, tab_valid_train = load_tabular_features(smiles_train)
    tab_test, _, tab_valid_test = load_tabular_features(smiles_test, verbose=False)

    # Train GNN with pretraining + tabular
    trainer = train_gnn_with_pretrain(
        smiles_train, y_train,
        tabular_train=tab_train,
        pretrain_smiles=pretrain_smiles,
        pretrain_tg=pretrain_tg,
        device=device,
        pretrain_epochs=pretrain_epochs,
        finetune_epochs=finetune_epochs,
        tab_valid_indices=tab_valid_train,
    )

    # Extract GNN embeddings
    print("  提取 GNN 嵌入...")
    emb_train, gnn_valid_train = extract_gnn_embeddings(
        trainer, smiles_train, device
    )
    emb_test, gnn_valid_test = extract_gnn_embeddings(
        trainer, smiles_test, device
    )

    # Fuse: 46d + 64d = 110d
    X_train_fused, y_train_fused, _ = fuse_features(
        tab_train, emb_train, tab_valid_train, gnn_valid_train, y_train
    )
    X_test_fused, y_test_fused, _ = fuse_features(
        tab_test, emb_test, tab_valid_test, gnn_valid_test, y_test
    )

    print(f"  融合特征: 训练 {X_train_fused.shape}, 测试 {X_test_fused.shape}")

    # Preprocess tabular part only (first 46d)
    pp = build_preprocessing()
    X_train_fused[:, :TABULAR_DIM] = pp.fit_transform(
        X_train_fused[:, :TABULAR_DIM]
    )
    X_test_fused[:, :TABULAR_DIM] = pp.transform(
        X_test_fused[:, :TABULAR_DIM]
    )

    # CatBoost on fused features
    estimator = get_estimator("CatBoost")
    cv_result = nested_cv_no_tuning(X_train_fused, y_train_fused, estimator)
    holdout_result = holdout_evaluate(
        X_train_fused, y_train_fused, X_test_fused, y_test_fused, estimator
    )

    result = {
        "experiment": "E27",
        "model": "CatBoost",
        "input": "M2M-V_46d + GNN_64d = 110d",
        "pretrain": True,
        "pretrain_size": len(pretrain_smiles),
        "n_features": FUSED_DIM,
        "n_train": len(y_train_fused),
        "n_test": len(y_test_fused),
        "cv": cv_result,
        "holdout": holdout_result,
    }
    _save_result(result, "exp_E27_fusion_catboost.json")
    return result


def run_e28_fusion_stacking(
    smiles_train: List[str], y_train: np.ndarray,
    smiles_test: List[str], y_test: np.ndarray,
    pretrain_smiles: List[str], pretrain_tg: List[float],
    device: str = "cuda",
    pretrain_epochs: int = PRETRAIN_EPOCHS,
    finetune_epochs: int = FINETUNE_EPOCHS,
) -> Dict[str, Any]:
    """E28: 110d → Stacking v2 (with TabPFN v2) — CORE EXPERIMENT."""
    print(f"\n{'='*60}")
    print("  E28: 110d → Stacking v2 (含 TabPFN v2) — 核心实验")
    print(f"{'='*60}")

    from src.ml.evaluation import simple_cv, holdout_evaluate
    from src.ml.sklearn_models import build_preprocessing

    # Compute tabular features
    print("  计算表格特征 (M2M-V 46d)...")
    tab_train, feat_names, tab_valid_train = load_tabular_features(smiles_train)
    tab_test, _, tab_valid_test = load_tabular_features(smiles_test, verbose=False)

    # Train GNN with pretraining + tabular
    trainer = train_gnn_with_pretrain(
        smiles_train, y_train,
        tabular_train=tab_train,
        pretrain_smiles=pretrain_smiles,
        pretrain_tg=pretrain_tg,
        device=device,
        pretrain_epochs=pretrain_epochs,
        finetune_epochs=finetune_epochs,
        tab_valid_indices=tab_valid_train,
    )

    # Extract GNN embeddings
    print("  提取 GNN 嵌入...")
    emb_train, gnn_valid_train = extract_gnn_embeddings(
        trainer, smiles_train, device
    )
    emb_test, gnn_valid_test = extract_gnn_embeddings(
        trainer, smiles_test, device
    )

    # Fuse: 46d + 64d = 110d
    X_train_fused, y_train_fused, _ = fuse_features(
        tab_train, emb_train, tab_valid_train, gnn_valid_train, y_train
    )
    X_test_fused, y_test_fused, _ = fuse_features(
        tab_test, emb_test, tab_valid_test, gnn_valid_test, y_test
    )

    print(f"  融合特征: 训练 {X_train_fused.shape}, 测试 {X_test_fused.shape}")

    # Preprocess tabular part only
    pp = build_preprocessing()
    X_train_fused[:, :TABULAR_DIM] = pp.fit_transform(
        X_train_fused[:, :TABULAR_DIM]
    )
    X_test_fused[:, :TABULAR_DIM] = pp.transform(
        X_test_fused[:, :TABULAR_DIM]
    )

    # Stacking per 方案B.2: CatBoost + TabPFN v2 + LightGBM + ExtraTrees → Ridge
    stacking, base_model_names = _build_stacking_b2()
    cv_result = simple_cv(
        X_train_fused, y_train_fused, stacking,
        n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS,
    )
    holdout_result = holdout_evaluate(
        X_train_fused, y_train_fused, X_test_fused, y_test_fused, stacking
    )

    result = {
        "experiment": "E28",
        "model": "Stacking_v2",
        "base_models": base_model_names,
        "meta_learner": "Ridge",
        "input": "M2M-V_46d + GNN_64d = 110d",
        "pretrain": True,
        "pretrain_size": len(pretrain_smiles),
        "n_features": FUSED_DIM,
        "n_train": len(y_train_fused),
        "n_test": len(y_test_fused),
        "cv": cv_result,
        "holdout": holdout_result,
    }
    _save_result(result, "exp_E28_fusion_stacking.json")
    return result


def run_e29_vpd_deep(
    smiles_train: List[str], y_train: np.ndarray,
    smiles_test: List[str], y_test: np.ndarray,
    pretrain_smiles: List[str], pretrain_tg: List[float],
    device: str = "cuda",
    pretrain_epochs: int = PRETRAIN_EPOCHS,
    finetune_epochs: int = FINETUNE_EPOCHS,
) -> Dict[str, Any]:
    """E29: VPD-Deep 64d replaces VPD 12d → Stacking v2.

    VPD-Deep: GNN(trimer) embedding as replacement for RDKit VPD descriptors.
    This is essentially E28 but the GNN embedding IS the VPD replacement.
    """
    print(f"\n{'='*60}")
    print("  E29: VPD-Deep (GNN 嵌入替换 VPD 12d) → Stacking v2")
    print(f"{'='*60}")

    from src.ml.evaluation import simple_cv, holdout_evaluate
    from src.ml.sklearn_models import build_preprocessing

    # Compute M2M-V features WITHOUT VPD (use M2M-P = 34d without VPD)
    # M2M-V = afsordeh(4) + rdkit(15) + hbond(15) + vpd(12) = 46d
    # M2M-P (no VPD) = afsordeh(4) + rdkit(15) + hbond(15) = 34d
    # VPD-Deep replaces VPD 12d with GNN 64d: 34d + 64d = 98d
    print("  计算表格特征 (不含 VPD)...")
    tab_train_no_vpd, _, tab_valid_train = load_tabular_features(
        smiles_train, layer="L1H", verbose=True
    )
    tab_test_no_vpd, _, tab_valid_test = load_tabular_features(
        smiles_test, layer="L1H", verbose=False
    )

    l1h_dim = tab_train_no_vpd.shape[1]

    # Train GNN with pretraining
    trainer = train_gnn_with_pretrain(
        smiles_train, y_train,
        tabular_train=None,
        pretrain_smiles=pretrain_smiles,
        pretrain_tg=pretrain_tg,
        device=device,
        pretrain_epochs=pretrain_epochs,
        finetune_epochs=finetune_epochs,
    )

    # Extract GNN embeddings (VPD-Deep: 64d replaces 12d VPD)
    print("  提取 VPD-Deep 嵌入...")
    emb_train, gnn_valid_train = extract_gnn_embeddings(
        trainer, smiles_train, device
    )
    emb_test, gnn_valid_test = extract_gnn_embeddings(
        trainer, smiles_test, device
    )

    # Fuse: L1H + GNN = (34 + 64)d = 98d
    X_train_fused, y_train_fused, _ = fuse_features(
        tab_train_no_vpd, emb_train, tab_valid_train, gnn_valid_train, y_train
    )
    X_test_fused, y_test_fused, _ = fuse_features(
        tab_test_no_vpd, emb_test, tab_valid_test, gnn_valid_test, y_test
    )

    vpd_deep_dim = l1h_dim + GNN_EMBEDDING_DIM
    print(f"  VPD-Deep 融合: 训练 {X_train_fused.shape}, "
          f"测试 {X_test_fused.shape} ({l1h_dim}d + {GNN_EMBEDDING_DIM}d)")

    # Preprocess tabular part only
    pp = build_preprocessing()
    X_train_fused[:, :l1h_dim] = pp.fit_transform(X_train_fused[:, :l1h_dim])
    X_test_fused[:, :l1h_dim] = pp.transform(X_test_fused[:, :l1h_dim])

    # Stacking per 方案B.2
    stacking, _ = _build_stacking_b2()
    cv_result = simple_cv(
        X_train_fused, y_train_fused, stacking,
        n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS,
    )
    holdout_result = holdout_evaluate(
        X_train_fused, y_train_fused, X_test_fused, y_test_fused, stacking
    )

    result = {
        "experiment": "E29",
        "model": "Stacking_v2",
        "input": f"L1H_{l1h_dim}d + VPD-Deep_{GNN_EMBEDDING_DIM}d = {vpd_deep_dim}d",
        "pretrain": True,
        "pretrain_size": len(pretrain_smiles),
        "n_features": vpd_deep_dim,
        "n_train": len(y_train_fused),
        "n_test": len(y_test_fused),
        "cv": cv_result,
        "holdout": holdout_result,
    }
    _save_result(result, "exp_E29_vpd_deep.json")
    return result


def run_e30_multitask(
    smiles_train: List[str], y_train: np.ndarray,
    smiles_test: List[str], y_test: np.ndarray,
    pretrain_smiles: List[str], pretrain_tg: List[float],
    device: str = "cuda",
    pretrain_epochs: int = PRETRAIN_EPOCHS,
    finetune_epochs: int = FINETUNE_EPOCHS,
) -> Dict[str, Any]:
    """E30: E28 + MultiTask (Tg+density+SOL).

    Uses MultiTaskTgModel for pretraining with auxiliary property targets,
    then extracts shared backbone embeddings for Stacking fusion.
    """
    print(f"\n{'='*60}")
    print("  E30: E28 + MultiTask (Tg+密度+SOL)")
    print(f"{'='*60}")

    import torch
    from torch_geometric.loader import DataLoader
    from src.gnn.graph_builder import batch_smiles_to_graphs
    from src.gnn.multitask import MultiTaskTgModel
    from src.gnn.tandem_m2m import TandemM2M
    from src.gnn.pretrainer import TgPretrainer
    from src.ml.evaluation import simple_cv, holdout_evaluate
    from src.ml.sklearn_models import build_preprocessing

    # Compute tabular features
    print("  计算表格特征 (M2M-V 46d)...")
    tab_train, feat_names, tab_valid_train = load_tabular_features(smiles_train)
    tab_test, _, tab_valid_test = load_tabular_features(smiles_test, verbose=False)

    # Stage 1: Pretrain with MultiTask model
    # The multi-task model shares the same PhysicsGAT backbone
    # After pretraining, we transfer the GNN backbone weights to TandemM2M
    print("  Stage 1: MultiTask 预训练...")
    mt_model = MultiTaskTgModel(
        in_dim=25, gnn_hidden=128, gnn_out=GNN_EMBEDDING_DIM,
        heads=4, dropout=0.1, edge_dim=6,
        aux_task_names=["density", "sol_param"],
        lambda_aux=0.1,
    ).to(device)

    # Build pretrain graphs
    if pretrain_smiles and len(pretrain_smiles) > 0:
        print(f"  构建预训练图 ({len(pretrain_smiles)} 条)...")
        pretrain_graphs, _ = batch_smiles_to_graphs(
            pretrain_smiles, y_list=pretrain_tg,
        )
        pretrain_loader = DataLoader(
            pretrain_graphs, batch_size=BATCH_SIZE_PRETRAIN, shuffle=True,
        )

        # Simple pretraining loop (Tg only, aux tasks need property labels)
        optimizer = torch.optim.Adam(mt_model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()

        mt_model.train()
        for epoch in range(1, pretrain_epochs + 1):
            total_loss = 0.0
            n = 0
            for batch in pretrain_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                preds = mt_model(batch)
                loss = criterion(preds["tg"].squeeze(), batch.y.squeeze())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mt_model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * batch.num_graphs
                n += batch.num_graphs

            if epoch % 20 == 0:
                print(f"    [MultiTask Pretrain] Epoch {epoch}/{pretrain_epochs}: "
                      f"loss={total_loss/max(n,1):.4f}")

    # Transfer backbone weights to TandemM2M for finetuning
    print("  权重迁移: MultiTask → TandemM2M...")
    tandem = TandemM2M(
        in_dim=25, tabular_dim=TABULAR_DIM,
        gnn_hidden=128, gnn_out=GNN_EMBEDDING_DIM,
        dropout=0.1, edge_dim=6,
    )
    # Copy shared GNN backbone weights
    tandem.gnn.load_state_dict(mt_model.gnn.state_dict())

    trainer = TgPretrainer(tandem, device=device, tabular_dim=TABULAR_DIM)

    # Finetune on target dataset
    finetune_graphs, valid_idx = batch_smiles_to_graphs(
        smiles_train, y_list=y_train.tolist(),
    )
    if tab_train is not None:
        tab_idx_map = {orig: pos for pos, orig in enumerate(tab_valid_train)}
        for i, g in enumerate(finetune_graphs):
            orig_i = valid_idx[i]
            if orig_i in tab_idx_map:
                g.tabular = torch.tensor(
                    tab_train[tab_idx_map[orig_i]], dtype=torch.float
                ).unsqueeze(0)
            else:
                g.tabular = torch.zeros(1, TABULAR_DIM, dtype=torch.float)
    finetune_loader = DataLoader(
        finetune_graphs, batch_size=BATCH_SIZE_FINETUNE, shuffle=True,
    )

    print(f"  Stage 2: 微调 ({finetune_epochs} epochs)...")
    trainer.finetune(finetune_loader, epochs=finetune_epochs, patience=PATIENCE)

    # Extract embeddings
    print("  提取 GNN 嵌入...")
    emb_train, gnn_valid_train = extract_gnn_embeddings(
        trainer, smiles_train, device
    )
    emb_test, gnn_valid_test = extract_gnn_embeddings(
        trainer, smiles_test, device
    )

    # Fuse: 46d + 64d = 110d
    X_train_fused, y_train_fused, _ = fuse_features(
        tab_train, emb_train, tab_valid_train, gnn_valid_train, y_train
    )
    X_test_fused, y_test_fused, _ = fuse_features(
        tab_test, emb_test, tab_valid_test, gnn_valid_test, y_test
    )

    print(f"  融合特征: 训练 {X_train_fused.shape}, 测试 {X_test_fused.shape}")

    # Preprocess tabular part
    pp = build_preprocessing()
    X_train_fused[:, :TABULAR_DIM] = pp.fit_transform(
        X_train_fused[:, :TABULAR_DIM]
    )
    X_test_fused[:, :TABULAR_DIM] = pp.transform(
        X_test_fused[:, :TABULAR_DIM]
    )

    # Stacking per 方案B.2
    stacking, base_names = _build_stacking_b2()
    cv_result = simple_cv(
        X_train_fused, y_train_fused, stacking,
        n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS,
    )
    holdout_result = holdout_evaluate(
        X_train_fused, y_train_fused, X_test_fused, y_test_fused, stacking
    )

    result = {
        "experiment": "E30",
        "model": f"Stacking_B2({'+'.join(base_names)}) + MultiTask_pretrain",
        "input": "M2M-V_46d + GNN_64d = 110d",
        "pretrain": True,
        "pretrain_type": "multitask",
        "aux_tasks": ["density", "sol_param"],
        "pretrain_size": len(pretrain_smiles) if pretrain_smiles else 0,
        "n_features": FUSED_DIM,
        "n_train": len(y_train_fused),
        "n_test": len(y_test_fused),
        "cv": cv_result,
        "holdout": holdout_result,
    }
    _save_result(result, "exp_E30_multitask.json")
    return result


def run_e31_deep_ensemble(
    smiles_train: List[str], y_train: np.ndarray,
    smiles_test: List[str], y_test: np.ndarray,
    pretrain_smiles: List[str], pretrain_tg: List[float],
    device: str = "cuda",
    pretrain_epochs: int = PRETRAIN_EPOCHS,
    finetune_epochs: int = FINETUNE_EPOCHS,
) -> Dict[str, Any]:
    """E31: Deep Ensemble ×5 + Conformal (E28 config)."""
    print(f"\n{'='*60}")
    print("  E31: Deep Ensemble ×5 + Conformal Prediction")
    print(f"{'='*60}")

    from src.ml.evaluation import holdout_evaluate
    from src.ml.sklearn_models import build_preprocessing
    from src.gnn.ensemble import DeepEnsembleTg
    from src.gnn.tandem_m2m import TandemM2M
    from src.gnn.graph_builder import batch_smiles_to_graphs
    import torch
    from torch_geometric.loader import DataLoader

    # Compute tabular features
    print("  计算表格特征 (M2M-V 46d)...")
    tab_train, _, tab_valid_train = load_tabular_features(smiles_train)
    tab_test, _, tab_valid_test = load_tabular_features(smiles_test, verbose=False)

    # Build graphs
    print("  构建微调图...")
    finetune_graphs, ft_valid = batch_smiles_to_graphs(
        smiles_train, y_list=y_train.tolist(),
    )
    if tab_train is not None:
        tab_idx_map = {orig: pos for pos, orig in enumerate(tab_valid_train)}
        for i, g in enumerate(finetune_graphs):
            orig_i = ft_valid[i]
            if orig_i in tab_idx_map:
                g.tabular = torch.tensor(
                    tab_train[tab_idx_map[orig_i]], dtype=torch.float
                ).unsqueeze(0)
            else:
                g.tabular = torch.zeros(1, TABULAR_DIM, dtype=torch.float)

    finetune_loader = DataLoader(
        finetune_graphs, batch_size=BATCH_SIZE_FINETUNE, shuffle=True,
    )

    # Build pretrain graphs
    pretrain_loader = None
    if pretrain_smiles and len(pretrain_smiles) > 0:
        print(f"  构建预训练图 ({len(pretrain_smiles)} 条)...")
        pretrain_graphs, _ = batch_smiles_to_graphs(
            pretrain_smiles, y_list=pretrain_tg,
        )
        pretrain_loader = DataLoader(
            pretrain_graphs, batch_size=BATCH_SIZE_PRETRAIN, shuffle=True,
        )

    # Create Deep Ensemble (5 models)
    def model_factory():
        return TandemM2M(
            in_dim=25, tabular_dim=TABULAR_DIM,
            gnn_hidden=128, gnn_out=GNN_EMBEDDING_DIM,
            dropout=0.1, edge_dim=6,
        )

    ensemble = DeepEnsembleTg(
        model_fn=model_factory,
        n_models=5,
        device=device,
        tabular_dim=TABULAR_DIM,
    )

    # Train all 5 models
    print("  训练 5 个 Ensemble 模型...")
    ensemble.fit(
        pretrain_loader=pretrain_loader,
        finetune_loader=finetune_loader,
        pretrain_epochs=pretrain_epochs,
        finetune_epochs=finetune_epochs,
        patience=PATIENCE,
    )

    # Calibrate conformal prediction
    print("  共形校准...")
    # Use last 20% of finetune data as calibration set
    n_cal = max(1, len(finetune_graphs) // 5)
    cal_graphs = finetune_graphs[-n_cal:]
    cal_loader = DataLoader(cal_graphs, batch_size=BATCH_SIZE_FINETUNE, shuffle=False)
    ensemble.calibrate(cal_loader, confidence=0.9)

    # Evaluate on test set
    print("  测试集评估...")
    test_graphs, test_valid = batch_smiles_to_graphs(
        smiles_test, y_list=y_test.tolist(),
    )
    if tab_test is not None:
        for i, g in enumerate(test_graphs):
            orig_i = test_valid[i]
            g.tabular = torch.tensor(
                tab_test[orig_i], dtype=torch.float
            ).unsqueeze(0)

    test_loader = DataLoader(
        test_graphs, batch_size=BATCH_SIZE_FINETUNE, shuffle=False
    )

    # Collect predictions
    all_preds = []
    all_true = []
    all_stds = []
    all_lower = []
    all_upper = []

    for batch in test_loader:
        batch = batch.to(device)
        tab = batch.tabular if hasattr(batch, "tabular") else torch.zeros(
            batch.num_graphs, TABULAR_DIM, device=device
        )
        mean_pred, std_pred, _ = ensemble.predict(batch, tab)
        pred_mean, lower, upper = ensemble.predict_interval(batch, tab)

        all_preds.extend(mean_pred.tolist())
        all_true.extend(batch.y.squeeze().cpu().numpy().tolist())
        all_stds.extend(std_pred.tolist())
        all_lower.extend(lower.tolist())
        all_upper.extend(upper.tolist())

    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    y_pred = np.array(all_preds)
    y_true = np.array(all_true)
    y_std = np.array(all_stds)
    y_lower = np.array(all_lower)
    y_upper = np.array(all_upper)

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Coverage
    in_interval = (y_true >= y_lower) & (y_true <= y_upper)
    coverage = np.mean(in_interval)
    avg_width = np.mean(y_upper - y_lower)

    print(f"  结果: R²={r2:.4f}, MAE={mae:.1f}K, RMSE={rmse:.1f}K")
    print(f"  UQ: 覆盖率={coverage:.3f}, 平均区间宽度={avg_width:.1f}K")
    print(f"  Epistemic 不确定性: 均值={y_std.mean():.1f}K, "
          f"最大={y_std.max():.1f}K")

    result = {
        "experiment": "E31",
        "model": "Deep_Ensemble_x5",
        "input": "M2M-V_46d + GNN_64d",
        "pretrain": True,
        "pretrain_size": len(pretrain_smiles) if pretrain_smiles else 0,
        "n_models": 5,
        "n_test": len(y_true),
        "point_metrics": {
            "R2": float(r2),
            "MAE": float(mae),
            "RMSE": float(rmse),
        },
        "uncertainty": {
            "coverage_90": float(coverage),
            "avg_interval_width": float(avg_width),
            "mean_epistemic_std": float(y_std.mean()),
            "max_epistemic_std": float(y_std.max()),
            "conformal_score": float(ensemble.conformal_scores)
                if ensemble.conformal_scores is not None else None,
        },
    }
    _save_result(result, "exp_E31_deep_ensemble.json")
    return result


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _save_result(result: Dict, filename: str):
    """Save result to JSON file."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"  结果已保存: {path}")


def generate_summary(all_results: Dict[str, Any]) -> str:
    """Generate markdown summary of all GNN experiment results."""
    lines = [
        "# Phase 5 GNN 实验结果汇总 (E25-E31)",
        "",
        f"特征层级: {FEATURE_LAYER} + GNN {GNN_EMBEDDING_DIM}d",
        f"数据集: unified_tg.parquet",
        "",
        "## 模型对比",
        "",
        "| 实验 | 配置 | 输入维度 | CV R² | Holdout R² | Holdout MAE (K) |",
        "|------|------|---------|-------|-----------|----------------|",
    ]

    for exp_id in ["E25", "E26", "E27", "E28", "E29", "E30"]:
        r = all_results.get(exp_id)
        if r is None:
            continue

        model = r.get("model", "?")
        n_feat = r.get("n_features", "?")

        cv = r.get("cv", {})
        cv_metrics = cv.get("metrics", cv)
        cv_r2 = cv_metrics.get("R2_mean", float("nan"))

        ho = r.get("holdout", {})
        ho_test = ho.get("test_metrics", {})
        ho_r2 = ho_test.get("R2", float("nan"))
        ho_mae = ho_test.get("MAE", float("nan"))

        lines.append(
            f"| {exp_id} | {model} | {n_feat} | {cv_r2:.4f} | "
            f"{ho_r2:.4f} | {ho_mae:.1f} |"
        )

    # E31 Deep Ensemble
    if "E31" in all_results:
        r = all_results["E31"]
        pm = r.get("point_metrics", {})
        uq = r.get("uncertainty", {})
        lines.extend([
            "",
            "## Deep Ensemble + Conformal (E31)",
            "",
            f"- R²: {pm.get('R2', 0):.4f}",
            f"- MAE: {pm.get('MAE', 0):.1f} K",
            f"- RMSE: {pm.get('RMSE', 0):.1f} K",
            f"- 覆盖率 (90%): {uq.get('coverage_90', 0):.3f}",
            f"- 平均区间宽度: {uq.get('avg_interval_width', 0):.1f} K",
            f"- 平均 Epistemic 不确定性: {uq.get('mean_epistemic_std', 0):.1f} K",
        ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 5 GNN experiments (E25-E31)"
    )
    parser.add_argument(
        "--exp", nargs="*", default=None,
        help="Experiment IDs to run (e.g., E25 E26). Default: all."
    )
    parser.add_argument(
        "--layer3", action="store_true",
        help="Enable Kuenneth copolymer data for pretraining (default: off)."
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Compute device: cuda or cpu (default: cuda)."
    )
    parser.add_argument(
        "--pretrain-epochs", type=int, default=PRETRAIN_EPOCHS,
        help=f"Pretraining epochs (default: {PRETRAIN_EPOCHS})."
    )
    parser.add_argument(
        "--finetune-epochs", type=int, default=FINETUNE_EPOCHS,
        help=f"Finetuning epochs (default: {FINETUNE_EPOCHS})."
    )
    parser.add_argument(
        "--skip-pretrain-data", action="store_true",
        help="Skip building pretrain data (for E25 only)."
    )
    args = parser.parse_args()

    # Validate environment
    if not _check_torch():
        sys.exit(1)

    device = _get_device(args.device)

    # Determine experiments to run
    all_exp_ids = ["E25", "E26", "E27", "E28", "E29", "E30", "E31"]
    if args.exp:
        run_ids = [e.upper() for e in args.exp]
        invalid = [e for e in run_ids if e not in all_exp_ids]
        if invalid:
            print(f"未知实验: {invalid}. 可用: {all_exp_ids}")
            sys.exit(1)
    else:
        run_ids = all_exp_ids

    print(f"\n{'#'*60}")
    print(f"  Phase 5 GNN: Tandem-M2M 嵌入融合实验")
    print(f"  实验: {run_ids}")
    print(f"  设备: {device}")
    print(f"  预训练: {args.pretrain_epochs} epochs")
    print(f"  微调: {args.finetune_epochs} epochs")
    print(f"  Layer 3 (Kuenneth): {'启用' if args.layer3 else '关闭'}")
    print(f"{'#'*60}")

    warnings.filterwarnings("ignore", message=".*deprecated.*", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*PYTORCH_CUDA_ALLOC_CONF.*")

    t_start = time.time()

    # Load unified dataset
    smiles_train, y_train, smiles_test, y_test = load_unified_data()

    # Build pretrain data (shared across E26-E31)
    pretrain_smiles, pretrain_tg = None, None
    needs_pretrain = any(e in run_ids for e in ["E26", "E27", "E28", "E29", "E30", "E31"])

    if needs_pretrain and not args.skip_pretrain_data:
        # Anti-leak: exclude unified dataset SMILES
        exclude_set = set(smiles_train + smiles_test)
        pretrain_smiles, pretrain_tg = build_pretrain_data(
            include_layer3=args.layer3,
            exclude_smiles=exclude_set,
        )

    all_results: Dict[str, Any] = {}

    # --- Run experiments ---
    for exp_id in run_ids:
        try:
            if exp_id == "E25":
                result = run_e25_gnn_baseline(
                    smiles_train, y_train, smiles_test, y_test,
                    device=device,
                    finetune_epochs=args.finetune_epochs,
                )
            elif exp_id == "E26":
                result = run_e26_gnn_pretrain(
                    smiles_train, y_train, smiles_test, y_test,
                    pretrain_smiles, pretrain_tg,
                    device=device,
                    pretrain_epochs=args.pretrain_epochs,
                    finetune_epochs=args.finetune_epochs,
                )
            elif exp_id == "E27":
                result = run_e27_fusion_catboost(
                    smiles_train, y_train, smiles_test, y_test,
                    pretrain_smiles, pretrain_tg,
                    device=device,
                    pretrain_epochs=args.pretrain_epochs,
                    finetune_epochs=args.finetune_epochs,
                )
            elif exp_id == "E28":
                result = run_e28_fusion_stacking(
                    smiles_train, y_train, smiles_test, y_test,
                    pretrain_smiles, pretrain_tg,
                    device=device,
                    pretrain_epochs=args.pretrain_epochs,
                    finetune_epochs=args.finetune_epochs,
                )
            elif exp_id == "E29":
                result = run_e29_vpd_deep(
                    smiles_train, y_train, smiles_test, y_test,
                    pretrain_smiles, pretrain_tg,
                    device=device,
                    pretrain_epochs=args.pretrain_epochs,
                    finetune_epochs=args.finetune_epochs,
                )
            elif exp_id == "E30":
                result = run_e30_multitask(
                    smiles_train, y_train, smiles_test, y_test,
                    pretrain_smiles, pretrain_tg,
                    device=device,
                    pretrain_epochs=args.pretrain_epochs,
                    finetune_epochs=args.finetune_epochs,
                )
            elif exp_id == "E31":
                result = run_e31_deep_ensemble(
                    smiles_train, y_train, smiles_test, y_test,
                    pretrain_smiles, pretrain_tg,
                    device=device,
                    pretrain_epochs=args.pretrain_epochs,
                    finetune_epochs=args.finetune_epochs,
                )
            else:
                continue

            if result is not None:
                all_results[exp_id] = result

        except Exception as e:
            print(f"\n  [X] {exp_id} 失败: {e}")
            import traceback
            traceback.print_exc()

    # --- Generate summary ---
    t_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  全部 GNN 实验完成! 总耗时: {t_total:.0f}s ({t_total/60:.1f}min)")
    print(f"{'='*60}")

    if all_results:
        summary = generate_summary(all_results)
        summary_path = RESULT_DIR / "phase5_gnn_summary.md"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"\n  汇总已保存: {summary_path}")
        # Safe print for Windows GBK terminals
        print(summary.encode("gbk", errors="replace").decode("gbk"))


if __name__ == "__main__":
    main()
