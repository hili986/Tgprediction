"""
End-to-end nucleic acid sequence -> Tg prediction.
核酸序列 → 玻璃化转变温度 (Tg) 端到端预测

Usage:
    # Interactive mode
    python scripts/predict_tg_from_sequence.py

    # Direct prediction
    python scripts/predict_tg_from_sequence.py --seq ACGT --type DNA

    # Predict single nucleotides
    python scripts/predict_tg_from_sequence.py --seq A
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sequence.nucleotide_smiles import (
    validate_sequence,
    build_full_smiles,
    get_monomer_smiles,
    sequence_to_smiles,
)
from src.features.feature_pipeline import compute_features, build_dataset_v2
from src.features.hbond_features import compute_hbond_features
from src.data.bridge_polymers import build_bridge_dataset


# ---------------------------------------------------------------------------
# Model training (on-the-fly, ~10s)
# ---------------------------------------------------------------------------

def _make_scaler():
    return Pipeline([
        ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
        ("scale", MinMaxScaler()),
    ])


def train_model(bridge_weight=0.8, verbose=True):
    """Train GBR model with bridge transfer for nucleic acid prediction.

    Returns:
        (model, scaler, feature_count)
    """
    if verbose:
        print("[1/3] 加载实验数据 (Bicerano 304 + H-bond features)...")
    X_exp, y_exp, _, feat_names, _ = build_dataset_v2(layer="L1H", verbose=verbose)

    if verbose:
        print("[2/3] 加载桥梁聚合物数据 (205 entries)...")
    X_bridge, y_bridge, _, _ = build_bridge_dataset(
        layer="L1", include_hbond=True, verbose=verbose,
    )

    if verbose:
        print(f"[3/3] 训练 GBR 模型 (bridge_weight={bridge_weight})...")

    # Combine data
    X_all = np.vstack([X_exp, X_bridge])
    y_all = np.concatenate([y_exp, y_bridge])
    w_all = np.concatenate([
        np.ones(len(y_exp)),
        np.full(len(y_bridge), bridge_weight),
    ])

    # Preprocess
    scaler = _make_scaler()
    scaler.fit(X_all)
    X_scaled = scaler.transform(X_all)

    # Train
    model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        min_samples_leaf=5, subsample=0.8, random_state=42,
    )
    model.fit(X_scaled, y_all, sample_weight=w_all)

    if verbose:
        print(f"  模型训练完成! 特征维度: {X_all.shape[1]}")

    return model, scaler, X_all.shape[1]


# ---------------------------------------------------------------------------
# Feature extraction for nucleic acid SMILES
# ---------------------------------------------------------------------------

def extract_features(smiles, layer="L1H"):
    """Extract L1H features from a SMILES string.

    Returns:
        1D numpy array or None if extraction fails.
    """
    try:
        x = compute_features(smiles, None, layer)
        if np.any(np.isnan(x)):
            return None
        return x
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_single_smiles(smiles, model, scaler, name=""):
    """Predict Tg for a single SMILES.

    Returns:
        dict with prediction result, or None on failure.
    """
    features = extract_features(smiles)
    if features is None:
        return None

    x_scaled = scaler.transform(features.reshape(1, -1))
    tg_pred = model.predict(x_scaled)[0]

    return {
        "name": name,
        "smiles": smiles,
        "tg_predicted_K": round(float(tg_pred), 1),
        "tg_predicted_C": round(float(tg_pred) - 273.15, 1),
    }


def predict_from_sequence(seq, seq_type, model, scaler):
    """Predict Tg from a nucleic acid sequence.

    Strategy:
    - Length 1: use monomer SMILES (most reliable)
    - Length 2-4: use full concatenated SMILES (reasonable)
    - Length >4: use full SMILES but warn about reliability

    Returns:
        dict with prediction results.
    """
    info = sequence_to_smiles(seq, seq_type)
    length = info["length"]
    clean_seq = info["sequence"]

    result = {
        "input_sequence": seq,
        "clean_sequence": clean_seq,
        "seq_type": seq_type,
        "length": length,
        "predictions": [],
        "warnings": [],
    }

    # -- Full sequence SMILES prediction --
    full_smiles = info["full_smiles"]

    # Validate SMILES with RDKit
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(full_smiles)
        if mol is None:
            result["warnings"].append(f"RDKit 无法解析完整序列 SMILES")
        else:
            result["atom_count"] = mol.GetNumAtoms()
    except ImportError:
        pass

    full_pred = predict_single_smiles(full_smiles, model, scaler, name=f"{seq_type}:{clean_seq}")
    if full_pred:
        full_pred["method"] = "full_sequence"
        result["predictions"].append(full_pred)
    else:
        result["warnings"].append("完整序列特征提取失败")

    # -- Per-monomer predictions --
    monomer_tgs = []
    for mono in info["monomers"]:
        pred = predict_single_smiles(mono["smiles"], model, scaler, name=mono["base"])
        if pred:
            pred["method"] = "monomer"
            monomer_tgs.append(pred["tg_predicted_K"])
            result["predictions"].append(pred)

    # -- Composition-weighted average (simple mixing rule) --
    if monomer_tgs:
        avg_tg = np.mean(monomer_tgs)
        result["monomer_avg_tg_K"] = round(float(avg_tg), 1)
        result["monomer_avg_tg_C"] = round(float(avg_tg) - 273.15, 1)

    # -- Confidence/reliability warnings --
    if length == 1:
        result["confidence"] = "HIGH"
        result["note"] = "单核苷酸预测，与训练集化学空间一致"
    elif length <= 4:
        result["confidence"] = "MEDIUM"
        result["note"] = "短序列预测，分子大小接近训练集范围"
    else:
        result["confidence"] = "LOW"
        result["warnings"].append(
            f"长序列 ({length}nt) 远超训练集分子大小范围，预测可能不可靠"
        )
        result["note"] = "建议参考单体平均值而非完整序列预测"

    return result


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_result(result):
    """Pretty-print prediction result."""
    print(f"\n{'=' * 60}")
    print(f"  核酸序列 Tg 预测结果")
    print(f"{'=' * 60}")
    print(f"  序列: {result['clean_sequence']}")
    print(f"  类型: {result['seq_type']}")
    print(f"  长度: {result['length']} nt")
    print(f"  置信度: {result.get('confidence', 'N/A')}")

    if result.get("atom_count"):
        print(f"  原子数: {result['atom_count']}")

    print(f"\n  --- 预测结果 ---")

    # Show full sequence prediction
    full_preds = [p for p in result["predictions"] if p["method"] == "full_sequence"]
    if full_preds:
        p = full_preds[0]
        print(f"  完整序列: {p['tg_predicted_K']:.1f} K ({p['tg_predicted_C']:.1f} C)")

    # Show monomer predictions
    mono_preds = [p for p in result["predictions"] if p["method"] == "monomer"]
    if mono_preds:
        print(f"\n  单体预测:")
        for p in mono_preds:
            print(f"    {p['name']:5s}: {p['tg_predicted_K']:.1f} K ({p['tg_predicted_C']:.1f} C)")

    if result.get("monomer_avg_tg_K"):
        print(f"\n  单体平均: {result['monomer_avg_tg_K']:.1f} K ({result['monomer_avg_tg_C']:.1f} C)")

    if result.get("note"):
        print(f"\n  备注: {result['note']}")

    for w in result.get("warnings", []):
        print(f"  [!] {w}")

    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def interactive_mode(model, scaler):
    """Interactive prediction loop."""
    print("\n" + "=" * 60)
    print("  核酸序列 Tg 预测系统 (交互模式)")
    print("  输入核酸序列 (如 A, AT, ACGT, AUGC)")
    print("  输入 'q' 退出, 'rna' 切换到 RNA 模式, 'dna' 切换到 DNA 模式")
    print("=" * 60)

    seq_type = "DNA"

    while True:
        try:
            user_input = input(f"\n[{seq_type}] 输入序列> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not user_input:
            continue
        if user_input.lower() == "q":
            print("再见!")
            break
        if user_input.lower() == "rna":
            seq_type = "RNA"
            print(f"  已切换到 RNA 模式")
            continue
        if user_input.lower() == "dna":
            seq_type = "DNA"
            print(f"  已切换到 DNA 模式")
            continue

        try:
            result = predict_from_sequence(user_input, seq_type, model, scaler)
            print_result(result)
        except ValueError as e:
            print(f"  [ERROR] {e}")
        except Exception as e:
            print(f"  [ERROR] 预测失败: {e}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="核酸序列 -> Tg 预测",
    )
    parser.add_argument("--seq", type=str, default=None,
                        help="核酸序列 (如 ACGT)")
    parser.add_argument("--type", choices=["DNA", "RNA"], default="DNA",
                        help="序列类型 (default: DNA)")
    parser.add_argument("--bridge-weight", type=float, default=0.8,
                        help="桥梁数据权重 (default: 0.8)")
    parser.add_argument("--json", action="store_true",
                        help="JSON 格式输出")

    args = parser.parse_args()

    # Train model
    print("=" * 60)
    print("  初始化 Tg 预测模型...")
    print("=" * 60)
    t0 = time.time()
    model, scaler, n_features = train_model(
        bridge_weight=args.bridge_weight, verbose=True,
    )
    print(f"  模型就绪! 耗时 {time.time() - t0:.1f}s\n")

    if args.seq:
        # Direct prediction mode
        result = predict_from_sequence(args.seq, args.type, model, scaler)
        if args.json:
            # Remove non-serializable items
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print_result(result)
    else:
        # Interactive mode
        interactive_mode(model, scaler)


if __name__ == "__main__":
    main()
