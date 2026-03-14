"""
数据集质量分析脚本
分析 data/external/ 目录下所有数据集的质量、重叠、异常值等。
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)
from rdkit import Chem

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "external"


# ============================================================
# 工具函数
# ============================================================

def read_csv(path, encoding="utf-8-sig", delimiter=","):
    """读取 CSV/TSV 文件，返回行列表。"""
    with open(path, "r", encoding=encoding, errors="replace") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        return list(reader)


def canonical_smiles(smi):
    """将 PSMILES/SMILES 规范化为 RDKit canonical SMILES。
    PSMILES 中的 [*] 替换为 * 后再处理。
    返回 (canonical_smi, is_valid)。
    """
    if not smi or not smi.strip():
        return None, False
    s = smi.strip()
    # 替换 PSMILES 标记
    s = s.replace("[*]", "*")
    # 尝试 RDKit 解析
    try:
        mol = Chem.MolFromSmiles(s, sanitize=False)
        if mol is not None:
            # 尝试 sanitize（可能因为 * 原子失败）
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                pass
            can = Chem.MolToSmiles(mol)
            if can:
                return can, True
        # 如果直接解析失败，去掉 * 后尝试
        s_no_star = s.replace("*", "[H]")
        mol2 = Chem.MolFromSmiles(s_no_star)
        if mol2 is not None:
            can2 = Chem.MolToSmiles(mol2)
            return can2, True
    except Exception:
        pass
    return s, False


def count_heavy_atoms(smi):
    """计算重原子数（非 H 非 *）。"""
    try:
        s = smi.strip().replace("[*]", "*")
        mol = Chem.MolFromSmiles(s.replace("*", "[H]"))
        if mol:
            return mol.GetNumHeavyAtoms()
    except Exception:
        pass
    return None


def tg_stats(values):
    """计算 Tg 统计量。"""
    arr = np.array(values, dtype=float)
    return {
        "count": len(arr),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
    }


# ============================================================
# 数据集加载器（返回 list of dict: smiles, tg_k, raw_smiles）
# ============================================================

def load_polymetrix():
    path = DATA_DIR / "polymetrix_tg.csv"
    rows = read_csv(path)
    results = []
    for r in rows:
        smi = r.get("PSMILES", "").strip()
        tg_str = r.get("Tg_K", "").strip()
        rel = r.get("reliability", "").strip()
        if not smi or not tg_str:
            continue
        try:
            tg_k = float(tg_str)
        except ValueError:
            continue
        results.append({
            "raw_smiles": smi, "tg_k": tg_k,
            "reliability": rel, "unit": "K",
        })
    return results


def load_point2():
    path = DATA_DIR / "point2_tg.csv"
    rows = read_csv(path)
    results = []
    for r in rows:
        smi = r.get("SMILES", "").strip()
        tg_str = r.get("Tg_K", "").strip()
        if not smi or not tg_str:
            continue
        try:
            tg_k = float(tg_str)
        except ValueError:
            continue
        results.append({"raw_smiles": smi, "tg_k": tg_k, "unit": "K"})
    return results


def load_neurips():
    path = DATA_DIR / "neurips_opp_tg.csv"
    rows = read_csv(path)
    results = []
    for r in rows:
        smi = r.get("SMILES", "").strip()
        tg_str = r.get("Tg", "").strip()
        if not smi or not tg_str:
            continue
        try:
            tg_c = float(tg_str)
            tg_k = tg_c + 273.15
        except ValueError:
            continue
        results.append({"raw_smiles": smi, "tg_k": tg_k, "unit": "C->K"})
    return results


def load_openpoly():
    path = DATA_DIR / "openpoly_properties.csv"
    rows = read_csv(path)
    results = []
    for r in rows:
        smi = r.get("PSMILES", "").strip()
        tg_str = r.get("Tg (K)", "").strip()
        if not smi or not tg_str:
            continue
        try:
            tg_k = float(tg_str)
        except ValueError:
            continue
        results.append({"raw_smiles": smi, "tg_k": tg_k, "unit": "K"})
    return results


def load_qiu_polymer():
    path = DATA_DIR / "qiu_polymer.csv"
    rows = read_csv(path)
    results = []
    for r in rows:
        smi_key = [k for k in r if "smiles" in k.lower()]
        tg_key = [k for k in r if "tg" in k.lower()]
        if not smi_key or not tg_key:
            continue
        smi = r[smi_key[0]].strip()
        tg_str = r[tg_key[0]].strip()
        if not smi or not tg_str:
            continue
        try:
            tg_c = float(tg_str)
            tg_k = tg_c + 273.15
        except ValueError:
            continue
        results.append({"raw_smiles": smi, "tg_k": tg_k, "unit": "C->K"})
    return results


def load_qiu_pi():
    path = DATA_DIR / "qiu_pi.csv"
    rows = read_csv(path)
    results = []
    for r in rows:
        smi_key = [k for k in r if "smiles" in k.lower()]
        tg_key = [k for k in r if "tg" in k.lower()]
        if not smi_key or not tg_key:
            continue
        smi = r[smi_key[0]].strip()
        tg_str = r[tg_key[0]].strip()
        if not smi or not tg_str:
            continue
        try:
            tg_c = float(tg_str)
            tg_k = tg_c + 273.15
        except ValueError:
            continue
        results.append({"raw_smiles": smi, "tg_k": tg_k, "unit": "C->K"})
    return results


def load_polyinfo_tg_median():
    path = DATA_DIR / "polyinfo_tg_median.csv"
    rows = read_csv(path)
    results = []
    for r in rows:
        smi = r.get("SMILES", "").strip()
        # Tg 列名不确定，检查多个可能
        tg_str = r.get("Tg", r.get("Tg_K", "")).strip()
        if not smi or not tg_str:
            continue
        try:
            tg_val = float(tg_str)
        except ValueError:
            continue
        # 判断单位：如果中位数在 -200~500 范围，可能是°C
        # 从前面 head 看到 -54, -3，确认是 °C
        if tg_val < 600:  # 大多数 Tg < 600°C
            tg_k = tg_val + 273.15
            unit = "C->K"
        else:
            tg_k = tg_val
            unit = "K"
        results.append({"raw_smiles": smi, "tg_k": tg_k, "unit": unit})
    return results


def load_jcim():
    path = DATA_DIR / "JCIM_sup_bigsmiles.csv"
    rows = read_csv(path)
    results = []
    for r in rows:
        smi = r.get("SMILES", "").strip()
        tg_str = r.get("Tg (C)", r.get("Tg", "")).strip()
        if not smi or not tg_str:
            continue
        try:
            tg_c = float(tg_str)
            tg_k = tg_c + 273.15
        except ValueError:
            continue
        results.append({"raw_smiles": smi, "tg_k": tg_k, "unit": "C->K"})
    return results


# ============================================================
# 分析 1：每个数据集的基本统计
# ============================================================

def analyze_basic_stats(name, data):
    """对单个数据集做基本统计。"""
    print(f"\n{'='*60}")
    print(f"数据集: {name}")
    print(f"{'='*60}")

    total = len(data)
    print(f"  总行数: {total}")

    if total == 0:
        print("  (空数据集)")
        return [], []

    # RDKit 解析
    valid_count = 0
    invalid_smiles = []
    can_smiles_list = []
    tg_values = []

    for entry in data:
        can, is_valid = canonical_smiles(entry["raw_smiles"])
        if is_valid:
            valid_count += 1
            can_smiles_list.append(can)
        else:
            invalid_smiles.append(entry["raw_smiles"][:80])
            can_smiles_list.append(None)
        tg_values.append(entry["tg_k"])

    parse_rate = valid_count / total * 100 if total > 0 else 0
    print(f"  有效 SMILES: {valid_count}/{total} ({parse_rate:.1f}%)")

    if invalid_smiles:
        print(f"  无法解析的 SMILES 示例 (最多 5 个):")
        for s in invalid_smiles[:5]:
            print(f"    {s}")

    # Tg 统计
    stats = tg_stats(tg_values)
    unit_info = data[0].get("unit", "unknown")
    print(f"  Tg 单位转换: {unit_info}")
    print(f"  Tg 统计 (K):")
    print(f"    min:    {stats['min']:.2f} K")
    print(f"    max:    {stats['max']:.2f} K")
    print(f"    mean:   {stats['mean']:.2f} K")
    print(f"    median: {stats['median']:.2f} K")
    print(f"    std:    {stats['std']:.2f} K")

    # 异常值
    outlier_low = sum(1 for t in tg_values if t < 100)
    outlier_high = sum(1 for t in tg_values if t > 900)
    print(f"  异常值: Tg < 100K: {outlier_low}, Tg > 900K: {outlier_high}")

    # 小分子检测
    short_smiles = sum(1 for e in data if len(e["raw_smiles"].strip()) < 5)
    small_mol = 0
    for e in data:
        ha = count_heavy_atoms(e["raw_smiles"])
        if ha is not None and ha < 3:
            small_mol += 1
    print(f"  小分子: SMILES 长度 < 5: {short_smiles}, 重原子数 < 3: {small_mol}")

    return can_smiles_list, tg_values


# ============================================================
# 分析 2：跨数据集去重分析
# ============================================================

def analyze_overlap(datasets_canonical):
    """计算每对数据集间的重叠率。"""
    print(f"\n{'='*60}")
    print("跨数据集去重分析")
    print(f"{'='*60}")

    names = list(datasets_canonical.keys())
    # 每个数据集的 unique canonical SMILES set
    sets = {}
    for name in names:
        s = {c for c in datasets_canonical[name] if c is not None}
        sets[name] = s
        print(f"  {name}: {len(s)} unique canonical SMILES")

    # 两两重叠
    print(f"\n  两两重叠矩阵 (重叠数 / 较小集大小的百分比):")
    print(f"  {'':>20s}", end="")
    for n2 in names:
        print(f"  {n2[:12]:>12s}", end="")
    print()

    for n1 in names:
        print(f"  {n1[:20]:>20s}", end="")
        for n2 in names:
            if n1 == n2:
                print(f"  {'---':>12s}", end="")
            else:
                overlap = len(sets[n1] & sets[n2])
                smaller = min(len(sets[n1]), len(sets[n2]))
                pct = overlap / smaller * 100 if smaller > 0 else 0
                print(f"  {overlap:>5d}({pct:4.1f}%)", end="")
        print()

    # 全部合并后的唯一分子数
    all_canonical = set()
    for s in sets.values():
        all_canonical.update(s)
    print(f"\n  所有数据集合并后: {len(all_canonical)} 个唯一 canonical SMILES")

    # 各源贡献的独有分子
    print(f"\n  各数据集独有分子（不出现在其他数据集中）:")
    for name in names:
        others = set()
        for n2 in names:
            if n2 != name:
                others.update(sets[n2])
        unique_to_this = sets[name] - others
        print(f"    {name}: {len(unique_to_this)} 独有")


# ============================================================
# 分析 3：数据质量评级
# ============================================================

def analyze_quality(datasets_raw, datasets_canonical, datasets_tg):
    """PolyMetriX reliability 分布，跨源 Tg 冲突。"""
    print(f"\n{'='*60}")
    print("数据质量评级")
    print(f"{'='*60}")

    # PolyMetriX reliability 分布
    print(f"\n  PolyMetriX reliability 分布:")
    if "polymetrix" in datasets_raw:
        rel_counts = defaultdict(int)
        for entry in datasets_raw["polymetrix"]:
            rel = entry.get("reliability", "unknown")
            rel_counts[rel] += 1
        for rel_type in sorted(rel_counts.keys()):
            print(f"    {rel_type}: {rel_counts[rel_type]}")
    else:
        print("    (数据集未加载)")

    # 跨源 Tg 冲突
    print(f"\n  跨源 Tg 冲突分析:")
    # 建立 canonical SMILES -> [(source, tg_k)] 映射
    smiles_tg_map = defaultdict(list)
    for name in datasets_canonical:
        can_list = datasets_canonical[name]
        tg_list = datasets_tg[name]
        for can, tg in zip(can_list, tg_list):
            if can is not None:
                smiles_tg_map[can].append((name, tg))

    # 找多源出现的分子
    multi_source = {k: v for k, v in smiles_tg_map.items() if len(set(src for src, _ in v)) > 1}
    print(f"    多源出现的分子: {len(multi_source)}")

    # Tg 差异 > 30K 的冲突
    conflicts_30 = []
    conflicts_50 = []
    for smi, entries in multi_source.items():
        tg_vals = [tg for _, tg in entries]
        tg_range = max(tg_vals) - min(tg_vals)
        if tg_range > 30:
            conflicts_30.append((smi, entries, tg_range))
        if tg_range > 50:
            conflicts_50.append((smi, entries, tg_range))

    print(f"    Tg 差异 > 30K 的冲突: {len(conflicts_30)}")
    print(f"    Tg 差异 > 50K 的冲突: {len(conflicts_50)}")

    if conflicts_30:
        print(f"\n    最严重的冲突 (Top 10):")
        sorted_conflicts = sorted(conflicts_30, key=lambda x: x[2], reverse=True)
        for smi, entries, tg_range in sorted_conflicts[:10]:
            sources_tg = ", ".join(f"{src}={tg:.1f}K" for src, tg in entries)
            print(f"      {smi[:60]:60s}  差异={tg_range:.1f}K  [{sources_tg}]")


# ============================================================
# 分析 4：polyinfo_database.tsv 和 JCIM 快速检查
# ============================================================

def analyze_special_files():
    """快速检查 polyinfo_database.tsv 和 JCIM_sup_bigsmiles.csv。"""
    print(f"\n{'='*60}")
    print("特殊文件快速检查")
    print(f"{'='*60}")

    # polyinfo_database.tsv
    print(f"\n  --- polyinfo_database.tsv ---")
    tsv_path = DATA_DIR / "polyinfo_database.tsv"
    if tsv_path.exists():
        with open(tsv_path, "r", encoding="utf-8-sig", errors="replace") as f:
            lines = f.readlines()
        print(f"  总行数: {len(lines)}")
        # 解析 TSV
        rows = read_csv(tsv_path, delimiter="\t")
        print(f"  数据行数: {len(rows)}")
        if rows:
            cols = list(rows[0].keys())
            print(f"  列数: {len(cols)}")
            # 找 Tg 相关列
            tg_cols = [c for c in cols if "tg" in c.lower() or "glass" in c.lower()]
            print(f"  Tg 相关列: {tg_cols}")
            # 找 SMILES 相关列
            smi_cols = [c for c in cols if "smiles" in c.lower() or "curly" in c.lower()]
            print(f"  SMILES 相关列: {smi_cols}")

            # 检查是否有可用 Tg 数据
            tg_count = 0
            for r in rows:
                for tc in tg_cols:
                    val = r.get(tc, "").strip()
                    if val and val != "" and val != "-":
                        try:
                            float(val.split(" ")[0].split("-")[0])
                            tg_count += 1
                            break
                        except ValueError:
                            pass
            print(f"  含 Tg 数据的行: {tg_count}")

            # 检查 CurlySMILES
            curly_count = 0
            for r in rows:
                for sc in smi_cols:
                    val = r.get(sc, "").strip()
                    if val and val != "":
                        curly_count += 1
                        break
            print(f"  含 SMILES 的行: {curly_count}")

            # 显示前 3 行的 SMILES 和 Tg
            print(f"\n  前 3 行示例:")
            for i, r in enumerate(rows[:3]):
                smi_val = ""
                for sc in smi_cols:
                    smi_val = r.get(sc, "").strip()
                    if smi_val:
                        break
                tg_val = ""
                for tc in tg_cols:
                    tg_val = r.get(tc, "").strip()
                    if tg_val:
                        break
                name = r.get("COMMON NAMES", r.get(list(r.keys())[0], ""))
                print(f"    [{i}] name={name[:40]}, SMILES={smi_val[:50]}, Tg={tg_val[:30]}")
    else:
        print("  文件不存在")

    # JCIM_sup_bigsmiles.csv
    print(f"\n  --- JCIM_sup_bigsmiles.csv ---")
    jcim_path = DATA_DIR / "JCIM_sup_bigsmiles.csv"
    if jcim_path.exists():
        rows = read_csv(jcim_path)
        print(f"  总行数: {len(rows)}")
        if rows:
            cols = list(rows[0].keys())
            print(f"  列名: {cols}")
            # 检查 Tg 列
            tg_cols = [c for c in cols if "tg" in c.lower()]
            smi_cols = [c for c in cols if "smiles" in c.lower()]
            print(f"  Tg 列: {tg_cols}")
            print(f"  SMILES 列: {smi_cols}")

            # 统计有效 Tg
            tg_count = 0
            tg_vals = []
            for r in rows:
                for tc in tg_cols:
                    val = r.get(tc, "").strip()
                    if val:
                        try:
                            tg_vals.append(float(val))
                            tg_count += 1
                            break
                        except ValueError:
                            pass
            print(f"  含 Tg 数据的行: {tg_count}")
            if tg_vals:
                stats = tg_stats(tg_vals)
                print(f"  Tg 统计 (原始单位可能是°C):")
                print(f"    min: {stats['min']:.2f}, max: {stats['max']:.2f}")
                print(f"    mean: {stats['mean']:.2f}, median: {stats['median']:.2f}")

            # 前 3 行
            print(f"\n  前 5 行示例:")
            for i, r in enumerate(rows[:5]):
                smi_val = r.get("SMILES", "")[:50]
                tg_val = r.get("Tg (C)", r.get("Tg", ""))
                bigsmiles = r.get("BigSMILES", "")[:50]
                print(f"    [{i}] SMILES={smi_val}, Tg={tg_val}, BigSMILES={bigsmiles}")
    else:
        print("  文件不存在")

    # PI1M.csv 快速检查
    print(f"\n  --- PI1M.csv ---")
    pi1m_path = DATA_DIR / "PI1M.csv"
    if pi1m_path.exists():
        with open(pi1m_path, "r", encoding="utf-8-sig") as f:
            header = f.readline().strip()
            line1 = f.readline().strip()
            # 计算行数（高效方式）
            count = 2
            for _ in f:
                count += 1
        print(f"  总行数: ~{count}")
        print(f"  列名: {header}")
        print(f"  示例行: {line1[:100]}")
        print(f"  注意: 此数据集只有 SMILES，无 Tg 数据，不参与 Tg 分析")

    # polyinfo_tc.csv
    print(f"\n  --- polyinfo_tc.csv ---")
    tc_path = DATA_DIR / "polyinfo_tc.csv"
    if tc_path.exists():
        rows = read_csv(tc_path)
        print(f"  总行数: {len(rows)}")
        if rows:
            cols = list(rows[0].keys())
            print(f"  列名: {cols[:10]}...")
            print(f"  注意: 此数据集是热导率 (TC)，非 Tg，不参与 Tg 分析")

    # Bicerano_bigsmiles.csv
    print(f"\n  --- Bicerano_bigsmiles.csv ---")
    bic_path = DATA_DIR / "Bicerano_bigsmiles.csv"
    if bic_path.exists():
        rows = read_csv(bic_path)
        print(f"  总行数: {len(rows)}")
        if rows:
            cols = list(rows[0].keys())
            print(f"  列名: {cols}")
            # 前 3 行
            for i, r in enumerate(rows[:3]):
                print(f"    [{i}] {list(r.values())[:3]}")


# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 60)
    print("外部数据集质量分析报告")
    print("=" * 60)

    # 定义所有有 Tg 数据的数据集
    loaders = {
        "polymetrix": load_polymetrix,
        "point2": load_point2,
        "neurips_opp": load_neurips,
        "openpoly": load_openpoly,
        "qiu_polymer": load_qiu_polymer,
        "qiu_pi": load_qiu_pi,
        "polyinfo_tg_median": load_polyinfo_tg_median,
        "jcim": load_jcim,
    }

    # 加载所有数据集
    datasets_raw = {}
    for name, loader in loaders.items():
        try:
            datasets_raw[name] = loader()
        except Exception as e:
            print(f"  ERROR loading {name}: {e}")
            datasets_raw[name] = []

    # ======== 分析 1：基本统计 ========
    print("\n" + "#" * 60)
    print("# 分析 1：每个数据集的基本统计")
    print("#" * 60)

    datasets_canonical = {}
    datasets_tg = {}

    for name, data in datasets_raw.items():
        can_list, tg_list = analyze_basic_stats(name, data)
        datasets_canonical[name] = can_list
        datasets_tg[name] = tg_list

    # ======== 分析 2：跨数据集去重 ========
    print("\n" + "#" * 60)
    print("# 分析 2：跨数据集去重分析")
    print("#" * 60)
    analyze_overlap(datasets_canonical)

    # ======== 分析 3：数据质量评级 ========
    print("\n" + "#" * 60)
    print("# 分析 3：数据质量评级")
    print("#" * 60)
    analyze_quality(datasets_raw, datasets_canonical, datasets_tg)

    # ======== 分析 4：特殊文件检查 ========
    print("\n" + "#" * 60)
    print("# 分析 4：特殊文件快速检查")
    print("#" * 60)
    analyze_special_files()

    # ======== 总结 ========
    print("\n" + "#" * 60)
    print("# 总结")
    print("#" * 60)

    print(f"\n  有 Tg 数据的数据集总结:")
    total_entries = 0
    for name, data in datasets_raw.items():
        print(f"    {name:25s}: {len(data):>7d} 条")
        total_entries += len(data)
    print(f"    {'合计':25s}: {total_entries:>7d} 条")

    # 合并所有 canonical SMILES
    all_can = set()
    for can_list in datasets_canonical.values():
        for c in can_list:
            if c is not None:
                all_can.add(c)
    print(f"\n  去重后唯一分子: {len(all_can)}")

    # 有效 Tg 范围内的数据
    valid_in_range = 0
    for tg_list in datasets_tg.values():
        for t in tg_list:
            if 100 <= t <= 900:
                valid_in_range += 1
    print(f"  Tg 在 [100K, 900K] 范围内: {valid_in_range}")

    print(f"\n分析完成。")


if __name__ == "__main__":
    main()
