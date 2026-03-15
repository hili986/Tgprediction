"""
Build unified homopolymer Tg dataset — Option B restructure.

Merges all available homopolymer Tg data sources:
  - Bicerano 304 (gold standard, priority in dedup)
  - PolyMetriX ~7,367
  - NeurIPS OPP ~14,000 (quality filtered)
  - Conjugated 32

Outputs:
  data/unified_tg.parquet — columns: [smiles, tg_k, source, canonical_smiles, split]
  80/20 stratified train/test split (random_state=42)

Usage:
  python scripts/build_unified_dataset.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.external_datasets import save_unified_parquet, load_unified_dataset


def main():
    print("=" * 60)
    print("  Building Unified Tg Dataset (Option B)")
    print("=" * 60)

    output_path, n_train, n_test = save_unified_parquet(
        test_size=0.2,
        random_state=42,
        verbose=True,
    )

    # Verification
    print("\n" + "=" * 60)
    print("  Verification")
    print("=" * 60)

    df = load_unified_dataset(output_path)
    print(f"  Total rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Null values: {df.isnull().sum().to_dict()}")
    print(f"  Tg range: [{df['tg_k'].min():.1f}, {df['tg_k'].max():.1f}]K")
    print(f"  Tg mean: {df['tg_k'].mean():.1f}K, std: {df['tg_k'].std():.1f}K")
    print(f"  Unique canonical SMILES: {df['canonical_smiles'].nunique()}")
    print(f"  Split distribution:")
    print(f"    {df['split'].value_counts().to_dict()}")
    print(f"\n  Source breakdown:")
    for split_name in ["train", "test"]:
        subset = df[df["split"] == split_name]
        print(f"    [{split_name}]:")
        for src, cnt in subset["source"].value_counts().items():
            print(f"      {src}: {cnt}")

    # Check for duplicates
    dup_count = df["canonical_smiles"].duplicated().sum()
    if dup_count > 0:
        print(f"\n  WARNING: {dup_count} duplicate canonical SMILES found!")
    else:
        print(f"\n  No duplicate canonical SMILES (dedup verified)")

    print("\n" + "=" * 60)
    print(f"  Dataset saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
