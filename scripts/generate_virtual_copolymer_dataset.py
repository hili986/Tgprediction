"""
Virtual copolymer Tg dataset generator using the best predictor in this repo.

Key runtime behavior:
    - One Python process per generation job
    - One BestTgPredictor instance per job
    - Lazy predictor.fit(): skipped entirely if --resume finds nothing new
    - Supports auto, csv, and hybrid recipe sources
    - Supports random, block, and both architecture expansion
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.predict_tg_tabpfn_186d import BestTgPredictor, InferencePaths, _require_paths
from src.data.virtual_copolymer_generation import load_recipe_specs_from_args, run_generation_job


def _default_path(*parts: str) -> str:
    return str(PROJECT_ROOT.joinpath(*parts))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate virtual copolymer Tg datasets with the best validated predictor."
    )
    parser.add_argument("--mode", choices=["auto", "csv", "hybrid"], required=True)
    parser.add_argument("--input-csv", type=str, default=None, help="Input CSV for csv or hybrid mode.")
    parser.add_argument("--output", type=str, required=True, help="Output path (.csv or .jsonl recommended).")
    parser.add_argument("--output-format", choices=["csv", "jsonl"], default="csv")
    parser.add_argument("--resume", action="store_true", help="Skip recipe_ids already present in output.")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--architecture", choices=["random", "block", "both"], default="random")

    parser.add_argument("--library", choices=["bicerano"], default="bicerano")
    parser.add_argument("--min-components", type=int, default=2)
    parser.add_argument("--max-components", type=int, default=2)
    parser.add_argument(
        "--weight-grid",
        type=str,
        default="0.5",
        help="Comma-separated positive fractions in (0,1), e.g. 0.1,0.3,0.5,0.7,0.9",
    )
    parser.add_argument("--max-recipes", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=42)

    parser.add_argument("--chain-physics-confs", type=int, default=50)
    parser.add_argument("--polybert-batch-size", type=int, default=64)

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


def build_inference_paths(args: argparse.Namespace) -> InferencePaths:
    return InferencePaths(
        data_path=Path(args.data_path),
        phyc_cache=Path(args.phyc_cache),
        gnn_cache=Path(args.gnn_cache),
        pbert_cache=Path(args.pbert_cache),
        chain_physics_cache=Path(args.chain_physics_cache),
        polybert_model_dir=Path(args.polybert_model_dir),
        gnn_checkpoint=Path(args.gnn_checkpoint),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mode in {"csv", "hybrid"} and not args.input_csv:
        parser.error("--input-csv is required for csv and hybrid modes.")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive.")
    if args.max_components < args.min_components:
        parser.error("--max-components must be >= --min-components.")

    paths = build_inference_paths(args)
    _require_paths(paths)

    recipes = load_recipe_specs_from_args(args)

    predictor = BestTgPredictor(
        paths=paths,
        device=args.device,
        chain_physics_confs=args.chain_physics_confs,
        polybert_batch_size=args.polybert_batch_size,
    )

    stats = run_generation_job(
        predictor=predictor,
        recipes=recipes,
        output_path=Path(args.output),
        output_format=args.output_format,
        chunk_size=args.chunk_size,
        resume=args.resume,
    )

    print(
        "Generation finished: "
        f"written={stats['written']}, "
        f"errors={stats['errors']}, "
        f"skipped_existing={stats['skipped_existing']}"
    )
    if stats["written"] == 0 and stats["skipped_existing"] == 0:
        print("No recipes generated. Nothing to do.")
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
