# Virtual Copolymer Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dedicated server-side generator that creates virtual copolymer Tg datasets with one shared `BestTgPredictor` per job, supports `auto/csv/hybrid`, supports binary and multicomponent recipes, supports `random/block`, and writes resumable chunked outputs.

**Architecture:** Keep model inference inside the existing `BestTgPredictor`, move recipe generation and output plumbing into a new reusable module under `src/data`, and expose a thin CLI in `scripts/generate_virtual_copolymer_dataset.py`. Test the reusable module with `unittest` and use a fake predictor for job-level smoke tests so the new behavior is validated without loading the heavy model during unit tests.

**Tech Stack:** Python 3, `argparse`, `dataclasses`, `itertools`, `csv`, `json`, `hashlib`, `pathlib`, `tempfile`, `unittest`, existing predictor in `scripts/predict_tg_tabpfn_186d.py`

---

## File Structure

### New Files

- `src/data/virtual_copolymer_generation.py`
  - recipe dataclass
  - weight-grid parsing
  - recipe canonicalization and `recipe_id`
  - auto enumeration
  - CSV recipe loading
  - chunked CSV/JSONL writing
  - resume ID loading
  - row flattening helpers
- `scripts/generate_virtual_copolymer_dataset.py`
  - CLI parser
  - predictor bootstrap
  - job loop
  - chunked flushing
- `tests/test_virtual_copolymer_generation.py`
  - deterministic unit tests for recipe logic and resume/writer behavior
  - fake-predictor smoke test for one-process generation flow

### Existing Files To Reuse Only

- `scripts/predict_tg_tabpfn_186d.py`
  - `BestTgPredictor`
  - `InferencePaths`
  - existing path defaults and runtime assumptions
- `src/data/bicerano_tg_dataset.py`
  - internal monomer library for `auto` mode

### Decomposition Notes

- Keep generation-specific logic out of `scripts/predict_tg_tabpfn_186d.py` unless implementation reveals a hard blocker.
- Keep the new script thin so most logic is testable without shelling out.
- Keep resume logic independent of the predictor so interrupted jobs can be resumed cheaply.

### Task 1: Build Recipe and Resume Utilities

**Files:**
- Create: `src/data/virtual_copolymer_generation.py`
- Test: `tests/test_virtual_copolymer_generation.py`

- [ ] **Step 1: Write the failing tests for weight grids, canonicalization, and recipe IDs**

```python
import unittest

from src.data.virtual_copolymer_generation import (
    RecipeSpec,
    canonicalize_recipe,
    make_recipe_id,
    parse_weight_grid,
)


class TestWeightGridAndIds(unittest.TestCase):
    def test_parse_weight_grid_keeps_positive_values(self):
        self.assertEqual(parse_weight_grid("0.2,0.5,0.8"), (0.2, 0.5, 0.8))

    def test_canonicalize_recipe_sorts_components_with_weights(self):
        components, weights = canonicalize_recipe(
            ["*CO(*)", "*CC(*)"],
            [0.25, 0.75],
        )
        self.assertEqual(components, ("*CC(*)", "*CO(*)"))
        self.assertEqual(weights, (0.75, 0.25))

    def test_recipe_id_is_order_invariant_for_random_mode(self):
        left = make_recipe_id(
            RecipeSpec(
                components=("*CC(*)", "*CO(*)"),
                weights=(0.6, 0.4),
                architecture="random",
                input_origin="auto",
                metadata={},
            )
        )
        right = make_recipe_id(
            RecipeSpec(
                components=("*CO(*)", "*CC(*)"),
                weights=(0.4, 0.6),
                architecture="random",
                input_origin="auto",
                metadata={},
            )
        )
        self.assertEqual(left, right)
```

- [ ] **Step 2: Run the new test target to verify the module does not exist yet**

Run: `python -m unittest tests.test_virtual_copolymer_generation.TestWeightGridAndIds -v`

Expected:

- import error for `src.data.virtual_copolymer_generation`, or
- `AttributeError` for missing functions

- [ ] **Step 3: Write the minimal recipe utility implementation**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple


@dataclass(frozen=True)
class RecipeSpec:
    components: Tuple[str, ...]
    weights: Tuple[float, ...]
    architecture: str
    input_origin: str
    metadata: Dict[str, object]


def parse_weight_grid(spec: str) -> Tuple[float, ...]:
    values = []
    for chunk in (spec or "").split(","):
        text = chunk.strip()
        if not text:
            continue
        value = float(text)
        if value <= 0 or value >= 1:
            raise ValueError("weight-grid values must be in (0, 1).")
        values.append(value)
    if not values:
        raise ValueError("weight-grid must contain at least one value.")
    return tuple(values)


def canonicalize_recipe(
    components: Sequence[str],
    weights: Sequence[float],
    decimals: int = 8,
) -> Tuple[Tuple[str, ...], Tuple[float, ...]]:
    paired = sorted(
        ((str(component).strip(), float(weight)) for component, weight in zip(components, weights)),
        key=lambda item: item[0],
    )
    ordered_components = tuple(component for component, _ in paired)
    ordered_weights = tuple(round(weight, decimals) for _, weight in paired)
    return ordered_components, ordered_weights


def make_recipe_id(recipe: RecipeSpec) -> str:
    components, weights = canonicalize_recipe(recipe.components, recipe.weights)
    joined_components = "|".join(components)
    joined_weights = "|".join(f"{weight:.8f}" for weight in weights)
    return f"{recipe.architecture}::{joined_components}::{joined_weights}"
```

- [ ] **Step 4: Re-run the focused test target**

Run: `python -m unittest tests.test_virtual_copolymer_generation.TestWeightGridAndIds -v`

Expected:

- all three tests pass

- [ ] **Step 5: Commit the utility baseline**

```bash
git add src/data/virtual_copolymer_generation.py tests/test_virtual_copolymer_generation.py
git commit -m "feat: add virtual copolymer recipe utilities"
```

### Task 2: Add Auto Enumeration, CSV Loading, Resume IDs, and Chunk Writers

**Files:**
- Modify: `src/data/virtual_copolymer_generation.py`
- Modify: `tests/test_virtual_copolymer_generation.py`

- [ ] **Step 1: Write failing tests for enumeration, resume loading, and CSV/JSONL append behavior**

```python
import json
import tempfile
from pathlib import Path

from src.data.virtual_copolymer_generation import (
    append_result_rows,
    iter_auto_recipe_specs,
    load_completed_recipe_ids,
)


class TestEnumerationAndResume(unittest.TestCase):
    def test_binary_auto_recipes_do_not_duplicate_ab_and_ba(self):
        library = [
            {"name": "A", "smiles": "*CC(*)"},
            {"name": "B", "smiles": "*CO(*)"},
        ]
        recipes = list(
            iter_auto_recipe_specs(
                library=library,
                min_components=2,
                max_components=2,
                weight_grid=(0.2, 0.8),
                architectures=("random",),
                max_recipes=None,
                random_seed=42,
            )
        )
        self.assertEqual(len(recipes), 2)
        self.assertEqual({recipe.components for recipe in recipes}, {("*CC(*)", "*CO(*)")})

    def test_resume_loader_reads_existing_csv_recipe_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "virtual.csv"
            append_result_rows(
                path,
                [{"recipe_id": "random::*CC(*)|*CO(*)::0.60000000|0.40000000", "status": "ok"}],
                output_format="csv",
            )
            loaded = load_completed_recipe_ids(path, output_format="csv")
            self.assertEqual(loaded, {"random::*CC(*)|*CO(*)::0.60000000|0.40000000"})

    def test_jsonl_writer_appends_one_object_per_line(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "virtual.jsonl"
            append_result_rows(path, [{"recipe_id": "r1", "status": "ok"}], output_format="jsonl")
            append_result_rows(path, [{"recipe_id": "r2", "status": "ok"}], output_format="jsonl")
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[1])["recipe_id"], "r2")
```

- [ ] **Step 2: Run the failing tests**

Run: `python -m unittest tests.test_virtual_copolymer_generation.TestEnumerationAndResume -v`

Expected:

- missing-function or assertion failures for enumeration/writer helpers

- [ ] **Step 3: Implement the reusable generation helpers**

```python
from itertools import combinations, product
import csv
import json
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Set


def iter_auto_recipe_specs(
    library: Sequence[dict],
    min_components: int,
    max_components: int,
    weight_grid: Sequence[float],
    architectures: Sequence[str],
    max_recipes: Optional[int],
    random_seed: int,
) -> Iterator[RecipeSpec]:
    emitted = 0
    for n_components in range(min_components, max_components + 1):
        for combo in combinations(library, n_components):
            smiles = tuple(sorted(entry["smiles"] for entry in combo))
            if n_components == 2:
                weight_sets = [(float(w), float(1.0 - w)) for w in weight_grid]
            else:
                seen = set()
                weight_sets = []
                for raw in product(weight_grid, repeat=n_components):
                    total = sum(raw)
                    if total <= 0:
                        continue
                    normalized = tuple(round(value / total, 8) for value in raw)
                    if normalized in seen:
                        continue
                    seen.add(normalized)
                    weight_sets.append(normalized)
            for architecture in architectures:
                for weights in weight_sets:
                    yield RecipeSpec(
                        components=smiles,
                        weights=tuple(weights),
                        architecture=architecture,
                        input_origin="auto",
                        metadata={},
                    )
                    emitted += 1
                    if max_recipes is not None and emitted >= max_recipes:
                        return


def load_completed_recipe_ids(path: Path, output_format: str) -> Set[str]:
    if not path.exists():
        return set()
    if output_format == "jsonl":
        return {
            json.loads(line)["recipe_id"]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["recipe_id"] for row in csv.DictReader(handle) if row.get("recipe_id")}


def append_result_rows(path: Path, rows: Sequence[dict], output_format: str) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "jsonl":
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    fieldnames = list(rows[0].keys())
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
```

- [ ] **Step 4: Run the full utility test module**

Run: `python -m unittest tests/test_virtual_copolymer_generation.py -v`

Expected:

- utility tests pass
- no model artifacts are required yet

- [ ] **Step 5: Commit the generation plumbing**

```bash
git add src/data/virtual_copolymer_generation.py tests/test_virtual_copolymer_generation.py
git commit -m "feat: add resume-safe virtual copolymer generators"
```

### Task 3: Add the Thin CLI and Shared-Predictor Job Loop

**Files:**
- Create: `scripts/generate_virtual_copolymer_dataset.py`
- Modify: `src/data/virtual_copolymer_generation.py`
- Modify: `tests/test_virtual_copolymer_generation.py`

- [ ] **Step 1: Write the failing smoke test for one-process execution with a fake predictor**

```python
from unittest import mock

from src.data.virtual_copolymer_generation import run_generation_job


class FakePredictor:
    fit_calls = 0

    def __init__(self, *args, **kwargs):
        self.fit_invocations = 0

    def fit(self):
        self.fit_invocations += 1
        FakePredictor.fit_calls += 1

    def predict_multicomponent(self, smiles_list, weights, architecture="random"):
        return {
            "mode": "multicomponent_copolymer",
            "architecture": architecture,
            "n_components": len(smiles_list),
            "weights_normalized": list(weights),
            "tg_k_pred": 350.0,
            "tg_c_pred": 76.85,
            "primary_method": "weighted_descriptor_embedding_mix",
            "descriptor_mix_tg_k": 350.0,
            "descriptor_mix_tg_c": 76.85,
            "fox_reference_tg_k": 345.0,
            "fox_reference_tg_c": 71.85,
            "component_tg_window_k": [300.0, 400.0],
            "component_tg_window_c": [26.85, 126.85],
            "model": "fake",
            "warning": "",
        }


class TestJobLoop(unittest.TestCase):
    def test_job_uses_one_predictor_fit_for_many_rows(self):
        recipes = [
            RecipeSpec(components=("*CC(*)", "*CO(*)"), weights=(0.6, 0.4), architecture="random", input_origin="auto", metadata={}),
            RecipeSpec(components=("*CC(*)", "*CN(*)"), weights=(0.6, 0.4), architecture="random", input_origin="auto", metadata={}),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "virtual.csv"
            run_generation_job(
                predictor=FakePredictor(),
                recipes=recipes,
                output_path=output,
                output_format="csv",
                chunk_size=1,
                resume=False,
            )
            self.assertEqual(FakePredictor.fit_calls, 1)
```

- [ ] **Step 2: Run the smoke test to verify the job runner does not exist yet**

Run: `python -m unittest tests.test_virtual_copolymer_generation.TestJobLoop -v`

Expected:

- import error or missing `run_generation_job`

- [ ] **Step 3: Implement the job runner and the new CLI**

```python
# src/data/virtual_copolymer_generation.py
from typing import Iterable, Sequence


def flatten_prediction_row(recipe: RecipeSpec, result: dict, status: str = "ok", error: str = "") -> dict:
    return {
        "recipe_id": make_recipe_id(recipe),
        "input_origin": recipe.input_origin,
        "architecture": recipe.architecture,
        "n_components": len(recipe.components),
        "components_serialized": "|".join(recipe.components),
        "weights_serialized": "|".join(f"{weight:.8f}" for weight in recipe.weights),
        "tg_k_pred": result.get("tg_k_pred"),
        "tg_c_pred": result.get("tg_c_pred"),
        "primary_method": result.get("primary_method"),
        "descriptor_mix_tg_k": result.get("descriptor_mix_tg_k"),
        "descriptor_mix_tg_c": result.get("descriptor_mix_tg_c"),
        "fox_reference_tg_k": result.get("fox_reference_tg_k"),
        "fox_reference_tg_c": result.get("fox_reference_tg_c"),
        "component_tg_window_k_min": (result.get("component_tg_window_k") or [None, None])[0],
        "component_tg_window_k_max": (result.get("component_tg_window_k") or [None, None])[1],
        "model": result.get("model"),
        "warning": result.get("warning", ""),
        "status": status,
        "error": error,
        **recipe.metadata,
    }


def run_generation_job(
    predictor,
    recipes: Sequence[RecipeSpec],
    output_path: Path,
    output_format: str,
    chunk_size: int,
    resume: bool,
) -> int:
    completed = load_completed_recipe_ids(output_path, output_format) if resume else set()
    predictor.fit()
    buffer = []
    written = 0
    for recipe in recipes:
        recipe_id = make_recipe_id(recipe)
        if recipe_id in completed:
            continue
        try:
            result = predictor.predict_multicomponent(
                list(recipe.components),
                list(recipe.weights),
                architecture=recipe.architecture,
            )
            row = flatten_prediction_row(recipe, result, status="ok", error="")
        except Exception as exc:
            row = flatten_prediction_row(recipe, {}, status="error", error=str(exc))
        buffer.append(row)
        if len(buffer) >= chunk_size:
            append_result_rows(output_path, buffer, output_format)
            written += len(buffer)
            buffer.clear()
    if buffer:
        append_result_rows(output_path, buffer, output_format)
        written += len(buffer)
    return written


# scripts/generate_virtual_copolymer_dataset.py
import argparse
from pathlib import Path

from scripts.predict_tg_tabpfn_186d import BestTgPredictor, InferencePaths


def build_parser():
    parser = argparse.ArgumentParser(description="Generate virtual copolymer Tg datasets.")
    parser.add_argument("--mode", choices=["auto", "csv", "hybrid"], required=True)
    parser.add_argument("--input-csv", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--output-format", choices=["csv", "jsonl"], default="csv")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--architecture", choices=["random", "block", "both"], default="random")
    parser.add_argument("--min-components", type=int, default=2)
    parser.add_argument("--max-components", type=int, default=2)
    parser.add_argument("--weight-grid", type=str, default="0.5")
    parser.add_argument("--max-recipes", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--data-path", type=str, default="data/unified_tg.parquet")
    parser.add_argument("--phyc-cache", type=str, default="data/feature_matrix_PHY-C.parquet")
    parser.add_argument("--gnn-cache", type=str, default="data/gnn_embeddings_64d.parquet")
    parser.add_argument("--pbert-cache", type=str, default="data/polybert_embeddings.parquet")
    parser.add_argument("--chain-physics-cache", type=str, default="data/chain_physics_features.parquet")
    parser.add_argument("--polybert-model-dir", type=str, default="data/polybert_model")
    parser.add_argument("--gnn-checkpoint", type=str, default="checkpoints/gnn_pretrained.pt")
    return parser


def build_inference_paths(args) -> InferencePaths:
    return InferencePaths(
        data_path=Path(args.data_path),
        phyc_cache=Path(args.phyc_cache),
        gnn_cache=Path(args.gnn_cache),
        pbert_cache=Path(args.pbert_cache),
        chain_physics_cache=Path(args.chain_physics_cache),
        polybert_model_dir=Path(args.polybert_model_dir),
        gnn_checkpoint=Path(args.gnn_checkpoint),
    )


def build_recipes_from_args(args):
    return load_recipe_specs_from_args(args)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    recipes = build_recipes_from_args(args)
    predictor = BestTgPredictor(paths=build_inference_paths(args), device=args.device)
    run_generation_job(
        predictor=predictor,
        recipes=recipes,
        output_path=Path(args.output),
        output_format=args.output_format,
        chunk_size=args.chunk_size,
        resume=args.resume,
    )
    return 0
```

- [ ] **Step 4: Run the smoke test and the utility suite**

Run: `python -m unittest tests/test_virtual_copolymer_generation.py -v`

Expected:

- fake-predictor smoke test passes
- fit count stays at one for the whole job

- [ ] **Step 5: Commit the executable generator**

```bash
git add src/data/virtual_copolymer_generation.py scripts/generate_virtual_copolymer_dataset.py tests/test_virtual_copolymer_generation.py
git commit -m "feat: add virtual copolymer dataset generator"
```

### Task 4: Finish CLI Coverage for CSV and Hybrid Inputs

**Files:**
- Modify: `src/data/virtual_copolymer_generation.py`
- Modify: `scripts/generate_virtual_copolymer_dataset.py`
- Modify: `tests/test_virtual_copolymer_generation.py`

- [ ] **Step 1: Write failing tests for CSV parsing, hybrid union, and resume skip behavior**

```python
class TestCsvAndHybridInputs(unittest.TestCase):
    def test_csv_components_column_builds_recipe_specs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "input.csv"
            path.write_text(
                "case_id,architecture,components\n"
                "row1,block,*CC(*)::0.6|*CO(*)::0.4\n",
                encoding="utf-8",
            )
            recipes = load_csv_recipe_specs(path)
            self.assertEqual(len(recipes), 1)
            self.assertEqual(recipes[0].architecture, "block")
            self.assertEqual(recipes[0].weights, (0.6, 0.4))

    def test_hybrid_mode_is_union_not_cross_product(self):
        auto_recipes = [
            RecipeSpec(components=("*CC(*)", "*CO(*)"), weights=(0.5, 0.5), architecture="random", input_origin="auto", metadata={})
        ]
        csv_recipes = [
            RecipeSpec(components=("*CC(*)", "*CN(*)"), weights=(0.5, 0.5), architecture="block", input_origin="csv", metadata={"case_id": "c1"})
        ]
        merged = merge_recipe_sources(auto_recipes, csv_recipes)
        self.assertEqual(len(merged), 2)

    def test_resume_skips_existing_recipe_ids(self):
        recipe = RecipeSpec(components=("*CC(*)", "*CO(*)"), weights=(0.6, 0.4), architecture="random", input_origin="auto", metadata={})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "virtual.csv"
            append_result_rows(path, [{"recipe_id": make_recipe_id(recipe), "status": "ok"}], output_format="csv")
            written = run_generation_job(
                predictor=FakePredictor(),
                recipes=[recipe],
                output_path=path,
                output_format="csv",
                chunk_size=1,
                resume=True,
            )
            self.assertEqual(written, 0)
```

- [ ] **Step 2: Run the new test cases**

Run: `python -m unittest tests.test_virtual_copolymer_generation.TestCsvAndHybridInputs -v`

Expected:

- failures for missing CSV helpers or hybrid merge logic

- [ ] **Step 3: Implement CSV and hybrid recipe-source helpers and wire them into the CLI**

```python
import csv
from pathlib import Path


def load_csv_recipe_specs(path: Path) -> List[RecipeSpec]:
    recipes = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            architecture = str(row.get("architecture", "random")).strip().lower() or "random"
            metadata = {
                key: value
                for key, value in row.items()
                if key not in {"architecture", "components", "smiles", "smiles1", "w1", "smiles2", "w2"}
                and value not in (None, "")
            }
            if row.get("components"):
                parts = [part.strip() for part in str(row["components"]).split("|") if part.strip()]
                components = []
                weights = []
                for part in parts:
                    smiles, weight = part.rsplit("::", 1)
                    components.append(smiles.strip())
                    weights.append(float(weight))
                recipes.append(
                    RecipeSpec(
                        components=tuple(components),
                        weights=tuple(weights),
                        architecture=architecture,
                        input_origin="csv",
                        metadata=metadata,
                    )
                )
                continue
            indexed_components = []
            indexed_weights = []
            for index in range(1, 10):
                smiles = (row.get(f"smiles{index}") or "").strip()
                weight = (row.get(f"w{index}") or "").strip()
                if not smiles:
                    continue
                indexed_components.append(smiles)
                indexed_weights.append(float(weight) if weight else 1.0)
            if indexed_components:
                recipes.append(
                    RecipeSpec(
                        components=tuple(indexed_components),
                        weights=tuple(indexed_weights),
                        architecture=architecture,
                        input_origin="csv",
                        metadata=metadata,
                    )
                )
    return recipes


def merge_recipe_sources(auto_recipes, csv_recipes):
    merged = {}
    for recipe in list(auto_recipes) + list(csv_recipes):
        merged[make_recipe_id(recipe)] = recipe
    return list(merged.values())


def load_recipe_specs_from_args(args):
    auto_recipes = []
    csv_recipes = []
    if args.mode in {"auto", "hybrid"}:
        library = [{"name": name, "smiles": smiles} for name, smiles, _, _ in BICERANO_DATA]
        architectures = ("random", "block") if args.architecture == "both" else (args.architecture,)
        auto_recipes = list(
            iter_auto_recipe_specs(
                library=library,
                min_components=args.min_components,
                max_components=args.max_components,
                weight_grid=parse_weight_grid(args.weight_grid),
                architectures=architectures,
                max_recipes=args.max_recipes,
                random_seed=args.random_seed,
            )
        )
    if args.mode in {"csv", "hybrid"}:
        csv_recipes = load_csv_recipe_specs(Path(args.input_csv))
    return merge_recipe_sources(auto_recipes, csv_recipes) if args.mode == "hybrid" else (csv_recipes or auto_recipes)
```

- [ ] **Step 4: Run the full test module again**

Run: `python -m unittest tests/test_virtual_copolymer_generation.py -v`

Expected:

- all generator tests pass

- [ ] **Step 5: Commit the completed input modes**

```bash
git add src/data/virtual_copolymer_generation.py scripts/generate_virtual_copolymer_dataset.py tests/test_virtual_copolymer_generation.py
git commit -m "feat: support csv and hybrid virtual copolymer jobs"
```

### Task 5: Manual Verification on the CLI

**Files:**
- Verify only: `scripts/generate_virtual_copolymer_dataset.py`

- [ ] **Step 1: Run a help check**

Run: `python scripts/generate_virtual_copolymer_dataset.py --help`

Expected:

- usage output shows `--mode`, `--output`, `--output-format`, `--resume`, `--chunk-size`, and architecture controls

- [ ] **Step 2: Run a tiny auto-mode smoke job on CPU**

Run:

```bash
python scripts/generate_virtual_copolymer_dataset.py ^
  --mode auto ^
  --output results/virtual_smoke.csv ^
  --output-format csv ^
  --device cpu ^
  --architecture random ^
  --min-components 2 ^
  --max-components 2 ^
  --weight-grid 0.5 ^
  --max-recipes 2 ^
  --chunk-size 1
```

Expected:

- predictor initialization messages appear once
- output file contains two result rows plus header

- [ ] **Step 3: Run the same job with resume**

Run:

```bash
python scripts/generate_virtual_copolymer_dataset.py ^
  --mode auto ^
  --output results/virtual_smoke.csv ^
  --output-format csv ^
  --device cpu ^
  --architecture random ^
  --min-components 2 ^
  --max-components 2 ^
  --weight-grid 0.5 ^
  --max-recipes 2 ^
  --chunk-size 1 ^
  --resume
```

Expected:

- existing `recipe_id`s are skipped
- no duplicate rows are appended

- [ ] **Step 4: Run the generator unit tests one final time**

Run: `python -m unittest tests/test_virtual_copolymer_generation.py -v`

Expected:

- PASS

- [ ] **Step 5: Commit final verification fixes if any**

```bash
git add scripts/generate_virtual_copolymer_dataset.py src/data/virtual_copolymer_generation.py tests/test_virtual_copolymer_generation.py
git commit -m "test: verify virtual copolymer generator flow"
```
