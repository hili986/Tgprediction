# Precomputed Feature Reuse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse precomputed 7k training features during inference and virtual copolymer generation, while keeping Bicerano as the default auto-generation library and adding an explicit unified-library option.

**Architecture:** Extend `BestTgPredictor` so `fit()` builds an in-memory `smiles -> precomputed component features` lookup from the aligned training artifacts, then let `featurize_component()` consult that lookup before falling back to on-the-fly computation. Extend the virtual-data generation CLI and loader so `auto` mode can use either the existing Bicerano library or unique `smiles` from `unified_tg.parquet`, with a hard safety gate requiring `--max-recipes` for unified enumeration.

**Tech Stack:** Python 3, `argparse`, `unittest`, `pandas`, existing predictor in `scripts/predict_tg_tabpfn_186d.py`, existing generator in `scripts/generate_virtual_copolymer_dataset.py`, recipe utilities in `src/data/virtual_copolymer_generation.py`

---

### Task 1: Add Predictor-Level Reuse Tests

**Files:**
- Create: `tests/test_predict_tg_tabpfn_186d.py`
- Modify: `scripts/predict_tg_tabpfn_186d.py`
- Test: `tests/test_predict_tg_tabpfn_186d.py`

- [ ] **Step 1: Write the failing tests**

```python
import unittest
import numpy as np

from scripts.predict_tg_tabpfn_186d import (
    FULL_DIM,
    GNN_DIM,
    PBERT_PCA_DIM,
    PHY_C_LIGHT_DIM,
    _build_precomputed_component_lookup,
    BestTgPredictor,
)


class TestPrecomputedLookup(unittest.TestCase):
    def test_build_lookup_keeps_only_finite_rows(self):
        smiles = np.array(["*CC(*)", "*CO(*)"])
        x_phyc = np.vstack([np.ones(PHY_C_LIGHT_DIM), np.full(PHY_C_LIGHT_DIM, np.nan)])
        x_gnn = np.ones((2, GNN_DIM))
        x_pbert = np.ones((2, PBERT_PCA_DIM))

        lookup = _build_precomputed_component_lookup(smiles, x_phyc, x_gnn, x_pbert)

        self.assertEqual(set(lookup), {"*CC(*)"})
        self.assertEqual(lookup["*CC(*)"]["chain_physics_source"], "precomputed")

    def test_featurize_component_uses_precomputed_lookup_before_recompute(self):
        predictor = BestTgPredictor.__new__(BestTgPredictor)
        predictor._component_cache = {}
        predictor._precomputed_component_lookup = {
            "*CC(*)": {
                "smiles": "*CC(*)",
                "phyc": np.ones(PHY_C_LIGHT_DIM),
                "gnn": np.ones(GNN_DIM),
                "pbert": np.ones(PBERT_PCA_DIM),
                "chain_physics_source": "precomputed",
            }
        }

        def _boom(*args, **kwargs):
            raise AssertionError("should not recompute")

        predictor._compute_phyc_light = _boom
        predictor._compute_gnn_embedding = _boom
        predictor._compute_polybert_pca = _boom

        result = predictor.featurize_component("*CC(*)")

        self.assertEqual(result["chain_physics_source"], "precomputed")
        self.assertIn("*CC(*)", predictor._component_cache)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests.test_predict_tg_tabpfn_186d -v`

Expected: `ImportError` or `AttributeError` because `_build_precomputed_component_lookup` and `_precomputed_component_lookup` do not exist yet.

- [ ] **Step 3: Add the minimal predictor reuse implementation**

```python
def _build_precomputed_component_lookup(smiles, x_phyc, x_gnn, x_pbert):
    lookup = {}
    for smi, phyc, gnn, pbert in zip(smiles, x_phyc, x_gnn, x_pbert):
        if not (np.isfinite(phyc).all() and np.isfinite(gnn).all() and np.isfinite(pbert).all()):
            continue
        lookup[str(smi)] = {
            "smiles": str(smi),
            "phyc": np.asarray(phyc, dtype=float).copy(),
            "gnn": np.asarray(gnn, dtype=float).copy(),
            "pbert": np.asarray(pbert, dtype=float).copy(),
            "chain_physics_source": "precomputed",
        }
    return lookup


class BestTgPredictor:
    def __init__(...):
        self._precomputed_component_lookup = {}

    def fit(self):
        smiles, X_phyc, X_gnn, X_pbert_raw, y = _load_training_blocks(self.paths)
        ...
        X_pbert[valid_pbert] = self.pca.transform(X_pbert_raw[valid_pbert])
        self._precomputed_component_lookup = _build_precomputed_component_lookup(
            smiles, X_phyc, X_gnn, X_pbert
        )

    def featurize_component(self, smiles):
        smi = _validate_repeat_unit_smiles(smiles)
        if smi in self._component_cache:
            return self._component_cache[smi]
        precomputed = self._precomputed_component_lookup.get(smi)
        if precomputed is not None:
            self._component_cache[smi] = precomputed
            return precomputed
        ...
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m unittest tests.test_predict_tg_tabpfn_186d -v`

Expected: all tests in `tests.test_predict_tg_tabpfn_186d` pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_predict_tg_tabpfn_186d.py scripts/predict_tg_tabpfn_186d.py
git commit -m "feat: reuse precomputed predictor features"
```

### Task 2: Add Unified Auto-Library Support

**Files:**
- Modify: `src/data/virtual_copolymer_generation.py`
- Modify: `scripts/generate_virtual_copolymer_dataset.py`
- Test: `tests/test_virtual_copolymer_generation.py`

- [ ] **Step 1: Write the failing tests**

```python
class TestAutoLibraries(unittest.TestCase):
    def test_load_recipe_specs_defaults_to_bicerano(self):
        class Args:
            mode = "auto"
            auto_library = "bicerano"
            min_components = 2
            max_components = 2
            weight_grid = "0.5"
            architecture = "random"
            max_recipes = 1
            random_seed = 42
            data_path = "unused.parquet"

        recipes = list(load_recipe_specs_from_args(Args()))
        self.assertEqual(len(recipes), 1)

    def test_unified_auto_library_requires_max_recipes(self):
        class Args:
            mode = "auto"
            auto_library = "unified"
            min_components = 2
            max_components = 2
            weight_grid = "0.5"
            architecture = "random"
            max_recipes = None
            random_seed = 42
            data_path = "data/unified_tg.parquet"

        with self.assertRaisesRegex(ValueError, "--max-recipes"):
            list(load_recipe_specs_from_args(Args()))

    def test_unified_auto_library_deduplicates_smiles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "unified.parquet"
            pd.DataFrame({"smiles": ["*CC(*)", "*CO(*)", "*CC(*)"]}).to_parquet(path)
            library = get_unified_auto_library(path)
            self.assertEqual([row["smiles"] for row in library], ["*CC(*)", "*CO(*)"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests.test_virtual_copolymer_generation.TestAutoLibraries -v`

Expected: failure because `auto_library` / `get_unified_auto_library` / unified safety validation do not exist.

- [ ] **Step 3: Implement the library selector and CLI plumbing**

```python
def get_unified_auto_library(data_path: Path) -> List[Dict[str, str]]:
    df = pd.read_parquet(data_path, columns=["smiles"])
    unique_smiles = sorted({str(smiles).strip() for smiles in df["smiles"] if str(smiles).strip()})
    return [{"name": smiles, "smiles": smiles} for smiles in unique_smiles]


def load_recipe_specs_from_args(args):
    auto_library = getattr(args, "auto_library", getattr(args, "library", "bicerano"))
    if args.mode in {"auto", "hybrid"} and auto_library == "unified" and args.max_recipes is None:
        raise ValueError("--max-recipes is required when --auto-library unified is used.")
    library = (
        get_default_auto_library()
        if auto_library == "bicerano"
        else get_unified_auto_library(Path(args.data_path))
    )
```

```python
parser.add_argument(
    "--auto-library",
    "--library",
    dest="auto_library",
    choices=["bicerano", "unified"],
    default="bicerano",
    help="Auto recipe source library. Default: bicerano.",
)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m unittest tests.test_virtual_copolymer_generation.TestAutoLibraries -v`

Expected: all tests in `TestAutoLibraries` pass.

- [ ] **Step 5: Commit**

```bash
git add src/data/virtual_copolymer_generation.py scripts/generate_virtual_copolymer_dataset.py tests/test_virtual_copolymer_generation.py
git commit -m "feat: add unified auto library selection"
```

### Task 3: Verify Integrated Behavior

**Files:**
- Modify: `tests/test_virtual_copolymer_generation.py`
- Modify: `scripts/predict_tg_tabpfn_186d.py`
- Modify: `src/data/virtual_copolymer_generation.py`
- Test: `tests/test_predict_tg_tabpfn_186d.py`
- Test: `tests/test_virtual_copolymer_generation.py`

- [ ] **Step 1: Add one integration-focused regression test**

```python
class TestReuseAndGenerationDefaults(unittest.TestCase):
    def test_legacy_library_alias_still_works(self):
        class Args:
            mode = "auto"
            library = "bicerano"
            min_components = 2
            max_components = 2
            weight_grid = "0.5"
            architecture = "random"
            max_recipes = 1
            random_seed = 42
            data_path = "unused.parquet"

        recipes = list(load_recipe_specs_from_args(Args()))
        self.assertEqual(len(recipes), 1)
```

- [ ] **Step 2: Run the targeted regression tests**

Run: `python -m unittest tests.test_predict_tg_tabpfn_186d tests.test_virtual_copolymer_generation -v`

Expected: pass once the reuse and auto-library behavior are wired together.

- [ ] **Step 3: Run the full baseline suite**

Run: `python -m unittest discover tests -v`

Expected: `OK` with the existing skip count, plus the new tests passing.

- [ ] **Step 4: Inspect the working tree**

Run: `git status --short`

Expected: only the intended predictor, generator, plan, and test files are modified in the worktree branch.

- [ ] **Step 5: Commit**

```bash
git add scripts/predict_tg_tabpfn_186d.py src/data/virtual_copolymer_generation.py scripts/generate_virtual_copolymer_dataset.py tests/test_predict_tg_tabpfn_186d.py tests/test_virtual_copolymer_generation.py docs/superpowers/plans/2026-04-24-precomputed-feature-reuse.md
git commit -m "feat: reuse precomputed training features in generator"
```
