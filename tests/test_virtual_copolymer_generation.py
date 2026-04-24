import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.virtual_copolymer_generation import (
    RecipeSpec,
    append_result_rows,
    build_recipe_spec,
    canonicalize_recipe,
    get_unified_auto_library,
    iter_auto_recipe_specs,
    load_completed_recipe_ids,
    load_csv_recipe_specs,
    load_recipe_specs_from_args,
    make_recipe_id,
    merge_recipe_sources,
    parse_weight_grid,
    run_generation_job,
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


class FakePredictor:
    def __init__(self):
        self.fit_calls = 0
        self.predict_calls = 0

    def fit(self):
        self.fit_calls += 1

    def predict_multicomponent(self, smiles_list, weights, architecture="random"):
        self.predict_calls += 1
        return {
            "mode": "binary_copolymer" if len(smiles_list) == 2 else "multicomponent_copolymer",
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


class FakeBatchPredictor(FakePredictor):
    def __init__(self):
        super().__init__()
        self.batch_predict_calls = 0

    def predict_multicomponent_batch(self, requests):
        self.batch_predict_calls += 1
        return [
            {
                "mode": "binary_copolymer" if len(smiles_list) == 2 else "multicomponent_copolymer",
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
            for smiles_list, weights, architecture in requests
        ]


class FakeCachedErrorBatchPredictor(FakeBatchPredictor):
    def __init__(self):
        super().__init__()
        self.errors_by_component = {"*CC(*)": "cached bad component"}

    def get_component_error(self, component):
        return self.errors_by_component.get(component)


class TestJobLoop(unittest.TestCase):
    def test_job_uses_one_predictor_fit_for_many_rows(self):
        predictor = FakePredictor()
        recipes = [
            build_recipe_spec(
                ("*CC(*)", "*CO(*)"),
                (0.6, 0.4),
                architecture="random",
                input_origin="auto",
                metadata={},
            ),
            build_recipe_spec(
                ("*CC(*)", "*CN(*)"),
                (0.6, 0.4),
                architecture="random",
                input_origin="auto",
                metadata={},
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "virtual.csv"
            stats = run_generation_job(
                predictor=predictor,
                recipes=recipes,
                output_path=output,
                output_format="csv",
                chunk_size=1,
                resume=False,
            )
            self.assertEqual(stats["written"], 2)
            self.assertEqual(predictor.fit_calls, 1)
            self.assertEqual(predictor.predict_calls, 2)

    def test_resume_skips_all_rows_without_fitting_predictor(self):
        predictor = FakePredictor()
        recipe = build_recipe_spec(
            ("*CC(*)", "*CO(*)"),
            (0.6, 0.4),
            architecture="random",
            input_origin="auto",
            metadata={},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "virtual.csv"
            append_result_rows(
                path,
                [{"recipe_id": make_recipe_id(recipe), "status": "ok"}],
                output_format="csv",
            )
            stats = run_generation_job(
                predictor=predictor,
                recipes=[recipe],
                output_path=path,
                output_format="csv",
                chunk_size=1,
                resume=True,
            )
            self.assertEqual(stats["written"], 0)
            self.assertEqual(stats["skipped_existing"], 1)
            self.assertEqual(predictor.fit_calls, 0)

    def test_job_uses_batch_predictor_once_per_chunk_when_available(self):
        predictor = FakeBatchPredictor()
        recipes = [
            build_recipe_spec(
                ("*CC(*)", "*CO(*)"),
                (0.6, 0.4),
                architecture="random",
                input_origin="auto",
                metadata={},
            ),
            build_recipe_spec(
                ("*CO(*)", "*CN(*)"),
                (0.6, 0.4),
                architecture="random",
                input_origin="auto",
                metadata={},
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "virtual.csv"
            stats = run_generation_job(
                predictor=predictor,
                recipes=recipes,
                output_path=output,
                output_format="csv",
                chunk_size=2,
                resume=False,
            )
            self.assertEqual(stats["written"], 2)
            self.assertEqual(predictor.fit_calls, 1)
            self.assertEqual(predictor.batch_predict_calls, 1)
            self.assertEqual(predictor.predict_calls, 0)

    def test_job_skips_cached_component_errors_before_batch(self):
        predictor = FakeCachedErrorBatchPredictor()
        recipes = [
            build_recipe_spec(
                ("*CC(*)", "*CO(*)"),
                (0.6, 0.4),
                architecture="random",
                input_origin="auto",
                metadata={},
            ),
            build_recipe_spec(
                ("*CO(*)", "*CN(*)"),
                (0.6, 0.4),
                architecture="random",
                input_origin="auto",
                metadata={},
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "virtual.csv"
            stats = run_generation_job(
                predictor=predictor,
                recipes=recipes,
                output_path=output,
                output_format="csv",
                chunk_size=2,
                resume=False,
            )
            rows = pd.read_csv(output)
            self.assertEqual(stats["written"], 2)
            self.assertEqual(stats["errors"], 1)
            self.assertEqual(predictor.batch_predict_calls, 1)
            self.assertEqual(set(rows["status"]), {"ok", "error"})


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

    def test_csv_indexed_columns_build_recipe_specs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "input.csv"
            path.write_text(
                "case_id,architecture,smiles1,w1,smiles2,w2\n"
                "row1,random,*CC(*),0.7,*CO(*),0.3\n",
                encoding="utf-8",
            )
            recipes = load_csv_recipe_specs(path)
            self.assertEqual(len(recipes), 1)
            self.assertEqual(recipes[0].weights, (0.7, 0.3))
            self.assertEqual(recipes[0].metadata["case_id"], "row1")

    def test_hybrid_mode_is_union_not_cross_product(self):
        auto_recipes = [
            build_recipe_spec(
                ("*CC(*)", "*CO(*)"),
                (0.5, 0.5),
                architecture="random",
                input_origin="auto",
                metadata={},
            )
        ]
        csv_recipes = [
            build_recipe_spec(
                ("*CC(*)", "*CN(*)"),
                (0.5, 0.5),
                architecture="block",
                input_origin="csv",
                metadata={"case_id": "c1"},
            )
        ]
        merged = merge_recipe_sources(auto_recipes, csv_recipes)
        self.assertEqual(len(merged), 2)


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


if __name__ == "__main__":
    unittest.main()
