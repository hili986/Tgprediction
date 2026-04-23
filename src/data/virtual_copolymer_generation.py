from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from src.data.bicerano_tg_dataset import BICERANO_DATA


SUPPORTED_ARCHITECTURES = ("random", "block")
COMPONENT_COL_RE = re.compile(r"^(?:smiles|component)_?(\d+)$", re.IGNORECASE)
WEIGHT_COL_RE = re.compile(r"^(?:w|weight)_?(\d+)$", re.IGNORECASE)
DEFAULT_CSV_FIELDNAMES = [
    "recipe_id",
    "status",
    "error",
    "mode",
    "input_origin",
    "architecture",
    "n_components",
    "components_serialized",
    "weights_serialized",
    "tg_k_pred",
    "tg_c_pred",
    "primary_method",
    "descriptor_mix_tg_k",
    "descriptor_mix_tg_c",
    "fox_reference_tg_k",
    "fox_reference_tg_c",
    "component_tg_window_k_min",
    "component_tg_window_k_max",
    "model",
    "warning",
    "case_id",
    "source",
    "notes",
    "metadata_json",
]


@dataclass(frozen=True)
class RecipeSpec:
    components: Tuple[str, ...]
    weights: Tuple[float, ...]
    architecture: str
    input_origin: str
    metadata: Dict[str, object]


def _validate_component_smiles(smiles: str) -> str:
    text = str(smiles or "").strip()
    if not text:
        raise ValueError("Component SMILES is empty.")
    if text.count("*") < 2:
        raise ValueError(
            f"SMILES '{text}' does not look like a repeat-unit SMILES with two attachment points."
        )
    return text


def _normalize_weights(weights: Sequence[float], decimals: int = 8) -> Tuple[float, ...]:
    if not weights:
        raise ValueError("Weights must be non-empty.")
    arr = [float(value) for value in weights]
    if any(not math.isfinite(value) for value in arr):
        raise ValueError("Weights contain NaN/inf.")
    if any(value < 0 for value in arr):
        raise ValueError("Weights must be non-negative.")
    total = sum(arr)
    if total <= 0:
        raise ValueError("Weights must sum to a positive value.")
    return tuple(round(value / total, decimals) for value in arr)


def parse_weight_grid(spec: str) -> Tuple[float, ...]:
    values: List[float] = []
    for chunk in (spec or "").split(","):
        text = chunk.strip()
        if not text:
            continue
        value = float(text)
        if not math.isfinite(value) or value <= 0 or value >= 1:
            raise ValueError("weight-grid values must be finite and in (0, 1).")
        values.append(value)
    if not values:
        raise ValueError("weight-grid must contain at least one comma-separated value.")
    return tuple(values)


def expand_architecture_choice(choice: str) -> Tuple[str, ...]:
    text = str(choice or "random").strip().lower()
    if text == "both":
        return SUPPORTED_ARCHITECTURES
    if text not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unsupported architecture '{text}'. Expected one of: {SUPPORTED_ARCHITECTURES + ('both',)}."
        )
    return (text,)


def canonicalize_recipe(
    components: Sequence[str],
    weights: Sequence[float],
    decimals: int = 8,
) -> Tuple[Tuple[str, ...], Tuple[float, ...]]:
    if len(components) != len(weights):
        raise ValueError("Number of components and weights must match.")
    normalized = _normalize_weights(weights, decimals=decimals)
    paired = sorted(
        zip((_validate_component_smiles(component) for component in components), normalized),
        key=lambda item: item[0],
    )
    ordered_components = tuple(component for component, _ in paired)
    ordered_weights = tuple(round(weight, decimals) for _, weight in paired)
    return ordered_components, ordered_weights


def build_recipe_spec(
    components: Sequence[str],
    weights: Sequence[float],
    architecture: str,
    input_origin: str,
    metadata: Optional[Dict[str, object]] = None,
) -> RecipeSpec:
    clean_arch = str(architecture or "").strip().lower()
    if clean_arch not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unsupported architecture '{clean_arch}'. Expected one of: {SUPPORTED_ARCHITECTURES}."
        )
    ordered_components, ordered_weights = canonicalize_recipe(components, weights)
    return RecipeSpec(
        components=ordered_components,
        weights=ordered_weights,
        architecture=clean_arch,
        input_origin=str(input_origin or "").strip().lower() or "unknown",
        metadata=dict(metadata or {}),
    )


def make_recipe_id(recipe: RecipeSpec) -> str:
    components, weights = canonicalize_recipe(recipe.components, recipe.weights)
    joined_components = "|".join(components)
    joined_weights = "|".join(f"{weight:.8f}" for weight in weights)
    return f"{recipe.architecture}::{joined_components}::{joined_weights}"


def serialize_components(components: Sequence[str]) -> str:
    return "|".join(str(component) for component in components)


def serialize_weights(weights: Sequence[float]) -> str:
    return "|".join(f"{float(weight):.8f}" for weight in weights)


def parse_component_specs(specs: Sequence[str]) -> Tuple[Tuple[str, ...], Tuple[float, ...]]:
    if not specs:
        raise ValueError("No component entries provided.")

    components: List[str] = []
    raw_weights: List[Optional[float]] = []
    for spec in specs:
        text = str(spec or "").strip()
        if not text:
            continue
        if "::" in text:
            smiles_part, weight_part = text.rsplit("::", 1)
            components.append(_validate_component_smiles(smiles_part.strip()))
            raw_weights.append(float(weight_part.strip()))
        else:
            components.append(_validate_component_smiles(text))
            raw_weights.append(None)

    if not components:
        raise ValueError("No valid component entries provided.")

    has_explicit = [weight is not None for weight in raw_weights]
    if any(has_explicit) and not all(has_explicit):
        raise ValueError("Either provide weights for all components or for none of them.")

    weights = [float(weight) for weight in raw_weights] if all(has_explicit) else [1.0] * len(components)
    return tuple(components), tuple(weights)


def get_default_auto_library() -> List[Dict[str, str]]:
    return [{"name": name, "smiles": smiles} for name, smiles, _, _ in BICERANO_DATA]


def iter_auto_recipe_specs(
    library: Sequence[dict],
    min_components: int,
    max_components: int,
    weight_grid: Sequence[float],
    architectures: Sequence[str],
    max_recipes: Optional[int],
    random_seed: int,
) -> Iterator[RecipeSpec]:
    del random_seed
    min_components = max(int(min_components), 2)
    max_components = int(max_components)
    if max_components < min_components:
        raise ValueError("max-components must be >= min-components.")

    unique_by_smiles: Dict[str, dict] = {}
    for entry in library:
        smiles = _validate_component_smiles(entry["smiles"])
        if smiles not in unique_by_smiles:
            unique_by_smiles[smiles] = dict(entry)
    ordered_library = [unique_by_smiles[smiles] for smiles in sorted(unique_by_smiles)]

    clean_architectures = tuple(expand_architecture_choice(arch)[0] for arch in architectures)
    grid = tuple(parse_weight_grid(",".join(str(value) for value in weight_grid)))

    emitted = 0
    seen_ids = set()

    for n_components in range(min_components, max_components + 1):
        for combo in combinations(ordered_library, n_components):
            component_smiles = tuple(entry["smiles"] for entry in combo)
            component_names = tuple(str(entry.get("name", "")) for entry in combo)

            if n_components == 2:
                weight_sets = [(value, 1.0 - value) for value in grid]
            else:
                seen_weight_sets = set()
                weight_sets = []
                for raw in product(grid, repeat=n_components):
                    try:
                        normalized = _normalize_weights(raw)
                    except ValueError:
                        continue
                    if normalized in seen_weight_sets:
                        continue
                    seen_weight_sets.add(normalized)
                    weight_sets.append(normalized)

            for architecture in clean_architectures:
                for weights in weight_sets:
                    metadata = {
                        "source": "bicerano_auto",
                        "component_names_serialized": serialize_components(component_names),
                    }
                    recipe = build_recipe_spec(
                        component_smiles,
                        weights,
                        architecture=architecture,
                        input_origin="auto",
                        metadata=metadata,
                    )
                    recipe_id = make_recipe_id(recipe)
                    if recipe_id in seen_ids:
                        continue
                    seen_ids.add(recipe_id)
                    yield recipe
                    emitted += 1
                    if max_recipes is not None and emitted >= int(max_recipes):
                        return


def _csv_row_metadata(row: Dict[str, str]) -> Dict[str, object]:
    excluded = {"architecture", "components"}
    metadata: Dict[str, object] = {}
    for key, value in row.items():
        text = "" if value is None else str(value).strip()
        if key in excluded or not text:
            continue
        if COMPONENT_COL_RE.match(key) or WEIGHT_COL_RE.match(key):
            continue
        metadata[key] = text
    return metadata


def _indexed_components_from_row(row: Dict[str, str]) -> Tuple[Tuple[str, ...], Tuple[float, ...]]:
    indexed_components: Dict[int, str] = {}
    indexed_weights: Dict[int, float] = {}
    for key, value in row.items():
        text = "" if value is None else str(value).strip()
        if not text:
            continue
        match_component = COMPONENT_COL_RE.match(key)
        if match_component:
            indexed_components[int(match_component.group(1))] = text
            continue
        match_weight = WEIGHT_COL_RE.match(key)
        if match_weight:
            indexed_weights[int(match_weight.group(1))] = float(text)

    if not indexed_components:
        return tuple(), tuple()

    ordered_indices = sorted(indexed_components)
    components = [indexed_components[index] for index in ordered_indices]
    has_any_weight = any(index in indexed_weights for index in ordered_indices)
    has_all_weight = all(index in indexed_weights for index in ordered_indices)

    if has_any_weight and not has_all_weight:
        raise ValueError("Partial component weights found. Provide all indexed weights or none of them.")

    weights = [indexed_weights[index] for index in ordered_indices] if has_all_weight else [1.0] * len(components)
    return tuple(components), tuple(weights)


def load_csv_recipe_specs(path: Path) -> List[RecipeSpec]:
    recipes: List[RecipeSpec] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_arch = str(row.get("architecture", "random")).strip().lower() or "random"
            architectures = expand_architecture_choice(raw_arch)
            metadata = _csv_row_metadata(row)

            components: Tuple[str, ...]
            weights: Tuple[float, ...]

            components_field = "" if row.get("components") is None else str(row.get("components")).strip()
            if components_field:
                specs = [part.strip() for part in components_field.split("|") if part.strip()]
                components, weights = parse_component_specs(specs)
            else:
                components, weights = _indexed_components_from_row(row)
                if not components:
                    smiles_field = "" if row.get("smiles") is None else str(row.get("smiles")).strip()
                    if not smiles_field:
                        raise ValueError("CSV row must provide either components or indexed smiles columns.")
                    components = (smiles_field,)
                    weights = (1.0,)

            for architecture in architectures:
                recipes.append(
                    build_recipe_spec(
                        components,
                        weights,
                        architecture=architecture,
                        input_origin="csv",
                        metadata=metadata,
                    )
                )
    return recipes


def merge_recipe_sources(
    auto_recipes: Iterable[RecipeSpec],
    csv_recipes: Iterable[RecipeSpec],
) -> List[RecipeSpec]:
    merged: Dict[str, RecipeSpec] = {}
    for recipe in list(auto_recipes) + list(csv_recipes):
        merged[make_recipe_id(recipe)] = recipe
    return list(merged.values())


def load_completed_recipe_ids(path: Path, output_format: str) -> set[str]:
    if not path.exists():
        return set()
    if output_format == "jsonl":
        recipe_ids = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("recipe_id"):
                recipe_ids.add(str(payload["recipe_id"]))
        return recipe_ids
    if output_format != "csv":
        raise ValueError("output_format must be 'csv' or 'jsonl'.")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            row["recipe_id"]
            for row in csv.DictReader(handle)
            if row.get("recipe_id")
        }


def _ensure_csv_row(row: Dict[str, object]) -> Dict[str, object]:
    ensured = {field: "" for field in DEFAULT_CSV_FIELDNAMES}
    ensured.update(row)
    return ensured


def append_result_rows(path: Path, rows: Sequence[Dict[str, object]], output_format: str) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "jsonl":
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    if output_format != "csv":
        raise ValueError("output_format must be 'csv' or 'jsonl'.")
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEFAULT_CSV_FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(_ensure_csv_row(row))


def flatten_prediction_row(
    recipe: RecipeSpec,
    result: Dict[str, object],
    status: str = "ok",
    error: str = "",
) -> Dict[str, object]:
    metadata_json = json.dumps(recipe.metadata, ensure_ascii=False, sort_keys=True)
    component_window = result.get("component_tg_window_k") or [None, None]
    return {
        "recipe_id": make_recipe_id(recipe),
        "status": status,
        "error": error,
        "mode": result.get("mode", "multicomponent_copolymer"),
        "input_origin": recipe.input_origin,
        "architecture": recipe.architecture,
        "n_components": len(recipe.components),
        "components_serialized": serialize_components(recipe.components),
        "weights_serialized": serialize_weights(recipe.weights),
        "tg_k_pred": result.get("tg_k_pred"),
        "tg_c_pred": result.get("tg_c_pred"),
        "primary_method": result.get("primary_method"),
        "descriptor_mix_tg_k": result.get("descriptor_mix_tg_k"),
        "descriptor_mix_tg_c": result.get("descriptor_mix_tg_c"),
        "fox_reference_tg_k": result.get("fox_reference_tg_k"),
        "fox_reference_tg_c": result.get("fox_reference_tg_c"),
        "component_tg_window_k_min": component_window[0],
        "component_tg_window_k_max": component_window[1],
        "model": result.get("model"),
        "warning": result.get("warning", ""),
        "case_id": recipe.metadata.get("case_id", ""),
        "source": recipe.metadata.get("source", ""),
        "notes": recipe.metadata.get("notes", ""),
        "metadata_json": metadata_json,
    }


def run_generation_job(
    predictor,
    recipes: Iterable[RecipeSpec],
    output_path: Path,
    output_format: str,
    chunk_size: int,
    resume: bool,
) -> Dict[str, int]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    seen_recipe_ids = load_completed_recipe_ids(output_path, output_format) if resume else set()
    fit_called = False
    buffer: List[Dict[str, object]] = []
    stats = {"written": 0, "errors": 0, "skipped_existing": 0}

    for recipe in recipes:
        recipe_id = make_recipe_id(recipe)
        if recipe_id in seen_recipe_ids:
            stats["skipped_existing"] += 1
            continue

        if not fit_called:
            predictor.fit()
            fit_called = True

        seen_recipe_ids.add(recipe_id)
        try:
            result = predictor.predict_multicomponent(
                list(recipe.components),
                list(recipe.weights),
                architecture=recipe.architecture,
            )
            row = flatten_prediction_row(recipe, result, status="ok", error="")
        except Exception as exc:
            stats["errors"] += 1
            row = flatten_prediction_row(recipe, {}, status="error", error=str(exc))

        buffer.append(row)
        if len(buffer) >= chunk_size:
            append_result_rows(output_path, buffer, output_format)
            stats["written"] += len(buffer)
            buffer.clear()

    if buffer:
        append_result_rows(output_path, buffer, output_format)
        stats["written"] += len(buffer)

    return stats


def load_recipe_specs_from_args(args) -> Iterable[RecipeSpec]:
    auto_recipes: Iterable[RecipeSpec] = []
    csv_recipes: List[RecipeSpec] = []

    if args.mode in {"auto", "hybrid"}:
        if getattr(args, "library", "bicerano") != "bicerano":
            raise ValueError("Only the 'bicerano' auto library is supported.")
        auto_recipes = iter_auto_recipe_specs(
            library=get_default_auto_library(),
            min_components=args.min_components,
            max_components=args.max_components,
            weight_grid=parse_weight_grid(args.weight_grid),
            architectures=expand_architecture_choice(args.architecture),
            max_recipes=args.max_recipes,
            random_seed=args.random_seed,
        )

    if args.mode in {"csv", "hybrid"}:
        input_csv = getattr(args, "input_csv", None)
        if not input_csv:
            raise ValueError("--input-csv is required for csv and hybrid modes.")
        csv_recipes = load_csv_recipe_specs(Path(input_csv))

    if args.mode == "hybrid":
        return merge_recipe_sources(auto_recipes, csv_recipes)
    if args.mode == "csv":
        return csv_recipes
    return auto_recipes
