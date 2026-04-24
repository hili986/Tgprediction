# Precomputed Feature Reuse Design

Date: 2026-04-24
Status: Approved in chat, written spec pending final user review

## Goal

Reuse the repository's existing `7,486`-row unified training artifacts during inference and virtual copolymer generation so that previously seen repeat-unit `smiles` do not trigger on-the-fly recomputation of:

- `PHY-C-light`
- `GNN 64d`
- `polyBERT PCA 64d`

The generator must keep its current default behavior of enumerating from the internal Bicerano library, while also adding an explicit option to enumerate from the full unified `7k` library.

## Problem Statement

The current inference stack already loads the training artifacts to fit the PCA, preprocessing pipeline, and TabPFN model. However, those loaded artifacts are used only as training matrices. They are not exposed as a runtime `smiles -> feature` lookup table.

As a result:

1. `BestTgPredictor.fit()` loads the cached feature matrices for training.
2. `BestTgPredictor.featurize_component()` starts from an empty in-process `_component_cache`.
3. For any query component not already seen in the current process, the predictor recomputes:
   - chain physics when the dedicated chain cache misses
   - GNN embedding
   - polyBERT embedding + PCA projection

This is inefficient for virtual-data generation because the default `auto` library is entirely contained in `unified_tg.parquet`, so those component features already exist in the cached training artifacts and should be reused directly.

## Scope

In scope:

- add runtime lookup tables for precomputed training features inside `BestTgPredictor`
- reuse precomputed features for any query SMILES found in the unified training artifacts
- keep on-the-fly computation as a fallback for truly new SMILES
- add generator support for `--auto-library bicerano|unified`
- keep `bicerano` as the default auto-generation source
- add a safety guard so `unified` auto-enumeration requires an explicit `--max-recipes`
- add tests covering reuse behavior and auto-library selection

Out of scope:

- changing the scientific meaning of the current copolymer approximation
- training new models
- creating new disk cache formats
- attempting to fully enumerate the `7k` library by default
- optimizing polyBERT/GNN batching beyond reuse of existing cached rows

## Existing Context

The relevant current behavior is:

1. `_load_training_blocks()` aligns `unified_tg.parquet` with:
   - `feature_matrix_PHY-C.parquet`
   - `gnn_embeddings_64d.parquet`
   - `polybert_embeddings.parquet`
2. `BestTgPredictor.fit()` uses those matrices to fit:
   - PCA on raw polyBERT embeddings
   - preprocessing pipeline
   - TabPFN regressor
3. `BestTgPredictor.featurize_component()` only consults:
   - `_component_cache` for same-process reuse
   - `chain_physics_cache` for chain-physics-only reuse

This means the code already has enough information to build precomputed lookup tables. It simply does not retain them in a queryable form.

## Requirements

### Functional Requirements

1. If a query `smiles` exists in `unified_tg.parquet` and the aligned feature artifacts contain valid rows for it, inference must reuse the precomputed features instead of recomputing them.
2. If a query `smiles` does not exist in the unified training artifacts, inference must continue to compute features on the fly exactly as it does today.
3. `predict_homopolymer`, `predict_multicomponent`, CSV batch inference, and virtual-data generation must all benefit from the same reuse path.
4. The virtual-data generator must default to the Bicerano auto library.
5. The virtual-data generator must support an explicit full-library auto source based on `unified_tg.parquet`.
6. If `--auto-library unified` is used without `--max-recipes`, the script must fail fast with a clear message.

### Performance Requirements

1. No precomputed component should trigger GNN or polyBERT recomputation.
2. The lookup structure should be built once per predictor initialization.
3. Memory overhead should remain proportional to the already-loaded training artifacts and avoid duplicate full-matrix copies where possible.

### Usability Requirements

1. Default behavior must remain compatible with existing `304`-component workflows.
2. The new `7k` option must be explicit and not silently change result set size.
3. Error messages for unsafe unified enumeration must tell the user to add `--max-recipes`.

## Approaches Considered

### Approach A: Generator-only preload hack

Teach only the generator to read the caches and prewarm `_component_cache`.

Pros:

- minimal change surface
- fastest to patch for one workflow

Cons:

- single-query inference and CSV inference still miss reuse
- duplicates cache logic outside the predictor
- makes future maintenance harder

### Approach B: Predictor-level precomputed lookup table

Teach `BestTgPredictor` to build a reusable `smiles -> features` lookup from the aligned training artifacts, then let every inference path use it.

Pros:

- one correct reuse path for all inference surfaces
- no duplication between CLI and generator
- matches the actual scientific intent of reusing the trained feature corpus

Cons:

- requires touching both predictor internals and generator CLI

### Approach C: Persist a new materialized `smiles -> 186d` cache file

Generate and read a separate disk artifact dedicated to inference lookup.

Pros:

- simple runtime lookup semantics

Cons:

- adds artifact lifecycle burden
- risks divergence from the existing training caches
- unnecessary because current artifacts already contain the needed data

## Recommendation

Choose Approach B.

The predictor should own feature reuse because it is the only place shared by all inference entry points. The generator should remain an orchestration layer and only decide which candidate library to enumerate.

## Proposed Design

### 1. Precomputed Lookup in `BestTgPredictor`

During `fit()`:

1. load aligned training blocks as today
2. fit PCA and preprocessing as today
3. build a lookup table keyed by exact training `smiles`
4. for each valid row, store:
   - `phyc`
   - `gnn`
   - `pbert` after PCA projection
   - `chain_physics_source = precomputed`

This lookup should be stored on the predictor instance and reused by `featurize_component()`.

### 2. Query Resolution Order

`featurize_component(smiles)` should resolve in this order:

1. `_component_cache` hit
2. precomputed unified-feature lookup hit
3. on-the-fly computation fallback

If step 2 hits, the method should populate `_component_cache` with the reused row so later accesses stay on the fast path.

### 3. Validity Rules for Reuse

A precomputed row is eligible for reuse only if:

- the `smiles` key exists in the aligned unified data
- `phyc`, `gnn`, and projected `pbert` are all finite

Rows with invalid values must not be inserted into the lookup table. They should fall back to on-the-fly computation if queried.

### 4. Auto-Library Selection in the Generator

Add a new CLI argument:

- `--auto-library {bicerano,unified}`

Behavior:

- `bicerano`: current behavior, default
- `unified`: enumerate unique `smiles` from `data/unified_tg.parquet`

`unified` must be deduplicated by exact `smiles` string before enumeration.

### 5. Unified Enumeration Safety Guard

If the user selects:

- `--mode auto` or `--mode hybrid`
- and `--auto-library unified`
- and `--max-recipes` is not provided

the script must exit with a clear error describing that unified enumeration is too large to run without an explicit cap.

This is a safety requirement, not a warning.

## Testing Plan

1. Add predictor-level tests showing that a `smiles` present in the unified artifacts is served from precomputed features rather than triggering recomputation.
2. Add fallback tests showing unseen `smiles` still use the old computation path.
3. Add generator tests for:
   - default `bicerano` auto library
   - explicit `unified` auto library
   - fast failure when `unified` is used without `--max-recipes`

## Risks and Mitigations

### Risk: duplicate SMILES entries

Mitigation:

- unified training data currently appears unique by exact `smiles`
- still build lookup defensively and let later duplicate insertion overwrite only if the row is equally valid

### Risk: row-order alignment differs between artifacts

Mitigation:

- preserve the existing `_align_block()` behavior used by training
- build the lookup only after alignment is complete

### Risk: memory growth from storing both matrices and lookup rows

Mitigation:

- build the lookup from already loaded arrays
- store only per-row vectors needed for runtime reuse
- do not materialize an additional full 186d file on disk

## Success Criteria

This task is complete when:

1. default virtual generation from the Bicerano library no longer recomputes features for rows already present in unified training artifacts
2. single-query and batch inference also benefit from the same reuse mechanism
3. `--auto-library unified` exists and is guarded by explicit `--max-recipes`
4. tests cover reuse, fallback, and auto-library selection behavior
