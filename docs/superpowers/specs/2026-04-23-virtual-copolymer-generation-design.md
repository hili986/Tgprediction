# Virtual Copolymer Data Generator Design

Date: 2026-04-23
Status: Approved in chat, written spec pending final user review

## Goal

Add a dedicated server-side script that generates virtual copolymer Tg datasets by reusing the current best predictor in this repository:

- model stack: `TabPFN v2 + 186d multiscale features`
- implementation base: `scripts/predict_tg_tabpfn_186d.py`
- predictor class: `BestTgPredictor`

The generator must keep the predictor alive for the full job lifetime so the model and preprocess artifacts are loaded once per task instead of once per sample.

## Problem Statement

The current prediction script is suitable for one-off inference and CSV batch inference, but it is not the right operational surface for large virtual-data generation jobs. When users call the CLI repeatedly from a shell loop, each command launches a new Python process, rebuilds the predictor, and repeats expensive setup work. This is acceptable for ad hoc predictions and inefficient for long-running virtual dataset generation on a server.

The new workflow needs:

1. One process for one generation task.
2. One predictor initialization per task.
3. Reuse of per-component features across many copolymer combinations.
4. Direct support for binary, multicomponent, random, and block-style entries.
5. Incremental output writing so long jobs can survive interruption and be resumed.

## Scope

In scope:

- new script: `scripts/generate_virtual_copolymer_dataset.py`
- reuse existing `BestTgPredictor`
- automatic candidate generation from internal homopolymer library
- user-supplied candidate generation from CSV
- mixed mode combining internal library plus user CSV
- binary and multicomponent copolymer recipes
- `random` and `block` architecture labels
- incremental CSV/JSONL output
- resumable execution based on stable recipe IDs

Out of scope for this task:

- training a new copolymer-specific supervised model
- changing the scientific meaning of the current copolymer approximation
- daemon/service deployment
- distributed multi-node scheduling
- database integration

## Existing Context

The current best inference path already contains the core mechanics needed for this generator:

- `BestTgPredictor.fit()` lazily builds preprocessing and the TabPFN regressor.
- `BestTgPredictor.featurize_component()` caches component-level features and embeddings in `_component_cache`.
- `BestTgPredictor.predict_multicomponent()` already supports `random` and `block` architecture modes and both binary and multicomponent composition vectors.

This means the missing piece is not a new predictor. The missing piece is an orchestration script that creates one predictor instance, reuses it for many recipes, and writes virtual data records efficiently.

## Requirements

### Functional Requirements

1. The script must support three input modes:
   - `auto`: enumerate recipes from an internal monomer library
   - `csv`: read recipes from a user-provided CSV
   - `hybrid`: combine internal monomer library with a user-provided CSV
2. The script must support:
   - binary copolymers
   - multicomponent copolymers
   - `random` architecture
   - `block` architecture
3. The script must be able to generate:
   - recipe metadata
   - normalized component weights
   - predictor outputs in K and C
   - method provenance fields from the predictor
4. The script must write results incrementally during long runs.
5. The script must be resumable without regenerating already completed recipes.

### Performance Requirements

1. Predictor initialization must happen once per script execution.
2. Repeated component featurization must hit cache within the same process.
3. Output must be flushed in chunks rather than retained entirely in memory.
4. The script must avoid materializing an unbounded Cartesian product in RAM when large enumeration settings are used.

### Usability Requirements

1. The CLI must be explicit and server-friendly.
2. The user must be able to limit generation size deterministically.
3. The user must be able to choose output format and output path.
4. Failures on individual recipes must be recorded without crashing the whole job unless the predictor cannot initialize.

## Approaches Considered

## Approach A: Extend `scripts/predict_tg_tabpfn_186d.py`

Add generation flags directly into the existing inference CLI.

Pros:

- minimal reuse overhead
- one fewer script

Cons:

- the file is already large and mixes single inference, CSV inference, and model setup
- generation concerns like enumeration, resume logic, and chunked writing would make it harder to maintain
- higher risk of breaking the existing inference surface

## Approach B: Add a dedicated generator script

Create `scripts/generate_virtual_copolymer_dataset.py` and import `BestTgPredictor` from the existing inference script.

Pros:

- keeps prediction logic and generation orchestration separate
- clean operational entrypoint for server jobs
- lower regression risk for existing CLI users
- easier to test resume logic and enumeration behavior

Cons:

- one more script to maintain

## Approach C: Long-running local service

Wrap predictor loading in a daemon or API service and submit generation jobs to it.

Pros:

- no repeated model initialization across separate jobs
- best throughput for repeated sessions

Cons:

- far more operational complexity than needed now
- adds lifecycle and failure-management burden
- not justified for the current repository stage

## Recommendation

Choose Approach B.

This isolates the server-scale virtual-data generation workflow without destabilizing the current predictor CLI. It also directly solves the observed operational issue: the predictor will be created once and kept alive until the generation task ends.

## Proposed CLI

## Entry Point

`python scripts/generate_virtual_copolymer_dataset.py ...`

## Core Arguments

- `--mode {auto,csv,hybrid}`
- `--output <path>`
- `--output-format {csv,jsonl}`
- `--resume`
- `--chunk-size <int>`
- `--device {cuda,cpu}`
- `--architecture {random,block,both}`

## Auto Mode Arguments

- `--library {bicerano}`
- `--min-components <int>`
- `--max-components <int>`
- `--weight-grid <spec>`
- `--max-recipes <int>`
- `--random-seed <int>`

Initial `--weight-grid` support will use a simple comma-separated fraction list such as:

- `0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9`

For multicomponent enumeration, candidate weight tuples are built from this fraction set and then filtered to those whose raw weights sum to a positive value before normalization. More advanced grid syntax is explicitly out of scope for the first implementation.

## CSV Mode Arguments

- `--input-csv <path>`

Accepted CSV row formats:

1. `components` column using `SMILES::weight|SMILES::weight|...`
2. indexed columns such as `smiles1,w1,smiles2,w2,smiles3,w3,...`
3. optional `architecture`
4. optional metadata columns such as `source`, `notes`, `case_id`

## Hybrid Mode Arguments

- `--input-csv <path>`
- auto-mode enumeration flags also active

Hybrid mode means a union of:

- recipes explicitly provided by the CSV
- recipes automatically enumerated from the internal library

It does not mean taking every CSV component and cross-producting it with every library component unless such rows are already present in the CSV.

## Output Schema

Each output row should include at least:

- `recipe_id`
- `mode`
- `architecture`
- `n_components`
- `components_serialized`
- `weights_serialized`
- `tg_k_pred`
- `tg_c_pred`
- `primary_method`
- `descriptor_mix_tg_k`
- `descriptor_mix_tg_c`
- `fox_reference_tg_k`
- `fox_reference_tg_c`
- `component_tg_window_k_min`
- `component_tg_window_k_max`
- `model`
- `warning`
- `status`
- `error`

If available, also include passthrough metadata:

- `source`
- `notes`
- `case_id`
- `input_origin`

## Recipe Enumeration Design

## Internal Library Source

Initial implementation will use the internal Bicerano repeat-unit list as the default monomer library because it already exists in the repository and matches the predictor input format.

## Binary Enumeration

For binary recipes:

- enumerate unique component pairs without A/B duplication
- apply the requested weight grid
- apply one or both architecture labels

Example:

- pair `(A, B)`
- weights `[0.1, 0.9]`, `[0.2, 0.8]`, ...
- architecture `random`, `block`, or both

## Multicomponent Enumeration

For multicomponent recipes:

- enumerate unique component sets up to `max-components`
- derive weight tuples from a user-specified grid
- discard invalid tuples whose normalized sum is zero
- canonicalize component ordering to stabilize recipe identity

For the first implementation, automatic enumeration should only generate recipes with at least two distinct components. Homopolymer rows are already handled by the existing predictor and are not the target of this virtual copolymer generator.

Because full combinatorics can explode, the script must support deterministic cap controls:

- `--max-recipes`
- optional seeded sampling after recipe enumeration

## Stable Recipe Identity

Each recipe will get a deterministic `recipe_id` built from:

- architecture
- ordered component SMILES
- normalized weights rounded to a fixed precision

This ID is used for:

- deduplication
- resume filtering
- stable output joins

## Runtime Architecture

## Predictor Lifetime

The script creates one `BestTgPredictor` instance and calls `fit()` once near startup. That predictor stays alive for the entire process lifetime.

## Component Cache Reuse

Recipe execution must route through `predict_multicomponent()` so the existing `_component_cache` inside `BestTgPredictor` can eliminate repeated feature and embedding work for recurring monomers.

## Chunked Execution

Execution pipeline:

1. build recipe iterator
2. skip already completed `recipe_id`s when `--resume` is enabled
3. predict recipes one by one using the shared predictor
4. accumulate `chunk-size` rows
5. flush chunk to output
6. continue until exhausted

This avoids storing all outputs in memory.

## Error Handling

### Fatal Errors

These should stop the job:

- missing required model artifacts
- predictor initialization failure
- unreadable input file
- invalid output path

### Per-Recipe Errors

These should be recorded as failed rows and the job should continue:

- invalid repeat-unit SMILES in one recipe
- graph construction failure for one component
- chain-physics failure for one component
- embedding extraction failure for one component

Each failed row should contain:

- `recipe_id` if available
- `status=error`
- `error=<message>`

## Resume Strategy

When `--resume` is passed:

1. if output file already exists, load the existing `recipe_id`s
2. skip those recipes during enumeration/execution
3. append only new results

For CSV output, the script should append rows after writing the header once.
For JSONL output, the script should append one JSON object per line.

Resume support is required for long server runs where interruption is expected.

## Testing Strategy

Add focused tests for the new script utilities, not full heavy-model inference.

Targeted tests should cover:

1. recipe ID canonicalization
2. binary recipe enumeration uniqueness
3. multicomponent weight parsing and normalization
4. resume skip behavior
5. chunk writer behavior
6. passthrough metadata retention

Heavy predictor execution can be mocked in unit tests.

## Risks and Constraints

1. The current copolymer prediction path is still an engineering approximation, especially for `block` mode. Output must preserve the warning/provenance fields and not overclaim scientific certainty.
2. Multicomponent enumeration can become combinatorially large. Cap controls are mandatory.
3. Runtime is dominated by per-component feature and embedding generation on cache misses. Reuse of recurring components is therefore central to performance.
4. The first run on a new candidate library may still be expensive even with a single predictor process because each previously unseen monomer must be featurized once.

## Acceptance Criteria

This task is complete when:

1. a dedicated generator script exists
2. one script execution uses one predictor initialization
3. the script supports `auto`, `csv`, and `hybrid`
4. the script supports binary and multicomponent recipes
5. the script supports `random` and `block`
6. the script writes incremental output
7. the script can resume from existing output
8. targeted tests pass for recipe generation and resume logic

## Implementation Notes

Preferred implementation boundary:

- keep `BestTgPredictor` in `scripts/predict_tg_tabpfn_186d.py` as the prediction engine
- put generation-specific helpers in the new script unless they prove reusable enough to move into `src/`

This minimizes churn while keeping the generation workflow isolated and maintainable.
