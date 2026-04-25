# Universal Single-Regressor Polymer Tg Model Design

Date: 2026-04-25
Status: Approved in chat, written spec pending final user review

## Goal

Build one universal Tg regression system that uses a single final regressor to predict:

- homopolymer Tg
- random, multicomponent, and block copolymer Tg
- nucleobase-functional polymer Tg

The system should preserve broad applicability instead of using separate production routes for each material family. The target is high accuracy across all families, with `R2 = 0.95` treated as an aspirational research target rather than a guaranteed metric.

## Problem Statement

The current repository has strong but separate capabilities:

- homopolymer prediction with 186d multiscale features and TabPFN
- general copolymer prediction with endpoint physics and same-system calibration
- nucleobase copolymer prediction with actual endpoint Tg physics

This is scientifically useful, but it is not the single-regressor model the user now wants. The new task is to express all polymers as the same supervised learning problem:

```text
Tg = f(component structures, composition, architecture, intrinsic descriptors, physical priors)
```

The final prediction must come from one trained regressor, not from task-specific routing logic.

## Design Principle

Every sample is converted into a component-set representation.

```text
polymer = {
  components: [repeat_unit_1, repeat_unit_2, ...],
  weights: [w1, w2, ...],
  architecture: homo | random | block | multicomponent,
  target_tg_c: measured_or_virtual_tg
}
```

Homopolymers are represented as a special case:

```text
n_components = 1
w1 = 1
composition_entropy = 0
endpoint_tg = homopolymer_tg_or_endpoint_estimate
```

This keeps one schema for all material classes.

## Scope

In scope:

- build a unified training table for homopolymers, virtual copolymers, clean real copolymers, and nucleobase data
- train one final regressor on that table
- support sample weighting so real experimental data can dominate virtual labels
- evaluate homopolymer, general copolymer, and nucleobase subsets separately
- create one prediction script for the trained single-regressor model
- run iterative experiments and save summaries under `results/`

Out of scope for this stage:

- claiming `R2 = 0.95` unless a leakage-safe split supports it
- replacing all existing route-based scripts immediately
- pretending block copolymers with phase separation always have one physical Tg
- distributed training infrastructure

## Data Sources

Initial training and evaluation should use the already available project resources:

- `data/unified_tg.parquet`
- cached PHY-C, GNN, and polyBERT-derived homopolymer features
- generated virtual copolymer files under `results/virtual_data/`
- clean PolyInfo copolymer table at `data/external/polyinfo_copolymer_tg_clean.csv`
- clean copolymer endpoint-physics details under `results/copolymer_residual_model/`
- nucleobase Excel-derived predictions and strategy details under `results/copolymer_residual_model/` or `results/universal_router_fulltest/`

The implementation must tolerate missing optional files and report which sources were used.

## Unified Feature Representation

Each sample should produce fixed-length features from these groups.

### Component Intrinsic Features

For each component, use available intrinsic descriptors:

- PHY-C or RDKit descriptors
- GNN embedding
- polyBERT PCA embedding
- hydrogen-bond donor and acceptor counts when available
- aromaticity and heteroatom indicators when available

For multi-component polymers, aggregate component features with permutation-invariant statistics:

- weighted mean
- weighted standard deviation
- min and max
- pairwise absolute contrast for binary systems

### Composition Features

Add composition-only fields:

- `n_components`
- `w_max`
- `w_min`
- `w_entropy`
- `w_herfindahl`
- sorted top-k weights with zero padding

### Architecture Features

Encode architecture as model features, not routes:

- `is_homopolymer`
- `is_random`
- `is_block`
- `is_multicomponent`

### Physical Prior Features

Physical priors are allowed because they are derived from material properties and composition. They must be features only, not separate prediction routes.

Include:

- endpoint Tg min, max, mean, and weighted mean
- endpoint Tg contrast
- Fox Tg estimate when endpoint Tg values are available
- linear mixing Tg estimate
- missing-endpoint indicators

Endpoint Tg values may come from measured homopolymer data where available or from the existing homopolymer predictor/cache for unknown components. The source must be recorded.

## Model Choice

The first implementation should compare several single-regressor candidates but ship one selected model:

- CatBoostRegressor, if installed
- HistGradientBoostingRegressor
- ExtraTreesRegressor
- XGBoost or LightGBM, if already installed
- TabPFNRegressor only if runtime remains manageable

Recommended initial production candidate:

```text
CatBoostRegressor single model
```

Fallback if CatBoost is unavailable:

```text
ExtraTreesRegressor or HistGradientBoostingRegressor
```

The final selected model must be trained as one regressor on the unified feature matrix.

## Sample Weighting

Virtual copolymer labels are teacher-model outputs, not independent experimental observations. They should broaden structural and composition coverage but should not dominate real data.

Initial weights:

```text
real homopolymer:       1.0
virtual copolymer:      0.1 to 0.3
real PolyInfo copolymer: 5.0 to 20.0
real nucleobase data:   10.0 to 30.0
```

Experiments should sweep these weights and record per-family metrics.

## Evaluation Protocol

The model must report both aggregate and family-specific metrics.

Required evaluations:

- homopolymer heldout split
- general copolymer leave-system-out, for new-system generalization
- general copolymer known-system composition split, for interpolation
- nucleobase leave-base-out or LOOCV, because the sample count is small

Metrics:

- `MAE`
- `RMSE`
- `R2`
- number of predicted rows
- number of skipped/error rows

Any metric using the same row in calibration/training and evaluation must be labeled as replay, not fair generalization.

## Training Workflow

The first training script should support:

```bash
python scripts/train_universal_tg_single_regressor.py \
  --output-dir results/universal_single_regressor \
  --device cuda
```

Recommended outputs:

- `unified_training_table.parquet`
- `feature_columns.json`
- `model.joblib`
- `summary.json`
- `predictions_by_split.csv`
- `experiment_log.md`

The script should cache expensive feature tables and reuse them across iterations.

## Prediction Workflow

The final prediction script should support:

```bash
python scripts/predict_tg_universal_single_regressor.py \
  --input-csv data/query.csv \
  --model-dir results/universal_single_regressor/best \
  --output results/predictions/universal_single_regressor.csv
```

The prediction script should accept:

- one-component homopolymer rows
- binary copolymer rows
- multicomponent rows
- architecture labels
- optional endpoint Tg overrides

It should output:

- predicted Tg in C and K
- feature coverage diagnostics
- endpoint source diagnostics
- model version
- uncertainty proxy if available

## Iteration Strategy

Run explicit guess-experiment-summary loops.

### Iteration 1

Hypothesis: physical priors plus component-set descriptors can match the current route-based baselines while using one regressor.

Experiment:

- train on real homopolymer + clean real copolymer + nucleobase data
- no virtual data or low virtual weight

### Iteration 2

Hypothesis: virtual copolymers improve composition smoothness without hurting real data if down-weighted.

Experiment:

- add virtual copolymer data
- sweep virtual weight
- compare per-family metrics

### Iteration 3

Hypothesis: real copolymer and nucleobase data need higher weights to correct teacher-model bias.

Experiment:

- sweep real copolymer and nucleobase sample weights
- keep homopolymer performance from collapsing

### Iteration 4

Hypothesis: endpoint-source quality explains many outliers.

Experiment:

- add endpoint source flags
- compare measured endpoint, cached endpoint, and predicted endpoint behavior

## Risks

1. A single regressor may underperform the route-based system on some families because routes encode domain-specific assumptions.
2. `R2 = 0.95` may be impossible for leakage-safe new-system copolymer prediction with the current real data volume.
3. Virtual labels can amplify the teacher model's bias if weighted too strongly.
4. Nucleobase data has very small sample count, so high `R2` can be unstable.
5. Block copolymers can have multiple Tg transitions; a scalar target may be physically incomplete.

## Acceptance Criteria

This stage is complete when:

1. a unified table builder exists
2. one final regressor is trained on the unified table
3. the training script saves the model, feature schema, predictions, and metrics
4. the prediction script uses that single model for homopolymers, copolymers, and nucleobase polymers
5. the result summary reports separate metrics for homopolymer, general copolymer, and nucleobase subsets
6. at least one guess-experiment-summary iteration is recorded in `results/`

## Implementation Boundaries

Prefer adding new scripts instead of destabilizing current route-based scripts:

- `scripts/train_universal_tg_single_regressor.py`
- `scripts/predict_tg_universal_single_regressor.py`

Existing route-based scripts remain as baselines and data providers.
