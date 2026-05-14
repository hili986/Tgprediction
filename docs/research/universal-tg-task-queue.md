# Universal Tg Agent Task Queue

This file is the persistent task queue for `scripts/codex_universal_tg_agent_loop.py`.

The agent loop should read this file at the start of every round, choose one bounded task, run exactly one iteration, then update `docs/research/universal-tg-iteration-log.md` and `results/universal_single_regressor/scoreboard.json`.

Default execution architecture:

```text
local Windows project = Codex controller and persistent prompt/log state
remote ~/Tgprediction = experiment execution and heavy result storage
```

When an experiment is needed, the local Codex round should connect to:

```text
ssh sheng-xiang@100.64.0.4
cd ~/Tgprediction
/home/sheng-xiang/miniconda3/envs/llm4graphgen/bin/python ...
```

The server does not need Codex CLI. It only needs the project code, data, and Python environment.

For unattended local control, prefer SSH keys. If keys are not configured, use the local helper without committing credentials:

```powershell
$env:TG_REMOTE_PASSWORD = "<local secret>"
python scripts/remote_tg_command.py --use-paramiko "pwd && git rev-parse --short HEAD"
```

## Round Decision Protocol

The loop is not a linear checklist runner. Every round must perform a fresh global analysis before choosing work.

Required sequence:

1. Read `AGENTS.md`, this queue, the iteration log, the current scoreboard, and the loop event/state files if present.
2. Summarize the current global state: best eligible model, bottleneck metric, known ineligible/leaky runs, failed hypotheses, and remaining high-value uncertainties.
3. Generate 2-4 competing next actions. Include at least one diagnostic action when the bottleneck cause is unclear, and at least one model/data intervention when there is enough evidence.
4. Rank the candidates by expected metric gain, risk of leakage/overfit, runtime cost, and whether the result will change the next decision.
5. Execute exactly one bounded chosen action.
6. Record in the iteration log: global analysis, competing hypotheses, chosen hypothesis, commands/code changes, metrics, decision, and next action.
7. If a listed task is no longer optimal, add a dated note and create a better task instead of blindly taking the next item.

This means the queue is a planning surface, not an instruction conveyor belt.

## Active Objective

Improve the current universal single-regressor Tg model while preserving one prediction path for homopolymers, general copolymers, and nucleobase / DNA-related copolymers.

Primary target:

```text
maximize min(homopolymer_holdout_R2, polyinfo_group_holdout_R2, nucleobase_group_holdout_R2)
```

Current best baseline:

```text
results/universal_single_regressor/exp56_homo_local_fox_pred_delta_nonhomo_cal_lowfox_shrink_nopure
```

Current serious bottlenecks:

- general copolymer group-holdout: R2=0.849, MAE=16.65
- nucleobase group-holdout: R2=0.817, MAE=6.27

## Candidate Tasks

### Task A: Scoreboard And Baseline Integrity

Create or update `results/universal_single_regressor/scoreboard.json` from existing experiment summaries. Confirm the current best model is still `exp45_homo_local_nopure` under the primary minimum-R2 objective.

Status note 2026-04-25 Round 1: completed for local controller state. Scoreboard created from 47 remote summaries; `exp45_homo_local_nopure` remains the best current-protocol eligible baseline after excluding historical pure-endpoint PoLyInfo runs.

Expected output:

- updated scoreboard
- iteration-log entry
- no model code changes unless a clear bug is found

### Task B: Nucleobase Error Diagnostics

Analyze `exp45_homo_local_nopure/predictions_by_split.csv`, especially nucleobase group-holdout outliers such as `AT2012-T-32`. Decide whether the bottleneck is data size, grouping protocol, endpoint prior, or model bias.

Status note 2026-04-25 Round 1: completed for local controller state. Diagnostics written under `results/universal_single_regressor/diagnostics/`. The bottleneck is a small-n leave-base-out bias problem: `AT2012-T-32` is the largest error, but T and G show systematic opposite-signed bias. Endpoint priors alone are weaker than `exp45` overall, so the next nucleobase action should test constrained residual shrinkage/calibration inside the single-regressor path rather than replacing the model with endpoint mixing.

Expected output:

- error table or diagnostic CSV under `results/universal_single_regressor/diagnostics/`
- iteration-log entry
- one next hypothesis

### Task C: PolyInfo Hard-System Diagnostics

Analyze high-error systems P900015, P900012, P900008, and P900025 under group-holdout. Check whether the error is caused by composition orientation, measurement conflict, semicrystallinity, endpoint mismatch, or model extrapolation.

Status note 2026-04-25 Round 3: completed for local controller state. Diagnostics written under `results/universal_single_regressor/diagnostics/`. P900015 is the dominant hard system (`MAE=45.12 C`, dropping it raises computed PolyInfo group R2 from `0.843781` to `0.895049`), with mixed-sign ethene/propylene errors consistent with composition-slope or phase-behavior mismatch. P900025 is underpredicted even though its endpoint Fox prior is strong (`MAE=5.30 C`), so the universal model/local residual is hurting that system. P900012 is overpredicted on low-Tg siloxane rows. The four requested hard systems were not explained by near-duplicate measurement conflicts; near-duplicate conflicts >=10 C were concentrated in P900016/P900017 instead.

Status note 2026-04-26 Round 12: exp53-anchored PolyInfo hard-system diagnostics were completed after the current baseline changed from exp45 to exp53. P900015 remains dominant under exp53 (`MAE=44.99 C`, mean error `-26.74 C`, dropping it raises PolyInfo group R2 from `0.844516` to `0.895234`). P900025 remains underpredicted (`MAE=18.03 C`) but has much smaller R2 influence. Cross-experiment comparison shows exp51 helps P900015/P900012/P900008 but was previously rejected because it hurts the primary minimum; exp54 slightly helps P900015/P900025 but hurts nucleobase versus exp53. Saved PolyInfo predictions contain non-unique `sample_id` values such as `polyinfo:nan`; before row-level feature joins, fix or audit stable PolyInfo row identifiers.

Status note 2026-04-26 Round 14: exp53 P900015 stable-row feature/phase diagnostics were completed after the Round 13 row-ID audit. P900015 remains the dominant hard system, and the endpoint Fox prior is much closer than exp53 on P900015 (`MAE=28.52 C` versus model `44.99 C`) and P900025 (`MAE=5.30 C` versus model `18.03 C`), so the learned residual path is moving some hard PolyInfo systems away from useful endpoint priors. However, P900015 has both overpredicted low-Tg/high-minor-fraction rows and underpredicted high-Tg/low-minor-fraction rows, so a system-wide intercept correction or system-specific route is unsafe. The next PolyInfo action should be a no-leak generic residual-reliability shrinkage/gate diagnostic toward endpoint/Fox priors, explicitly preserving exp53 nucleobase R2.

Status note 2026-04-26 Round 15: exp53 residual-reliability shrinkage screen was completed as a no-code diagnostic. After correcting two diagnostic script bugs, the baseline reproduced exp53 exactly (`PolyInfo group R2=0.844516`, `nucleobase group R2=0.791624`). The best target-free rule, `low_fox_m35_0.75`, shrinks non-homopolymer predictions 25% toward endpoint/Fox only when `endpoint_tg_fox_c < -35 C`; on saved exp53 predictions it would improve PolyInfo group R2 to `0.852330`, nucleobase group R2 to `0.810383`, and P900015 MAE to `40.87 C`. This is not yet eligible because it is post-hoc on saved predictions. The next action should be one real model/fold experiment implementing this generic gate inside the single-regressor path, with unit tests first and rejection if full evaluation does not beat exp53 primary min-R2.

Status note 2026-04-26 Round 16: the `low_fox_m35_0.75` rule was implemented inside `PhysicsResidualKernelRegressor` as a target-free final shrinkage stage and tested as `exp55_homo_local_fox_nonhomo_cal_lowfox_shrink_nopure`. The experiment is kept as the new best eligible baseline: primary min-R2 improved from exp53 `0.791624` to `0.810383`, PolyInfo group R2 improved from `0.844516` to `0.852330`, and nucleobase group R2 improved from `0.791624` to `0.810383`. Homopolymer holdout R2 stayed `0.887069`; PolyInfo random holdout fell from `0.932283` to `0.929629`, so the next action should diagnose shrinkage side effects and search only target-free variants that preserve exp55 nucleobase.

Status note 2026-04-26 Round 17: exp55 shrinkage side-effect/refinement screen was completed as a no-code diagnostic. The screen reproduced exp55 exactly (`homopolymer R2=0.887069`, `PolyInfo group R2=0.852330`, `nucleobase group R2=0.810383`) and evaluated 45 target-free variants. No variant improved primary min-R2 while preserving exp55 nucleobase. Stronger low-Fox shrinkage improved PolyInfo group slightly but lowered nucleobase, while broader large-delta shrinkage could improve P900025/P900008 or PolyInfo random holdout but damaged nucleobase strongly. Do not broaden the exp55 shrinkage gate unless a new no-leak feature screen first beats exp55.

Status note 2026-04-26 Round 18: exp55 remaining nucleobase bias was diagnosed with leave-base-out residual feature screens. Exp55 remains the eligible baseline (`nucleobase group R2=0.810383`, `MAE=6.379`), with T still underpredicted (`mean error=-10.27 C`) and G overpredicted (`mean error=+5.65 C`). A diagnostic-only nucleobase residual calibrator using `endpoint_tg_fox_c` plus `model_delta_vs_fox_c` reached `R2=0.901811`, `MAE=5.199`; `endpoint_tg_fox_c` plus `w_min` was close (`R2=0.895972`). This is not eligible because it is nucleobase-only fitting, but it identifies the next model-side action: test one universal non-homopolymer low-capacity residual calibration using Fox level and model-vs-Fox displacement inside each fold, and reject if full primary min-R2 does not beat exp55.

Status note 2026-04-26 Round 19: the universal non-homopolymer prediction-delta final calibration was implemented and tested as `exp56_homo_local_fox_pred_delta_nonhomo_cal_lowfox_shrink_nopure`. Keep exp56 as the new eligible baseline: primary min-R2 improved from exp55 `0.810383` to `0.817088`, driven by nucleobase group R2 improving to `0.817088` and MAE to `6.266`. Homopolymer holdout stayed `R2=0.887069`; PolyInfo group R2 regressed from `0.852330` to `0.849235`, with P900025 and P900008 worsening even though P900015/P900012 slightly improved. Do not stack more calibration terms blindly; the next action should diagnose exp56's PolyInfo side effect and screen source-balanced or lower-penalty prediction-delta variants before another model experiment.

Status note 2026-04-26 Round 20: exp56 prediction-delta final-calibration penalty variants were screened as a no-code diagnostic. The diagnostic reproduced exp56 exactly and tested `final_calibration_lambda` values `0.03`, `0.3`, and `1.0`, plus an exp55-equivalent Fox-only control. Lambda-only variants were numerically indistinguishable from exp56; the best `lambda=1.0` changed primary min-R2 by only about `+2.7e-7` and did not recover PolyInfo group R2 or P900025/P900008. Do not run a full model experiment for lambda-only exp56 variants. The next action should be a genuinely different no-leak mechanism such as source-balanced final calibration or a filtered weak-supervision/data diagnostic.

Status note 2026-04-26 Round 21: exp56 source-balanced final-calibration variants were screened as a no-code diagnostic. The diagnostic reproduced exp56 exactly (`primary min-R2=0.817087549`) and tested `source_balanced_nonhomo`, `nucleobase_x2`, `nucleobase_x4`, and `source_balanced_plus_nb2` calibration weights. The best candidate, `nucleobase_x4`, only raised primary min-R2 to `0.817307625` (`+0.000220`) while lowering PolyInfo group R2 from `0.849235` to `0.848606` and worsening P900015 MAE from `40.75 C` to `41.07 C`. Do not implement a source-balanced or nucleobase-boosted final-calibration alias by itself. The next action should pivot to a filtered weak-supervision/data diagnostic or a genuinely new non-calibration mechanism.

Expected output:

- diagnostic CSV under `results/universal_single_regressor/diagnostics/`
- iteration-log entry
- proposed filtering or feature hypothesis

### Task D: Virtual Data Filtering Hypothesis

Do not mix raw virtual data directly. Design a filtered virtual-data experiment that selects samples close to the real PolyInfo component/endpoint distribution and uses virtual labels only as weak supervision.

Status note 2026-04-26 Round 22 / loop round 10: structural/composition-near virtual filtering was tested as `exp57_filtered_virtual_polyinfo_near400_w005`. The filter selected 400 of 5000 existing HYBRID-HOMO186 virtual rows nearest to real PolyInfo rows in robust-scaled non-target feature space, excluded exact PolyInfo component-pair matches, and used `virtual-weight=0.05`. Raw primary min-R2 rose from exp56 `0.817088` to `0.827203` because nucleobase group R2 improved strongly to `0.857903`, but PolyInfo group R2 degraded from `0.849235` to `0.827203` and PolyInfo random R2 fell to `0.911590`. Under the virtual-data rule, reject promotion when real group-holdout degrades. Existing virtual feature tables have missing endpoint Tg columns for virtual rows, so a true endpoint-distribution filter requires rebuilding or annotating virtual rows with endpoint features and teacher-consistency/uncertainty checks.

Status note 2026-04-26 Round 23 / loop round 11: teacher-consistency filtered virtual weak supervision was tested as `exp58_virtual_teacher_consistent200_w002`. A real-data exp56-path model was fit on the 7652 real no-pure rows, then existing virtual5k candidates were filtered to non-leaky component pairs, broad target sanity, PolyInfo-like composition range, and `abs(virtual_label - real_model_prediction) <= 30 C`; 200 rows were selected and trained with `virtual-weight=0.02`. The filter confirmed severe label conflict in the existing virtual pool (median teacher/model disagreement `145.05 C` before filtering versus `12.65 C` selected), and it avoided exp57's large PolyInfo collapse, but it did not improve the primary objective: PolyInfo group R2 rose slightly from exp56 `0.849235` to `0.849643`, while nucleobase group R2 fell slightly from `0.817088` to `0.816982`, lowering primary min-R2. Reject promotion. Do not continue by merely downweighting existing endpoint-missing virtual rows; next virtual work should regenerate or annotate endpoint-aware virtual rows, or pivot to a non-virtual constrained model mechanism.

Expected output:

- filter script or config
- audit JSON/CSV
- one controlled experiment
- reject if real group-holdout degrades

### Task E: Constrained Multi-Kernel Search

Try one constrained multi-kernel or local residual change that does not increase free residual capacity enough to overfit group-holdout.

Status note 2026-04-26 Round 24 / loop round 12: an exp56-compatible low-weight additive physical/embedding sum-kernel diagnostic was tested without production code changes. The control reproduced exp56 exactly (`primary min-R2=0.817087549`). The candidate `exp56_additive_phys_emb_sum_low` lowered primary min-R2 to `0.815844`, PolyInfo group R2 to `0.844016`, and nucleobase group R2 to `0.815844`; P900025 improved modestly but P900015/P900012/P900008 and both primary group metrics worsened. Do not promote additive-kernel stacking on exp56. The next independent action should be endpoint-aware virtual regeneration/annotation or a different target-free local-residual reliability diagnostic.

Status note 2026-04-26 Round 25 / loop round 13: the autonomous loop was stopped with a bottleneck report after repeated post-exp56 non-improvement. Exp56 remains the best eligible baseline (`primary min-R2=0.817087549`). Since exp56, calibration penalty/source-balanced screens, filtered existing-virtual experiments, and additive-kernel diagnostics all failed to produce an eligible primary-objective improvement. Do not continue with small exp56 tuning. Resume only with a new data or representation change, preferably endpoint-aware virtual row regeneration/annotation or genuinely new no-leak copolymer phase/reliability features.

Expected output:

- one code/config change
- unit tests
- one controlled experiment
- comparison against `exp45_homo_local_nopure`

### Task F: Constrained Nucleobase Leave-Base-Out Calibration

Use the Task B diagnostics to test one bounded single-regressor change that reduces nucleobase leave-base-out bias without increasing free residual capacity. Candidate direction: shrink or gate nucleobase local residual corrections against endpoint/composition priors so G rows are not over-corrected upward and high-composition T rows retain enough slope.

Status note 2026-04-25 Round 2: scalar global nucleobase-weight shrinkage was tested as `exp50_homo_local_nopure_nw40` and rejected. Lowering nucleobase weight from 60 to 40 improved PolyInfo group R2 slightly (`0.843781 -> 0.844728`) and nucleobase MAE slightly (`6.849 -> 6.800`), but nucleobase group R2 fell (`0.788792 -> 0.775479`) because T rows became more underpredicted. Do not repeat simple scalar nucleobase-weight shrinkage; any further Task F work should be a true local-residual gate/shrinkage change or be preceded by Task C diagnostics.

Status note 2026-04-25 Round 4: global local-residual removal was tested as `exp51_homo_correction_strong_nopure` and rejected. Removing the light local residual while keeping strong homopolymer correction improved PolyInfo group R2 slightly (`0.843781 -> 0.847893`) but lowered nucleobase group R2 (`0.788792 -> 0.785098`) and homopolymer R2 (`0.887069 -> 0.878961`). Do not repeat full local-residual ablation; any future Task F work should be a targeted reliability gate/calibration rather than disabling local corrections everywhere.

Status note 2026-04-26 Round 5: exp45 nucleobase group-fold component decomposition was completed. The diagnostic reproduced saved exp45 nucleobase predictions exactly and showed that removing the local-blend delta would lower nucleobase R2 to `0.785098`. The worst T row `AT2012-T-32` changes by only `0.077 C` without the local delta, so simple local-residual shrinkage/gating is not the right next intervention. Future Task F work should target nucleobase prior/composition calibration or data augmentation/curation for leave-base-out base-family slopes rather than another local residual gate.

Status note 2026-04-26 Round 6: a no-leak nucleobase endpoint/prior calibration diagnostic was completed. For each held-out base family, affine calibrators were fit only on the other base families. The best diagnostic candidate, `linear_fox_lbo`, improved nucleobase group R2 from exp45 `0.788792` to `0.917554` and MAE from `6.849` to `4.348`. This is not an eligible final model by itself because it is a post-hoc diagnostic, but it shows the next highest-value intervention is an internal low-capacity endpoint/Fox affine calibration in the single regressor. Do not implement it as a task-specific prediction route; it must be fit inside each training fold and re-evaluated against all three primary metrics.

Status note 2026-04-26 Round 7: internal endpoint/Fox residual calibration was implemented and tested as `exp52_homo_local_fox_cal_nopure`. It reused the exp45 no-pure data, weights, light local residual, and homopolymer correction, adding only a low-capacity final residual correction on `endpoint_tg_fox_c` inside `PhysicsResidualKernelRegressor`. The experiment is kept as the new best eligible baseline because primary min-R2 improved slightly from `0.788792` to `0.789460`; however, the gain is far smaller than the Round 6 diagnostic, so the next Task F action should diagnose the dilution and test one stronger constrained calibration/composition-prior variant rather than assuming endpoint/Fox calibration is solved.

Status note 2026-04-26 Round 8: exp52 final-calibration dilution was diagnosed without changing model code. The diagnostic reproduced saved exp52 nucleobase predictions exactly. Current all-row final calibration gives only a near-constant `~0.06 C` correction on held-out nucleobase rows because calibration weight is dominated by homopolymer rows; average normalized fold weights were approximately homopolymer `5877.9`, PolyInfo `1169.9`, nucleobase `599.9`. A nucleobase-only counterfactual final calibrator on the same base predictions improves nucleobase R2 to `0.834109`, but source-balanced all-row calibration (`0.788940`) and 10x nucleobase all-row calibration (`0.791003`) remain near exp52. The next Task F model experiment should be a stronger generic calibration gate or feature interaction, not simple mild global calibration reweighting.

Status note 2026-04-26 Round 9: a non-homopolymer-gated endpoint/Fox final calibration was implemented and tested as `exp53_homo_local_fox_nonhomo_cal_nopure`. The calibration design multiplies the low-capacity Fox residual correction by a model-internal `1 - is_homopolymer` gate, preserving one regressor path while preventing homopolymer rows from dominating the final calibrator. The experiment is kept as the new best eligible baseline: primary min-R2 improved from exp52 `0.789460` to `0.791624`; PolyInfo group R2 also improved from `0.843849` to `0.844516`; homopolymer R2 returned to exp45-level `0.887069`, still far above the minimum. The gain is real but still far below the Round 6 post-hoc diagnostic, so the next Task F action should diagnose whether the remaining limit is the single Fox feature and test at most one additional low-capacity gated endpoint/composition feature variant before pivoting to PolyInfo robustness if nucleobase gains plateau.

Status note 2026-04-26 Round 10: adding `endpoint_tg_weighted_mean_c` beside `endpoint_tg_fox_c` in the same non-homopolymer-gated final calibrator was tested as `exp54_homo_local_fox_wmean_nonhomo_cal_nopure` and rejected. PolyInfo group R2 improved from exp53 `0.844516` to `0.845397`, but nucleobase group R2 fell from `0.791624` to `0.789852`, lowering the primary minimum. Do not keep stacking endpoint columns into the final calibrator. The next Task F action should be a diagnostic for a no-leak low-capacity feature or calibration form that can move overpredicted G rows downward while still lifting underpredicted T rows; if no such feature exists, pivot to PolyInfo robustness while preserving exp53 nucleobase performance.

Status note 2026-04-26 Round 11: a no-code feature-screen diagnostic refit exp53-style nucleobase leave-base-out folds for 8 low-capacity non-homopolymer-gated final-calibration forms. The diagnostic reproduced saved exp53 predictions exactly (`max abs diff=3.55e-15 C`). None beat exp53: the best non-control candidate, `fox_delta_nonhomo`, gave nucleobase R2 `0.790067` versus exp53 `0.791624`. Endpoint span/composition features tended to move G upward or T downward, the wrong direction. Do not continue endpoint/composition final-calibrator stacking unless a new no-leak diagnostic feature beats exp53 first. Pivot to PolyInfo robustness targeting P900015/P900025 while preserving exp53 nucleobase performance.

Expected output:

- one code/config change
- unit tests before any experiment
- one controlled experiment
- comparison against `exp45_homo_local_nopure`
- reject if PolyInfo group-holdout or homopolymer holdout degrades enough to lower the primary minimum-R2 objective

## Queue Rules

- Pick one task per loop round.
- Prefer diagnostics before code changes when the cause is unclear.
- If a task is completed, mark it with a dated note rather than deleting it.
- If three consecutive serious iterations do not improve the primary target, stop and write a bottleneck report.
