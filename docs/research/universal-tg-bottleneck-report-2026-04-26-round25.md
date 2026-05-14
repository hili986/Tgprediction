# Universal Tg Bottleneck Report - 2026-04-26 Round 25 / Loop Round 13

- date: 2026-04-26T09:54:59+08:00
- AGENTS.md SHA256: 4da4e3a7c7cac29fdb9bf3c5fe68b630a1a970fe3f716941d922810dde9d84a4
- local controller git hash: 8b66bec
- remote runner git hash: 5f063be
- current best eligible model: `exp56_homo_local_fox_pred_delta_nonhomo_cal_lowfox_shrink_nopure`
- current primary min-R2: 0.817087549

## Best Metrics

Current exp56 primary metrics:

- homopolymer random holdout: R2=0.887069, MAE=27.164 C
- PolyInfo group holdout: R2=0.849235, MAE=16.654 C
- nucleobase group holdout: R2=0.817088, MAE=6.266 C

The limiting metric is nucleobase group-holdout R2. PolyInfo group-holdout is the second bottleneck, with P900015 still dominating PolyInfo error.

## What Was Tried Since The Last Improvement

Exp56 was the last eligible improvement, raising primary min-R2 from exp55 0.810383 to 0.817088 by adding universal non-homopolymer prediction-delta final calibration.

After exp56, the loop tested independent branches:

- Final-calibration penalty variants: lambda-only changes were numerical noise; best changed primary min-R2 by about +2.7e-7 and did not recover PolyInfo.
- Source-balanced or nucleobase-boosted final calibration: best diagnostic raised primary min-R2 only to 0.817308 while lowering PolyInfo group R2 and worsening P900015.
- Filtered virtual weak supervision, structural-near: nucleobase improved to R2=0.857903, but PolyInfo group R2 collapsed to 0.827203, so it was rejected by the virtual-data rule.
- Teacher-consistency filtered virtual weak supervision: PolyInfo group R2 slightly improved to 0.849643, but nucleobase fell to 0.816982, lowering the primary objective.
- Low-weight additive physical/embedding sum-kernel diagnostic: primary min-R2 fell to 0.815844; both PolyInfo and nucleobase group metrics worsened.

## Why This Is Blocked

Three or more serious post-exp56 iterations failed to improve the primary objective. The remaining bottleneck is not responding to small calibration-weight, penalty, additive-kernel, or existing-virtual-row filtering changes.

The evidence points to a data/representation limit:

- Nucleobase has only 17 group-holdout rows and systematic leave-base-out family bias. T remains underpredicted and G overpredicted.
- Existing virtual rows are endpoint-incomplete and composition-narrow. High-Tg virtual rows can move nucleobase, but without endpoint-aware filtering they damage real PolyInfo group-holdout.
- PolyInfo hard systems, especially P900015, contain composition/phase behavior that current endpoint and residual features do not model reliably. Generic shrinkage helped, but broader shrinkage trades off against nucleobase.

## Needed Next Change

The next useful work should not be another small tweak of exp56. It should be one of:

- rebuild or annotate endpoint-aware virtual rows, including endpoint Tg features, teacher consistency, and uncertainty checks before training;
- add genuinely new no-leak features for copolymer phase/composition reliability, then screen them diagnostically before model training;
- expand or curate nucleobase/DNA-related copolymer data so leave-base-out calibration is not determined by 17 rows.

Until one of those data or representation changes is available, the current loop should stop under the project rule for repeated non-improvement.

