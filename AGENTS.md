# Tgprediction Agent Protocol

## Mission

Build one universal Tg single-regressor system that can predict:

- homopolymers
- general copolymers
- nucleobase / DNA-related copolymers

The target is `R2 >= 0.95` on all three categories under non-leaky evaluation. Real generalization is more important than random holdout improvement.

When the user says "继续迭代全能 Tg 模型", continue this protocol automatically.

## Hard Scope

Only work inside the `Tgprediction` project.

On the server, only operate under:

```text
~/Tgprediction
```

Do not modify files outside this directory.

## Primary Metrics

Use these metrics as the main decision criteria:

- Homopolymer: random holdout `R2`, `MAE`.
- General copolymer: PolyInfo group-holdout `R2`, `MAE`.
- Nucleobase copolymer: nucleobase group-holdout `R2`, `MAE`.

Primary optimization target:

```text
maximize min(homopolymer_R2, polyinfo_group_R2, nucleobase_group_R2)
```

Do not claim progress based only on random holdout if group-holdout degrades.

## Current Baseline

Before starting a new iteration, read:

```text
docs/research/universal-single-regressor-tg-progress-2026-04-25.md
```

Current best baseline:

```text
results/universal_single_regressor/exp45_homo_local_nopure
```

Baseline configuration:

```text
model: physics_homo_local_light
feature layer: HYBRID-HOMO186
max-virtual: 0
virtual-weight: 0
copolymer-weight: 10
nucleobase-weight: 60
```

Baseline metrics:

```text
homopolymer random holdout: R2=0.887, MAE=27.16
general copolymer random holdout: R2=0.932, MAE=10.18
general copolymer group-holdout: R2=0.844, MAE=16.36
nucleobase group-holdout: R2=0.789, MAE=6.85
```

Treat general copolymer group-holdout and nucleobase group-holdout as the serious bottlenecks.

## Iteration Loop

For every serious iteration:

1. Read the latest experiment summaries, iteration log, and relevant docs.
2. State one explicit hypothesis.
3. Make the smallest code, data, or configuration change needed to test it.
4. Run relevant unit tests before long experiments.
5. Run one controlled experiment.
6. Compare against the current best baseline using the primary metrics.
7. Record the result with command, git hash, data inputs, metrics, and conclusion.
8. Keep the change only if it improves the primary target or provides clear diagnostic value.

Do not stack unrelated changes in one experiment.

## Agent-Controlled Long Loop

`AGENTS.md` is a rule file, not a loop engine. It cannot, by itself, guarantee a 5+ hour chat session or force the agent to reread itself after every experiment.

For true long-running autonomous iteration, use an external loop harness that repeatedly starts a fresh local Codex exec session. Each round must reload persistent project state from disk, run one controlled iteration, write logs, and emit a terminal signal.

Default architecture:

```text
local Windows project = Codex controller, reasoning, code edits, logs
remote ~/Tgprediction = experiment runner and result storage
```

Do not require Codex CLI on the server. The local Codex agent should use SSH when it needs to run experiments on the server.

Preferred agent loop launcher:

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/codex_universal_tg_agent_loop.ps1 `
  -MaxHours 5 `
  -MaxRounds 20 `
  -RoundTimeoutMinutes 45
```

Linux/macOS shell:

```bash
mkdir -p logs
python -u scripts/codex_universal_tg_agent_loop.py \
  --max-hours 5 \
  --max-rounds 20 \
  --codex-cmd codex \
  --codex-sandbox danger-full-access \
  --codex-approval never \
  > logs/codex_universal_tg_agent_loop.nohup.log 2>&1 &
```

Every loop round must:

1. Re-read `AGENTS.md` from disk.
2. Record the `AGENTS.md` SHA256 hash.
3. Re-read `docs/research/universal-tg-task-queue.md`.
4. Re-read recent `docs/research/universal-tg-iteration-log.md`.
5. Re-read `results/universal_single_regressor/scoreboard.json` when present.
6. Select exactly one hypothesis or task.
7. Execute one bounded experiment or one bounded code/data improvement.
8. Update the iteration log and scoreboard.
9. End the final answer with exactly one signal:

```text
TG_CONTINUE
TG_BLOCKED
TG_CONVERGED
```

Use `TG_CONTINUE` only when another independent iteration is justified.
Use `TG_BLOCKED` when user input, missing data, broken environment, or repeated non-improvement blocks progress.
Use `TG_CONVERGED` only when all target categories reach `R2 >= 0.95`.

Rules for long-running agent loops:

- Use `nohup` or `tmux` for jobs expected to exceed 30 minutes.
- Write all outputs under `results/`, `logs/`, and `docs/research/`.
- Never rely only on chat memory; every experiment must be recoverable from disk.
- If an experiment fails, log the failure and choose a new independent hypothesis unless the failure indicates a broken shared dependency.
- Stop launching new experiments when the configured time budget is nearly exhausted.
- Do not start GPU-heavy jobs if another project job is already using the GPU unless explicitly approved.
- Prefer several 20-40 minute agent rounds over one huge irreversible run.
- If three serious iterations do not improve the primary target, stop with `TG_BLOCKED` and write a bottleneck report.

Remote experiment rules:

- SSH target: `sheng-xiang@100.64.0.4`.
- Remote project directory: `~/Tgprediction`.
- Remote Python: `/home/sheng-xiang/miniconda3/envs/llm4graphgen/bin/python`.
- Run remote commands only after `cd ~/Tgprediction`.
- Do not operate outside `~/Tgprediction` on the remote server.
- For long remote training jobs, use `nohup` or `tmux`, but the local agent loop remains the controller.
- Pull remote metrics and write local summaries after every remote experiment.
- Prefer SSH key authentication for unattended local loops.
- If SSH keys are unavailable, use `scripts/remote_tg_command.py` with `TG_REMOTE_PASSWORD` set in the local environment. Never write passwords into repository files.

Remote helper examples:

```powershell
$env:TG_REMOTE_PASSWORD = "<set locally, do not commit>"
python scripts/remote_tg_command.py --use-paramiko "git rev-parse --short HEAD"
python scripts/remote_tg_command.py --use-paramiko "/home/sheng-xiang/miniconda3/envs/llm4graphgen/bin/python -V"
```

After reconnecting, inspect:

```bash
tail -n 120 logs/codex_universal_tg_agent_loop.nohup.log
cat results/universal_single_regressor/agent_loop_state.json
tail -n 120 docs/research/universal-tg-iteration-log.md
```

The older `scripts/run_universal_tg_iteration_queue.py` is only an experiment subprocess queue. Prefer `scripts/codex_universal_tg_agent_loop.py` when the goal is sustained agent reasoning and self-directed iteration.

## Experiment Logging

Every experiment must write or update:

```text
docs/research/universal-tg-iteration-log.md
```

Each entry must include:

```text
date
git hash
experiment name
hypothesis
command
data sources
metrics
comparison to baseline
decision: keep / reject / investigate
next step
```

Experiment outputs should go under:

```text
results/universal_single_regressor/expNN_descriptive_name
```

## Data Rules

Do not silently change datasets.

If filtering or normalizing data:

- write an audit CSV or JSON
- document why rows were removed or transformed
- compare metrics before and after the change

Never optimize on leaked splits.

For copolymers:

- use group-holdout by system as the serious metric
- exclude pure endpoint rows from PolyInfo copolymer training unless explicitly testing endpoint behavior
- keep raw data and cleaned data separate

For virtual data:

- do not directly mix large virtual datasets into final training unless a filtering or uncertainty rule is used
- treat virtual data as weak supervision
- reject virtual-data experiments if real group-holdout degrades

## Model Rules

The final system should remain one single-regressor prediction path.

Allowed:

- new feature construction
- physics-informed priors
- kernel and residual models
- robust losses
- uncertainty filtering
- pretraining then fine-tuning inside one model family
- constrained multi-kernel learning

Avoid:

- task-specific routing as the main solution
- choosing different external predictors for different material classes
- improving one category by badly degrading another

## Stopping And Escalation

Continue iterating until one of these happens:

- all three target categories reach `R2 >= 0.95`
- three consecutive serious iterations fail to improve the primary target
- evidence shows the target is blocked by data quality or dataset size
- a long experiment needs user approval due to runtime or resource cost

If blocked, produce a concise report with:

- what was tried
- best metrics
- why the current bottleneck exists
- what data or algorithmic change is needed next
