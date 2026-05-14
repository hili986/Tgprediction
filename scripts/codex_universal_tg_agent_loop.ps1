param(
    [double]$MaxHours = 5,
    [int]$MaxRounds = 20,
    [double]$RoundTimeoutMinutes = 45,
    [string]$CodexCmd = "codex",
    [string]$CodexSandbox = "danger-full-access",
    [string]$CodexApproval = "never",
    [string]$Model = "gpt-5.5",
    [string]$ReasoningEffort = "xhigh",
    [string]$RemoteHost = "sheng-xiang@100.64.0.4",
    [string]$RemoteProjectDir = "~/Tgprediction",
    [string]$RemotePython = "/home/sheng-xiang/miniconda3/envs/llm4graphgen/bin/python",
    [string]$ExtraInstruction = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

New-Item -ItemType Directory -Force -Path "logs" | Out-Null

$argsList = @(
    "-u", "scripts/codex_universal_tg_agent_loop.py",
    "--max-hours", "$MaxHours",
    "--max-rounds", "$MaxRounds",
    "--round-timeout-minutes", "$RoundTimeoutMinutes",
    "--codex-cmd", "$CodexCmd",
    "--codex-sandbox", "$CodexSandbox",
    "--codex-approval", "$CodexApproval",
    "--model", "$Model",
    "--reasoning-effort", "$ReasoningEffort",
    "--remote-host", "$RemoteHost",
    "--remote-project-dir", "$RemoteProjectDir",
    "--remote-python", "$RemotePython"
)

if ($ExtraInstruction) {
    $argsList += @("--extra-instruction", "$ExtraInstruction")
}

if ($DryRun) {
    $argsList += @("--dry-run")
}

python @argsList
