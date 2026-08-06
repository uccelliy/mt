param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Model = "meta-llama/Llama-3.1-8B",
    [string]$Data = "data\psych-101-test\prompts_testing_t1.jsonl",
    [Alias("Output")]
    [string]$OutputDir = "outputs\runs\llama31_8b_base_e0_full_4bit",
    [string]$Summary = "outputs\scoring\llama31_8b_base_e0_full_4bit_summary.csv",
    [int]$BatchTokens = 16384,
    [int]$ChunkSize = 8
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}
if (-not (Test-Path -LiteralPath $Data -PathType Leaf)) {
    throw "Psych-101 test prompts not found: $Data"
}

Write-Host "Running Llama-3.1-8B base E0"
Write-Host "  model:   $Model"
Write-Host "  data:    $Data"
Write-Host "  output:  $OutputDir"
Write-Host "  runtime: CUDA, NF4 weights, FP16 compute"
Write-Host ""
Write-Host "Gated-model prerequisite:"
Write-Host "  1. Accept access at https://huggingface.co/meta-llama/Llama-3.1-8B"
Write-Host "  2. Authenticate this environment with: .\.venv\Scripts\hf.exe auth login"
Write-Host "  3. Verify the account with: .\.venv\Scripts\hf.exe auth whoami"
Write-Host ""
Write-Host "The output is resumable. Re-running this script skips completed sessions."

$runnerArgs = @(
    "scripts\experiments\run_transcript_scoring.py",
    "--model", $Model,
    "--choice-readout", "greedy-unconstrained-1token",
    "--data", $Data,
    "--resume",
    "--chunk-size", $ChunkSize,
    "--batch-tokens", $BatchTokens,
    "--dtype", "fp16",
    "--load", "4bit",
    "--device", "cuda",
    "--output-dir", $OutputDir,
    "--summary", $Summary
)

& $Python @runnerArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "If the error above is HTTP 403 / GatedRepoError, the active Hugging Face"
    Write-Host "account has not yet been granted access to the official Llama checkpoint."
    Write-Host "Accept the model license in the browser, log in again, then rerun this script."
    throw "Llama base E0 exited with code $LASTEXITCODE"
}
