param(
    [string]$Suites = "retrieval,memory,session,answer_smoke",
    [int]$MaxCases = 2,
    [string]$OutputPrefix = "dev_eval"
)

$python = ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Missing .venv\\Scripts\\python.exe"
    exit 1
}

& $python evaluation\run_project_eval.py --profile dev --suites $Suites --max-cases $MaxCases --output-prefix $OutputPrefix
