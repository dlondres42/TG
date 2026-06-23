# Overnight runner for the CPU-mechanism subset (item 3 of the thesis
# statistical follow-ups). Registered via schtasks to fire once at 02:00.
#
# - Verifies Docker Desktop is reachable; aborts cleanly if not.
# - Clears any prior layer3_with_cpu tree so the overnight run starts fresh.
# - Invokes the WSL2 bench script under Ubuntu with stdout teed to a log.
# - Regenerates analyze.py outputs so cpu_mechanism.csv lands by morning.
#
# Re-arm with: schtasks /Change /TN "TG-BenchCpuMechanism" /ENABLE
$ErrorActionPreference = "Stop"

$Repo = "C:\Users\David\Documents\learning_repos\TG"
$Bench = "$Repo\project\3-bench"
$ResultsCpu = "$Bench\results\layer3_with_cpu"
$LogDir = "$Bench\results\layer3_with_cpu_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Log = "$LogDir\run_$Stamp.log"

function Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Output $line
    Add-Content -Path $Log -Value $line
}

try {
    Log "starting cpu-mechanism overnight run"

    # Sanity check: Docker must respond before we kick off ~90 min of bench work.
    $dockerOk = $false
    try {
        $null = docker info 2>$null
        if ($LASTEXITCODE -eq 0) { $dockerOk = $true }
    } catch {}
    if (-not $dockerOk) {
        Log "docker info failed; Docker Desktop not running. Aborting."
        exit 1
    }
    Log "docker is reachable"

    if (Test-Path $ResultsCpu) {
        Log "clearing prior layer3_with_cpu tree"
        Remove-Item -Recurse -Force $ResultsCpu
    }

    # Run the bench under WSL2 (Ubuntu). uv is on PATH via .bashrc; the venv
    # uses UV_PROJECT_ENVIRONMENT=.venv-wsl per the WSL setup script.
    Log "launching L3_cpu_mechanism.py via WSL"
    $WslCmd = "cd /mnt/c/Users/David/Documents/learning_repos/TG/project/3-bench && " +
              "uv run python layers/L3_cpu_mechanism.py --duration 45 --repeats 3"
    wsl -d Ubuntu -- bash -lc $WslCmd 2>&1 | Tee-Object -FilePath $Log -Append
    if ($LASTEXITCODE -ne 0) {
        Log "bench exited non-zero ($LASTEXITCODE)"
        exit $LASTEXITCODE
    }
    Log "bench finished cleanly"

    # Regenerate analyze.py outputs so cpu_mechanism.csv is ready by morning.
    Log "running analyze.py to produce cpu_mechanism.csv"
    $AnalyzeCmd = "cd /mnt/c/Users/David/Documents/learning_repos/TG/project/3-bench && " +
                  "uv run python analyze.py --no-encode"
    wsl -d Ubuntu -- bash -lc $AnalyzeCmd 2>&1 | Tee-Object -FilePath $Log -Append
    if ($LASTEXITCODE -ne 0) {
        Log "analyze.py exited non-zero ($LASTEXITCODE); skipping report step"
        exit $LASTEXITCODE
    }

    # Produce the morning deliverable: 4 PNG figures + phase3_cpu_mechanism_report.md
    # with numbers pulled live from cpu_mechanism.csv. Templated so re-runs
    # regenerate cleanly without manual editing.
    Log "running cpu_mechanism_report.py to produce figures + report"
    $ReportCmd = "cd /mnt/c/Users/David/Documents/learning_repos/TG/project/3-bench && " +
                 "uv run python cpu_mechanism_report.py"
    wsl -d Ubuntu -- bash -lc $ReportCmd 2>&1 | Tee-Object -FilePath $Log -Append

    Log "all done"
} catch {
    Log "uncaught error: $($_.Exception.Message)"
    exit 1
}
