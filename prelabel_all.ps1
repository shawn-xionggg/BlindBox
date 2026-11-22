# Pre-label sessions: auto_label.py, then combo_candidates.py (adds combo follow-ups and
# builds the flow cache). Sessions named on the command line go first, then every other
# session with a video. Safe to rerun: steps already done are skipped
# (.auto_candidates.json exists = auto_label done; flow cache exists = combo pass done).
param([string[]]$First = @())
$py = "C:\Hack Western\Punch AI\boxing-assistant\.venv\Scripts\python.exe"
$all = Get-ChildItem data\session_*.npz | Sort-Object Name | ForEach-Object { $_.BaseName }
$order = @($First) + @($all | Where-Object { $First -notcontains $_ })
foreach ($s in $order) {
    $npz = "data\$s.npz"
    if (-not (Test-Path "data\$s.mp4")) { continue }
    if (-not (Test-Path "data\$s.auto_candidates.json")) {
        Write-Output "=== auto_label $s"
        & $py auto_label.py $npz | Select-Object -Last 2
    }
    if (-not (Test-Path "features\$s.flow_s0.25.npz")) {
        Write-Output "=== combo_candidates $s"
        & $py combo_candidates.py $npz | Select-Object -Last 2
    }
    Write-Output "READY $s"
}
Write-Output "PRELABEL DONE"
