$ErrorActionPreference = "Stop"
$exe = "C:\Users\admin\Documents\deepseek-harness\JJ\vendor\pikafish\pikafish.exe"

function New-Engine {
  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName               = $exe
  $psi.RedirectStandardInput  = $true
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError  = $true
  $psi.UseShellExecute        = $false
  $psi.CreateNoWindow         = $true
  $psi.WorkingDirectory       = Split-Path $exe
  $p = [System.Diagnostics.Process]::Start($psi)
  return $p
}

function Send-Cmd($p, $s) { $p.StandardInput.WriteLine($s); $p.StandardInput.Flush() }

function Read-Until($p, $prefix, $timeoutSec) {
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $lines = New-Object System.Collections.ArrayList
  while ($sw.Elapsed.TotalSeconds -lt $timeoutSec) {
    $line = $p.StandardOutput.ReadLine()
    if ($null -eq $line) { Start-Sleep -Milliseconds 20; continue }
    [void]$lines.Add($line)
    if ($line.StartsWith($prefix)) { return $lines }
  }
  throw "timeout waiting for '$prefix'"
}

function Analyse($p, $fen, $goCmd, $timeoutSec) {
  Send-Cmd $p "position fen $fen"
  Send-Cmd $p $goCmd
  $lines = Read-Until $p "bestmove" $timeoutSec
  return $lines
}

# ---------------- 1) handshake ----------------
$p = New-Engine
Send-Cmd $p "uci"
$h = Read-Until $p "uciok" 20
"--- handshake ---"
$h | Where-Object { $_ -match "^(id name|id author|uciok)" }
Send-Cmd $p "setoption name Threads value 16"
Send-Cmd $p "setoption name Hash value 512"
Send-Cmd $p "setoption name EvalFile value pikafish.nnue"
Send-Cmd $p "isready"
[void](Read-Until $p "readyok" 30)
"readyok OK"

# ---------------- 2) opening position, fixed depth ----------------
$startFen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
"`n--- opening, go depth 18 ---"
$l = Analyse $p $startFen "go depth 18" 120
$last = $l | Where-Object { $_ -match "^info depth 18 " } | Select-Object -Last 1
if (-not $last) { $last = $l | Where-Object { $_ -match "^info depth" } | Select-Object -Last 1 }
($last -replace "\s+", " ")
$l | Where-Object { $_ -match "^bestmove" }

# ---------------- 3) MATE-IN-1 mapping test ----------------
# red 帅 d0, red 车 a8, red 车 h5, black 将 e9, red to move
# predicted unique mate: h5h9  (车 h5 -> h9, rook mates along rank 9 with a8 covering rank 8)
$mateFen = "4k4/R8/9/9/7R1/9/9/9/9/3K5 w - - 0 1"
"`n--- mate-in-1 mapping test ---"
"fen = $mateFen"
"expected bestmove = h5h9"
$m = Analyse $p $mateFen "go depth 12" 60
$m | Where-Object { $_ -match "^info depth (1[0-2]|[0-9]) " -and $_ -match "score mate" } | Select-Object -Last 1 | ForEach-Object { $_ -replace "\s+", " " }
$m | Where-Object { $_ -match "^bestmove" }

# ---------------- 4) strict FEN validation ----------------
"`n--- strict validation: illegal FEN (kings facing on e-file) ---"
$badFen = "4k4/9/9/9/9/9/9/9/9/4K4 w - - 0 1"
Send-Cmd $p "position fen $badFen"
Send-Cmd $p "go depth 2"
Start-Sleep -Milliseconds 1200
$exited = $p.HasExited
"engine exited after illegal FEN? $exited"
if ($exited) {
  "exit code = $($p.ExitCode)"
  $err = $p.StandardError.ReadToEnd()
  if ($err) { "stderr: $err" }
} else {
  Send-Cmd $p "quit"
}

"`nPROBE DONE"
