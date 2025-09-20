param(
  [int]$Port = 8000,
  [string]$BindHost = "127.0.0.1"
)

# Ensure we are in the script directory
Set-Location -Path $PSScriptRoot

# Prepare E:-only caches/temp
$env:HF_HOME = Join-Path (Get-Location) ".cache/hf"
$env:TRANSFORMERS_CACHE = $env:HF_HOME
$env:HUGGINGFACE_HUB_CACHE = Join-Path $env:HF_HOME "hub"
$env:TORCH_HOME = Join-Path (Get-Location) ".cache/torch"
$env:PIP_CACHE_DIR = Join-Path (Get-Location) ".cache/pip"
$env:TEMP = Join-Path (Get-Location) ".cache/tmp"
$env:TMP  = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:HF_HOME,$env:HUGGINGFACE_HUB_CACHE,$env:TORCH_HOME,$env:PIP_CACHE_DIR,$env:TEMP | Out-Null

# Add portable Python to PATH for this session
$portablePy = Join-Path (Get-Location) "python-embed"
$env:PATH = (Join-Path $portablePy "Scripts") + ";" + $portablePy + ";" + $env:PATH

# Set model paths if present and not already set
if (-not $env:FINBERT_PATH) {
  $fin = Join-Path (Get-Location) "FINBERT_FINAL.BIN"
  if (Test-Path $fin) { $env:FINBERT_PATH = $fin }
}
if (-not $env:SVM_PATH) {
  $svm = Join-Path (Get-Location) "SVM_FINAL.PKL"
  if (Test-Path $svm) { $env:SVM_PATH = $svm }
}
if (-not $env:TFIDF_PATH) {
  $tfidf = Join-Path (Get-Location) "TFIDF_VECTORIZER_FINAL.PKL"
  if (Test-Path $tfidf) { $env:TFIDF_PATH = $tfidf }
}

# If desired port is taken, bump to next
function Test-PortFree($p) {
  try {
    $inUse = (Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue)
    return -not $inUse
  } catch { return $true }
}
while (-not (Test-PortFree $Port)) { $Port++ }

Write-Host ("Starting API on http://{0}:{1} ..." -f $BindHost, $Port)
uvicorn app.main:app --host $BindHost --port $Port
