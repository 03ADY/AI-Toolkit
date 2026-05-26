$Root = Split-Path $PSScriptRoot -Parent
$Py = Join-Path $Root "venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
  python -m venv (Join-Path $Root "venv")
  & (Join-Path $Root "venv\Scripts\pip.exe") install -r (Join-Path $Root "requirements.txt")
}
$env:AI_TOOLKIT_API_BASE = "http://127.0.0.1:8000/api"
Start-Process "http://127.0.0.1:8506"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$Root'; `$env:AI_TOOLKIT_API_BASE='http://127.0.0.1:8000/api'; & '$Py' -m uvicorn merged_backend:app --port 8000"
Start-Sleep -Seconds 8
& $Py -m streamlit run (Join-Path $Root "app.py") --server.port 8506
