$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")
$py = ".\.venv\Scripts\python.exe"
$log = Join-Path $PSScriptRoot "..\install.log"
"[$(Get-Date -Format 'HH:mm:ss')] start" | Out-File $log -Encoding utf8

# 1) CUDA 版 torch/torchvision（官方源，速度 1.4 MB/s，约 2.5 GB）
& $py -m pip install --no-cache-dir --retries 8 --timeout 120 `
    "torch==2.5.1+cu121" "torchvision==0.20.1+cu121" `
    --index-url https://download.pytorch.org/whl/cu121 2>&1 |
    Tee-Object -FilePath $log -Append
"[$(Get-Date -Format 'HH:mm:ss')] torch done exit=$LASTEXITCODE" | Out-File $log -Append -Encoding utf8

# 2) 其余依赖（PyPI）
& $py -m pip install --no-cache-dir --retries 8 --timeout 120 `
    ultralytics opencv-python-headless matplotlib pandas tqdm 2>&1 |
    Tee-Object -FilePath $log -Append
"[$(Get-Date -Format 'HH:mm:ss')] rest done exit=$LASTEXITCODE" | Out-File $log -Append -Encoding utf8

& $py -c "import torch, torchvision, ultralytics, cv2; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('ultralytics', ultralytics.__version__); print('cv2', cv2.__version__)" 2>&1 |
    Tee-Object -FilePath $log -Append
"[$(Get-Date -Format 'HH:mm:ss')] verify done" | Out-File $log -Append -Encoding utf8
