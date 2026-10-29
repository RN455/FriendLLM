@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo Python Launcher was not found. Install Python 3.14 from python.org and enable the py launcher.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating the FriendLLM environment with Python 3.14...
  py -3.14 -m venv .venv
  if errorlevel 1 goto failed
)

echo Updating pip...
".venv\Scripts\python.exe" -m pip install --no-cache-dir --upgrade pip
if errorlevel 1 goto failed

rem Preserve a working custom PyTorch backend unless an explicit backend is requested.
if not "%~1"=="" goto choose_backend
".venv\Scripts\python.exe" -c "import torch" >nul 2>nul
if not errorlevel 1 goto dependencies

:choose_backend
set "FRIENDLLM_BACKEND=%~1"
if not defined FRIENDLLM_BACKEND set "FRIENDLLM_BACKEND=auto"
if /i "%FRIENDLLM_BACKEND%"=="auto" goto detect_backend
if /i "%FRIENDLLM_BACKEND%"=="cuda" goto cuda_backend
if /i "%FRIENDLLM_BACKEND%"=="xpu" goto xpu_backend
if /i "%FRIENDLLM_BACKEND%"=="cpu" goto cpu_backend
echo Usage: Setup.bat [auto^|cuda^|xpu^|cpu]
goto failed

:detect_backend
rem A custom wheel index allows an AMD or other supported PyTorch distribution.
if defined FRIENDLLM_TORCH_INDEX goto install_torch
nvidia-smi --query-gpu=name --format=csv,noheader >nul 2>nul
if not errorlevel 1 goto cuda_backend
powershell -NoProfile -Command "$cards = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue; if ($cards.Name -match 'NVIDIA') {exit 0}; exit 1" >nul 2>nul
if not errorlevel 1 goto cuda_backend
powershell -NoProfile -Command "$cards = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue; if ($cards.Name -match 'Intel.*Arc') {exit 0}; exit 1" >nul 2>nul
if not errorlevel 1 goto xpu_backend
goto cpu_backend

:cuda_backend
set "FRIENDLLM_TORCH_INDEX=https://download.pytorch.org/whl/cu132"
goto install_torch

:xpu_backend
set "FRIENDLLM_TORCH_INDEX=https://download.pytorch.org/whl/xpu"
goto install_torch

:cpu_backend
set "FRIENDLLM_TORCH_INDEX=https://download.pytorch.org/whl/cpu"

:install_torch
echo Installing PyTorch from %FRIENDLLM_TORCH_INDEX%...
".venv\Scripts\python.exe" -m pip install --upgrade --no-cache-dir "torch>=2.6,<3" --index-url "%FRIENDLLM_TORCH_INDEX%"
if errorlevel 1 goto failed

:dependencies
echo Installing FriendLLM training dependencies...
".venv\Scripts\python.exe" -m pip install --no-cache-dir -r requirements.txt
if errorlevel 1 goto failed

".venv\Scripts\python.exe" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() and not torch.version.hip else 1)" >nul 2>nul
if errorlevel 1 goto ready
echo Installing optional NVIDIA 4-bit training support...
".venv\Scripts\python.exe" -m pip install --no-cache-dir "bitsandbytes>=0.47,<2"
if errorlevel 1 echo Optional 4-bit support could not be installed. Standard LoRA training remains available.

:ready
".venv\Scripts\python.exe" -c "from accelerate import PartialState; print('Detected training device:', PartialState().device)"
if errorlevel 1 goto failed
echo.
echo Setup is complete. Double-click Run FriendLLM.bat to start the app.
pause
exit /b 0

:failed
echo.
echo Setup failed. Review the messages above and check your internet connection and available disk space.
pause
exit /b 1
