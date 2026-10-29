@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run Setup.bat first.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" -m pip install --no-cache-dir pyinstaller
if errorlevel 1 goto failed

".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean FriendLLM.spec
if errorlevel 1 goto failed

echo.
echo App created at dist\FriendLLM\FriendLLM.exe. Keep the FriendLLM folder together when launching.
pause
exit /b 0

:failed
echo.
echo Build failed. Review the messages above.
pause
exit /b 1
