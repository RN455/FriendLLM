@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
  ".venv\Scripts\pythonw.exe" app.py
  exit /b 0
)
where pyw >nul 2>nul
if errorlevel 1 (
  echo Install Python 3.14 or run Setup.bat first.
  pause
  exit /b 1
)
pyw -3.14 app.py
if errorlevel 1 (
  echo.
  echo FriendLLM closed with an error. Review the message above.
  pause
)
