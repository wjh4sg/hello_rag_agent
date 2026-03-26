@echo off
setlocal
set PYTHONIOENCODING=utf-8

set "PYTHON_EXE=python"

if exist ".venv\Scripts\python.exe" (
  set "PYTHON_EXE=.venv\Scripts\python.exe"
)

if exist "venv\Scripts\python.exe" (
  if "%PYTHON_EXE%"=="python" (
    set "PYTHON_EXE=venv\Scripts\python.exe"
  )
)

%PYTHON_EXE% start_services.py %*
