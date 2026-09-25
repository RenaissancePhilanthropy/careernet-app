@echo off
cd /d "%~dp0"
set "PY="
py -3 -V >nul 2>nul && set "PY=py -3"
if not defined PY ( python -V >nul 2>nul && set "PY=python" )
if not defined PY ( python3 -V >nul 2>nul && set "PY=python3" )
if not defined PY (
  echo.
  echo  CareerNet needs Python 3 to show the page.
  echo  Install it from https://www.python.org/downloads/  ^(tick "Add python.exe to PATH"^)
  echo  then double-click this file again.
  echo.
  pause
  exit /b 1
)
%PY% serve.py
