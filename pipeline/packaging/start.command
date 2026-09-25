#!/bin/sh
cd "$(dirname "$0")" || exit 1
if command -v python3 >/dev/null 2>&1; then PY=python3
elif command -v python  >/dev/null 2>&1; then PY=python
else
  echo ""
  echo " CareerNet needs Python 3 to show the page."
  echo " On macOS, run:  xcode-select --install"
  echo " Then double-click this file again."
  echo ""
  printf " Press return to close. "; read -r _; exit 1
fi
exec "$PY" serve.py
