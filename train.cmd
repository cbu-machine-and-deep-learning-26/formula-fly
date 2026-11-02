@echo off
rem `train start`, `train status`, `train stop`: see the Lab PCs section of AGENTS.md.
setlocal
cd /d "%~dp0"
set "FLYVIS_ROOT_DIR=%~dp0.cache\flyvis"
set "PYTHONUNBUFFERED=1"
.venv-flyvis\Scripts\python.exe scripts\lab_train.py %*
