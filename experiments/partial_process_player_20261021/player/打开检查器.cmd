@echo off
cd /d "%~dp0"
python -B open_player.py
if errorlevel 1 pause
