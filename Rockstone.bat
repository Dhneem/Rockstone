@echo off
rem Rockstone launcher: double-click to open the desktop app.
cd /d "%~dp0"
".venv\Scripts\pythonw.exe" -m rockstone.gui
if errorlevel 1 echo Rockstone exited with an error. See rockstone_gui.log
if errorlevel 1 pause
