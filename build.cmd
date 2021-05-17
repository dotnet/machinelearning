@echo off
powershell -ExecutionPolicy ByPass -NoProfile -command "& """%~dp0eng\common\build.ps1""" -restore -build -warnAsError 0 %*"
exit /b %ErrorLevel%
