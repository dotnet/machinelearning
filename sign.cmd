@echo off
powershell -ExecutionPolicy ByPass -NoProfile -command "& """%~dp0eng\common\Build.ps1""" -sign -warnAsError 0 %*"
exit /b %ErrorLevel% 