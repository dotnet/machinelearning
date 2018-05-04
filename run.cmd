@if not defined _echo @echo off
setlocal

:: Clear the 'Platform' env variable for this session, as it's a per-project setting within the build, and
:: misleading value (such as 'MCD' in HP PCs) may lead to build breakage (corefx issue: #69).
set Platform=

:: Disable telemetry, first time experience, and global sdk look for the CLI
set DOTNET_CLI_TELEMETRY_OPTOUT=1
set DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1
set DOTNET_MULTILEVEL_LOOKUP=0

:: Restore the Tools directory
call %~dp0init-tools.cmd
if NOT [%ERRORLEVEL%]==[0] exit /b 1

set _toolRuntime=%~dp0Tools
set _dotnet=%_toolRuntime%\dotnetcli\dotnet.exe
set _json=%~dp0config.json

:: run.exe depends on running in the root directory, notably because the config.json specifies
:: a relative path to the binclash logger

pushd %~dp0
call %_dotnet% %_toolRuntime%\run.exe "%_json%" %*
popd

exit /b %ERRORLEVEL%