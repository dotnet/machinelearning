git submodule update --init
@call "%~dp0run.cmd" build %*
@exit /b %ERRORLEVEL%
