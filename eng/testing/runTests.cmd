@echo off
setlocal ENABLEEXTENSIONS ENABLEDELAYEDEXPANSION

set EXECUTION_DIR=%~dp0


:: ========================= BEGIN Test Execution =============================
echo ----- start %DATE% %TIME% ===============  To repro directly: =====================================================
echo pushd %EXECUTION_DIR%
echo ${runCommand}
echo popd
echo ===========================================================================================================
pushd %EXECUTION_DIR%
@echo on
${runCommand}
@set _exit_code=%ERRORLEVEL%
@echo off
if exist testResults.xml (
  set HAS_TEST_RESULTS=1
)
popd
echo ----- end %DATE% %TIME% ----- exit code %_exit_code% ----------------------------------------------------------
:: ========================= END Test Execution ===============================

:: The tests either failed or crashed, copy output files
if not %_exit_code%==0 (
    if not "%HELIX_WORKITEM_UPLOAD_ROOT%" == "" (
        powershell Compress-Archive %EXECUTION_DIR%\TestOutput %HELIX_WORKITEM_UPLOAD_ROOT%\TestOutput.zip
    )
)

:: The helix work item should not exit with non-zero if tests ran and produced results
:: The xunit console runner returns 1 when tests fail
if %_exit_code%==1 (
  if %HAS_TEST_RESULTS%==1 (
    if not "%HELIX_WORKITEM_PAYLOAD%"=="" (
      exit /b 0
    )
  )
)

exit /b %_exit_code%