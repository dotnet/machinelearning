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
@echo on
if exist testResults.xml (
  set HAS_TEST_RESULTS=1
)
popd
echo ----- end %DATE% %TIME% ----- exit code %_exit_code% ----------------------------------------------------------
:: ========================= END Test Execution ===============================

:: The tests either failed or crashed, copy output files
echo --- HELIX_WORKITEM_UPLOAD_ROOT %HELIX_WORKITEM_UPLOAD_ROOT% ---------------------------------
echo --- EXECUTION_DIR %EXECUTION_DIR% ---------------------------------
if not %_exit_code%==0 (
    if not "%HELIX_WORKITEM_UPLOAD_ROOT%" == "" (
      if exist "%EXECUTION_DIR%TestOutput\NUL" (
        powershell Compress-Archive "%EXECUTION_DIR%TestOutput" "%HELIX_WORKITEM_UPLOAD_ROOT%\TestOutput.zip"
      ) else (
        echo No test output directory found to compress.
      )
    )
)

:: The helix work item should not exit with non-zero if tests ran and produced results
:: The xunit console runner returns 1 when tests fail
echo --- HAS_TEST_RESULTS %HAS_TEST_RESULTS% ----- HELIX_WORKITEM_PAYLOAD %HELIX_WORKITEM_PAYLOAD% ---------------------------------
if %_exit_code%==1 (
  if %HAS_TEST_RESULTS%==1 (
    if not "%HELIX_WORKITEM_PAYLOAD%"=="" (
      exit /b 0
    )
  )
)

exit /b %_exit_code%