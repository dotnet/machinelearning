@if not defined _echo @echo off
setlocal

set INIT_TOOLS_LOG=%~dp0init-tools.log
if [%PACKAGES_DIR%]==[] set PACKAGES_DIR=%~dp0packages
if [%TOOLRUNTIME_DIR%]==[] set TOOLRUNTIME_DIR=%~dp0Tools
set DOTNET_PATH=%TOOLRUNTIME_DIR%\dotnetcli\
if [%DOTNET_CMD%]==[] set DOTNET_CMD=%DOTNET_PATH%dotnet.exe
if [%BUILDTOOLS_SOURCE%]==[] set BUILDTOOLS_SOURCE=https://dotnet.myget.org/F/dotnet-buildtools/api/v3/index.json
set /P BUILDTOOLS_VERSION=< "%~dp0BuildToolsVersion.txt"
set BUILD_TOOLS_PATH=%PACKAGES_DIR%\Microsoft.DotNet.BuildTools\%BUILDTOOLS_VERSION%\lib
set INIT_TOOLS_RESTORE_PROJECT=%~dp0init-tools.msbuild
set BUILD_TOOLS_SEMAPHORE_DIR=%TOOLRUNTIME_DIR%\%BUILDTOOLS_VERSION%
set BUILD_TOOLS_SEMAPHORE=%BUILD_TOOLS_SEMAPHORE_DIR%\init-tools.completed
set ARCH=x64

:: if force option is specified then clean the tool runtime and build tools package directory to force it to get recreated
if [%1]==[force] (
  if exist "%TOOLRUNTIME_DIR%" rmdir /S /Q "%TOOLRUNTIME_DIR%"
  if exist "%PACKAGES_DIR%\Microsoft.DotNet.BuildTools" rmdir /S /Q "%PACKAGES_DIR%\Microsoft.DotNet.BuildTools"
)

:: If semaphore exists do nothing
if exist "%BUILD_TOOLS_SEMAPHORE%" (
  echo Tools are already initialized.
  goto :EOF
)

if exist "%TOOLRUNTIME_DIR%" rmdir /S /Q "%TOOLRUNTIME_DIR%"

if exist "%DotNetBuildToolsDir%" (
  echo Using tools from '%DotNetBuildToolsDir%'.
  mklink /j "%TOOLRUNTIME_DIR%" "%DotNetBuildToolsDir%"

  if not exist "%DOTNET_CMD%" (
    echo ERROR: Ensure that '%DotNetBuildToolsDir%' contains the .NET Core SDK at '%DOTNET_PATH%'
    exit /b 1
  )

  echo Done initializing tools.
  if NOT exist "%BUILD_TOOLS_SEMAPHORE_DIR%" mkdir "%BUILD_TOOLS_SEMAPHORE_DIR%"
  echo Using tools from '%DotNetBuildToolsDir%'. > "%BUILD_TOOLS_SEMAPHORE%"
  exit /b 0
)

echo Running %0 > "%INIT_TOOLS_LOG%"

set /p DOTNET_VERSION=< "%~dp0DotnetCLIVersion.txt"
set /p DOTNET_EXTRA_RUNTIME_VERSION=< "%~dp0DotnetExtraRuntimeVersion.txt"

:Arg_Loop
if [%1] == [] goto :ArchSet
if /i [%1] == [x86]         ( set ARCH=x86)
shift
goto :Arg_Loop

:ArchSet
if exist "%DOTNET_CMD%" goto :afterdotnetrestore

if NOT exist "%DOTNET_PATH%" mkdir "%DOTNET_PATH%"

:: set registry to take dump automatically when test process crashes
if NOT [%AGENT_ID%] == [] (
  reg add "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps" /f
  reg add "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps" /f /v DumpType /t REG_DWORD /d 2
  reg add "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps" /f /v DumpCount /t REG_DWORD /d 2
  reg add "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps" /f /v DumpFolder /t REG_SZ /d "%~dp0CrashDumps"
)

:: install procdump.exe to take process dump when test crashes, hangs or fails
echo Installing procdump.exe
powershell -Command "Invoke-WebRequest https://download.sysinternals.com/files/Procdump.zip -UseBasicParsing -outfile procdump.zip | Out-Null"
powershell -Command "Expand-Archive -Force procdump.zip Tools"
del /f procdump.zip
echo Finish install procdump.exe

:: install the extra runtime first, so the SDK install will overwrite the root dotnet executable
echo Installing dotnet runtime %DOTNET_EXTRA_RUNTIME_VERSION%...
set DOTNET_EXTRA_RUNTIME_ZIP_NAME=dotnet-runtime-%DOTNET_EXTRA_RUNTIME_VERSION%-win-%ARCH%.zip
set DOTNET_EXTRA_RUNTIME_REMOTE_PATH=https://dotnetcli.azureedge.net/dotnet/Runtime/%DOTNET_EXTRA_RUNTIME_VERSION%/%DOTNET_EXTRA_RUNTIME_ZIP_NAME%
set DOTNET_EXTRA_RUNTIME_LOCAL_PATH=%DOTNET_PATH%%DOTNET_EXTRA_RUNTIME_ZIP_NAME%
echo Installing '%DOTNET_EXTRA_RUNTIME_REMOTE_PATH%' to '%DOTNET_EXTRA_RUNTIME_LOCAL_PATH%' >> "%INIT_TOOLS_LOG%"
powershell -NoProfile -ExecutionPolicy unrestricted -Command "$retryCount = 0; $success = $false; $proxyCredentialsRequired = $false; do { try { $wc = New-Object Net.WebClient; if ($proxyCredentialsRequired) { [Net.WebRequest]::DefaultWebProxy.Credentials = [Net.CredentialCache]::DefaultNetworkCredentials; } $wc.DownloadFile('%DOTNET_EXTRA_RUNTIME_REMOTE_PATH%', '%DOTNET_EXTRA_RUNTIME_LOCAL_PATH%'); $success = $true; } catch { if ($retryCount -ge 6) { throw; } else { $we = $_.Exception.InnerException -as [Net.WebException]; $proxyCredentialsRequired = ($we -ne $null -and ([Net.HttpWebResponse]$we.Response).StatusCode -eq [Net.HttpStatusCode]::ProxyAuthenticationRequired); Start-Sleep -Seconds (5 * $retryCount); $retryCount++; } } } while ($success -eq $false); Expand-Archive '%DOTNET_EXTRA_RUNTIME_LOCAL_PATH%' '%DOTNET_PATH%';" >> "%INIT_TOOLS_LOG%"

echo Installing dotnet cli %DOTNET_VERSION%...
set DOTNET_ZIP_NAME=dotnet-sdk-%DOTNET_VERSION%-win-%ARCH%.zip
set DOTNET_REMOTE_PATH=https://dotnetcli.azureedge.net/dotnet/Sdk/%DOTNET_VERSION%/%DOTNET_ZIP_NAME%
set DOTNET_LOCAL_PATH=%DOTNET_PATH%%DOTNET_ZIP_NAME%
echo Installing '%DOTNET_REMOTE_PATH%' to '%DOTNET_LOCAL_PATH%' >> "%INIT_TOOLS_LOG%"
powershell -NoProfile -ExecutionPolicy unrestricted -Command "$retryCount = 0; $success = $false; $proxyCredentialsRequired = $false; do { try { $wc = New-Object Net.WebClient; if ($proxyCredentialsRequired) { [Net.WebRequest]::DefaultWebProxy.Credentials = [Net.CredentialCache]::DefaultNetworkCredentials; } $wc.DownloadFile('%DOTNET_REMOTE_PATH%', '%DOTNET_LOCAL_PATH%'); $success = $true; } catch { if ($retryCount -ge 6) { throw; } else { $we = $_.Exception.InnerException -as [Net.WebException]; $proxyCredentialsRequired = ($we -ne $null -and ([Net.HttpWebResponse]$we.Response).StatusCode -eq [Net.HttpStatusCode]::ProxyAuthenticationRequired); Start-Sleep -Seconds (5 * $retryCount); $retryCount++; } } } while ($success -eq $false); Expand-Archive '%DOTNET_LOCAL_PATH%' '%DOTNET_PATH%' -Force; " >> "%INIT_TOOLS_LOG%"

if NOT exist "%DOTNET_LOCAL_PATH%" (
  echo ERROR: Could not install dotnet cli correctly. 1>&2
  goto :error
)

:afterdotnetrestore

if exist "%BUILD_TOOLS_PATH%" goto :afterbuildtoolsrestore
echo Restoring BuildTools version %BUILDTOOLS_VERSION%...
echo Running: "%DOTNET_CMD%" restore "%INIT_TOOLS_RESTORE_PROJECT%" --no-cache --packages "%PACKAGES_DIR%" --source "%BUILDTOOLS_SOURCE%" /p:BuildToolsPackageVersion=%BUILDTOOLS_VERSION% /p:ToolsDir=%TOOLRUNTIME_DIR% >> "%INIT_TOOLS_LOG%"
call "%DOTNET_CMD%" restore "%INIT_TOOLS_RESTORE_PROJECT%" --no-cache --packages "%PACKAGES_DIR%" --source "%BUILDTOOLS_SOURCE%" /p:BuildToolsPackageVersion=%BUILDTOOLS_VERSION% /p:ToolsDir="%TOOLRUNTIME_DIR%" >> "%INIT_TOOLS_LOG%"
if NOT exist "%BUILD_TOOLS_PATH%\init-tools.cmd" (
  echo ERROR: Could not restore build tools correctly. 1>&2
  goto :error
)

:afterbuildtoolsrestore

echo Initializing BuildTools...
echo Running: "%BUILD_TOOLS_PATH%\init-tools.cmd" "%~dp0" "%DOTNET_CMD%" "%TOOLRUNTIME_DIR%" "%PACKAGES_DIR%" >> "%INIT_TOOLS_LOG%"
call "%BUILD_TOOLS_PATH%\init-tools.cmd" "%~dp0" "%DOTNET_CMD%" "%TOOLRUNTIME_DIR%" "%PACKAGES_DIR%" >> "%INIT_TOOLS_LOG%"
set INIT_TOOLS_ERRORLEVEL=%ERRORLEVEL%
if not [%INIT_TOOLS_ERRORLEVEL%]==[0] (
  echo ERROR: An error occured when trying to initialize the tools. 1>&2
  goto :error
)

:: Create semaphore file
echo Done initializing tools.
if NOT exist "%BUILD_TOOLS_SEMAPHORE_DIR%" mkdir "%BUILD_TOOLS_SEMAPHORE_DIR%"
echo Init-Tools.cmd completed for BuildTools Version: %BUILDTOOLS_VERSION% > "%BUILD_TOOLS_SEMAPHORE%"
exit /b 0

:error
echo Please check the detailed log that follows. 1>&2
type "%INIT_TOOLS_LOG%" 1>&2
exit /b 1
