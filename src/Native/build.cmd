@if not defined _echo @echo off
setlocal

:: Store current script directory before %~dp0 gets affected by another process later.
set __currentScriptDir=%~dp0

:SetupArgs
:: Initialize the args that will be passed to cmake
set __binDir=%__currentScriptDir%..\..\bin
set __rootDir=%__currentScriptDir%..\..
set __CMakeBinDir=""
set __IntermediatesDir=""
set __BuildArch=x64
set __VCBuildArch=x86_amd64
set CMAKE_BUILD_TYPE=Debug
set MKL_LIB_PATH=""

:Arg_Loop
if [%1] == [] goto :ToolsVersion
if /i [%1] == [Release]     ( set CMAKE_BUILD_TYPE=Release&&shift&goto Arg_Loop)
if /i [%1] == [Release-netcoreapp3_0]     ( set CMAKE_BUILD_TYPE=Release-netcoreapp3_0&&shift&goto Arg_Loop)
if /i [%1] == [Release-netfx]     ( set CMAKE_BUILD_TYPE=Release-netfx&&shift&goto Arg_Loop)
if /i [%1] == [Debug]       ( set CMAKE_BUILD_TYPE=Debug&&shift&goto Arg_Loop)
if /i [%1] == [Debug-netcoreapp3_0]       ( set CMAKE_BUILD_TYPE=Debug-netcoreapp3_0&&shift&goto Arg_Loop)
if /i [%1] == [Debug-netfx]       ( set CMAKE_BUILD_TYPE=Debug-netfx&&shift&goto Arg_Loop)

if /i [%1] == [x86]         ( set __BuildArch=x86&&set __VCBuildArch=x86&&shift&goto Arg_Loop)
if /i [%1] == [x64]         ( set __BuildArch=x64&&set __VCBuildArch=x86_amd64&&shift&goto Arg_Loop)
if /i [%1] == [amd64]       ( set __BuildArch=x64&&set __VCBuildArch=x86_amd64&&shift&goto Arg_Loop)

if /i [%1] == [--mkllibpath] ( set MKL_LIB_PATH=%2&&shift&goto Arg_Loop)
shift
goto :Arg_Loop

:ToolsVersion
if defined VisualStudioVersion goto :RunVCVars

set _VSWHERE="%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist %_VSWHERE% (
  for /f "usebackq tokens=*" %%i in (`%_VSWHERE% -latest -prerelease -property installationPath`) do set _VSCOMNTOOLS=%%i\Common7\Tools
)
if not exist "%_VSCOMNTOOLS%" set _VSCOMNTOOLS=%VS140COMNTOOLS%
if not exist "%_VSCOMNTOOLS%" goto :MissingVersion


set "VSCMD_START_DIR=%__currentScriptDir%"
call "%_VSCOMNTOOLS%\VsDevCmd.bat"

:RunVCVars
if "%VisualStudioVersion%"=="16.0" (
    goto :VS2019
) else if "%VisualStudioVersion%"=="15.0" (
    goto :VS2017
) else if "%VisualStudioVersion%"=="14.0" (
    goto :VS2015
)

:MissingVersion
:: Can't find VS 2015, 2017 or 2019
echo Error: Visual Studio 2015, 2017 or 2019 required
echo        Please see https://github.com/dotnet/machinelearning/tree/master/Documentation for build instructions.
exit /b 1

:VS2019
:: Setup vars for VS2019
set __PlatformToolset=v142
set __VSVersion=16 2019
if NOT "%__BuildArch%" == "arm64" (
    :: Set the environment for the native build
    call "%VS160COMNTOOLS%..\..\VC\Auxiliary\Build\vcvarsall.bat" %__VCBuildArch%
)
goto :SetupDirs

:VS2017
:: Setup vars for VS2017
set __PlatformToolset=v141
set __VSVersion=15 2017
if NOT "%__BuildArch%" == "arm64" (
    :: Set the environment for the native build
    call "%VS150COMNTOOLS%..\..\VC\Auxiliary\Build\vcvarsall.bat" %__VCBuildArch%
)
goto :SetupDirs

:VS2015
:: Setup vars for VS2015build
set __PlatformToolset=v140
set __VSVersion=14 2015
if NOT "%__BuildArch%" == "arm64" (
    :: Set the environment for the native build
    call "%VS140COMNTOOLS%..\..\VC\vcvarsall.bat" %__VCBuildArch%
)

:SetupDirs
:: Setup to cmake the native components
echo Commencing native build of dotnet/machinelearning
echo.

if %__CMakeBinDir% == "" (
    set "__CMakeBinDir=%__binDir%\%__BuildArch%.%CMAKE_BUILD_TYPE%\Native"
)
if %__IntermediatesDir% == "" (
    set "__IntermediatesDir=%__binDir%\obj\%__BuildArch%.%CMAKE_BUILD_TYPE%\Native"
)
set "__CMakeBinDir=%__CMakeBinDir:\=/%"
set "__IntermediatesDir=%__IntermediatesDir:\=/%"

:: Strip the "-netcoreapp3_0" suffix from the build type
if [%CMAKE_BUILD_TYPE:~-14%] == [-netcoreapp3_0] (
	set CMAKE_BUILD_TYPE=%CMAKE_BUILD_TYPE:~0,-14%
)

:: Strip the "-netfx" suffix from the build type
if [%CMAKE_BUILD_TYPE:~-6%] == [-netfx] (
	set CMAKE_BUILD_TYPE=%CMAKE_BUILD_TYPE:~0,-6%
)

:: Check that the intermediate directory exists so we can place our cmake build tree there
if not exist "%__IntermediatesDir%" md "%__IntermediatesDir%"

:: Regenerate the VS solution

set "__gen-buildsys-win-path=%__currentScriptDir%\gen-buildsys-win.bat"
set "__source-code-path=%__currentScriptDir%"

echo Calling "%__gen-buildsys-win-path%" "%__source-code-path%" "%__VSVersion%" %__BuildArch%
pushd "%__IntermediatesDir%"
call "%__gen-buildsys-win-path%" "%__source-code-path%" "%__VSVersion%" %__BuildArch%
popd

:CheckForProj
:: Check that the project created by Cmake exists
if exist "%__IntermediatesDir%\INSTALL.vcxproj" goto BuildNativeProj
goto :Failure

:BuildNativeProj
:: Build the project created by Cmake
set __msbuildArgs=/p:Platform=%__BuildArch% /p:PlatformToolset="%__PlatformToolset%"

cd %__rootDir%

echo msbuild "%__IntermediatesDir%\INSTALL.vcxproj" /t:rebuild /p:Configuration=%CMAKE_BUILD_TYPE% %__msbuildArgs%
call msbuild "%__IntermediatesDir%\INSTALL.vcxproj" /t:rebuild /p:Configuration=%CMAKE_BUILD_TYPE% %__msbuildArgs%
IF ERRORLEVEL 1 (
    goto :Failure
)
echo Done building Native components
exit /B 0

:Failure
:: Build failed
echo Failed to generate native component build project!
exit /b 1
