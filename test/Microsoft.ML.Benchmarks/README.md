# ML.NET Benchmarks

This project contains performance benchmarks.

## Run the Performance Tests

**Pre-requisite:** On a clean repo, `build.cmd` at the root installs the right version of dotnet.exe and builds the solution. You need to build the solution in `Release` with native dependencies. 

    build.cmd -release -buildNative

**Pre-requisite:** To use dotnet cli from the `Tools\dotnetcli` directory remember to set `DOTNET_MULTILEVEL_LOOKUP` environment variable to `0`!

    $env:DOTNET_MULTILEVEL_LOOKUP=0

1. Navigate to the benchmarks directory (machinelearning\test\Microsoft.ML.Benchmarks)

2. Run the benchmarks in Release, choose one of the benchmarks when prompted

```log
    ..\..\Tools\dotnetcli\dotnet.exe run -c Release
```
   
3. To run specific tests only, pass in the filter to the harness:

```log
    ..\..\Tools\dotnetcli\dotnet.exe run -c Release -- --filter namespace*
    ..\..\Tools\dotnetcli\dotnet.exe run -c Release -- --filter *typeName*
    ..\..\Tools\dotnetcli\dotnet.exe run -c Release -- --filter *.methodName
    ..\..\Tools\dotnetcli\dotnet.exe run -c Release -- --filter namespace.typeName.methodName
```

4. To find out more about supported command line arguments run

```log
    ..\..\Tools\dotnetcli\dotnet.exe run -c Release -- --help
```
