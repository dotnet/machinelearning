# ML.NET Benchmarks

This project contains performance benchmarks.

## Run the Performance Tests

**Pre-requisite:** On a clean repo, `build.cmd` at the root installs the right version of dotnet.exe and builds the solution. You need to build the solution in `Release` with native dependencies. 

    build.cmd -release -buildNative
    
Moreover, to run some of the benchmarks you have to download external dependencies.

    build.cmd -- /t:DownloadExternalTestFiles /p:IncludeBenchmarkData=true

1. Navigate to the benchmarks directory (machinelearning\test\Microsoft.ML.Benchmarks)

2. Run the benchmarks in Release, choose one of the benchmarks when prompted

```log
    dotnet run -c Release
```
   
3. To run specific tests only, pass in the filter to the harness:

```log
    dotnet run -c Release -- --filter namespace*
    dotnet run -c Release -- --filter *typeName*
    dotnet run -c Release -- --filter *.methodName
    dotnet run -c Release -- --filter namespace.typeName.methodName
```

4. GC Statistics

To get the total number of allocated managed memory please pass additional console argument: `--memory` or just `-m`. This feature is disabled by default because it requires an additional iteration (which is expensive for time consuming benchmarks).

|       Gen 0 |      Gen 1 |     Gen 2 | Allocated |
|------------:|-----------:|----------:|----------:|
| 175000.0000 | 33000.0000 | 7000.0000 | 238.26 MB |

5. To find out more about supported command line arguments run

```log
    dotnet run -c Release -- --help
```
