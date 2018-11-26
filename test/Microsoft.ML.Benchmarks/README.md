# ML.NET Benchmarks

This project contains performance benchmarks.

## Run the Performance Tests

**Pre-requisite:** In order to fetch dependencies which come through Git submodules the following command needs to be run before building:

    git submodule update --init

**Pre-requisite:** On a clean repo with initalized submodules, `build.cmd` at the root installs the right version of dotnet.exe and builds the solution. You need to build the solution in `Release` with native dependencies. 

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

## .NET Core 3.0

**Pre-requisite:** Follow the [netcoreapp3.0 instructions](../../docs/building/netcoreapp3.0-instructions.md).

**Pre-requisite:** To use dotnet cli from the root directory remember to set `DOTNET_MULTILEVEL_LOOKUP` environment variable to `0`!

    $env:DOTNET_MULTILEVEL_LOOKUP=0

1. Navigate to the benchmarks directory (machinelearning\test\Microsoft.ML.Benchmarks)

2. Run the benchmarks in `Release-Intrinsics` configuration, choose one of the benchmarks when prompted

```log
    ..\..\Tools\dotnetcli\dotnet.exe run -c Release-Intrinsics
```
## Authoring new benchmarks

1. The type which contains benchmark(s) has to be a public, non-sealed, non-static class.
2. Put the initialization logic into a separate public method with `[GlobalSetup]` attribute. You can use `Target` property to make it specific for selected benchmark. Example: `[GlobalSetup(Target = nameof(MakeIrisPredictions))]`.
3. Put the benchmarked code into a separate public method with `[Benchmark]` attribute. If the benchmark method computes some result, please return it from the benchmark. Harness will consume it to avoid dead code elimination. 
4. If given benchmark is a Training benchmark, please apply `[Config(typeof(TrainConfig))]` to the class. It will tell BenchmarkDotNet to run the benchmark only once in a dedicated process to mimic the real-world scenario for training.

Examples:

```cs
public class NonTrainingBenchmark
{
    [GlobalSetup(Target = nameof(TheBenchmark))]
    public void Setup() { /* setup logic goes here */ }

    [Benchmark]
    public SomeResult TheBenchmark() { /* benchmarked code goes here */  }
}

[Config(typeof(TrainConfig))]
public class TrainingBenchmark
```
