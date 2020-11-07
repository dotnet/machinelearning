# ML.NET Benchmarks/Performance Tests

This project contains performance benchmarks.

## Run the Performance Tests

**Pre-requisite:** In order to fetch dependencies which come through Git submodules the following command needs to be run before building:

    git submodule update --init

**Pre-requisite:** On a clean repo with initialized submodules, `build.cmd` at the root installs the right version of dotnet.exe and builds the solution. You need to build the solution in `Release`. 

    build.cmd -configuration Release

1. Navigate to the performance tests directory (machinelearning\test\Microsoft.ML.PerformanceTests)

2. Run the benchmarks in Release:

```log
    build.cmd -configuration Release -performanceTest
```

## .NET Core 3.1

**Pre-requisite:** Follow the [netcoreapp3.1 instructions](../../docs/building/netcoreapp3.1-instructions.md).

**Pre-requisite:** To use dotnet cli from the root directory remember to set `DOTNET_MULTILEVEL_LOOKUP` environment variable to `0`!

    $env:DOTNET_MULTILEVEL_LOOKUP=0

1. Navigate to the benchmarks directory (machinelearning\test\Microsoft.ML.PerformanceTests)

2. Run the benchmarks in `Release-netcoreapp3_1` configuration:

```log
    build.cmd -configuration Release-netcoreapp3_1 -performanceTest
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
## Running the `BenchmarksProjectIsNotBroken`  test

If your build is failing in the build machines, in the release configuration due to the `BenchmarksProjectIsNotBroken` test failing, 
you can debug this test locally by:

1- Building the solution in the release mode locally

build.cmd -configuration Release -performanceTest

2- Changing the configuration in Visual Studio from Debug -> Release
3- Changing the annotation in the `BenchmarksProjectIsNotBroken` to replace `BenchmarkTheory` with `Theory`, as below. 

```cs
[Theory]
[MemberData(nameof(GetBenchmarks))]
public void BenchmarksProjectIsNotBroken(Type type)

```

4- Restart Visual Studio
5- Proceed to running the tests normally from the Test Explorer view. 