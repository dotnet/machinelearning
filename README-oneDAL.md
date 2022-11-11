# oneDAL supported algorithms

oneAPI Data Analytics Library (oneDAL) is a library providing highly optimized machine learning and data analytics kernels. Part of these kernels is integrated into ML.NET via C++/C# interoperability.

[oneDAL Documentation](http://oneapi-src.github.io/oneDAL/) | [oneDAL Repository](https://github.com/oneapi-src/oneDAL)

[`onedal-integration`](https://github.com/dotnet/machinelearning/tree/onedal-integration) branch provides work-in-progress integration of oneDAL into ML.NET.

Integration consists of:

* A "native" component (under `src/Native/Microsoft.ML.OneDal`) implementing wrapper to pass data and parameters to oneDAL;
* Dispatching to oneDAL kernels inside relevant learners: `OLS` (`src/Microsoft.ML.Mkl.Components`), `Logistic Regression` (`src/Microsoft.ML.StandardTrainers`), `Random Forest` (`src/Microsoft.ML.FastTree`);

## Building from source

This is instruction how to build ML.NET with integrated oneDAL on Linux, Windows and MacOS platforms:

1. Clone oneDAL from `develop` branch: `git clone -b develop https://github.com/oneapi-src/oneDAL.git`

2. Build oneDAL using [this instruction](https://github.com/oneapi-src/oneDAL/blob/develop/INSTALL.md)

3. Produce Nuget packages using next command template:

        deploy/nuget/prepare_dal_nuget.sh --template ./deploy/nuget/inteldal.nuspec.tpl  --release-dir __release_{system}[_{compiler}] --platform {system} --ver {version} --major-binary-ver 1 --minor-binary-ver 1 --build-nupkg yes

4. Add source path for produced Nuget packages in ML.NET NuGet.Config by adding line to `packageSources`: `<add key="onedal-local" value="{oneDAL repo path}/__nuget" />` 

5. Build ML.NET from source as usual.

## Running ML.NET trainers with dispatching to oneDAL kernels

Currently, dispatching to oneDAL inside ML.NET is regulated by `MLNET_BACKEND` environment variable. If it's set to `ONEDAL`, oneDAL kernel will be used, otherwise - default ML.NET.
