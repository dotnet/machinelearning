# OneDAL-supported algorithms

The [Intel OneDAL](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/api-based-programming/intel-oneapi-data-analytics-library-onedal.html) ("DAL" stands for "Data Analytics Library") includes accelerated implementations of several popular Machine Learning kernels that complement those existing in ML.NET.  In [this branch](https://github.com/dotnet/machinelearning/tree/onedal-integration) we provide a work-in-progress integration of OneDAL into ML.NET.  Currently, this work consists of:

* A "native" component (under `src/Native/Microsoft.ML.OneDal`)
* Integration into the relevant learners (currently OLS (in `src/Microsoft.ML.Mkl.Components` and  `Microsoft.ML.StandardTrainers`))
* Unit tests (under `test/Microsoft.ML.Tests/TrainerEstimators/OneDALComponentsTests.cs` currently, but will extend)
* A "sample" that doubles as a benchmark driver and a convenient end-to-end target for debugging (in `docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Regression/TestingOLS.cs`)

## Building

This code is meant to be built and used on Windows or Linux (bare-metal, testing within the WSL is underway)
1. Get the OneDAL (this step is temporary, while we sort out how to share `nupkg`s)
    * On Linux (Ubuntu), you can use the most recent `intel-oneapi-dal-common-devel`
    * On Windows (and Linux) you can use the [OneAPI installer](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html), making sure to select the "OneDAL" component
1. The above installation options include the OneDAL CMake modules that the ML.NET `build.*` script relies on.  Note that this is *not* the case for OneDAL installed via `conda`, so we discourage that route.
1. No build requirements beyond the ones of ML.NET are expected, issuing the `build` script that corresponds to your platform should suffice

## Testing

For an example on how to make use of the algorithms the OneDAL provides you can take a look at the samples included in `docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Regression/TestingOLS.cs`.  To engage the OneDAL-based implementations, you'll need to set the `MLNET_BACKEND` environment variable with the value "ONEDAL" (if you leave `MLNET_BACKEND` unset or with any other value, the behavior from the "main" branch is unchanged).  This is expected to be the main "switching" control for implementations for the time being for accelerated implementations of algorithms that already exist in ML.NET.

*Only for the purposes of testing*, the sample expects the specification of the task and the dataset name in the env variables `MLNET_BENCHMARK_TASK` and `MLNET_BENCHMARK_DATASET` respectively.  Which task to exercise is encoded in the values "reg" (for regression), "bin" (for binary classification) and "mcl" (for multi-class classification).  The datasets we've been using to try out our implementations are pre-split in training/testing sets, encoded in csv (with the "_train", "_test" suffix in the name of the file) and living in `test/data`.  These datasets are *not* versioned due to the files being too big, but can be shared (probably using the same mechanism we decide to use for communicating nupkgs)

<blockquote>
More details in this section forthcoming.
</blockquote>
