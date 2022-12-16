# oneDAL supported algorithms

oneAPI Data Analytics Library (oneDAL) is a library providing highly optimized machine learning and data analytics kernels. Some of these kernels is integrated into ML.NET via C++/C# interoperability.

[oneDAL Documentation](http://oneapi-src.github.io/oneDAL/) | [oneDAL Repository](https://github.com/oneapi-src/oneDAL)

> Please note that oneDAL acceleration paths are only available in x64 architectures

Integration consists of:

* A "native" component (under `src/Native/Microsoft.ML.OneDal`) implementing wrapper to pass data and parameters to oneDAL;
* Dispatching to oneDAL kernels inside relevant learners: `OLS` (`src/Microsoft.ML.Mkl.Components`), `Logistic Regression` (`src/Microsoft.ML.StandardTrainers`), `Random Forest` (`src/Microsoft.ML.FastTree`);

## Running ML.NET trainers with dispatching to oneDAL kernels

Currently, dispatching to oneDAL inside ML.NET is regulated by `MLNET_BACKEND` environment variable. If it's set to `ONEDAL`, oneDAL kernel will be used, otherwise - default ML.NET.
