ML.NET is a modular set of libraries that enables building a pipeline from data loaders, trainers/estimators (in case of training), transformers (in case of inferencing), and various data structures to facilitate the building of pipelines and representing data.  The core of ML.NET – the Microsoft.ML package has no external dependencies. It's largely managed code.  

Microsoft.ML does have a helper native math library CPUMath - which is only used on .NETFramework. .NET 6.0 and later have a managed implementation using intrinsics/TensorPrimitives and do not require the native build of CPUMath.

Microsoft.ML contains one other native library, LDANative, which is used by the LatentDirichletAllocationEstimator/Transform to support the LightLDA algorithm.  If this component is used it will require the LightLDA native library.  The native library is built for linux-arm, linux-arm64, linux-x64, osx-arm64 (M1), osx-x64, win-arm64, win-x64, win-x86.  This library has only platform/CRT dependencies.

Some components that represent an algorithm or binding to another framework are factored into separate packages to allow opt-in to using those and their dependencies.

ML.NET redistributes Intel MKL as Microsoft.ML.MKL.Redist in which is a minimized MKL library linked with just exports needed by ML.NET.  This component follows the support matrix of Intel MKL and is only supported on x86 and x64 architectures: linux-x64, osx-x64 (no longer supported by Intel), win-x64, and win-x86.  Similarly some components have light-up to use an Intel OneDAL implementation which is only supported on x64.

| NuGet Package                       | Entry-Point Components                                         | Native Dependencies                            | Status   | Notes                                                                                      |
|-------------------------------------|----------------------------------------------------------------|------------------------------------------------|----------|--------------------------------------------------------------------------------------------|
| `Microsoft.ML`                      | `MLContext`, core transforms, trainers                         | None                                           | Stable   |                                                                                            |
| `Microsoft.Extensions.ML`           | `PredictionEnginePool`                                         | None                                           | Stable   |                                                                                            |
| `Microsoft.ML.AutoML`               | `AutoCatalog` for AutoML                                       | *As required by other components*              | Preview  | Support varies based on components used                                                    |
| `Microsoft.ML.CodeGenerator`        |                                                                | None                                           | Preview  | Part of AutoML                                                                             |
| `Microsoft.ML.CpuMath`              |                                                                | Optional native                                | Stable   | Internal implementation; only used on .NET Framework                                       |
| `Microsoft.ML.DataView`             | `IDataView`                                                    | None                                           | Stable   |                                                                                            |
| `Microsoft.ML.DnnImageFeaturizer.*` |                                                                | None                                           | Preview  | Data-only                                                                                  |
| `Microsoft.ML.Ensemble`             |                                                                | None                                           | Preview  | Supports ML.NET component catalog                                                          |
| `Microsoft.ML.EntryPoints`          |                                                                | None                                           | Preview  | Supports ML.NET component catalog                                                          |
| `Microsoft.ML.Experimental`         |                                                                | None                                           | Preview  | Experimental API                                                                           |
| `Microsoft.ML.FairLearn`            | `FairlearnCatalog`                                             | None                                           | Preview  |                                                                                            |
| `Microsoft.ML.FastTree`             | `FastTreeRankingTrainer`                                       | Optional native acceleration                   | Stable   | Native library used on x86/x64; managed fallback                                           |
| `Microsoft.ML.ImageAnalytics`       | `MLImage` (image exchange type)                                | `libSkiaSharp`                                 | Stable   | Wrapper over SkiaSharp / Google Skia; supported where dependency is supported              |
| `Microsoft.ML.LightGBM`             | `LightGbm\*Trainer`                                            | `LightGBM`                                     | Stable   | Wrapper over LightGBM; supported where dependency is supported                             |
| `Microsoft.ML.MKL.Components`       | `SymbolicSgdLogisticRegressionBinaryTrainer`                   | Intel MKL                                      | Stable   | Only works where Intel MKL works                                                           |
| `Microsoft.ML.MKL.Redist`           | Internal native Intel MKL                                      | `libomp`                                       | Stable   | Not for direct reference; win-x86/x64 only                                                 |
| `Microsoft.ML.OneDal`               | Internal native Intel OneDal                                   | Intel OneDAL                                   | Preview  | Not for direct reference; x64 only                                                         |
| `Microsoft.ML.OnnxConverter`        | Adds ONNX export support                                       | `Microsoft.ML.OnnxRuntime`                     | Stable   | Wrapper over ONNX Runtime; supports "bring your own" runtime                               |
| `Microsoft.ML.OnnxTransformer`      | `OnnxCatalog`                                                  | `Microsoft.ML.OnnxRuntime`                     | Stable   | Wrapper over ONNX Runtime; supports "bring your own" runtime                               |
| `Microsoft.ML.Parquet`              | `ParquetLoader`                                                | None                                           | Preview  | Uses managed Parquet.Net (port of Apache Parquet)                                          |
| `Microsoft.ML.Recommender`          | `MatrixFactorizationTrainer`                                   | LIBMF (bundled)                                | Stable   | Includes libmf built for all runtimes supported by ML.NET                                  |
| `Microsoft.ML.TensorFlow`           | `TensorFlowModel`, `Transformer`, `Estimator`                  | TensorFlow via `TensorFlow.NET`                | Stable   | Wrapper over TensorFlow; supports "bring your own" runtime                                 |
| `Microsoft.ML.TimeSeries`           | `ForecastingCatalog`                                           | Intel MKL, `libomp`                            | Stable   | Only works where Intel MKL works                                                           |
| `Microsoft.ML.TorchSharp`           | `QATrainer`, `TextClassificationTrainer`, `SentenceSimilarityTrainer` | libTorch via `TorchSharp`               | Preview  | Wrapper over libTorch; supported where TorchSharp is supported                             |
| `Microsoft.ML.Vision`               | `ImageClassificationTrainer`                                   | TensorFlow                                     | Stable   | Depends on `Microsoft.ML.TensorFlow` for implementation                                    |


Other packages:
| NuGet Package                    | Entry-Point Components                           | Native Dependencies                           | Status     | Notes   |
|----------------------------------|--------------------------------------------------|-----------------------------------------------|------------|---------|
| `Microsoft.Data.Analysis`        | `DataFrame`                                      | `Apache.Arrow`                                | Preview    |         |
| `Microsoft.ML.GenAI.*`           |                                                  |                                               | Preview    |         |
| `Microsoft.ML.Tokenizers.*`      | `Tokenizer`                                      |                                               | Stable     |         |
| `Microsoft.ML.SampleUtils`       |                                                  |                                               | Preview    |         |





