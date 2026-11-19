# [ML.NET](https://dot.net/ml) 6.0.0

Package versions: stable ML.NET packages **6.0.0**, experimental packages (including DataFrame, AutoML, and TorchSharp) **0.24.0**, and Tokenizers and its data packages **3.0.0**.

## **New Features**
- **Support GPT-5.1 model in Tiktoken tokenizer** ([#7556](https://github.com/dotnet/machinelearning/pull/7556))
- **Support GPT-5.2 in Tokenizer library** ([#7571](https://github.com/dotnet/machinelearning/pull/7571))
- **Add GPT-5.3 model support to TiktokenTokenizer** ([#7579](https://github.com/dotnet/machinelearning/pull/7579))
- **Add GPT-5.4 to Tiktoken tokenizer** ([#7591](https://github.com/dotnet/machinelearning/pull/7591))
- **Add Melt method to DataFrame** ([#7578](https://github.com/dotnet/machinelearning/pull/7578)) - Thanks @sevenzees!
- **Add GPT-5.5 and GPT-5.6 tokenizer mappings** ([#7661](https://github.com/dotnet/machinelearning/pull/7661))
- **Add SentencePieceTokenizer factory methods for Unigram from a vocabulary list and tokenizer.json** ([#7625](https://github.com/dotnet/machinelearning/pull/7625))
- **Add GPT-6 support to TiktokenTokenizer** ([#7693](https://github.com/dotnet/machinelearning/pull/7693))
- **Enable SymbolicSgdLogisticRegression on ARM64** ([#7671](https://github.com/dotnet/machinelearning/pull/7671))


## **Enhancements**
- **Add heap-based BPE merge path for pre-tokenized pieces longer than 128 bytes** ([#7580](https://github.com/dotnet/machinelearning/pull/7580))
- **Remove Google.Protobuf dependency from Microsoft.ML.Tokenizers** ([#7587](https://github.com/dotnet/machinelearning/pull/7587))
- **Optimize managed FastTree Sumup on ARM64** ([#7670](https://github.com/dotnet/machinelearning/pull/7670))
- **Avoid Microsoft.Bcl.Memory on modern .NET targets** ([#7703](https://github.com/dotnet/machinelearning/pull/7703))

  .NET 8+ applications no longer require `Microsoft.Bcl.Memory` through these packages. .NET Standard 2.0 support remains, with the dependency updated to 9.0.14. This change was also included in ML.NET 5.0.1.

- **Use Unsafe.As for FloatUtils bit conversions** ([#7676](https://github.com/dotnet/machinelearning/pull/7676)) - Thanks @abhishekdahad-1!

## **Bug Fixes**
- **Fix Merge failures with empty DataFrames** ([#7573](https://github.com/dotnet/machinelearning/pull/7573)) - Thanks @sevenzees!
- **Fix memory leaks and stale-model results when hot-swapping PredictionEnginePool** ([#7626](https://github.com/dotnet/machinelearning/pull/7626))
- **Fix SR-CNN missing anomalies when Period > 0** ([#7610](https://github.com/dotnet/machinelearning/pull/7610)) - Thanks @FanBaoMS!
- **Fix the endless loop in Join and Merge when both suffixes are the same** ([#7692](https://github.com/dotnet/machinelearning/pull/7692)) - Thanks @HuzaifaChaudary!
- **Preserve LightGBM validation groups after dataset loading** ([#7782](https://github.com/dotnet/machinelearning/pull/7782))

## **Deprecation**

- **Remove GenAI packages** ([#7738](https://github.com/dotnet/machinelearning/pull/7738))

  `Microsoft.ML.GenAI.Core`, `Microsoft.ML.GenAI.LLaMA`, `Microsoft.ML.GenAI.Mistral`, and `Microsoft.ML.GenAI.Phi` are not included in this release and have no corresponding 0.24.0 upgrade.
  These were experimental packages and are no longer planned for further development, so they are not part of ML.NET 6 and will not receive further updates.
  `Microsoft.ML.TorchSharp` and `Microsoft.ML.Tokenizers` remain available.

**Full changelog:** [ML.NET 5.0.0 to the 6.0.0 release build](https://github.com/dotnet/machinelearning/compare/v5.0.0...6eff5ccedce81728c927bfa0155284aadd38531d)
