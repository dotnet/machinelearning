# [ML.NET](http://dot.net/ml) 5.0.1

Package versions: stable ML.NET packages **5.0.1**, experimental packages (including DataFrame and GenAI) **0.23.1**, and Tokenizers and its data packages **2.0.1**.

## **Enhancements**
- **[release/5.0] Avoid Microsoft.Bcl.Memory on modern .NET targets** ([#7773](https://github.com/dotnet/machinelearning/pull/7773))

  Compatibility note: .NET 8+ applications no longer require `Microsoft.Bcl.Memory` through these packages. .NET Standard 2.0 builds remain available with the dependency updated to **9.0.14**.

- **[release/5.0] Update Semantic Kernel to 1.71.0 with GenAI-scoped dependencies** ([#7783](https://github.com/dotnet/machinelearning/pull/7783))

## **Build / Test updates**
- **[release/5.0] Restore CI infrastructure and test downloads, and stabilize SDCA tests** ([#7695](https://github.com/dotnet/machinelearning/pull/7695), [#7729](https://github.com/dotnet/machinelearning/pull/7729))
- **[release/5.0] Update macOS build machines and Arcade dependencies** ([#7560](https://github.com/dotnet/machinelearning/pull/7560), [#7776](https://github.com/dotnet/machinelearning/pull/7776))
