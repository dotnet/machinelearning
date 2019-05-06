# ML.NET 1.0.0-preview Release Notes

This release is `Release Candidate` for version `1.0.0` of [ML.NET](https://aka.ms/mlnet). We have closed our main [API project](https://github.com/dotnet/machinelearning/projects/13). The next release will be `1.0.0` and during this sprint we are focusing on improving documentation and samples and consider addressing major critical issues. The goal is to avoid any new breaking changes going forward. One change in this release is that we have moved `IDataView` back into `Microsoft.ML` namespace based on some feedback that we received.

### Installation

ML.NET supports Windows, MacOS, and Linux. See [supported OS versions of .NET Core 2.0](https://github.com/dotnet/core/blob/master/release-notes/2.0/2.0-supported-os.md) for more details.

You can install ML.NET NuGet from the CLI using:
```
dotnet add package Microsoft.ML
```

From package manager:
```
Install-Package Microsoft.ML
```

### Release Notes

Below are a few of the highlights from this release. There are many other improvements in the API.

* Move `IDataView` into `Microsoft.ML` namespace. ([#2987](https://github.com/dotnet/machinelearning/pull/2987))
* Move KeyType, VectorType and VBuffer to `ML.DataView`. ([#3022](https://github.com/dotnet/machinelearning/pull/3022))
* Remove ConcurrencyFactor from `IHostEnvironment`. ([#2846](https://github.com/dotnet/machinelearning/pull/2846))
* More work in reorganizing namespaces related to issue: ([#2751](https://github.com/dotnet/machinelearning/issues/2751))
* Remove Value-tuples in the public API. ([#2950](https://github.com/dotnet/machinelearning/pull/2950))
* Categorizing NuGets into preview and stable. ([#2951](https://github.com/dotnet/machinelearning/pull/2951))
* Hiding `ColumnOptions`. ([#2959](https://github.com/dotnet/machinelearning/pull/2959))
* Asynchronous cancellation mechanism. ([#2797](https://github.com/dotnet/machinelearning/pull/2797))

### Acknowledgements

Shoutout to [MarcinJuraszek](https://github.com/MarcinJuraszek),
[llRandom](https://github.com/llRandom),
[jwood803](https://github.com/jwood803), [Potapy4](https://github.com/Potapy4) and the [ML.NET](https://aka.ms/mlnet) team for their
contributions as part of this release!
