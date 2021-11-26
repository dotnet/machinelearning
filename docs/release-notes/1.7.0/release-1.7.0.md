# [ML.NET](http://dot.net/ml) 1.7.0

Moving forward, we are going to be aligning more with the overall .NET release schedule. As such, this is a smaller release since we had a larger one just about 3 months ago but it aligns us with the release of .NET 6.

## **New Features**
### ML.NET
- **Switched to getting version from assembly custom attributes**- ([#4512](https://github.com/dotnet/machinelearning/pull/4512)) Remove reliance on getting product version for model.zip/version.txt from FileVersionInfo and replace with using assembly custom attributes. This will help in supporting single file applications. (**Thanks @r0ss88**)
- **Can now optionally not dispose of the underlying model when you dispose a prediction engine**. ([#5964](https://github.com/dotnet/machinelearning/pull/5964)) A new prediction engine options class has been added that lets you determine if the underlying model should be disposed of or not when the prediction engine itself is disposed of.
- **Can now set the number of threads that onnx runtime uses** ([#5962](https://github.com/dotnet/machinelearning/pull/5962)) This lets you specify the number of parallel threads ONNX runtime will use to execute the graph and run the model. (**Thanks @yaeldekel**)
- **The PFI API has been completely reworked and is now much more user friendly** ([#5934](https://github.com/dotnet/machinelearning/pull/5934)) You can now get the output from PFI as a dictionary mapping the column name (or the slot name) to its PFI result.
### DataFrame
- **Can now merge using multiple columns in a JOIN condition** ([#5838](https://github.com/dotnet/machinelearning/pull/5838)) (**Thanks @asmirnov82**)

## **Enhancements**
### ML.NET
- Run formatting on all src projects ([#5937](https://github.com/dotnet/machinelearning/pull/5937)) (**Thanks @jwood803**)
- Added BufferedStream for reading from DeflateStream - reduces loading time for .NET core ([#5924](https://github.com/dotnet/machinelearning/pull/5924)) (**Thanks @martintomasek**)
- Update editor config to match Roslyn and format samples ([#5893](https://github.com/dotnet/machinelearning/pull/5893)) (**Thanks @jwood803**)
- Few more minor editor config changes ([#5933](https://github.com/dotnet/machinelearning/pull/5933))
### DataFrame
- Use Equals and = operator for DataViewType comparison ([#5942](https://github.com/dotnet/machinelearning/pull/5942)) (**Thanks @thoron**)



## **Bug Fixes**
- Initialize _bestMetricValue when using the Loss metric ([#5939](https://github.com/dotnet/machinelearning/pull/5939)) (**Thanks @MiroslavKabat**)


## **Build / Test updates**
- Changed the queues used for building/testing from Ubuntu 16.04 to 18.04 ([#5970](https://github.com/dotnet/machinelearning/pull/5970))
- Add in support for building with VS 2022. ([#5956](https://github.com/dotnet/machinelearning/pull/5956))
- Codecov yml token was added ([#5950](https://github.com/dotnet/machinelearning/pull/5950))
- Move from XliffTasks to Microsoft.DotNet.XliffTasks ([#5887](https://github.com/dotnet/machinelearning/pull/5887))


## **Documentation Updates**
- Fixed up Readme, updated the roadmap, and new doc detailing some platform limitations. ([#5892](https://github.com/dotnet/machinelearning/pull/5892))


## **Breaking Changes**
- None
