ML.NET source code files build for .NET Core 3.1 and .NET Standard 2.0. However, ML.NET test files only build on one target framework below at a time:
- .NET Core 2.1
- .NET Core 3.1
- .NET Framework 4.6.1

To run tests on .NET Core 3.1, you need to do a few manual steps.

1. Run `.\build.cmd -configuration Debug-netcoreapp3_1` or `.\build.cmd -configuration Release-netcoreapp3_1` from the root of the repo.
2. If you want to build the NuGet packages, `.\build.cmd -pack` after step 1.

If you are using Visual Studio, you will need to do the following:

1. In the Configuration Manager, switch the current configuration to `Debug-netcoreapp3_1` or `Release-netcoreapp3_1`.
2. Build and test as usual.