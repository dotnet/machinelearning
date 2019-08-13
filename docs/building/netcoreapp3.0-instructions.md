ML.NET now builds for .NET Core 3.0 by default. However to run tests on .NET Core 3.0, you need to do a few manual steps.

1. Run `.\build.cmd -- /p:Configuration=Release-netcoreapp3_0` or `.\build.cmd -Release-netcoreapp3_0` from the root of the repo.
2. If you want to build the NuGet packages, `.\build.cmd -buildPackages` after step 2.

If you are using Visual Studio, you will need to do the following:

1. In the Configuration Manager, switch the current configuration to `Debug-netcoreapp3_0` or `Release-netcoreapp3_0`.
2. Build and test as usual.