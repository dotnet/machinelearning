In order to build ML.NET for .NET Core 3.0, you need to do a few manual steps.

1. Pick a version of the .NET Core 3.0 SDK you want to use. As of this writing, I'm using `3.0.100-alpha1-009622`. You can get the latest available version from the [dotnet/core-sdk README](https://github.com/dotnet/core-sdk#installers-and-binaries) page.
2. Change the [DotnetCLIVersion.txt](https://github.com/dotnet/machinelearning/blob/master/DotnetCLIVersion.txt) file to use that version number.
3. Delete the local `.\Tools\` folder from the root of the repo, to ensure you download the new version.
4. Run `.\build.cmd -- /p:Configuration=Release-Intrinsics` from the root of the repo.
5. If you want to build the NuGet packages, `.\build.cmd -buildPackages` after step 4.

If you are using Visual Studio, you will need to do the following:

1. Install the above .NET Core 3.0 SDK into %Program Files%. Or extract it to a directory and put that directory at the front of your %PATH%, so it is the first `dotnet.exe` on the PATH.
2. In the Configuration Manager, switch the current configuration to `Debug-Intrinsics` or `Release-Intrinsics`.
3. Build and test as usual.