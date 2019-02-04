In order to build ML.NET for .NET Core 3.0, you need to do a few manual steps.

1. Delete the local `.\Tools\` folder from the root of the repo, to ensure you download the new version of the .NET Core SDK.
2. Run `.\build.cmd -- /p:Configuration=Release-Intrinsics` or `.\build.cmd -Release-Intrinsics` from the root of the repo.
3. If you want to build the NuGet packages, `.\build.cmd -buildPackages` after step 2.

If you are using Visual Studio, you will need to do the following:

1. Install the above .NET Core 3.0 SDK into %Program Files%. Or extract it to a directory and put that directory at the front of your %PATH%, so it is the first `dotnet.exe` on the PATH.
2. In the Configuration Manager, switch the current configuration to `Debug-Intrinsics` or `Release-Intrinsics`.
3. Build and test as usual.