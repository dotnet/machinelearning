ML.NET source and test code files build for .NET Core 3.1 and NET Framework 4.6.1.

If you want to run command line tests on only 1 of the frameworks or on a different framework, do this:
 add `/p:TestTargetFramework=<Framework Names>` to the command line build.

1. Run `.\build.cmd -configuration Debug /p:TestTargetFramework=<Framework Names>` or `.\build.cmd -configuration Release /p:TestTargetFramework=<Framework Names>` from the root of the repo.
2. If you want to build the NuGet packages, `.\build.cmd -pack` after step 1.

If you are using Visual Studio, the tests for both .NET Core 3.1 and NET Framework 4.6.1 can both be run from the UI with no changes. If you want to test using different frameworks, you will have to edit the `<TargetFrameworks>` found in `test/Directory.build.props`