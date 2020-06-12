Developer Guide
===============

The repo can be built for the following platforms, using the provided setup and the following instructions.

| Chip  | Windows | Linux | OS X |
| :---- | :-----: | :---: | :--: |
| x64   | [Instructions](../building/windows-instructions.md) | [Instructions](../building/unix-instructions.md) | [Instructions](../building/unix-instructions.md) |


Building the repository
=======================

The ML.NET repo can be built from a regular, non-admin command prompt. The build produces multiple binaries that make up the ML.NET libraries and the accompanying tests.

Developer Workflow
------------------
The dev workflow describes the [development process](https://github.com/dotnet/buildtools/blob/master/Documentation/Dev-workflow.md) to follow. It is divided into specific tasks that are fast, transparent and easy to understand.
The tasks are represented in scripts (cmd/sh) in the root of the repo.

For more information about the different options that each task has, use the argument `-?` when calling the script.  For example:
```
build -?
```

**Examples**

- Initialize the repo to make build possible (if the build fails because it can't find `mf.cpp` then perhaps you missed this step)

```
git submodule update --init
```

- Building in release mode for platform x64
```
build.cmd -Release -TargetArchitecture:x64
```

- Building the src and then building and running the tests
```
build.cmd
build.cmd -runTests
```

### Building individual projects

**Note**: Before working on individual projects or test projects you **must** run `build` from the root once before beginning that work. It is also a good idea to run `build` whenever you pull a large set of unknown changes into your branch.

Under the src directory is a set of directories, each of which represents a particular assembly in ML.NET.  

For example the src\Microsoft.MachineLearning.Core directory holds the source code for the Microsoft.MachineLearning.Core.dll assembly.

You can build the DLL for Microsoft.MachineLearning.Core.dll by going to the `src\Microsoft.MachineLearning.Core` directory and typing `dotnet build`.

You can build the tests for Microsoft.MachineLearning.Core.dll by going to
`test\Microsoft.MachineLearning.Core.Tests` directory and typing `dotnet test`.

**Note:** We use build/vsts-ci.yml to define our official build

### Building in Release or Debug

By default, building from the root or within a project will build the libraries in Debug mode.
One can build in Debug or Release mode from the root by doing `build.cmd -Release` or `build.cmd -Debug`.

### Building other Architectures

We only support 64-bit binaries right now.

### Updating manifest and ep-list files

During development, there may arise a need to update the current baseline `core_manifest.json` and/or `core_ep-list.tsv` files. For example, a change in the name or type of a variable in a given class in the API that is not reflected in `core_manifest.json` will trigger the following failure:

`*** Failure: Output and baseline mismatch at line 123 , expected ' ...x... ' but got ' ...y...' : '../Common/EntryPoints/core_manifest.json'`

Steps to update `core_manifest.json` and `core_ep-list.tsv`:
1. Unskip the `RegenerateEntryPointCatalog` unit test in `test/Microsoft.ML.Core.Tests/UnitTests/TestEntryPoints.cs`. This can be done by temporarily commenting out the skip attribute on the unit test for `RegenerateEntryPointCatalog` (`[Fact(Skip = "Execute this test if you want to regenerate the core_manifest and core_ep_list files")]`).
2. Run the unit tests `build.cmd -runTests` (alternatively, run the `RegenerateEntryPointCatalog` unit test natively on Visual Studio through the Test Explorer or through Shortcuts by clicking on `RegenerateEntryPointCatalog` in `test/Microsoft.ML.Core.Tests/UnitTests/TestEntryPoints.cs` and pressing Ctrl+R,T).
3. Verify the changes to `core_manifest.json` and `core_ep-list.tsv` are correct.
4. Re-enable the skip attribute on the `RegenerateEntryPointCatalog` test.
5. Commit the updated `core_manifest.json` and `core_ep-list.tsv` files to your branch.

### Running unit tests through VSTest Task & Collecting memory dumps

During development, there may also arise a need to debug hanging tests. In this scenario, it can be beneficial to collect the memory dump while a given test is hanging.

In this case, the given needs needs to be implemented according to the Microsoft test framework. Please check out the [Microsoft test framework walkthrough](https://docs.microsoft.com/en-us/visualstudio/test/walkthrough-creating-and-running-unit-tests-for-managed-code?view=vs-2019) and the VSTest [sample](https://github.com/dotnet/samples/tree/master/core/getting-started/unit-testing-using-mstest) demonstrating the "TestClass", "TestMethod", "DataTestMethod", and "DataRow" attributes.

Once the unit test(s) are implemented according to VSTest and ready to be debugged, the `useVSTestTask` parameter in `build\ci\job-template.yml` needs to be set to `True`. Once these steps are completed and pushed in your pull request, the unit test(s) will run and produce a full memory dump. At the end of a run, the memory dump `.dmp` file will be availible for downloading and inspection in the published artifacts of the build, in the folder `TestResults`.

Note: this is only supported on Windows builds, as [ProcDump](https://docs.microsoft.com/en-us/sysinternals/downloads/procdump) is officially only available on Windows.