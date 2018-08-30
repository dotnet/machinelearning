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