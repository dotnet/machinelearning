Building ML.NET on Linux and macOS
==========================================
## Building

1. Install the prerequisites ([Linux](#user-content-linux), [macOS](#user-content-macos))
2. Clone the machine learning repo `git clone https://github.com/dotnet/machinelearning.git`
3. Navigate to the `machinelearning` directory
4. Run the build script `./build.sh`

Calling the script `./build.sh` builds both the native and managed code.

For more information about the different options when building, run `./build.sh -?` and look at examples in the [developer-guide](../project-docs/developer-guide.md).

## Minimum Hardware Requirements
- 2GB RAM
- x64

## Prerequisites

### Linux

The following components are needed:

* git
* clang-3.9
* cmake 2.8.12
* libunwind8
* curl
* All the requirements necessary to run .NET Core 2.0 applications: libssl1.0.0 (1.0.2 for Debian 9) and libicu5x (libicu52 for ubuntu 14.x, libicu55 for ubuntu 16.x, and libicu57 for ubuntu 17.x). For more information on prerequisites in different linux distributions click [here](https://docs.microsoft.com/en-us/dotnet/core/linux-prerequisites?tabs=netcore2x).

e.g. for Ubuntu 16.x:

```sh
sudo apt-get update
sudo apt-get install git clang-3.9 cmake libunwind8 curl
sudo apt-get install libssl1.0.0 libicu55
```

### macOS

macOS 10.12 (Sierra) or higher is needed to build dotnet/machinelearning.

On macOS a few components are needed which are not provided by a default developer setup:
* cmake 3.10.3
* gcc
* All the requirements necessary to run .NET Core 2.0 applications. To view macOS prerequisites click [here](https://docs.microsoft.com/en-us/dotnet/core/macos-prerequisites?tabs=netcore2x).

One way of obtaining CMake and gcc is via [Homebrew](https://brew.sh):
```sh
$ brew install cmake
$ brew install gcc
```
