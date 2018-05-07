Building ML.NET on Linux and macOS
==========================================
## Building

1. Install the prerequisites ([Linux](#user-content-linux), [macOS](#user-content-macos))
2. Clone the machine learning repo `git clone https://github.com/dotnet/machinelearning.git`
3. Navigate to the `machinelearning` directory
4. Run the build script `./build.sh`

Calling the script `build.sh` builds both the native and managed code.

For more information about the different options when building, run `build.sh -?` and look at examples in the [developer-guide](../project-docs/developer-guide.md).

## Minimum Hardware Requirements
- 2GB RAM
- x64

## Prerequisites

### Linux

On Linux, the following components are needed:

* git
* clang-3.9
* cmake 2.8.12
* libunwind8
* curl
* All the requirements necessary to run .NET Core 2.0 applications

e.g. for Ubuntu 14.04, follow the steps below:

`sudo apt-get update`

`sudo apt-get install git clang-3.9 cmake libunwind8 curl`

Follow instructions on how to [install .NET Core SDK 2.0+ on Ubuntu](https://www.microsoft.com/net/learn/get-started/linux/ubuntu14-04).

### macOS

macOS 10.12 or higher is needed to build dotnet/machinelearning.

On macOS a few components are needed which are not provided by a default developer setup:
* cmake 3.10.3

One way of obtaining CMake is via [Homebrew](http://brew.sh):
```sh
$ brew install cmake
```