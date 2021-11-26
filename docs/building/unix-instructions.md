Building ML.NET on Linux and macOS
==========================================
## Building

1. Install the prerequisites ([Linux](#user-content-linux), [macOS](#user-content-macos))
2. Clone the machine learning repo `git clone --recursive https://github.com/dotnet/machinelearning.git`
3. Navigate to the `machinelearning` directory
4. Run `git submodule update --init` if you have not previously done so
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
* libomp-dev
* curl
* All the requirements necessary to run .NET Core 3.1 applications: libssl1.0.0 (1.0.2 for Debian 9) and libicu5x (libicu55 for ubuntu 16.x, and libicu57 for ubuntu 17.x). For more information on prerequisites in different linux distributions click [here](https://docs.microsoft.com/en-us/dotnet/core/linux-prerequisites?tabs=netcore30).

For example, for Ubuntu 16.x:

```sh
sudo apt-get update
sudo apt-get install git clang-3.9 cmake libunwind8 curl
sudo apt-get install libssl1.0.0 libicu55
sudo apt-get install libomp-dev
```

#### Cross compiling for ARM

Cross compilation is only supported on an Ubuntu host, 18.x and newer, and only .Net Core 3.1 or newer. You will need to install debootstrap and qemu-user-static to facilitate the process. Once they are installed you will need to build the cross compiling rootfs. We provide a script, `build-rootfs.sh`, to do this. You will also need to set the ROOTFS_DIR environment variable to the location of the rootfs you just created. The general process is as follows:

```sh
sudo apt-get update
sudo apt-get install debootstrap qemu-user-static
sudo apt-get install binutils-aarch64-linux-gnu
sudo ./eng/common/cross/build-rootfs.sh <target architecture> <ubuntu distro name> --rootfsdir <new rootfs location>
export ROOTFS_DIR=<new rootfs location>

# The cross compiling environment is now setup and you can proceed with a normal build
./build.sh -c Release /p:TargetArchitecture=<target architecture>
```

Note that the `<target architecture>` will usually be arm or arm64 and the `<ubuntu distro name>` is bionic for 18.04.

Alternatively, use the following Docker image which contains all the software packages and configurations required to cross-compile for ARM `mcr.microsoft.com/dotnet-buildtools/prereqs:ubuntu-18.04-mlnet-cross-arm64-20210519131124-2e59a5f`.

### macOS

macOS 10.13 (High Sierra) or higher is needed to build dotnet/machinelearning. We are using a .NET Core 3.1 SDK to build, which supports 10.13 or higher.

On macOS a few components are needed which are not provided by a default developer setup:
* cmake 3.10.3
* libomp 7
* libgdiplus
* gettext
* All the requirements necessary to run .NET Core 3.1 applications. To view macOS prerequisites click [here](https://docs.microsoft.com/en-us/dotnet/core/install/macos?tabs=netcore31#dependencies).

One way of obtaining CMake and other required libraries is via [Homebrew](https://brew.sh):
```sh
$ brew update && brew install cmake https://raw.githubusercontent.com/dotnet/machinelearning/main/build/libomp.rb mono-libgdiplus gettext && brew link gettext --force && brew link libomp --force
```

Please note that newer versions of Homebrew [don't allow installing directly from a URL](https://github.com/Homebrew/brew/issues/8791). If you run into this issue, you may need to download libomp.rb first and install it with the local file instead.

Also, libomp version 7.0.0 doesn't have a cask for Big Sur. You can work around this by downloading the libomp.rb file and then calling `brew install libomp.rb --build-from-source --formula`.
