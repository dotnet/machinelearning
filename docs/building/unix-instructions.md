# Building ML.NET on Linux and macOS

## Building

1. Install the prerequisites described below
2. Clone the machine learning repo `git clone --recursive https://github.com/dotnet/machinelearning.git`
3. Navigate to the `machinelearning` directory
4. Run `git submodule update --init` if you have not previously done so
5. Run the build script `./build.sh`

Calling the script `./build.sh` builds both the native and managed code.

For more information about the different options when building, run `./build.sh -?` and look at examples in the [developer-guide](../project-docs/developer-guide.md).

## Minimum hardware requirements

- 2GB RAM
- x64 or ARM64

## Prerequisites

Install Git and the native build dependencies for your operating system. The repository's dependency script is the source of truth for the required packages:

```sh
sudo ./eng/common/native/install-dependencies.sh
```

The script supports common Debian/Ubuntu, Fedora/RHEL, Azure Linux, Amazon Linux, Alpine, and macOS environments. On macOS, install Xcode command-line tools and [Homebrew](https://brew.sh/) first.

The build scripts acquire the .NET SDK selected by `global.json`; you do not need to install that exact SDK separately.

### Cross compiling for ARM

Cross-compilation requires an Ubuntu host, `debootstrap`, and `qemu-user-static`. Build a root file system with the repository script, then set `ROOTFS_DIR` before building:

```sh
sudo apt-get update
sudo apt-get install debootstrap qemu-user-static
sudo apt-get install binutils-aarch64-linux-gnu
sudo ./eng/common/cross/build-rootfs.sh <target architecture> <ubuntu distro name> --rootfsdir <new rootfs location>
export ROOTFS_DIR=<new rootfs location>

# The cross compiling environment is now setup and you can proceed with a normal build
./build.sh -c Release /p:TargetArchitecture=<target architecture>
```

The `<target architecture>` is typically `arm` or `arm64`. Use an Ubuntu codename supported by `eng/common/cross/build-rootfs.sh`.
