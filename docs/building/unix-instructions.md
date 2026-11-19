# Building ML.NET on Linux and macOS

## Building

1. Install Git
2. Clone the machine learning repo `git clone --recurse-submodules https://github.com/dotnet/machinelearning.git`
3. Navigate to the `machinelearning` directory
4. Run `git submodule update --init` if you have not previously done so
5. Install the native build prerequisites described below
6. Run the build script `./build.sh`

Calling the script `./build.sh` builds both the native and managed code.

For more information about the different options when building, run `./build.sh -?` and look at examples in the [developer-guide](../project-docs/developer-guide.md).

## Minimum hardware requirements

- 2GB RAM
- x64 or ARM64

## Prerequisites

After cloning and entering the repository, install the native build dependencies for your operating system.

On Linux:

```sh
sudo ./eng/install-dependencies.sh
```

The dependency script supports Debian/Ubuntu, Fedora, Azure Linux, Amazon Linux, and Alpine. On Azure Linux, install `libomp` manually before building.

On macOS, install Xcode command-line tools and [Homebrew](https://brew.sh/) first, then run:

```sh
./eng/install-dependencies.sh
```

The build scripts acquire the .NET SDK selected by `global.json`; you do not need to install that exact SDK separately.

### Cross compiling on Linux

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

The `<target architecture>` can be `x64`, `arm`, or `arm64`. Use an Ubuntu codename supported by `eng/common/cross/build-rootfs.sh`.

Linux cross-build jobs using `build/ci/job-template.yml` must pass the sysroot path through `rootfsDir` (for example, `/crossrootfs/arm64`). The template does not infer it from the agent image or job name.

### CI build containers

CI cross-builds use these Azure Linux 4 images:

- `mcr.microsoft.com/dotnet-buildtools/prereqs:azurelinux-4.0-net11.0-cross-amd64`
- `mcr.microsoft.com/dotnet-buildtools/prereqs:azurelinux-4.0-net11.0-cross-arm`
- `mcr.microsoft.com/dotnet-buildtools/prereqs:azurelinux-4.0-net11.0-cross-arm64`

They target Ubuntu Bionic sysroots for x64 and arm64 and Ubuntu Jammy for arm.
The sysroots include the target OpenMP packages. Set `ROOTFS_DIR` to the matching
directory (`/crossrootfs/x64`, `/crossrootfs/arm`, or `/crossrootfs/arm64`) and
set `LDFLAGS=-fuse-ld=lld` to use the bundled LLVM linker.

The nightly pipeline still references the Ubuntu 22.04 ML.NET build image.

The official x64 native-package build retains its CentOS Stream 8 image to avoid
raising the native libraries' minimum system-library requirements. Updating a
build container must not silently change the compatibility baseline of shipped
native assets.
