#!/usr/bin/env bash
set -e

usage()
{
    echo "Usage: $0 --arch <Architecture> "
    echo ""
    echo "Options:"
    echo "  --arch <Architecture>             Target Architecture (x64, x86)"
    echo "  --configuration <Configuration>   Build Configuration (Debug, Release)"
    echo "  --stripSymbols                    Enable symbol stripping (to external file)"
    echo "  --mkllibpath                      Path to mkl library."
    exit 1
}

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ "$SOURCE" != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
RootRepo="$DIR/../.."

__build_arch=
__strip_argument=
__configuration=Debug
__rootBinPath="$RootRepo/artifacts/bin"
__baseIntermediateOutputPath="$RootRepo/artifacts/obj"
__versionSourceFile="$__baseIntermediateOutputPath/version.c"
__mkllibpath=""
__mkllibrpath=""

while [ "$1" != "" ]; do
        lowerI="$(echo $1 | awk '{print tolower($0)}')"
        case $lowerI in
        -h|--help)
            usage
            exit 1
            ;;
        --arch)
            shift
            __build_arch=$1
            ;;
        --configuration)
            shift
            __configuration=$1
            ;;
        --mkllibpath)
            shift
            __mkllibpath=$1
            ;;
        --mkllibrpath)
            shift
            __mkllibrpath=$1
            ;;
        --stripsymbols)
            __strip_argument="-DSTRIP_SYMBOLS=true"
            ;;
        *)
        echo "Unknown argument to build.sh $1"; usage; exit 1
    esac
    shift
done

__cmake_defines="-DCMAKE_BUILD_TYPE=${__configuration} ${__strip_argument} -DMKL_LIB_PATH=${__mkllibpath} -DMKL_LIB_RPATH=${__mkllibrpath}"

__IntermediatesDir="$__baseIntermediateOutputPath/Native/$__build_arch.$__configuration"
__BinDir="$__rootBinPath/Native/$__build_arch.$__configuration"

mkdir -p "$__BinDir"
mkdir -p "$__IntermediatesDir"

export __IntermediatesDir=$__IntermediatesDir

# Set up the environment to be used for building with clang.
if command -v "clang-9" > /dev/null 2>&1; then
    export CC="$(command -v clang-9)"
    export CXX="$(command -v clang++-9)"
elif command -v "clang-6.0" > /dev/null 2>&1; then
    export CC="$(command -v clang-6.0)"
    export CXX="$(command -v clang++-6.0)"
elif command -v "clang-3.5" > /dev/null 2>&1; then
    export CC="$(command -v clang-3.5)"
    export CXX="$(command -v clang++-3.5)"
elif command -v "clang-3.6" > /dev/null 2>&1; then
    export CC="$(command -v clang-3.6)"
    export CXX="$(command -v clang++-3.6)"
elif command -v "clang-3.9" > /dev/null 2>&1; then
    export CC="$(command -v clang-3.9)"
    export CXX="$(command -v clang++-3.9)"
elif command -v clang > /dev/null 2>&1; then
    export CC="$(command -v clang)"
    export CXX="$(command -v clang++)"
else
    echo "Unable to find Clang Compiler"
    echo "Install clang-3.5 or clang3.6 or clang3.9"
    exit 1
fi

# Specify path to be set for CMAKE_INSTALL_PREFIX.
# This is where all built native libraries will copied to.
export __CMakeBinDir="$__BinDir"

if [ ! -f $__versionSourceFile ]; then
    __versionSourceLine="static char sccsid[] __attribute__((used)) = \"@(#)No version information produced\";"
    echo $__versionSourceLine > $__versionSourceFile
fi

__cmake_defines="${__cmake_defines} -DVERSION_FILE_PATH:STRING=${__versionSourceFile}"

OS_ARCH=$(uname -m)
OS=$(uname)

# If we are cross compiling on Linux we need to set the CMAKE_TOOLCHAIN_FILE
if [[ ( $OS_ARCH == "amd64" || $OS_ARCH == "x86_64" ) && ( $__build_arch == "arm64" || $__build_arch == "arm" ) && $OS != "Darwin" ]] ; then
    __cmake_defines="${__cmake_defines} -DCMAKE_TOOLCHAIN_FILE=$RootRepo/eng/common/cross/toolchain.cmake"
    export TARGET_BUILD_ARCH=$__build_arch

# If we are on a Mac we need to let it know the cross architecture to build for.
# We use x64 for our 64 bit code, but Mac defines it as x86_64.
elif [[  $OS == "Darwin" && $__build_arch == "x64" ]] ; then
    __build_arch="x86_64"
fi

# Set the ARCHITECTURE for all builds
__cmake_defines="${__cmake_defines} -DARCHITECTURE=${__build_arch}"

cd "$__IntermediatesDir"

echo "Building Machine Learning native components from $DIR to $(pwd)"
set -x # turn on trace
cmake "$DIR" -G "Unix Makefiles" $__cmake_defines
set +x # turn off trace
make install