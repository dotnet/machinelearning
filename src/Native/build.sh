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
__rootBinPath="$RootRepo/bin"
__baseIntermediateOutputPath="$__rootBinPath/obj"
__versionSourceFile="$__baseIntermediateOutputPath/version.c"

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
        --stripsymbols)
            __strip_argument="-DSTRIP_SYMBOLS=true"
            ;;
        *)
        echo "Unknown argument to build.sh $1"; usage; exit 1
    esac
    shift
done

__cmake_defines="-DCMAKE_BUILD_TYPE=${__configuration} ${__strip_argument}"

__IntermediatesDir="$__baseIntermediateOutputPath/$__build_arch.$__configuration/Native"
__BinDir="$__rootBinPath/$__build_arch.$__configuration/Native"

mkdir -p "$__BinDir"
mkdir -p "$__IntermediatesDir"

# Set up the environment to be used for building with clang.
if command -v "clang-3.5" > /dev/null 2>&1; then
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

__cmake_defines="${__cmake_defines} -DBUILD_WITH_OpenMP=ON -DVERSION_FILE_PATH:STRING=${__versionSourceFile}"

cd "$__IntermediatesDir"

echo "Building Machine Learning native components from $DIR to $(pwd)"
set -x # turn on trace
cmake "$DIR" -G "Unix Makefiles" $__cmake_defines
set +x # turn off trace
make install
