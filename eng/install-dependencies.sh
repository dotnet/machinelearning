#!/bin/sh

set -e

os="$(echo "$1" | tr "[:upper:]" "[:lower:]")"

if [ -z "$os" ]; then
    . "$(dirname "$0")"/common/native/init-os-and-arch.sh
fi

if [ "$os" = "linux" ]; then
    if [ -e /etc/os-release ]; then
        . /etc/os-release
    fi

    if [ "$ID" = "rhel" ] || [ "$ID" = "centos" ] || [ "$ID" = "azurelinux" ]; then
        echo "warning: libomp was not installed automatically for $ID. Install it manually before building ML.NET." >&2
    fi
fi

"$(dirname "$0")"/common/native/install-dependencies.sh "$os"

case "$os" in
    linux)
        if [ "$ID" = "debian" ] || [ "$ID_LIKE" = "debian" ]; then
            apt install -y libomp-dev
        elif [ "$ID" = "fedora" ]; then
            dnf install -y libomp-devel
        elif [ "$ID" = "amzn" ]; then
            dnf install -y libomp-devel
        elif [ "$ID" = "alpine" ]; then
            apk add openmp-dev
        fi
        ;;

    osx|maccatalyst|ios|iossimulator|tvos|tvossimulator)
        brew install libomp
        brew link libomp --force
        ;;
esac
