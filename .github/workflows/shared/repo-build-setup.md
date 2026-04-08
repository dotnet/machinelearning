---
# Repo-specific build & environment setup for the add-tests workflow.
#
# This shared component is imported by add-tests.md to keep repo-specific
# configuration (native deps, build commands, checkout options) separate
# from the generic workflow logic.
#
# To adapt for your own repository:
#   1. Adjust checkout settings (submodules, LFS, fetch-depth) as needed
#   2. Replace the steps with your repo's build commands
#   3. Ensure dotnet (or your toolchain) is on PATH after the steps run

description: "Repository-specific build setup for add-tests workflow"

steps:
  - name: Initialize git submodules
    run: git submodule update --init --recursive

  - name: Install native dependencies
    run: |
      sudo ./eng/common/native/install-dependencies.sh
      sudo apt-get install -qq -y libomp-dev

  - name: Build
    run: ./build.sh

  - name: Put dotnet on the path
    run: echo "PATH=$PWD/.dotnet:$PATH" >> $GITHUB_ENV
---

# Repository Build Setup

This file contains the repo-specific build configuration for the `/add-tests` workflow.

## What this provides

- **Checkout**: Configures `submodules: recursive` (required for native C/C++ dependencies in `src/Native/`)
- **Native dependencies**: Installs system libraries needed by the native build (MKL, OpenMP)
- **Build**: Runs the Arcade SDK build script (`build.sh`) to compile all managed and native code
- **PATH**: Adds the locally-installed .NET SDK to `$PATH` so the agent can invoke `dotnet` directly

## Customization

To use the `/add-tests` workflow in a different repository, replace this file with your own
build setup steps. The workflow only requires that the repo is built and the
appropriate toolchain is available on `$PATH` after these steps run.
