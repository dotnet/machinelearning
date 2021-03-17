# The MlNetMklDeps nuget
ML.NET's repository takes a dependency of the MlNetMklDeps nuget, which contains the MKL binaries of the MKL functions that ML.NET uses. This other nuget is actually also built and managed by the team. In the next section, the steps to create those binaries are described. In this section the contents of the nuget are mentioned.

**MlNetMklDeps nuget follows this layout:**
* licensing (contains Intel's license.txt they ship MKL with along with any third party licenses)
* MlNetMklDeps.nuspec
* runtimes
   * linux-x64
      * native (contains linux binaries)
   * osx-x64
      * native (cntains osx binaries)
   * win-x64
      * native (contains windows x64 binaries)
   * win-x86
      * native (contains windows ia32 binaries)

The .nuspec can be found on this folder:
https://github.com/dotnet/machinelearning/tree/main/docs/building/MlNetMklDeps

If actually publishing a new version of MlNetMklDeps, remember to update this other file to document any changes:
https://github.com/dotnet/machinelearning/blob/main/docs/building/MlNetMklDeps/version.md

# Instructions to build the binaries using Intel's MKL SDK
ML.NET MKL implementation uses Intel MKL Custom Builder to produce the binaries for the functions that we select. Follow the instructions below to produce the binaries for each platform, which will then be added to the MlNetMklDeps nuget described on the previus section.

**Download Intel MKL SDK** before following the instructions below on each platform:
https://software.intel.com/en-us/mkl

**NOTE about TLC**: The previous version of this instructions said to set the `TLCROOT` variable to "your TLC_Resources folder", since in ML.NET we don't have a `TLC_Resources` folder this seems to be stale instructions, which aren't needed any more. But it might become relevant if trying to test anything MKL related with TLC.

## Windows

1. In the Intel MKL SDK install directory, go to the Builder folder, found in `compilers_and_libraries\windows\mkl\tools\builder`
2. Replace the contents of the `user_example_list` file, found in that folder, with the contents of the [mlnetmkl.list
](mlnetmkl.list) file.
3. Initialize your environment by running the `vcvarsall.bat x86` command found on your Visual Studio installation directory. E.g., it might be found on any path similar to these:
   * `"C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" x86`
   * `"C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x86`
4. Back to the Builder folder run the following command to create the binary files:
`nmake libia32 name=MklImports`. Add `threading=sequential` if you are building without openmp (**NOTE:** it seems that starting on [PR #2867](https://github.com/dotnet/machinelearning/pull/2867) we always want to build using openmp, so there's no need to use this `threading` flag). This will produce `MklImports.dll, MklImports.dll.manifest, MklImports.exp, and MklImports.lib`. To also create `MklImports.pdb` see the note at the end of this section.
5. Copy the MKLImports files to the folder that will host the binaries inside the nuget: `copy /Y MklImports.*`
6. Also copy the Openmp library found inside the MKL SDK installation folder to the folder with the binaries:  `copy /Y ..\..\..\redist\ia32_win\compiler\libiomp5md*`
7. Delete the x86 MklImports files from the Builder folder: `del MklImports.*`
8. Initialize your environment with `vcvarsall.bat amd64`. It should be in the same path as found on step 3.
9. On the Builder folder: `nmake intel64 name=MklImports` (add `threading=sequential` if you are building without openmp)
10. Copy the MKL files to the folder for x64 binaries: `copy /Y MklImports.*`
11. Copy the Openmp library to the folder with the x64 binaries:  `copy /Y ..\..\..\redist\intel64_win\compiler\libiomp5md* `

**NOTE to create MklImports.pdb:** If the symbols for the built MklImports.dlls are required, add `/DEBUG:FULL /PDB:MklImports.pdb \` on the makefile after `mkl_custom_vers.res \`, in both the `libintel64` and `libia32` targets, to get the symbols for both x86 and x64 binaries.

## Linux
**NOTE:** Do not copy the libiomp5 file for Linux builds as this relies on OpenMP to be installed on the system.
1. Untar the linux Intel MKL SDK: `tar -zxvf name_of_downloaded_file`
2. Run the installation script and follow the instuctions in the dialog screens that are presented: `./install.sh`
3. Go to the Builder directory found in `/opt/intel/mkl/tools/builder` (it might be in another path, such as `/home/username/intel/compilers_and_libraries/linux/mkl/tools/builder` depending on your installation).
4. Modify the makefile found on the Builder directory: add `-Wl,-rpath,'$$ORIGIN' \ -Wl,-z,origin \`  after `-Wl,--end-group \`
5. Modify `user_example_list` file in the Builder directory to contain all the required functions, that are present in the [mlnetmkl.list](mlnetmkl.list) file
6. Run `make intel64 name=libMklImports` (add `threading=sequential` if you are building without openmp)

## OSX
**NOTE:** Do not copy the libiomp5 file for OSX builds as this relies on OpenMP to be installed on the system.
1. Extract and install the Intel MKL SDK dmg (double-click and drag it in the `Applications` folder)
2. Go to the Builder directory: `/opt/mkl/tools/builder`.
3. Modify user_example_list file in the Builder directory to contain all the required functions, that are present in the [mlnetmkl.list](mlnetmkl.list) file
4. Run `make libintel64 name=libMklImports` (add `threading=sequential` if you are building without openmp)
5. Copy `libMklImports.dylib` from the builder directory to the folder containing the OSX binaries.
6. Fix the id and the rpath running the following commands:
   * `sudo install_name_tool -id "@loader_path/libMklImports.dylib" libMklImports.dylib`
   * `sudo install_name_tool -id "@rpath/libMklImports.dylib" libMklImports.dylib`