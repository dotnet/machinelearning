#Instructions to build a custom DLL from Intel's MKL SDK
ML.NET MKL implementation uses Intel MKL Custom DLL builder to produce a single DLL which contains all of the MKL functions used by it.
To update the DLL, follow the steps below:

Windows (32 and 64 bit):
- Ensure you have Intel's MKL SDK installed, you can find it here: https://software.intel.com/en-us/mkl.
- Open an admin command prompt and run the following commands, CAREFULLY INSPECTING THE COMMAND OUTPUT FOR ERRORS.
- TLCROOT should be the root of your TLC_Resources folder.

Directory layout for nuget file is as follows:
* licensing (contains Intel's license.txt they ship MKL with along with any third party licenses)
* runtimes
**  linux-x64
*** native (contains linux binaries)
** osx-x64
*** native (cntains osx binaries)
** win-x64
*** native (contains windows x64 binaries)
** win-x86
*** native (contains windows ia32 binaries)

##Windows
1. In the Intel install directory, go to compilers_and_libraries\windows\mkl\tools\builder
2. Modify user_example_list file in directory to contain all required functions, that are present in the [mlnetmkl.list
](mlnetmkl.list) file
3. "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" x86
4. nmake libia32 name=MklImports (add threading=sequential if you are building without openmp)
5. Copy MKL library: copy /Y MklImports.* to the folder that will host the x86 binaries.
6. Copy openmp library:  copy /Y ..\..\..\redist\ia32_win\compiler\libiomp5md* to the folder for x86 binaries.
7. del MklImports.*
8. "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64
9. nmake intel64 name=MklImports (add threading=sequential if you are building without openmp)
10. Copy mkl library: copy /Y MklImports.* to the folder that will host the x64 binaries.
11. Copy openmp library:  copy /Y ..\..\..\redist\intel64_win\compiler\libiomp5md* to the folder for x86 binaries.

##Linux
NOTE: Do not copy the libiomp5 file for Linux builds as this relies on OpenMP to be installed on the system.
1. untar the linux sdk (tar -zxvf name_of_downloaded_file)
2. Run installation script and follow the instuctions in the dialog screens that are presented ./install.sh
3. Go to /opt/intel/mkl/tools/builder.
4. Modify makefile add -Wl,-rpath,'$$ORIGIN' \ -Wl,-z,origin \  after -Wl,--end-group \
5. Modify user_example_list file in directory to contain all the required functions, that are present in the [mlnetmkl.list](mlnetmkl.list) file
6. Run make intel64 name=libMklImports (add threading=sequential if you are building without openmp)

##OSX
NOTE: Do not copy the libiomp5 file for OSX builds as this relies on OpenMP to be installed on the system.
1. extract and install the dmg (double-click and drag it in the Applications folder)
2. Go to /opt/mkl/tools/builder.
3. Modify user_example_list file in directory to contain all the required functions, that are present in the [mlnetmkl.list](mlnetmkl.list) file
4. Run make libintel64 name=libMklImports (add threading=sequential if you are building without openmp)
5. Copy libMklImports.dylib from the builder directory to the folder containign the OSX binaries.
6. Fix the id and the rpath running the following commands:
   sudo install_name_tool -id "@loader_path/libMklImports.dylib" libMklImports.dylib 
   sudo install_name_tool -id "@rpath/libMklImports.dylib" libMklImports.dylib
