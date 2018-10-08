#Instructions to build a custom DLL from Intel's MKL SDK
ML.NET MKL implementation uses Intel MKL Custom DLL builder to produce a single DLL which contains all of the MKL functions used by it.
To update the DLL, follow the steps below:

Windows (32 and 64 bit):
- Ensure you have Intel's MKL SDK installed, you can find it here: https://software.intel.com/en-us/mkl.
- Open an admin command prompt and run the following commands, CAREFULLY INSPECTING THE COMMAND OUTPUT FOR ERRORS.
- TLCROOT should be the root of your TLC_Resources folder.

##Windows
1. Modify user_example_list file in directory to contain all required functions, that are present in the [mlnetmkl.list](mlnetmkl.list) file
2. "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" x86
3. nmake libia32 name=MklImports  (add threading=sequential if you are building without openmp)
4. copy /Y Microsoft.ML.MklImports.* to the folder that will host the x86 binaries. 
5. del Microsoft.ML.MklImports.*
6. "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64
7. nmake intel64 name=MklImports (add threading=sequential if you are building without openmp)
8. copy /Y Microsoft.ML.MklImports.* to the folder that will host the x64 binaries.

##Linux
1. untar the linux sdk (tar -zxvf name_of_downloaded_file)
2. Run installation script and follow the instuctions in the dialog screens that are presented ./install.sh
3. Go to /opt/mkl/tools/builder.
4. Modify makefile add -Wl,-rpath,'$$ORIGIN' \ -Wl,-z,origin \  after -Wl,--end-group \
5. Modify user_example_list file in directory to contain all the required functions, that are present in the [mlnetmkl.list](mlnetmkl.list) file
6. Run make intel64 name=libMklImports (add threading=sequential if you are building without openmp)
7. Copy libMklImports.so the folder containing the Linux binaries.

##OSX
1. extract and install the dmg (double-click and drag it in the Applications folder)
2. Go to /opt/mkl/tools/builder.
3. Modify user_example_list file in directory to contain all the required functions, that are present in the [mlnetmkl.list](mlnetmkl.list) file
4. Run make intel64 name=libMklImports (add threading=sequential if you are building without openmp)
5. Copy libMklImports.dylib from the builder directory to the folder containign the OSX binaries.
6. Fix the id and the rpath running the following commands:
   sudo install_name_tool -id "@loader_path/libMklImports.dylib" libMklImports.dylib 
   sudo install_name_tool -id "@rpath/libMklImports.dylib" libMklImports.dylib