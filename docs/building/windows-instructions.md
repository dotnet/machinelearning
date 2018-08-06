Building ML.NET on Windows
==========================

You can build ML.NET either via the command line or by using Visual Studio.

## Required Software

1. **[Visual Studio 2017](https://www.visualstudio.com/downloads/) (Community, Professional, Enterprise)**.  The Community version is completely free.
2. **[CMake](https://cmake.org/)** must be installed from [the CMake download page](https://cmake.org/download/#latest) and added to your path.

### Visual Studio 2017

#### Visual Studio 2017 - 'Workloads' based install

The following are the minimum requirements:
  * .NET desktop development
    * All Required Components
    * .NET Framework 4-4.6 Development Tools
  * Desktop development with C++
    * All Required Components
    * VC++ 2017 v141 Toolset (x86, x64)
    * Windows 8.1 SDK and UCRT SDK
  * .NET Core cross-platform development
    * All Required Components

Note: If you have both VS 2017 and 2015 installed, you need to copy DIA SDK directory from VS 2015 installation into VS 2017 (VS installer bug).

#### Visual Studio 2017 - 'Individual components' based install

The following are the minimum requirements:
  * C# and Visual Basic Roslyn Compilers
  * Static Analysis Tools
  * .NET Portable Library Targeting Pack
  * Visual Studio C++ Core Features
  * VC++ 2017 v141 Toolset (x86, x64)
  * MSBuild
  * .NET Framework 4.6 Targeting Pack
  * Windows Universal CRT SDK
  
In order to build in the Visual Studio IDE, need to call “build.cmd” from the command line first. Tests can be executed from the VS Test Explorer or command line.
  
## Building From the Command Line

You can use the Developer Command Prompt, Powershell or work in any regular cmd. The Developer Command Prompt will have a name like "Developer Command Prompt for VS 2017" or similar in your start menu. 

From a (non-admin) Command Prompt window:

- `build.cmd` - builds the assemblies
- `build.cmd -runTests` - called after a normal "build.cmd" will run all tests
- `build.cmd -buildPackages` called after a normal “build.cmd” will create the NuGet packages with the assemblies in “bin"

**Note**: Before working on individual projects or test projects you **must** run `build.cmd` from the root once before beginning that work. It is also a good idea to run `build.cmd` whenever you pull a large set of unknown changes into your branch.

### Running tests from the command line

From the root, run `build.cmd` and then `build.cmd -runTests`.
For more details, or to test an individual project, you can navigate to the test project directory and then use `dotnet test`
 
### Running tests from Visual Studio

You need to run tests in the Test Explorer window.

### Known Issues

CMake 3.7 or higher is required for Visual Studio 2017.

You need to run `build` from the root of the repo first prior to opening the solution file and building in Visual Studio.