# Cross-Platform and Architecture design doc

### Table of contents  <!-- omit in toc -->
- [Cross-Platform and Architecture design doc](#cross-platform-and-architecture-design-doc)
	- [1. Why cross-platform/architecture](#1-why-cross-platformarchitecture)
	- [2. Current status](#2-current-status)
		- [2.1 Problems](#21-problems)
		- [2.2 Build](#22-build)
		- [2.3 Managed Code](#23-managed-code)
		- [2.4 Native Projects](#24-native-projects)
		- [2.5 3rd Party Dependencies](#25-3rd-party-dependencies)
	- [3 Possible Solutions](#3-possible-solutions)
		- [3.1 Eliminate native components](#31-eliminate-native-components)
		- [3.2 Rewrite native components to work on other platforms](#32-rewrite-native-components-to-work-on-other-platforms)
		- [3.3 Rewrite native components to be only managed code](#33-rewrite-native-components-to-be-only-managed-code)
		- [3.4 Hybrid Between Finding Replacements and Software Fallbacks](#34-hybrid-between-finding-replacements-and-software-fallbacks)
	- [My Suggestion](#my-suggestion)
		- [Improving managed fallback through intrinsics](#improving-managed-fallback-through-intrinsics)
		- [3rd Party Dependencies](#3rd-party-dependencies)
		- [Helix](#helix)
		- [Mobile Support](#mobile-support)
		- [Support Grid](#support-grid)

## 1. Why cross-platform/architecture
ML.NET is an open-source machine learning framework which makes machine learning accessible to .NET developers with the same code that powers machine learning across many Microsoft products, including Power BI, Windows Defender, and Azure.

ML.NET allows .NET developers to develop/train their own models and infuse custom machine learning into their applications using .NET.

Currently, while .NET is able to run on many different platforms and architectures, ML.NET is only able to run on Windows, Mac, and Linux, either x86 or x64. This excludes many architectures such as Arm, Arm64, M1, and web assembly, places that .NET is currently able to run.

The goal is to enable ML.NET to run everywhere that .NET itself is able to run.

## 2. Current status
There are several problems complicating us from moving to a fully cross-platform solution. At a high level these include:
1. We only build and test for a subset of platforms that .NET supports.
2. Native components must be explicitly built for additional architectures which we wish to support. This limits our ability to support new platforms without doing work.
3. Building our native components for new platforms faces challenges due to lack of support for those components dependencies. This limits our ability to support the current set of platforms.
4. Some of our external dependencies have limited support for .NET's supported platforms.

### 2.1 Problems
ML.NET has a hard dependency on x86/x64. Some of the dependency is on Intel MKL, while other parts depend on x86/x64 SIMD instructions. To make things easier I will refer to these as just the x86/x64 dependencies. This is to perform many optimized math functions and enables several transformers/trainers to be run natively for improved performance. The problem is that these dependencies can only run on x86/x64 machines and are the main blockers for expanding to other architectures. While you can run the managed code on other architectures, there is no good way to know which parts will run and which ones wont. This includes the build process as well, which currently has these same hard dependencies and building on non x86/x64 machines is not supported.

ML.NET also has dependencies on things that either don't build on other architectures or have to be compiled by the user if it's wanted. For example:
 - LightGBM
 - TensorFlow
 - ONNX Runtime

I will go over these in more depth below.

### 2.2 Build
Since ML.NET has a hard dependency on x86/x64, the build process assumes it's running there. For example, the build process will try and copy native DLLs without checking if they exist because it assumes the build for them succeeded or that they are available. The build process will need to be modified so that it doesn't fail when it can't find these files. It does the same copy for our own Native DLLs, so this will need to be fixed for those as well.

### 2.3 Managed Code
Since ML.NET has a hard dependency on x86/x64, the managed code imports DLLs without checking whether or not they exist. If the DLLs don't exist you get a hard failure. For example, if certain columns are active, the `MulticlassClassificationScorer` will call `CalculateIntermediateVariablesNative` which is loaded from `CpuMathNative`, but all of this is done without any checks to see if the DLL actually exists. The tests also run into this problem, for instance, the base test class imports and sets up Intel MKL even if the test itself does not need it.

### 2.4 Native Projects
ML.NET has 6 native projects. They are:
 - CpuMathNative
   - Partial managed fallback when using NetCore 3.1.
   - A large amount of work would be required to port the native code to other platforms. We would have to change all the SIMD instructions for each platform.
   - A small amount of work required for a full managed fallback.
   - This was created before we had hardware intrinsics so we used native code for performance. The managed fallback uses x86/x64 intrinsics where possible.
 - FastTreeNative
   - Full managed fallback by changing a C# compiler flag. This flag is hardcoded to always use the native code.
   - Small amount of work required to change build process and verify it's correct.
   - This was created before we had hardware intrinsics so we used native code for performance. The managed fallback does not use hardware intrinsics, so this will be slower than the native solution.
 - LdaNative
   - No managed fallback, but builds successfully on non x86/x64 without changes.
   - Large amount of work to have a managed fallback. This solution is about 3000 lines of code not including any code from the dependencies.
   - This was created before we had hardware intrinsics so we used native code for performance.
 - MatrixFactorizationNative
   - No software fallback.
   - Currently we are hardcoding the "USESSE" flag which requires x86/x64 SIMD commands. Removing this flag allows MatrixFactorizationNative to build for other platforms.
   - Small amount of work required to change the build process and verify it's working.
   - Large/Xlarge amount of work to have a managed fallback. This uses libmf, so we would have to not only port our code, but we would have to understand which parts of libmf are being used and port those as well.
   - This was created before we had hardware intrinsics so we used native code for performance as well as take advantage of libmf.
 - MklProxyNative
   - Wrapper for Intel MKL. When Intel MKL is not present, this is not needed. However, the build is hardcoded to always compile this code.
   - Small amount of work required to change the build process to exclude this as needed.
 - SymSgdNative
   - No managed fallback.
   - Medium amount of work required to have a managed fallback. Only about 500 lines of code plus having to implmement 4 vector operations.
   - Small amount of work required port to other platforms. Only uses 4 Intel MKL methods which we can replace. I was only able to find 2 direct replacements on other architectures, but we could easily write our own for either those 2 or even all 4.
   - This was created before we had hardware intrinsics so we used native code for performance as well as take advantage of IntelMKL.

Of these 6, only LdaNative builds successfully for other architectures without changing anything.

### 2.5 3rd Party Dependencies
As mentioned above, there are several 3rd party packages that don't have support for non x86/x64 machines.
 - LightGBM. LightGBM doesn't offer packages for non x86/x64. I was able to build the code for Arm64, but we would either have to build it ourselves, convince them to create more packages for us, or annotate that this doesn't work on non x86/64 machines.
 - TensorFlow. The full version of TensorFlow only runs on x86/x64. There is a [lite](https://www.tensorflow.org/lite/guide/build_arm64) version that supports Arm64, and you can install it directly with python, but this isn't the full version so not all models will run. We would also have to verify if the C# library we use to interface with TensorFlow will work with the lite version.
 - OnnxRuntime. OnnxRuntime doesn't have prebuilt packages for more than just x86/x64. It does support Arm, but we have to build it ourselves or get the OnnxRuntime team to package Arm assemblies. This is the same situation as with LightGBM.


## 3 Possible Solutions
We have several possible options we can use to resolve this:
 - Eliminate ML.NET native components and implement all functionality in managed code.
 - Keep ML.NET native components and rewrite them to a avoid problematic dependencies completely.
 - Keep ML.NET native components and ifdef/rewrite to avoid problematic dependencies on platforms/architectures only where they are not supported.
 - Hybrid approach of replacement code and software fallback. This is the approach I recommend.

None of these approaches resolve the 3rd party dependency issues. These solutions only deal with first party ML.NET code itself.

I have lots more info about our dependency on x86/x64 in another document if required.

### 3.1 Eliminate native components
This is the most complicated solution and provides the least amount of short term benefit. Since x86/x64 run fine and gain performance benefits with these components, it doesn't make sense to spend the time to fully remove it. This is a possible solution, but not one that I would recommend, so I am not going to give more details on it unless we explicitly decide to go this route.

### 3.2 Rewrite native components to work on other platforms
This will still allow us to gain the benefits of the X86/x64 SIMD instructions and Intel MKL on architectures that support it but will also keep the benefits of native code in the other places. The downside is that we would have to build the native code for, potentially, a lot of different architectures.

At a high level, this solution would require us to:
 - Fix the build so it's not hardcoded to look for specific native dependencies.
 - Fix the managed code so that if it can't find the native components the code behaves correctly.
 - All the native code would need to have the x86/x64 dependencies change to other libraries or be re-written by us.
   - CpuMathNative would need to be re-written to not use x86/x64 dependencies.
   - FastTreeNative would need to be re-written to not use x86/x64 dependencies.
   - LdaNative builds just find without x86/x64 dependencies, so no replacements would need to be found.
   - MatrixFactorizationNative builds with the "USESSE" flag that requires x86/x64. We can conditionally enable/disable that flag and no other work would be required.
   - MklProxyNative is only a wrapper for Intel MKL and can be ignored on non x86/x64 platforms. We will need to modify the build to exclude it as needed.
   - SymSgdNative would need to be re-written to not use Intel MKL. There are only 4 methods that would need to be changed.

I was unable to find all the replacements we would need. We would end up having to write many native methods ourselves with this approach.

### 3.3 Rewrite native components to be only managed code
This will truly allow ML.NET to run anywhere that .NET runs. The only downside is the speed of execution, and the time to rewrite the existing native code we have.  If we restrict new architectures to .NET core 3.1 or newer, we will have an easier time with the software fallbacks as some of this code has already been written. This solution will also require a lot of code rewrite from native code to managed code.

At a high level, this solution would require us to:
 - Fix the build so it's not hardcoded to look for any native dependencies or binaries.
 - Fix the managed code so that if it can't find the native binaries, the code behaves correctly and performs the software fallback.
 - CpuMathNative mostly has software fallbacks already in place for .NET core 3.1, so only a little work is needed.
 - FastTreeNative has a flag for a software fallback. We would need to conditionally enable this. Alternatively, we could change the C# code so the software fallback is always enabled for the cases when it can't find the native binaries.
 - LdaNative would need to be re-written.
 - MatrixFactorizationNative would need to be re-written.
 - MklProxyNative can be ignored. It is not needed with software fallbacks. We will need to modify the build to exclude it as needed.
 - SymSgdNative would need to be re-written.

### 3.4 Hybrid Between Finding Replacements and Software Fallbacks
Since some of the native code already has replacements and some of the code already has software fallbacks, we can leverage this work by doing a hybrid between the prior 2 solutions.

At a high level, this solution would require us to:
 - Fix the build so it's not hardcoded to look for any native dependencies or binaries.
 - Fix the managed code so that if it can't find the native binaries, the code behaves correctly and performs the software fallback. This includes software fallbacks and/or description error messages.
 - CpuMathNative mostly has software fallbacks already in place for .NET core 3.1, so only a little work is needed.
 - FastTreeNative has a flag for a software fallback. We would need to conditionally enable this. Alternatively, we could change the C# code so the software fallback is always enabled for the cases when it can't find the native binaries.
 - LdaNative builds just find without x86/x64 dependencies, so no replacements would need to be found.
 - MatrixFactorizationNative builds with the "USESSE" flag that requires x86/x64. We can conditionally enable/disable that flag and no other work would be required.
 - MklProxyNative is only a wrapper for Intel MKL and can be ignored on non x86/x64 platforms. We will need to modify the build to exclude it as needed.
 - SymSgdNative needs to be either re-written in managed code, or re-write 4 Intel MKL methods. The 4 methods are just dealing with vector manipulation and shouldn't be hard to do.

## My Suggestion
My suggestion would be to start with the hybrid approach. It will require the least amount of work to get ML.NET running elsewhere, while still being able to support a large majority of devices out of the gate. This solution will still limit the platforms we can run-on to what we build the native components for, initially Arm64 devices, but we can do a generic Arm64 compile so it should work for all Arm64 v8 devices. The goal is to eventually have a general purpose implementation which can work everywhere .NET does and accelerated components to increase performance where possible, such as running on Web Assembly and when .NET 6 comes out on mobile as well.

### Improving managed fallback through intrinsics
We should also target .NET 5 so that we gain access to the Arm64 intrinsics. Rather than implementing special-purpose native libraries to take advantage of architecture-specific instructions we should instead enhance performance ensuring our managed implementation leverage intrinsics.

### 3rd Party Dependencies
I think initially we should annotate that they don't work on non x86/x64 devices. This includes logging an error when they try and run an unsupported 3rd party dependency, and then failing gracefully with a helpful and descriptive error. The user should be able to compile the 3rd party dependency, for the ones that support it, and have ML.NET still be able to pick it up and run it if it exists. OnnxRuntime is something that we will probably want, but we can look more into this as we get requests for it in the future.

### Helix
In order to fully test everything we need to, we would also need to change how we test to use the Helix testing servers. Currently, Helix doesn't have the capability to test Apple's new M1 code, but that is in the works.

### Mobile Support
.NET Core 6 will allow us to run nativly on mobile. Since we are making these changes before .NET 6 is released, I propose we don't include that work as of yet. As long as we handle the native binaries correctly and make sure ML.NET provides descriptive error methods, we should be able to have mobile support as soon as .NET Core 6 releases for everything that currently has a software fallback. Since the native projects we are proposing to keep build for Arm64, they should work on mobile as well.

### Support Grid
This is what I propose for the support grid. Since .NET Core 2.1 is end-of-life this year, I am putting much less emphasis on it. Since .NET Core 5 will be out of support before .NET Core 3.1 will be, I am putting less CI emphasis on .NET Core 5.

| Platform | Architecture | Intel MKL | .NET Framework | .NET Core 2.1 | .NET Core 3.1 | .NET Core 5 |
| ---------| -------------| --------- | -------------- | ------------- | ------------- | ----------- |
| Windows  | x64          | Yes       | Yes            | Yes, no CI    | Yes           | Yes         |
| Windows  | x86          | Yes       | Yes, no CI     | Yes, no CI    | Yes, no CI    | Yes, no CI  |
| Mac      | x64          | Yes       | No             | Yes, no CI    | Yes           | Yes, no CI  |
| Mac      | Arm64        | No        | No             | No            | Yes           | Yes, no CI  |
| Ios      | Arm64        | No        | No             | No            | No            | No          |
| Ios      | x64          | No        | No             | No            | No            | No          |
| Linux    | x64          | Yes       | No             | Yes, no CI    | Yes           | Yes, no CI  |
| Linux    | Arm64        | No        | No             | No            | Yes           | Yes         |
| Android  | Arm64        | No        | No             | No            | No            | No          |
