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
		- [3.1 Fully Remove Intel MKL](#31-fully-remove-intel-mkl)
		- [3.2 Find Replacements as needed](#32-find-replacements-as-needed)
		- [3.3 Software fallbacks](#33-software-fallbacks)
		- [3.4 Hybrid Between Finding Replacements and Software Fallbacks](#34-hybrid-between-finding-replacements-and-software-fallbacks)
	- [My Suggestion](#my-suggestion)
		- [NET Core 5](#net-core-5)
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
### 2.1 Problems
ML.NET has a hard dependency on Intel MKL. It performs many optimized math functions and enables several transformers/trainers to be run natively for improved performance. The problem is that Intel MKL can only run on x86/x64 machines and is the main blocker for expanding to other architectures.

ML.NET also has dependencies on things that either don't build on other architectures or have to be compiled by the user if its wanted. For example:
 - LightGBM
 - TensorFlow
 - OnnxRuntime

I will go over these in more depth below.

### 2.2 Build
Since ML.NET has a hard dependency on Intel MKL, the build process assumes it's always there. For example, the build process will try and copy dlls without checking if they exist. The build process will need to be modified so that it doesn't fail when it can't find these files. It does the same copy check for our own Native dlls, so this will need to be fixed for those as well.

### 2.3 Managed Code
Since ML.NET has a hard dependency on Intel MKL, the managed code imports the dlls without checking whether or not they exist. If the dlls don't exist you get a hard failure. For example, the base test class imports and sets up Intel MKL even if the test itself does not need it. This same situation applies to our own native dlls. When they aren't present, the imports fail. We will need to make ML.NET correctly handle the dll imports.

### 2.4 Native Projects
ML.NET has 6 native projects. They are:
 - CpuMathNative
   - CpuMathNative relies heavily on IntelMKL. Using NetCore 3.1, however, there is a full software fallback.
 - FastTreeNative
   - FastTreeNative has a software fallback using a C# compiler flag. This flag is hardcoded to always use the native code.
 - LdaNative
   - Builds successfully without native dependencies.
 - MatrixFactorizationNative
   - Uses libmf. Currently, we are hardcoding the "USESSE" flag which requires Intel MKL. Removing this flag allows MatrixFactorizationNative to build for Arm.
 - MklProxyNative
   - Wrapper for Intel MKL. When Intel MKL is not present, this is not needed. However, the build is hardcoded to always compile this code.
 - SymSgdNative
   - Only uses 4 Intel MKL methods. For 2 of them, I can't find direct replacements on other architectures.

Of these 6, only LdaNative builds successfully for other architectures without changing anything.

### 2.5 3rd Party Dependencies
As mentioned above, there are several 3rd party packages that don't have support for non x86/x64 machines.
 - LightGBM. LightGBM doesn't offer packages for non x86/x64. I was able to build the code for Arm64, but we would either have to build it ourselves, convince them to create more packages for us, or annotate that this doesn't work on non x86/64 machines.
 - TensorFlow. The full version of TensorFlow only runs on x86/x64. There is a [lite](https://www.tensorflow.org/lite/guide/build_arm64) version that supports Arm64, and you can install it directly with python, but this isn't the full version so not all models will run. We would also have to verify if the C# library we use to interface with TensorFlow will work with the lite version.
 - OnnxRuntime. OnnxRuntime doesn't have prebuilt packages for more than just x86/x64. It does support Arm, but we have to build it ourselves. This is the same situation as with LightGBM.


## 3 Possible Solutions
We have several possible options we can use to resolve this:
 - Fully remove Intel MKL as a dependency and use software that supports all platforms/architectures.
 - Continue to use Intel MKL for what it supports, and find a replacement for all platforms/architectures that don't support it.
 - Create software fallbacks, so that if Intel MKL is not found, then ML.NET will run fully in managed code.
 - Hybrid approach of replacement code and software fallback. This is the approach I recommend.

None of these approaches resolve the 3rd party dependency issues. These solutions only deal with first party ML.NET code itself.

I have lots more info about our dependency on Intel MKL in another document if required.

### 3.1 Fully Remove Intel MKL
This is the most complicated solution and provides the least amount of short term benefit. Since x86/x64 run fine and gain performance benefits with Intel MKL, it doesn't make sense to spend the time to fully remove it. This is a possible solution, but not one that I would recommend, so I am not going to give more details on it unless we explicitly decide to go this route.

### 3.2 Find Replacements as needed
This will still allow us to gain the benefits of Intel MKL on architectures that support it but will also keep the benefits of native code in the other places. The downside is that we would have to build the native code for, potentially, a lot of different architectures.

At a high level, this solution would require us to:
 - Fix the build so it's not hardcoded to look for Intel MKL.
 - Fix the managed code so that if it can't find Intel MKL, the code behaves correctly.
 - All the native code would need to have the Intel MKL dependency change to another library or be re-written by us.
   - CpuMathNative would need to be re-written to not use Intel MKL.
   - FastTreeNative would need to be re-written to not use Intel MKL.
   - LdaNative builds just find without Intel MKL, so no replacements would need to be found.
   - MatrixFactorizationNative builds with the "USESSE" flag that requires Intel MKL. We can conditionally enable/disable that flag and no other work would be required.
   - MklProxyNative is only a wrapper for Intel MKL and can be ignored on non x86/x64 platforms. We will need to modify the build to exclude it as needed.
   - SymSgdNative would need to be re-written. There are only 4 methods that would need to be changed.

I was unable to find all the replacements we would need. We would end up having to write many native methods ourselves with this approach.

### 3.3 Software fallbacks
This will truly allow ML.NET to run anywhere that .NET runs. The only downside is the speed of execution, and the time to rewrite the existing native code we have.  If we restrict new architectures to .NET core 3.1 or newer, we will have an easier time with the software fallbacks as some of this code has already been written. This solution will also require a lot of code rewrite from native code to managed code.

At a high level, this solution would require us to:
 - Fix the build so it's not hardcoded to look for Intel MKL or any native binaries.
 - Fix the managed code so that if it can't find the native binaries, the code behaves correctly and performs the software fallback.
 - CpuMathNative has software fallbacks already in place for .NET core 3.1, so no work is needed.
 - FastTreeNative has a flag for a software fallback. We would need to conditionally enable this. Alternatively, we could change the C# code so the software fallback is always enabled for the cases when it can't find the native binaries.
 - LdaNative would need to be re-written.
 - MatrixFactorizationNative would need to be re-written.
 - MklProxyNative can be ignored. It is not needed with software fallbacks. We will need to modify the build to exclude it as needed.
 - SymSgdNative would need to be re-written.

### 3.4 Hybrid Between Finding Replacements and Software Fallbacks
Since some of the native code already has replacements and some of the code already has software fallbacks, we can leverage this work by doing a hybrid between the prior 2 solutions.

At a high level, this solution would require us to:
 - Fix the build so it's not hardcoded to look for Intel MKL or any native binaries.
 - Fix the managed code so that if it can't find Intel MKL or any native binaries, the code behaves correctly. This includes software fallbacks and/or description error messages.
 - CpuMathNative has software fallbacks already in place for .NET core 3.1, so no work is needed.
 - FastTreeNative has a flag for a software fallback. We would need to conditionally enable this. Alternatively, we could change the C# code so the software fallback is always enabled for the cases when it can't find the native binaries.
 - LdaNative builds just fine for Arm, so no work would be required.
 - MatrixFactorizationNative builds with the "USESSE" flag that requires Intel MKL. We can conditionally enable/disable that flag and no other work would be required.
 - MklProxyNative can be removed/ignored for Arm builds. We will need to modify the build to exclude it as needed.
 - SymSgdNative needs to be either re-written in managed code, or re-write 4 Intel MKL methods. The 4 methods are just dealing with vector manipulation and shouldn't be hard to do.

## My Suggestion
My suggestion would be to start with the hybrid approach. It will require the least amount of work to get ML.NET running elsewhere, while still being able to support a large majority of devices out of the gate. This solution will still limit the platforms we can run-on to Arm64 devices, but we can do a generic Arm64 compile, so it should work for all Arm64 v8 devices. It would be a good idea to have full software fallbacks so that truly anywhere .NET Core runs, ML.NET will run, such as Web Assembly and when .NET 6 comes out on mobile as well.

### NET Core 5
I think we should also add support for .NET Core 5 during this process so that we gain access to the Arm64 intrinsics. However, since .Net Core 5 goes out of support before .NET core 3.1 I don't think .NET Core 5 should be a huge focus.

### 3rd Party Dependencies
I think initially we should annotate that they don't work on non x86/x64 devices. This includes logging an error when they try and run an unsupported 3rd party dependency, and then failing gracefully with a helpful and descriptive error. The user should be able to compile the 3rd party dependency, for the ones that support it, and have ML.NET still be able to pick it up and run it if it exists. OnnxRuntime is something that we will probably want, but we can look more into this as we get requests for it in the future.

### Helix
In order to fully test everything we need to, we would also need to change how we test to use the Helix testing servers. Currently, Helix doesn't have the capability to test Apple's new M1 code, but that is in the works.

### Mobile Support
.NET Core 6 will allow us to run nativly on mobile. Since we are making these changes before .NET 6 is released, I propose we don't include that work as of yet. As long as we handle the native binaries correctly and make sure ML.NET provides descriptive error methods, we should be able to have mobile support as soon as .NET Core 6 releases for everything that currently has a software fallback.

### Support Grid
This is what I propose for the support grid. Since .NET Core 2.1 is end-of-life this year, I am putting much less emphasis on it. Since .NET Core 5 will be out of support before .NET Core 3.1 will be, I am putting less CI emphasis on .NET Core 5.

| Platform | Architecture | Intel MKL | .NET Framework | .NET Core 2.1 | .NET Core 3.1 | .NET Core 5 |
| ---------| -------------| --------- | -------------- | ------------- | ------------- | ----------- |
| Windows  | x64          | Yes       | Yes            | Yes, no CI    | Yes           | Yes         |
| Windows  | x86          | Yes       | Yes, no CI     | Yes, no CI    | Yes, no CI    | Yes, no CI  |
| Mac      | x64          | Yes       | No             | Yes, no CI    | Yes           | Yes, no CI  |
| Mac      | Arm64        | No        | No             | No            | Yes           | Yes, no CI  |
| Ios      | Arm64        | No        | No             | No            | No            | No          |
| Linux    | x64          | Yes       | No             | Yes, no CI    | Yes           | Yes, no CI  |
| Linux    | Arm64        | No        | No             | No            | Yes           | Yes         |
| Android  | Arm64        | No        | No             | No            | No            | No          |