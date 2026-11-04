// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Runtime.InteropServices;
using Microsoft.ML.TestFrameworkCommon.Attributes;
using Xunit;

namespace Microsoft.ML.NightlyBuild.Tests
{
    /// <summary>
    /// Verifies that supported native package assets exist and can be loaded.
    /// </summary>
    public class PackageNativeDependenciesTests
    {
        private static bool IsOnnxRuntimeSupported =>
            (RuntimeInformation.RuntimeIdentifier is "win-x64" or "win-arm64" or "linux-x64" or "linux-arm64" or "osx-x64" or "osx-arm64")
            && (!RuntimeInformation.IsOSPlatform(OSPlatform.Linux)
                || AttributeHelpers.CheckLibcVersionGreaterThanMinimum(new Version(2, 23)));

        // Preserve the integration suite's exclusion of MKL on macOS x64.
        private static bool IsMklSupported =>
            RuntimeInformation.RuntimeIdentifier is "win-x64" or "win-x86" or "linux-x64";

        [ConditionalFact(nameof(IsOnnxRuntimeSupported))]
        public void OnnxRuntimeNativeLibraryLoads()
        {
            AssertPackagedNativeLibraryLoads("onnxruntime");
        }

        [ConditionalFact(nameof(IsMklSupported))]
        public void MklNativeLibraryLoads()
        {
            AssertPackagedNativeLibraryLoads("MklImports");
        }

        private static void AssertPackagedNativeLibraryLoads(string libraryName)
        {
            string path = Path.Combine(
                AppContext.BaseDirectory, "runtimes", RuntimeInformation.RuntimeIdentifier, "native", GetNativeFileName(libraryName));

            Assert.True(File.Exists(path), $"The packaged native library was not found: {path}");

            // Load by absolute path so a same-named library on the system search path cannot hide a missing package asset.
            IntPtr handle = NativeLibrary.Load(path);
            NativeLibrary.Free(handle);
        }

        private static string GetNativeFileName(string libraryName)
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return libraryName + ".dll";
            if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                return "lib" + libraryName + ".dylib";
            return "lib" + libraryName + ".so";
        }
    }
}
