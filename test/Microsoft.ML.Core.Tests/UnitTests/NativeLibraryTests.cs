// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Runtime.InteropServices;
using Microsoft.ML.TestFramework;
using Microsoft.ML.TestFramework.Attributes;
using Xunit;
using Xunit.Abstractions;
using NativeLibrary = Microsoft.ML.TestFrameworkCommon.Utility.NativeLibrary;

namespace Microsoft.ML.RunTests
{
    public sealed class NativeLibraryTests : BaseTestClass
    {
        private static bool IsOneDalSupported =>
            RuntimeInformation.ProcessArchitecture == Architecture.X64 &&
            // We don't test macOS x64 for OneDAL.
            !RuntimeInformation.IsOSPlatform(OSPlatform.OSX);

        public NativeLibraryTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void SystemNativeLibraryCanBeLoaded()
        {
            string library = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                ? "kernel32"
                : RuntimeInformation.IsOSPlatform(OSPlatform.Linux) ? "libc.so.6" : "libSystem.B.dylib";

            Assert.True(NativeLibrary.NativeLibraryExists(library));
        }

        [Fact]
        public void MissingNativeLibraryReturnsFalse()
        {
            Assert.False(NativeLibrary.NativeLibraryExists("MLNetNonexistentNativeLibrary"));
        }

        [X64Fact("The LightGBM native package supports x64 only.")]
        public void PackageNativeLibraryCanBeLoaded()
        {
            Assert.True(NativeLibrary.NativeLibraryExists("lib_lightgbm"));
        }

        [ConditionalFact(nameof(IsOneDalSupported))]
        public void OneDalNativeLibraryCanBeLoaded()
        {
            Assert.True(NativeLibrary.NativeLibraryExists("OneDalNative"));
        }
    }
}
