// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.InteropServices;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public class DumpCaptureProbeTests : BaseTestClass
    {
        public DumpCaptureProbeTests(ITestOutputHelper output) : base(output)
        {
        }

        private static bool IsMacOSHelix => RuntimeInformation.IsOSPlatform(OSPlatform.OSX)
            && RuntimeInformation.ProcessArchitecture == Architecture.X64
            && !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("HELIX_DUMP_FOLDER"));

        [ConditionalFact(nameof(IsMacOSHelix))]
        public void CaptureSegmentationFault()
        {
            Console.WriteLine("Dump capture probe: intentionally writing to an invalid address in native memset.");
            Memset(new IntPtr(-1), 0, new UIntPtr(1));
            throw new InvalidOperationException("The SIGSEGV probe unexpectedly returned.");
        }

        [DllImport("libSystem.dylib", EntryPoint = "memset", ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr Memset(IntPtr destination, int value, UIntPtr count);
    }
}
