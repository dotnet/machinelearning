// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.InteropServices;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.TestFrameworkCommon;

namespace Microsoft.ML.TestFrameworkCommon.Attributes
{
    /// <summary>
    /// A fact for tests requiring Onnx.
    /// </summary>
    public sealed class OnnxFactAttribute : EnvironmentSpecificFactAttribute
    {
        public OnnxFactAttribute() : base("Onnx is not supported on Linux with libc < v2.23")
        {
        }

        /// <inheritdoc />
        protected override bool IsEnvironmentSupported()
        {
            return IsOnnxRuntimeSupported;
        }

        public static bool IsOnnxRuntimeSupported { get; } =
            (!RuntimeInformation.IsOSPlatform(OSPlatform.Linux)
                || AttributeHelpers.CheckLibcVersionGreaterThanMinimum(new Version(2, 23)))
            && Utility.NativeLibrary.NativeLibraryExists("onnxruntime");
    }
}
