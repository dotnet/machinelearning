// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.InteropServices;
using Microsoft.ML.TestFrameworkCommon.Attributes;

namespace Microsoft.ML.TestFramework.Attributes
{
    /// <summary>
    /// A fact for tests requiring Onnx.
    /// </summary>
    public sealed class OnnxTheoryAttribute : EnvironmentSpecificTheoryAttribute
    {
        public OnnxTheoryAttribute() : base("Onnx is 64-bit Windows only")
        {
        }

        /// <inheritdoc />
        protected override bool IsEnvironmentSupported()
        {
            return (!RuntimeInformation.IsOSPlatform(OSPlatform.Linux)
                || AttributeHelpers.CheckLibcVersionGreaterThanMinimum(new Version(2, 23)))
            && Microsoft.ML.TestFrameworkCommon.Utility.NativeLibrary.NativeLibraryExists("onnxruntime");
        }
    }
}
