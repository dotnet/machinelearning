// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.TestFrameworkCommon.Attributes;

namespace Microsoft.ML.TestFramework.Attributes
{
    /// <summary>
    /// A theory for tests requiring Onnx.
    /// </summary>
    public sealed class OnnxTheoryAttribute : EnvironmentSpecificTheoryAttribute
    {
        public OnnxTheoryAttribute() : base("Onnx is 64-bit only and is not supported on Linux with libc < v2.23")
        {
        }

        /// <inheritdoc />
        protected override bool IsEnvironmentSupported()
        {
            return OnnxFactAttribute.IsOnnxRuntimeSupported;
        }
    }
}
