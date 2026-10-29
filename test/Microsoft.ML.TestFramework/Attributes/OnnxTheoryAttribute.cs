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
        public OnnxTheoryAttribute() : base("ONNX Runtime requires a supported platform and a loadable native library.")
        {
        }

        /// <inheritdoc />
        protected override bool IsEnvironmentSupported()
        {
            return OnnxFactAttribute.IsOnnxRuntimeSupported;
        }
    }
}
