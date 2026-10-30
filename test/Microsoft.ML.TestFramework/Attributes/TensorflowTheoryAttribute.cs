// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.InteropServices;
using Microsoft.ML.TestFrameworkCommon.Attributes;

namespace Microsoft.ML.TestFramework.Attributes
{
    /// <summary>
    /// A theory for tests requiring TensorFlow.
    /// </summary>
    public sealed class TensorFlowTheoryAttribute : EnvironmentSpecificTheoryAttribute
    {
        public TensorFlowTheoryAttribute() : base("TensorFlow requires a 64-bit process, a supported platform, and a loadable native library.")
        {
        }

        /// <inheritdoc />
        protected override bool IsEnvironmentSupported()
        {
            return (Environment.Is64BitProcess &&
                  (RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ||
                    AttributeHelpers.CheckLibcVersionGreaterThanMinimum(new Version(2, 23))))
                    && Microsoft.ML.TestFrameworkCommon.Utility.NativeLibrary.NativeLibraryExists("tensorflow");
        }
    }
}
