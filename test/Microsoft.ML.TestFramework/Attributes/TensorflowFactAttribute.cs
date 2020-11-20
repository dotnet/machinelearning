﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.InteropServices;
using Microsoft.ML.TestFrameworkCommon.Attributes;

namespace Microsoft.ML.TestFramework.Attributes
{
    /// <summary>
    /// A fact for tests requiring TensorFlow.
    /// </summary>
    public sealed class TensorFlowFactAttribute : EnvironmentSpecificFactAttribute
    {
        public TensorFlowFactAttribute() : base("TensorFlow is 64-bit only and is not supported on Linux with libc < v2.23")
        {
        }

        /// <inheritdoc />
        protected override bool IsEnvironmentSupported()
        {
            return Environment.Is64BitProcess &&
                   ( RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ||
                     AttributeHelpers.CheckLibcVersionGreaterThanMinimum(new Version(2, 23)));
                   
        }
    }
}
