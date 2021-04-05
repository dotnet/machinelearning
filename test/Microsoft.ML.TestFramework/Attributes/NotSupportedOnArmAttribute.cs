// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using static System.Runtime.InteropServices.RuntimeInformation;
using System.Runtime.InteropServices;
using Microsoft.ML.TestFrameworkCommon.Attributes;

namespace Microsoft.ML.TestFramework.Attributes
{
    /// <summary>
    /// A fact for tests that can't run on Arm/Arm64.
    /// </summary>
    public sealed class NotSupportedOnArmAttribute : EnvironmentSpecificFactAttribute
    {
        public NotSupportedOnArmAttribute() : base("This test is not supported on Arm/Arm64")
        {
        }

        /// <inheritdoc />
        protected override bool IsEnvironmentSupported()
        {
            var architecture = ProcessArchitecture;
            return architecture != Architecture.Arm64 && architecture != Architecture.Arm;
        }
    }
}