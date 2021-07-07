// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.InteropServices;
using Microsoft.ML.TestFrameworkCommon.Attributes;

namespace Microsoft.ML.TestFramework.Attributes
{
    class NotArm32FactAttribute : EnvironmentSpecificFactAttribute
    {
        public NotArm32FactAttribute(string skipMessage) : base(skipMessage)
        {
        }

        /// <inheritdoc />
        protected override bool IsEnvironmentSupported()
        {
            return RuntimeInformation.ProcessArchitecture != Architecture.Arm;
        }
    }
}
