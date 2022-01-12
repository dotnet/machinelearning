// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Runtime.InteropServices;

namespace Microsoft.ML.TestFramework.Attributes
{
    class NotArmFactAttribute : EnvironmentSpecificFactAttribute
    {
        public NotArmFactAttribute(string skipMessage) : base(skipMessage)
        {
        }

        /// <inheritdoc />
        protected override bool IsEnvironmentSupported()
        {
            return RuntimeInformation.ProcessArchitecture != Architecture.Arm && RuntimeInformation.ProcessArchitecture != Architecture.Arm64;
        }
    }
}
