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
