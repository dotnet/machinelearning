using System;
using System.Runtime.InteropServices;
using Microsoft.ML.TestFrameworkCommon.Attributes;

namespace Microsoft.ML.TestFramework.Attributes
{
    public class NotAppleSiliconFactAttribute : EnvironmentSpecificFactAttribute
    {
        public NotAppleSiliconFactAttribute(string skipMessage) : base(skipMessage)
        {
        }

        protected override bool IsEnvironmentSupported()
        {
            return !(RuntimeInformation.ProcessArchitecture == Architecture.Arm64 && RuntimeInformation.IsOSPlatform(OSPlatform.OSX));
        }
    }
}
