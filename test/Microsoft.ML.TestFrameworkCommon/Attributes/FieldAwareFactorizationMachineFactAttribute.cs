// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.TestFrameworkCommon.Attributes;
using Microsoft.ML.TestFrameworkCommon.Utility;

namespace Microsoft.ML.TestFramework.Attributes
{
    public class FieldAwareFactorizationMachineFactAttribute : EnvironmentSpecificFactAttribute
    {
        private const string SkipMessage = "FieldAwareFactorizationMachine requires AVX or CpuMathNative.";

        public FieldAwareFactorizationMachineFactAttribute() : base(SkipMessage)
        {
        }

        protected override bool IsEnvironmentSupported()
        {
#if NET8_0_OR_GREATER
            if (System.Runtime.Intrinsics.X86.Avx.IsSupported)
                return true;
#endif
            return NativeLibrary.NativeLibraryExists("CpuMathNative");
        }
    }
}
