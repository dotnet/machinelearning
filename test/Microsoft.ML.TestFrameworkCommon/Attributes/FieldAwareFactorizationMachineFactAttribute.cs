// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Runtime.InteropServices;
using Microsoft.ML.TestFrameworkCommon;
using Microsoft.ML.TestFrameworkCommon.Attributes;
using NativeLibrary = Microsoft.ML.TestFrameworkCommon.Utility.NativeLibrary;

namespace Microsoft.ML.TestFramework.Attributes
{
    public class FieldAwareFactorizationMachineFactAttribute : EnvironmentSpecificFactAttribute
    {
        private const string SkipMessage = "FieldAwareFactorizationMachine doesn't currently support non x86/x64. https://github.com/dotnet/machinelearning/issues/5871";

        public FieldAwareFactorizationMachineFactAttribute() : base(SkipMessage)
        {
        }

        protected override bool IsEnvironmentSupported()
        {
            if (TestCommon.RequireNativeDependencies)
                return RuntimeInformation.ProcessArchitecture == Architecture.X86 || RuntimeInformation.ProcessArchitecture == Architecture.X64;
            return NativeLibrary.NativeLibraryExists("CpuMathNative");
        }
    }
}
