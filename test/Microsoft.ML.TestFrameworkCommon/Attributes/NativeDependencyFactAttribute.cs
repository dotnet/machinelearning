// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.TestFrameworkCommon.Attributes;
using Microsoft.ML.TestFrameworkCommon.Utility;

namespace Microsoft.ML.TestFramework.Attributes
{
    public sealed class NativeDependencyFactAttribute : EnvironmentSpecificFactAttribute
    {
        private readonly string _library;

        public NativeDependencyFactAttribute(string library) : base($"This test requires a native library {library} that wasn't found.")
        {
            _library = library;
        }

        /// <inheritdoc />
        protected override bool IsEnvironmentSupported()
        {
            // Starting to drop native support for X64 OSX since intel no longer makes them.
            if (System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.OSX) && System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture == System.Runtime.InteropServices.Architecture.X64)
                return false;
            return NativeLibrary.NativeLibraryExists(_library);
        }
    }
}
