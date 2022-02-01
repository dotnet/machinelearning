// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.TestFrameworkCommon.Attributes;
using Microsoft.ML.TestFrameworkCommon.Utility;

namespace Microsoft.ML.TestFramework.Attributes
{
    public sealed class NativeDependencyTheoryAttribute : EnvironmentSpecificTheoryAttribute
    {
        private readonly string _library;

        public NativeDependencyTheoryAttribute(string library) : base($"This test requires a native library {library} that wasn't found.")
        {
            _library = library;
        }

        /// <inheritdoc />
        protected override bool IsEnvironmentSupported()
        {
            return NativeLibrary.NativeLibraryExists(_library);
        }
    }
}
