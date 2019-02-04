// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.TestFramework.Attributes
{
    /// <summary>
    /// A fact for tests requiring full .NET Framework.
    /// </summary>
    public sealed class NotFullFrameworkFactAttribute : EnvironmentSpecificFactAttribute
    {
        public NotFullFrameworkFactAttribute(string skipMessage) : base(skipMessage)
        {
        }

        /// <inheritdoc />
        protected override bool IsEnvironmentSupported()
        {
#if NETFRAMEWORK
            return false;
#else
            return true;
#endif
        }
    }
}