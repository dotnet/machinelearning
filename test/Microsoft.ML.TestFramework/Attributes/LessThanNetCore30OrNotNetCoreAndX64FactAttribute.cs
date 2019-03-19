// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
namespace Microsoft.ML.TestFramework.Attributes
{
    /// <summary>
    /// A fact for tests requiring x64 environment and either .NET Core version lower than 3.0 or framework other than .NET Core.
    /// </summary>
    public sealed class LessThanNetCore30OrNotNetCoreAndX64FactAttribute : EnvironmentSpecificFactAttribute
    {
        public LessThanNetCore30OrNotNetCoreAndX64FactAttribute(string skipMessage) : base(skipMessage)
        {
        }

        /// <inheritdoc />
        protected override bool IsEnvironmentSupported()
        {
            return Environment.Is64BitProcess && AppDomain.CurrentDomain.GetData("FX_PRODUCT_VERSION") == null;
        }
    }

    /// <summary>
    /// A fact for tests requiring x64 environment and either .NET Core version lower than 3.0 or framework other than .NET Core.
    /// </summary>
    public sealed class LessThanNetCore30OrNotNetCore : EnvironmentSpecificFactAttribute
    {
        public LessThanNetCore30OrNotNetCore() : base("Skipping test on .net core version > 3.0 ")
        {
        }

        /// <inheritdoc />
        protected override bool IsEnvironmentSupported()
        {
            return AppDomain.CurrentDomain.GetData("FX_PRODUCT_VERSION") == null;
        }
    }
}