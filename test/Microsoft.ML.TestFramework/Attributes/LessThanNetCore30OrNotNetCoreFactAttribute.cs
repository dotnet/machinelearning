// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
namespace Microsoft.ML.TestFramework.Attributes
{
    /// <summary>
    /// A fact for tests requiring either .NET Core version lower than 3.0 or framework other than .NET Core.
    /// </summary>
    public sealed class LessThanNetCore30OrNotNetCoreFactAttribute : EnvironmentSpecificFactAttribute
    {
        public LessThanNetCore30OrNotNetCoreFactAttribute(string skipMessage) : base(skipMessage)
        {
        }

        /// <inheritdoc />
        protected override bool IsEnvironmentSupported()
        {
            return AppDomain.CurrentDomain.GetData("FX_PRODUCT_VERSION") == null;
        }
    }
}