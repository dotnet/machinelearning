// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
namespace Microsoft.ML.TestFramework.Attributes
{
    /// <summary>
    /// A fact for tests requiring matrix factorization.
    /// </summary>
    public sealed class MatrixFactorizationFactAttribute : EnvironmentSpecificFactAttribute
    {
        public MatrixFactorizationFactAttribute() : base("Disabled - this test is being fixed as part of https://github.com/dotnet/machinelearning/issues/1441")
        {
        }

        /// <inheritdoc />
        protected override bool IsEnvironmentSupported()
        {
            return Environment.Is64BitProcess;
        }
    }
}