// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Runtime
{
    /// <summary>
    /// A catalog of operations to load and save data.
    /// </summary>
    public sealed class DataLoadSaveOperations
    {
        internal IHostEnvironment Environment { get; }

        internal DataLoadSaveOperations(IHostEnvironment env)
        {
            Contracts.AssertValue(env);
            Environment = env;
        }
    }
}
