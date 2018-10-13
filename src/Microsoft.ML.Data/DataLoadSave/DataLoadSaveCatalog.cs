using System;
using System.Collections.Generic;
using System.Text;

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
