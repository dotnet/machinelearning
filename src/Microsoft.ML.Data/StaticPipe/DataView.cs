// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Data.StaticPipe
{
    public class DataView<TTupleShape>
    {
        private readonly IHostEnvironment _env;
        public IDataView Wrapped { get; }

        public DataView(IHostEnvironment env, IDataView view)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(view, nameof(view));

            _env = env;
            Wrapped = view;
        }
    }
}
