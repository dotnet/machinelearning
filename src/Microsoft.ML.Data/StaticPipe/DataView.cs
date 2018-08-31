// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Data.StaticPipe
{
    public class DataView<TTupleShape> : SchemaBearing<TTupleShape>
    {
        public IDataView AsDynamic { get; }

        public DataView(IHostEnvironment env, IDataView view)
            : base(env)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(view, nameof(view));

            AsDynamic = view;
        }
    }
}
