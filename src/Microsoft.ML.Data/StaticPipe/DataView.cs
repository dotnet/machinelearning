// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Data.StaticPipe.Runtime;

namespace Microsoft.ML.Data.StaticPipe
{
    public class DataView<TTupleShape> : SchemaBearing<TTupleShape>
    {
        public IDataView AsDynamic { get; }

        internal DataView(IHostEnvironment env, IDataView view, StaticSchemaShape shape)
            : base(env, shape)
        {
            Env.AssertValue(view);

            AsDynamic = view;
            Shape.Check(Env, AsDynamic.Schema);
        }
    }
}
