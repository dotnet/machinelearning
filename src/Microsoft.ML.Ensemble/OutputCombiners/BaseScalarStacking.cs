// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

namespace Microsoft.ML.Runtime.Ensemble.OutputCombiners
{
    public abstract class BaseScalarStacking<TSigBase> : BaseStacking<Single, TSigBase>
    {
        internal BaseScalarStacking(IHostEnvironment env, string name, ArgumentsBase args)
            : base(env, name, args)
        {
        }

        internal BaseScalarStacking(IHostEnvironment env, string name, ModelLoadContext ctx)
            : base(env, name, ctx)
        {
        }

        protected override void FillFeatureBuffer(Single[] src, ref VBuffer<Single> dst)
        {
            Contracts.AssertNonEmpty(src);
            int len = src.Length;
            var values = dst.Values;
            if (Utils.Size(values) < len)
                values = new Single[len];
            Array.Copy(src, values, len);
            dst = new VBuffer<Single>(len, values, dst.Indices);
        }
    }
}
