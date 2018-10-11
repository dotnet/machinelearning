// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Numeric;

namespace Microsoft.ML.Runtime.Ensemble.OutputCombiners
{
    public abstract class BaseMultiAverager : BaseMultiCombiner
    {
        internal BaseMultiAverager(IHostEnvironment env, string name, ArgumentsBase args)
            : base(env, name, args)
        {
        }

        internal BaseMultiAverager(IHostEnvironment env, string name, ModelLoadContext ctx)
            : base(env, name, ctx)
        {
        }

        protected void CombineCore(ref VBuffer<Single> dst, VBuffer<Single>[] src, Single[] weights = null)
        {
            Host.AssertNonEmpty(src);
            Host.Assert(weights == null || Utils.Size(weights) == Utils.Size(src));

            // REVIEW: Should this be tolerant of NaNs?
            int len = GetClassCount(src);
            if (!TryNormalize(src))
            {
                GetNaNOutput(ref dst, len);
                return;
            }

            var values = dst.Values;
            if (Utils.Size(values) < len)
                values = new Single[len];
            else
                Array.Clear(values, 0, len);

            // Set the output to values.
            dst = new VBuffer<Single>(len, values, dst.Indices);

            Single weightTotal;
            if (weights == null)
            {
                weightTotal = src.Length;
                for (int i = 0; i < src.Length; i++)
                    VectorUtils.Add(src[i], ref dst);
            }
            else
            {
                weightTotal = 0;
                for (int i = 0; i < src.Length; i++)
                {
                    var w = weights[i];
                    weightTotal += w;
                    VectorUtils.AddMult(src[i], w, ref dst);
                }
            }

            VectorUtils.ScaleBy(ref dst, 1 / weightTotal);
        }
    }
}
