// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Numeric;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Trainers.Ensemble
{
    internal abstract class BaseMultiAverager : BaseMultiCombiner
    {
        private protected BaseMultiAverager(IHostEnvironment env, string name, OptionsBase options)
            : base(env, name, options)
        {
        }

        protected BaseMultiAverager(IHostEnvironment env, string name, ModelLoadContext ctx)
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

            var editor = VBufferEditor.Create(ref dst, len);
            if (!editor.CreatedNewValues)
                editor.Values.Clear();
            // Set the output to values.
            dst = editor.Commit();

            Single weightTotal;
            if (weights == null)
            {
                weightTotal = src.Length;
                for (int i = 0; i < src.Length; i++)
                    VectorUtils.Add(in src[i], ref dst);
            }
            else
            {
                weightTotal = 0;
                for (int i = 0; i < src.Length; i++)
                {
                    var w = weights[i];
                    weightTotal += w;
                    VectorUtils.AddMult(in src[i], w, ref dst);
                }
            }

            VectorUtils.ScaleBy(ref dst, 1 / weightTotal);
        }
    }
}
