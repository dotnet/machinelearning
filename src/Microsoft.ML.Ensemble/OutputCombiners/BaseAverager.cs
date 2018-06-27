// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime.Model;

namespace Microsoft.ML.Runtime.Ensemble.OutputCombiners
{
    public abstract class BaseAverager : IBinaryOutputCombiner
    {
        protected readonly IHost Host;
        public BaseAverager(IHostEnvironment env, string name)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckNonWhiteSpace(name, nameof(name));
            Host = env.Register(name);
        }

        protected BaseAverager(IHostEnvironment env, string name, ModelLoadContext ctx)
        {
            Contracts.AssertValue(env);
            env.AssertNonWhiteSpace(name);
            Host = env.Register(name);
            Host.CheckValue(ctx, nameof(ctx));

            // *** Binary format ***
            // int: sizeof(Single)
            int cbFloat = ctx.Reader.ReadInt32();
            Host.CheckDecode(cbFloat == sizeof(Single));
        }

        public void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            SaveCore(ctx);
        }

        protected virtual void SaveCore(ModelSaveContext ctx)
        {
            // *** Binary format ***
            // int: sizeof(Single)
            ctx.Writer.Write(sizeof(Single));
        }

        public abstract Combiner<Single> GetCombiner();

        protected void CombineCore(ref Single dst, Single[] src, Single[] weights = null)
        {
            Single sum = 0;
            Single weightTotal = 0;
            if (weights == null)
            {
                for (int i = 0; i < src.Length; i++)
                {
                    if (!Single.IsNaN(src[i]))
                    {
                        sum += src[i];
                        weightTotal++;
                    }
                }
            }
            else
            {
                for (int i = 0; i < src.Length; i++)
                {
                    if (!Single.IsNaN(src[i]))
                    {
                        sum += weights[i] * src[i];
                        weightTotal += weights[i];
                    }
                }
            }
            dst = sum / weightTotal;
        }
    }
}
