﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Numeric;

namespace Microsoft.ML.Trainers.Ensemble
{
    internal abstract class BaseMultiCombiner : IMultiClassOutputCombiner, ICanSaveModel
    {
        protected readonly IHost Host;

        public abstract class OptionsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to normalize the output of base models before combining them",
                ShortName = "norm", SortOrder = 50)]
            public bool Normalize = true;
        }

        protected readonly bool Normalize;

        internal BaseMultiCombiner(IHostEnvironment env, string name, OptionsBase options)
        {
            Contracts.AssertValue(env);
            env.AssertNonWhiteSpace(name);
            Host = env.Register(name);
            Host.CheckValue(options, nameof(options));

            Normalize = options.Normalize;
        }

        internal BaseMultiCombiner(IHostEnvironment env, string name, ModelLoadContext ctx)
        {
            Contracts.AssertValue(env);
            env.AssertNonWhiteSpace(name);
            Host = env.Register(name);
            Host.AssertValue(ctx);

            // *** Binary format ***
            // int: sizeof(Single)
            // bool: _normalize
            int cbFloat = ctx.Reader.ReadInt32();
            Host.CheckDecode(cbFloat == sizeof(Single));
            Normalize = ctx.Reader.ReadBoolByte();
        }

        void ICanSaveModel.Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            SaveCore(ctx);
        }

        protected virtual void SaveCore(ModelSaveContext ctx)
        {
            // *** Binary format ***
            // int: sizeof(Single)
            // bool: _normalize
            ctx.Writer.Write(sizeof(Single));
            ctx.Writer.WriteBoolByte(Normalize);
        }

        public abstract Combiner<VBuffer<Single>> GetCombiner();

        protected int GetClassCount(VBuffer<Single>[] values)
        {
            int len = 0;
            foreach (var item in values)
            {
                if (len < item.Length)
                    len = item.Length;
            }
            return len;
        }

        protected bool TryNormalize(VBuffer<Single>[] values)
        {
            if (!Normalize)
                return true;

            for (int i = 0; i < values.Length; i++)
            {
                // Leave a zero vector as all zeros. Otherwise, make the L1 norm equal to 1.
                var sum = VectorUtils.L1Norm(in values[i]);
                if (!FloatUtils.IsFinite(sum))
                    return false;
                if (sum > 0)
                    VectorUtils.ScaleBy(ref values[i], 1 / sum);
            }
            return true;
        }

        protected void GetNaNOutput(ref VBuffer<Single> dst, int len)
        {
            Contracts.Assert(len >= 0);
            var editor = VBufferEditor.Create(ref dst, len);
            for (int i = 0; i < len; i++)
                editor.Values[i] = Single.NaN;
            dst = editor.Commit();
        }
    }
}
