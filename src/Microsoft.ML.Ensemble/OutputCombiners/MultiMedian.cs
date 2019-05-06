// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.Ensemble;

[assembly: LoadableClass(typeof(MultiMedian), typeof(MultiMedian.Options), typeof(SignatureCombiner),
    Median.UserName, MultiMedian.LoadName)]
[assembly: LoadableClass(typeof(MultiMedian), null, typeof(SignatureLoadModel), Median.UserName, MultiMedian.LoaderSignature)]

namespace Microsoft.ML.Trainers.Ensemble
{
    /// <summary>
    /// Generic interface for combining outputs of multiple models
    /// </summary>
    internal sealed class MultiMedian : BaseMultiCombiner
    {
        public const string LoadName = "MultiMedian";
        public const string LoaderSignature = "MultiMedianCombiner";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "MMEDCOMB",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(MultiMedian).Assembly.FullName);
        }

        [TlcModule.Component(Name = LoadName, FriendlyName = Median.UserName)]
        public sealed class Options : OptionsBase, ISupportMulticlassOutputCombinerFactory
        {
            public IMulticlassOutputCombiner CreateComponent(IHostEnvironment env) => new MultiMedian(env, this);
        }

        public MultiMedian(IHostEnvironment env, Options options)
            : base(env, LoaderSignature, options)
        {
        }

        private MultiMedian(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, LoaderSignature, ctx)
        {
        }

        public static MultiMedian Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new MultiMedian(env, ctx);
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
        }

        public override Combiner<VBuffer<Single>> GetCombiner()
        {
            Single[] raw = null;
            return
                (ref VBuffer<Single> dst, VBuffer<Single>[] src, Single[] weights) =>
                {
                    Host.AssertNonEmpty(src);
                    Host.Assert(weights == null || Utils.Size(weights) == Utils.Size(src));

                    int len = GetClassCount(src);
                    if (!TryNormalize(src))
                    {
                        GetNaNOutput(ref dst, len);
                        return;
                    }

                    var editor = VBufferEditor.Create(ref dst, len);

                    int count = src.Length;
                    if (Utils.Size(raw) < count)
                        raw = new Single[count];
                    for (int i = 0; i < len; i++)
                    {
                        for (int j = 0; j < count; j++)
                            raw[j] = i < src[j].Length ? src[j].GetItemOrDefault(i) : 0;
                        editor.Values[i] = MathUtils.GetMedianInPlace(raw, count);
                    }

                    // Set the output to values.
                    dst = editor.Commit();
                };
        }
    }
}
