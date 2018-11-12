// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.TimeSeriesProcessing;

[assembly: LoadableClass(PValueTransform.Summary, typeof(PValueTransform), typeof(PValueTransform.Arguments), typeof(SignatureDataTransform),
    PValueTransform.UserName, PValueTransform.LoaderSignature, PValueTransform.ShortName)]
[assembly: LoadableClass(PValueTransform.Summary, typeof(PValueTransform), null, typeof(SignatureLoadDataTransform),
    PValueTransform.UserName, PValueTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.TimeSeriesProcessing
{
    /// <summary>
    /// PValueTransform is a sequential transform that computes the empirical p-value of the current value in the series based on the other values in
    /// the sliding window.
    /// </summary>
    public sealed class PValueTransform : SequentialTransformBase<Single, Single, PValueTransform.State>
    {
        internal const string Summary = "This P-Value transform calculates the p-value of the current input in the sequence with regard to the values in the sliding window.";
        public const string LoaderSignature = "PValueTransform";
        public const string UserName = "p-Value Transform";
        public const string ShortName = "PVal";

        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "The name of the source column", ShortName = "src",
                SortOrder = 1, Purpose = SpecialPurpose.ColumnName)]
            public string Source;

            [Argument(ArgumentType.Required, HelpText = "The name of the new column", ShortName = "name",
                SortOrder = 2)]
            public string Name;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The seed value of the random generator", ShortName = "seed",
                SortOrder = 3)]
            public int Seed = 0;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The flag that determines whether the p-values are calculated on the positive side", ShortName = "pos",
                SortOrder = 4)]
            public bool PositiveSide = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The size of the sliding window for computing the p-value", ShortName = "wnd",
                SortOrder = 5)]
            public int WindowSize = 1;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The size of the initial window for computing the p-value. The default value is set to 0, which means there is no initial window considered.",
                ShortName = "initwnd", SortOrder = 6)]
            public int InitialWindowSize = 0;
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "PVALTRNS",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(PValueTransform).Assembly.FullName);
        }

        private readonly int _seed;
        private readonly bool _isPositiveSide;

        public PValueTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(args.WindowSize, args.InitialWindowSize, args.Source, args.Name, LoaderSignature, env, input)
        {
            Host.CheckUserArg(args.WindowSize >= 1, nameof(args.WindowSize), "The size of the sliding window should be at least 1.");
            _seed = args.Seed;
            _isPositiveSide = args.PositiveSide;
        }

        public PValueTransform(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            : base(env, ctx, LoaderSignature, input)
        {
            // *** Binary format ***
            // int: _percentile
            // byte: _isPositiveSide

            _seed = ctx.Reader.ReadInt32();
            _isPositiveSide = ctx.Reader.ReadBoolByte();
            Host.CheckDecode(WindowSize >= 1);
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.Assert(WindowSize >= 1);
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>
            // int: _percentile
            // byte: _isPositiveSide

            base.Save(ctx);
            ctx.Writer.Write(_seed);
            ctx.Writer.WriteBoolByte(_isPositiveSide);
        }

        public sealed class State : StateBase
        {
            private IRandom _randomGen;

            private PValueTransform _parent;

            private protected override void SetNaOutput(ref Single dst)
            {
                dst = Single.NaN;
            }

            private protected override void TransformCore(ref Single input, FixedSizeQueue<Single> windowedBuffer, long iteration, ref Single dst)
            {
                int count;
                int equalCount;
                int totalCount;

                PercentileThresholdTransform.CountGreaterOrEqualValues(windowedBuffer, input, out count, out equalCount, out totalCount);
                count = (_parent._isPositiveSide) ? count : totalCount - count - equalCount;

                dst = (Single)((count + _randomGen.NextDouble() * equalCount) / (totalCount + 1));
                // Based on the equation in http://arxiv.org/pdf/1204.3251.pdf
            }

            private protected override void InitializeStateCore()
            {
                _parent = (PValueTransform)ParentTransform;
                _randomGen = RandomUtils.Create(_parent._seed);
            }

            private protected override void LearnStateFromDataCore(FixedSizeQueue<Single> data)
            {
                // This method is empty because there is no need for parameter learning from the initial windowed buffer for this transform.
            }
        }
    }
}
