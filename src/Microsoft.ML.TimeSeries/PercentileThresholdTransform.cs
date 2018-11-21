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

[assembly: LoadableClass(PercentileThresholdTransform.Summary, typeof(PercentileThresholdTransform), typeof(PercentileThresholdTransform.Arguments), typeof(SignatureDataTransform),
    PercentileThresholdTransform.UserName, PercentileThresholdTransform.LoaderSignature, PercentileThresholdTransform.ShortName)]
[assembly: LoadableClass(PercentileThresholdTransform.Summary, typeof(PercentileThresholdTransform), null, typeof(SignatureLoadDataTransform),
    PercentileThresholdTransform.UserName, PercentileThresholdTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.TimeSeriesProcessing
{
    /// <summary>
    /// PercentileThresholdTransform is a sequential transform that decides whether the current value of the time-series belongs to the 'percentile' % of the top values in
    /// the sliding window. The output of the transform will be a boolean flag.
    /// </summary>
    public sealed class PercentileThresholdTransform : SequentialTransformBase<Single, bool, PercentileThresholdTransform.State>
    {
        public const string Summary = "Detects the values of time-series that are in the top percentile of the sliding window.";
        public const string LoaderSignature = "PercentThrTransform";
        public const string UserName = "Percentile Threshold Transform";
        public const string ShortName = "TopPcnt";

        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "The name of the source column", ShortName = "src",
                SortOrder = 1, Purpose = SpecialPurpose.ColumnName)]
            public string Source;

            [Argument(ArgumentType.Required, HelpText = "The name of the new column", ShortName = "name",
                SortOrder = 2)]
            public string Name;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The percentile value for thresholding in the range [0, 100]", ShortName = "pcnt",
                SortOrder = 3)]
            public Double Percentile = 1;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The size of the sliding window for computing the percentile threshold. " +
                                                          "The default value is set to 1.", ShortName = "wnd",
                SortOrder = 4)]
            public int WindowSize = 1;
        }

        public const Double MinPercentile = 0;
        public const Double MaxPercentile = 100;

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "PCNTTRNS",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(PercentileThresholdTransform).Assembly.FullName);
        }

        private readonly Double _percentile;

        public PercentileThresholdTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(args.WindowSize, args.WindowSize, args.Source, args.Name, LoaderSignature, env, input)
        {
            Host.CheckUserArg(args.WindowSize >= 1, nameof(args.WindowSize), "The size of the sliding window should be at least 1.");
            Host.CheckUserArg(MinPercentile <= args.Percentile && args.Percentile <= MaxPercentile, nameof(args.Percentile), "The percentile value should be in [0, 100].");
            _percentile = args.Percentile;
        }

        public PercentileThresholdTransform(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            : base(env, ctx, LoaderSignature,input)
        {
            // *** Binary format ***
            // Double: _percentile

            _percentile = ctx.Reader.ReadDouble();

            Host.CheckDecode(WindowSize >= 1);
            Host.CheckDecode(MinPercentile <= _percentile && _percentile <= MaxPercentile);
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.Assert(MinPercentile <= _percentile && _percentile <= MaxPercentile);
            Host.Assert(WindowSize >= 1);
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>
            // Double: _percentile

            base.Save(ctx);
            ctx.Writer.Write(_percentile);
        }

        public static void CountGreaterOrEqualValues(FixedSizeQueue<Single> others, Single theValue, out int greaterVals, out int equalVals, out int totalVals)
        {
            // The current linear algorithm for counting greater and equal elements takes O(n),
            // but it can be improved to O(log n) if a separate Binary Search Tree data structure is used.

            greaterVals = 1;
            equalVals = 0;
            totalVals = 0;

            var n = others.Count;

            for (int i = 0; i < n; ++i)
            {
                if (!Single.IsNaN(others[i]))
                {
                    greaterVals += (others[i] > theValue) ? 1 : 0;
                    equalVals += (others[i] == theValue) ? 1 : 0;
                    totalVals++;
                }
            }
        }

        public sealed class State : StateBase
        {
            /// <summary>
            /// The number of elements in the top 'percentile' % of the top values.
            /// </summary>
            private PercentileThresholdTransform _parent;

            protected override void SetNaOutput(ref bool dst)
            {
                dst = false;
            }

            protected override void TransformCore(ref Single input, FixedSizeQueue<Single> windowedBuffer, long iteration, ref bool dst)
            {
                int greaterCount;
                int equalCount;
                int totalCount;

                CountGreaterOrEqualValues(windowedBuffer, input, out greaterCount, out equalCount, out totalCount);
                dst = greaterCount < (int)(_parent._percentile * totalCount / 100);
            }

            protected override void InitializeStateCore()
            {
                _parent = (PercentileThresholdTransform)ParentTransform;
            }

            protected override void LearnStateFromDataCore(FixedSizeQueue<Single> data)
            {
                // This method is empty because there is no need for parameter learning from the initial windowed buffer for this transform.
            }
        }
    }
}
