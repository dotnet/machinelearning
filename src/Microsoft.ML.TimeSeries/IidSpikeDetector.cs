// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.TimeSeriesProcessing;

[assembly: LoadableClass(IidSpikeDetector.Summary, typeof(IidSpikeDetector), typeof(IidSpikeDetector.Arguments), typeof(SignatureDataTransform),
    IidSpikeDetector.UserName, IidSpikeDetector.LoaderSignature, IidSpikeDetector.ShortName)]
[assembly: LoadableClass(IidSpikeDetector.Summary, typeof(IidSpikeDetector), null, typeof(SignatureLoadDataTransform),
    IidSpikeDetector.UserName, IidSpikeDetector.LoaderSignature)]

namespace Microsoft.ML.Runtime.TimeSeriesProcessing
{
    /// <summary>
    /// This class implements the spike detector transform for an i.i.d. sequence based on adaptive kernel density estimation.
    /// </summary>
    public sealed class IidSpikeDetector : IidAnomalyDetectionBase, ITransformTemplate
    {
        internal const string Summary = "This transform detects the spikes in a i.i.d. sequence using adaptive kernel density estimation.";
        public const string LoaderSignature = "IidSpikeDetector";
        public const string UserName = "IID Spike Detection";
        public const string ShortName = "ispike";

        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "The name of the source column.", ShortName = "src",
                SortOrder = 1, Purpose = SpecialPurpose.ColumnName)]
            public string Source;

            [Argument(ArgumentType.Required, HelpText = "The name of the new column.",
                SortOrder = 2)]
            public string Name;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The argument that determines whether to detect positive or negative anomalies, or both.", ShortName = "side",
                SortOrder = 101)]
            public AnomalySide Side = AnomalySide.TwoSided;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The size of the sliding window for computing the p-value.", ShortName = "wnd",
                SortOrder = 102)]
            public int PvalueHistoryLength = 100;

            [Argument(ArgumentType.Required, HelpText = "The confidence for spike detection in the range [0, 100].",
                ShortName = "cnf", SortOrder = 3)]
            public double Confidence = 99;
        }

        private sealed class BaseArguments : ArgumentsBase
        {
            public BaseArguments(Arguments args)
            {
                Source = args.Source;
                Name = args.Name;
                Side = args.Side;
                WindowSize = args.PvalueHistoryLength;
                AlertThreshold = 1 - args.Confidence / 100;
                AlertOn = SequentialAnomalyDetectionTransformBase<float, State>.AlertingScore.PValueScore;
                Martingale = MartingaleType.None;
            }

            public BaseArguments(IidSpikeDetector transform)
            {
                Source = transform.InputColumnName;
                Name = transform.OutputColumnName;
                Side = transform.Side;
                WindowSize = transform.WindowSize;
                AlertThreshold = transform.AlertThreshold;
                AlertOn = AlertingScore.PValueScore;
                Martingale = MartingaleType.None;
            }
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "ISPKTRNS",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        public IidSpikeDetector(IHostEnvironment env, Arguments args, IDataView input)
            : base(new BaseArguments(args), LoaderSignature, env, input)
        {
            // This constructor is empty.
        }

        public IidSpikeDetector(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            : base(env, ctx, LoaderSignature, input)
        {
            // *** Binary format ***
            // <base>

            Host.CheckDecode(ThresholdScore == AlertingScore.PValueScore);
        }
        private IidSpikeDetector(IHostEnvironment env, IidSpikeDetector transform, IDataView newSource)
           : base(new BaseArguments(transform), LoaderSignature, env, newSource)
        {
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            Host.Assert(ThresholdScore == AlertingScore.PValueScore);

            // *** Binary format ***
            // <base>

            base.Save(ctx);
        }

        public IDataTransform ApplyToData(IHostEnvironment env, IDataView newSource)
        {
            return new IidSpikeDetector(env, this, newSource);
        }
    }
}
