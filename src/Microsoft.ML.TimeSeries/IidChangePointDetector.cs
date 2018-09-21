// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.TimeSeriesProcessing;

[assembly: LoadableClass(IidChangePointDetector.Summary, typeof(IidChangePointDetector), typeof(IidChangePointDetector.Arguments), typeof(SignatureDataTransform),
    IidChangePointDetector.UserName, IidChangePointDetector.LoaderSignature, IidChangePointDetector.ShortName)]
[assembly: LoadableClass(IidChangePointDetector.Summary, typeof(IidChangePointDetector), null, typeof(SignatureLoadDataTransform),
    IidChangePointDetector.UserName, IidChangePointDetector.LoaderSignature)]

namespace Microsoft.ML.Runtime.TimeSeriesProcessing
{
    /// <summary>
    /// This class implements the change point detector transform for an i.i.d. sequence based on adaptive kernel density estimation and martingales.
    /// </summary>
    public sealed class IidChangePointDetector : IidAnomalyDetectionBase
    {
        internal const string Summary = "This transform detects the change-points in an i.i.d. sequence using adaptive kernel density estimation and martingales.";
        public const string LoaderSignature = "IidChangePointDetector";
        public const string UserName = "IID Change Point Detection";
        public const string ShortName = "ichgpnt";

        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "The name of the source column.", ShortName = "src",
                SortOrder = 1, Purpose = SpecialPurpose.ColumnName)]
            public string Source;

            [Argument(ArgumentType.Required, HelpText = "The name of the new column.",
                SortOrder = 2)]
            public string Name;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The change history length.", ShortName = "wnd",
                SortOrder = 102)]
            public int ChangeHistoryLength = 20;

            [Argument(ArgumentType.Required, HelpText = "The confidence for change point detection in the range [0, 100].",
                ShortName = "cnf", SortOrder = 3)]
            public double Confidence = 95;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The martingale used for scoring.", ShortName = "mart", SortOrder = 103)]
            public MartingaleType Martingale = SequentialAnomalyDetectionTransformBase<float, State>.MartingaleType.Power;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The epsilon parameter for the Power martingale.",
                ShortName = "eps", SortOrder = 104)]
            public double PowerMartingaleEpsilon = 0.1;
        }

        private sealed class BaseArguments : ArgumentsBase
        {
            public BaseArguments(Arguments args)
            {
                Source = args.Source;
                Name = args.Name;
                Side = SequentialAnomalyDetectionTransformBase<float, State>.AnomalySide.TwoSided;
                WindowSize = args.ChangeHistoryLength;
                Martingale = args.Martingale;
                PowerMartingaleEpsilon = args.PowerMartingaleEpsilon;
                AlertOn = SequentialAnomalyDetectionTransformBase<float, State>.AlertingScore.MartingaleScore;
            }
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(modelSignature: "ICHGTRNS",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        public IidChangePointDetector(IHostEnvironment env, Arguments args, IDataView input)
            : base(new BaseArguments(args), LoaderSignature, env, input)
        {
            switch (Martingale)
            {
            case MartingaleType.None:
                AlertThreshold = Double.MaxValue;
                break;
            case MartingaleType.Power:
                AlertThreshold = Math.Exp(WindowSize * LogPowerMartigaleBettingFunc(1 - args.Confidence / 100, PowerMartingaleEpsilon));
                break;
            case MartingaleType.Mixture:
                AlertThreshold = Math.Exp(WindowSize * LogMixtureMartigaleBettingFunc(1 - args.Confidence / 100));
                break;
            default:
                throw Host.ExceptParam(nameof(args.Martingale),
                    "The martingale type can be only (0) None, (1) Power or (2) Mixture.");
            }
        }

        public IidChangePointDetector(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            : base(env, ctx, LoaderSignature,  input)
        {
            // *** Binary format ***
            // <base>

            Host.CheckDecode(ThresholdScore == AlertingScore.MartingaleScore);
            Host.CheckDecode(Side == AnomalySide.TwoSided);
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            Host.Assert(ThresholdScore == AlertingScore.MartingaleScore);
            Host.Assert(Side == AnomalySide.TwoSided);

            // *** Binary format ***
            // <base>

            base.Save(ctx);
        }
    }
}
