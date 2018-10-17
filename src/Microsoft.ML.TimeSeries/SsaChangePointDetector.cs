// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.TimeSeriesProcessing;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.StaticPipe.Runtime;

[assembly: LoadableClass(SsaChangePointDetector.Summary, typeof(IDataTransform), typeof(SsaChangePointDetector), typeof(SsaChangePointDetector.Arguments), typeof(SignatureDataTransform),
    SsaChangePointDetector.UserName, SsaChangePointDetector.LoaderSignature, SsaChangePointDetector.ShortName)]

[assembly: LoadableClass(SsaChangePointDetector.Summary, typeof(IDataTransform), typeof(SsaChangePointDetector), null, typeof(SignatureLoadDataTransform),
    SsaChangePointDetector.UserName, SsaChangePointDetector.LoaderSignature)]

[assembly: LoadableClass(SsaChangePointDetector.Summary, typeof(SsaChangePointDetector), null, typeof(SignatureLoadModel),
    SsaChangePointDetector.UserName, SsaChangePointDetector.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(SsaChangePointDetector), null, typeof(SignatureLoadRowMapper),
   SsaChangePointDetector.UserName, SsaChangePointDetector.LoaderSignature)]

namespace Microsoft.ML.Runtime.TimeSeriesProcessing
{
    /// <summary>
    /// This class implements the change point detector transform based on Singular Spectrum modeling of the time-series.
    /// For the details of the Singular Spectrum Analysis (SSA), refer to http://arxiv.org/pdf/1206.6910.pdf.
    /// </summary>
    public sealed class SsaChangePointDetector : SsaAnomalyDetectionBase
    {
        internal const string Summary = "This transform detects the change-points in a seasonal time-series using Singular Spectrum Analysis (SSA).";
        public const string LoaderSignature = "SsaChangePointDetector";
        public const string UserName = "SSA Change Point Detection";
        public const string ShortName = "chgpnt";

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

            [Argument(ArgumentType.Required, HelpText = "The number of points from the beginning of the sequence used for training.",
                ShortName = "twnd", SortOrder = 3)]
            public int TrainingWindowSize = 100;

            [Argument(ArgumentType.Required, HelpText = "The confidence for change point detection in the range [0, 100].",
                ShortName = "cnf", SortOrder = 4)]
            public double Confidence = 95;

            [Argument(ArgumentType.Required, HelpText = "An upper bound on the largest relevant seasonality in the input time-series.", ShortName = "swnd", SortOrder = 5)]
            public int SeasonalWindowSize = 10;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The function used to compute the error between the expected and the observed value.", ShortName = "err", SortOrder = 103)]
            public ErrorFunctionUtils.ErrorFunction ErrorFunction = ErrorFunctionUtils.ErrorFunction.SignedDifference;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The martingale used for scoring.", ShortName = "mart", SortOrder = 104)]
            public MartingaleType Martingale = SequentialAnomalyDetectionTransformBase<float, State>.MartingaleType.Power;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The epsilon parameter for the Power martingale.",
                ShortName = "eps", SortOrder = 105)]
            public double PowerMartingaleEpsilon = 0.1;
        }

        private sealed class BaseArguments : SsaArguments
        {
            public BaseArguments(Arguments args)
            {
                Source = args.Source;
                Name = args.Name;
                Side = SequentialAnomalyDetectionTransformBase<float, State>.AnomalySide.TwoSided;
                WindowSize = args.ChangeHistoryLength;
                InitialWindowSize = args.TrainingWindowSize;
                SeasonalWindowSize = args.SeasonalWindowSize;
                Martingale = args.Martingale;
                PowerMartingaleEpsilon = args.PowerMartingaleEpsilon;
                AlertOn = SequentialAnomalyDetectionTransformBase<float, State>.AlertingScore.MartingaleScore;
                DiscountFactor = 1;
                IsAdaptive = false;
                ErrorFunction = args.ErrorFunction;
            }
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(modelSignature: "SCHGTRNS",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(SsaChangePointDetector).Assembly.FullName);
        }

        // Factory method for SignatureDataTransform.
        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));

            var transform = new SsaChangePointDetector(env, args);
            transform.Fit(input);
            return transform.MakeDataTransform(input);
        }

        internal SsaChangePointDetector(IHostEnvironment env, Arguments args)
            : base(new BaseArguments(args), LoaderSignature, env)
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
                    Host.Assert(!Enum.IsDefined(typeof(MartingaleType), Martingale));
                    throw Host.ExceptUserArg(nameof(args.Martingale), "Value not defined.");
            }
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            env.CheckValue(input, nameof(input));

            return new SsaChangePointDetector(env, ctx).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadModel.
        private static SsaChangePointDetector Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new SsaChangePointDetector(env, ctx);
        }

        internal SsaChangePointDetector(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, ctx, LoaderSignature)
        {
            // *** Binary format ***
            // <base>

            Host.CheckDecode(ThresholdScore == AlertingScore.MartingaleScore);
            Host.CheckDecode(Side == AnomalySide.TwoSided);
            Host.CheckDecode(DiscountFactor == 1);
            Host.CheckDecode(IsAdaptive == false);
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            Host.Assert(ThresholdScore == AlertingScore.MartingaleScore);
            Host.Assert(Side == AnomalySide.TwoSided);
            Host.Assert(DiscountFactor == 1);
            Host.Assert(IsAdaptive == false);

            // *** Binary format ***
            // <base>

            base.Save(ctx);
        }

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, Schema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);
    }

    public sealed class SsaChangePointEstimator : IEstimator<ITransformer>
    {
        private readonly IHost _host;
        private readonly string _inputColumnName;
        private readonly string _outputColumnName;
        private readonly int _confidence;
        private readonly int _changeHistoryLength;
        private readonly int _trainingWindowSize;
        private readonly int _seasonalityWindowSize;

        public SsaChangePointEstimator(
            IHostEnvironment env,
            int confidence,
            int changeHistoryLength,
            int trainingWindowSize,
            int seasonalityWindowSize,
            string input,
            string output)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(SsaChangePointEstimator));

            _host.CheckNonEmpty(input, nameof(input));
            _host.CheckNonEmpty(output, nameof(output));

            _confidence = confidence;
            _changeHistoryLength = changeHistoryLength;
            _trainingWindowSize = trainingWindowSize;
            _seasonalityWindowSize = seasonalityWindowSize;
            _inputColumnName = input;
            _outputColumnName = output;
        }

        public ITransformer Fit(IDataView input)
        {
            _host.CheckValue(input, nameof(input));
            var transformer = new SsaChangePointDetector(_host,
                new SsaChangePointDetector.Arguments {
                    Confidence = _confidence,
                    ChangeHistoryLength = _changeHistoryLength,
                    TrainingWindowSize = _trainingWindowSize,
                    SeasonalWindowSize = _seasonalityWindowSize,
                    Source = _inputColumnName,
                    Name = _outputColumnName
                });
            transformer.Fit(input);
            return transformer;
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var resultDic = inputSchema.Columns.ToDictionary(x => x.Name);

            resultDic[_outputColumnName] = new SchemaShape.Column(
                _outputColumnName, SchemaShape.Column.VectorKind.Vector, NumberType.R8, false);

            return new SchemaShape(resultDic.Values);
        }
    }

    public static class SsaChangePointStaticExtensions
    {
        private sealed class OutColumn : Vector<float>
        {
            public PipelineColumn Input { get; }

            public OutColumn(Vector<float> input,
                int confidence,
                int changeHistoryLength,
                int trainingWindowSize,
                int seasonalityWindowSize)
                : base(new Reconciler(confidence, changeHistoryLength, trainingWindowSize, seasonalityWindowSize), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            private readonly int _confidence;
            private readonly int _changeHistoryLength;
            private readonly int _trainingWindowSize;
            private readonly int _seasonalityWindowSize;

            public Reconciler(
                int confidence,
                int changeHistoryLength,
                int trainingWindowSize,
                int seasonalityWindowSize)
            {
                _confidence = confidence;
                _changeHistoryLength = changeHistoryLength;
                _trainingWindowSize = trainingWindowSize;
                _seasonalityWindowSize = seasonalityWindowSize;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);
                var outCol = (OutColumn)toOutput[0];
                return new SsaChangePointEstimator(env,
                    _confidence,
                    _changeHistoryLength,
                    _trainingWindowSize,
                    _seasonalityWindowSize,
                    inputNames[outCol.Input], outputNames[outCol]);
            }
        }
    }
}
