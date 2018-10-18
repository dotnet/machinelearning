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

[assembly: LoadableClass(IidChangePointDetector.Summary, typeof(IDataTransform), typeof(IidChangePointDetector), typeof(IidChangePointDetector.Arguments), typeof(SignatureDataTransform),
    IidChangePointDetector.UserName, IidChangePointDetector.LoaderSignature, IidChangePointDetector.ShortName)]

[assembly: LoadableClass(IidChangePointDetector.Summary, typeof(IDataTransform), typeof(IidChangePointDetector), null, typeof(SignatureLoadDataTransform),
    IidChangePointDetector.UserName, IidChangePointDetector.LoaderSignature)]

[assembly: LoadableClass(IidChangePointDetector.Summary, typeof(IidChangePointDetector), null, typeof(SignatureLoadModel),
    IidChangePointDetector.UserName, IidChangePointDetector.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(IidChangePointDetector), null, typeof(SignatureLoadRowMapper),
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

            public BaseArguments(IidChangePointDetector transform)
            {
                Source = transform.InputColumnName;
                Name = transform.OutputColumnName;
                Side = AnomalySide.TwoSided;
                WindowSize = transform.WindowSize;
                Martingale = transform.Martingale;
                PowerMartingaleEpsilon = transform.PowerMartingaleEpsilon;
                AlertOn = AlertingScore.MartingaleScore;
                AlertThreshold = transform.AlertThreshold;
            }
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(modelSignature: "ICHGTRNS",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(IidChangePointDetector).Assembly.FullName);
        }

        // Factory method for SignatureDataTransform.
        private static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));

            return new IidChangePointDetector(env, args).MakeDataTransform(input);
        }

        internal IidChangePointDetector(IHostEnvironment env, Arguments args)
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
                    throw Host.ExceptParam(nameof(args.Martingale),
                        "The martingale type can be only (0) None, (1) Power or (2) Mixture.");
            }
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            env.CheckValue(input, nameof(input));

            return new IidChangePointDetector(env, ctx).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadModel.
        private static IidChangePointDetector Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new IidChangePointDetector(env, ctx);
        }

        internal IidChangePointDetector(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, ctx, LoaderSignature)
        {
            // *** Binary format ***
            // <base>

            Host.CheckDecode(ThresholdScore == AlertingScore.MartingaleScore);
            Host.CheckDecode(Side == AnomalySide.TwoSided);
        }

        private IidChangePointDetector(IHostEnvironment env, IidChangePointDetector transform)
            : base(new BaseArguments(transform), LoaderSignature, env)
        {
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

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);
    }

    public sealed class IidChangePointEstimator : IEstimator<IidChangePointDetector>
    {
        private readonly IHost _host;
        private readonly string _inputColumnName;
        private readonly string _outputColumnName;
        private readonly int _confidence;
        private readonly int _changeHistoryLength;

        public IidChangePointEstimator(
            IHostEnvironment env,
            int confidence,
            int changeHistoryLength,
            string input,
            string output)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(IidChangePointEstimator));

            _host.CheckNonEmpty(input, nameof(input));
            _host.CheckNonEmpty(output, nameof(output));

            _confidence = confidence;
            _changeHistoryLength = changeHistoryLength;
            _inputColumnName = input;
            _outputColumnName = output;
        }

        public IidChangePointDetector Fit(IDataView input)
        {
            _host.CheckValue(input, nameof(input));
            return new IidChangePointDetector(_host,
                new IidChangePointDetector.Arguments
                {
                    Confidence = _confidence,
                    ChangeHistoryLength = _changeHistoryLength,
                    Source = _inputColumnName,
                    Name = _outputColumnName,
                });
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var metadata = new List<SchemaShape.Column>() {
                new SchemaShape.Column(MetadataUtils.Kinds.SlotNames, SchemaShape.Column.VectorKind.Vector, TextType.Instance, false)
            };
            var resultDic = inputSchema.Columns.ToDictionary(x => x.Name);
            resultDic[_outputColumnName] = new SchemaShape.Column(
                _outputColumnName, SchemaShape.Column.VectorKind.Vector, NumberType.R8, false, new SchemaShape(metadata));

            return new SchemaShape(resultDic.Values);
        }
    }
}
