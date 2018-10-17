﻿// Licensed to the .NET Foundation under one or more agreements.
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

[assembly: LoadableClass(IidSpikeDetector.Summary, typeof(IDataTransform), typeof(IidSpikeDetector), typeof(IidSpikeDetector.Arguments), typeof(SignatureDataTransform),
    IidSpikeDetector.UserName, IidSpikeDetector.LoaderSignature, IidSpikeDetector.ShortName)]

[assembly: LoadableClass(IidSpikeDetector.Summary, typeof(IDataTransform), typeof(IidSpikeDetector), null, typeof(SignatureLoadDataTransform),
    IidSpikeDetector.UserName, IidSpikeDetector.LoaderSignature)]

[assembly: LoadableClass(IidSpikeDetector.Summary, typeof(IidSpikeDetector), null, typeof(SignatureLoadModel),
    IidSpikeDetector.UserName, IidSpikeDetector.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(IidSpikeDetector), null, typeof(SignatureLoadRowMapper),
   IidSpikeDetector.UserName, IidSpikeDetector.LoaderSignature)]

namespace Microsoft.ML.Runtime.TimeSeriesProcessing
{
    /// <summary>
    /// This class implements the spike detector transform for an i.i.d. sequence based on adaptive kernel density estimation.
    /// </summary>
    public sealed class IidSpikeDetector : IidAnomalyDetectionBase
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
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(IidSpikeDetector).Assembly.FullName);
        }

        // Factory method for SignatureDataTransform.
        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));

            return new IidSpikeDetector(env, args).MakeDataTransform(input);
        }

        internal IidSpikeDetector(IHostEnvironment env, Arguments args)
            : base(new BaseArguments(args), LoaderSignature, env)
        {
            // This constructor is empty.
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            env.CheckValue(input, nameof(input));

            return new IidSpikeDetector(env, ctx).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadModel.
        private static IidSpikeDetector Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new IidSpikeDetector(env, ctx);
        }

        public IidSpikeDetector(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, ctx, LoaderSignature)
        {
            // *** Binary format ***
            // <base>

            Host.CheckDecode(ThresholdScore == AlertingScore.PValueScore);
        }
        private IidSpikeDetector(IHostEnvironment env, IidSpikeDetector transform)
           : base(new BaseArguments(transform), LoaderSignature, env)
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

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, Schema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);
    }

    public sealed class IidSpikeEstimator : IEstimator<ITransformer>
    {
        private readonly IHost _host;
        private readonly string _inputColumnName;
        private readonly string _outputColumnName;
        private readonly int _confidence;
        private readonly int _pvalueHistoryLength;

        public IidSpikeEstimator(
            IHostEnvironment env,
            int confidence,
            int pvalueHistoryLength,
            string input,
            string output)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(IidSpikeEstimator));

            _host.CheckNonEmpty(input, nameof(input));
            _host.CheckNonEmpty(output, nameof(output));

            _confidence = confidence;
            _pvalueHistoryLength = pvalueHistoryLength;
            _inputColumnName = input;
            _outputColumnName = output;
        }

        public ITransformer Fit(IDataView input)
        {
            _host.CheckValue(input, nameof(input));
            return new IidSpikeDetector(_host,
                new IidSpikeDetector.Arguments
                {
                    Confidence = _confidence,
                    PvalueHistoryLength = _pvalueHistoryLength,
                    Source = _inputColumnName,
                    Name = _outputColumnName
                });
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

    public static class IidSpikeStaticExtensions
    {
        private sealed class OutColumn : Vector<float>
        {
            public PipelineColumn Input { get; }

            public OutColumn(Vector<float> input,
                int confidence,
                int pvalueHistoryLength)
                : base(new Reconciler(confidence, pvalueHistoryLength), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            private readonly int _confidence;
            private readonly int _pvalueHistoryLength;

            public Reconciler(
                int confidence,
                int pvalueHistoryLength)
            {
                _confidence = confidence;
                _pvalueHistoryLength = pvalueHistoryLength;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);
                var outCol = (OutColumn)toOutput[0];
                return new IidSpikeEstimator(env,
                    _confidence,
                    _pvalueHistoryLength,
                    inputNames[outCol.Input], outputNames[outCol]);
            }
        }
    }
}
