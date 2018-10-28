// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.TimeSeriesProcessing;
using static Microsoft.ML.Runtime.TimeSeriesProcessing.SequentialAnomalyDetectionTransformBase<System.Single, Microsoft.ML.Runtime.TimeSeriesProcessing.IidAnomalyDetectionBase.State>;

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
        private static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
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
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);
    }

    /// <summary>
    /// Estimator for <see cref="IidSpikeDetector"/>
    /// </summary>
    public sealed class IidSpikeEstimator : TrivialEstimator<IidSpikeDetector>
    {
        /// <summary>
        /// Create a new instance of <see cref="IidSpikeEstimator"/>
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="inputColumn">Name of the input column.</param>
        /// <param name="outputColumn">The name of the new column.</param>
        /// <param name="confidence">The confidence for spike detection in the range [0, 100].</param>
        /// <param name="pvalueHistoryLength">The size of the sliding window for computing the p-value.</param>
        /// <param name="side">The argument that determines whether to detect positive or negative anomalies, or both.</param>
        public IidSpikeEstimator(IHostEnvironment env, string inputColumn, string outputColumn, int confidence, int pvalueHistoryLength, AnomalySide side = AnomalySide.TwoSided)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(IidSpikeDetector)),
                new IidSpikeDetector(env, new IidSpikeDetector.Arguments
                {
                    Name = outputColumn,
                    Source = inputColumn,
                    Confidence = confidence,
                    PvalueHistoryLength = pvalueHistoryLength,
                    Side = side
                }))
        {
        }

        public IidSpikeEstimator(IHostEnvironment env, IidSpikeDetector.Arguments args)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(IidSpikeEstimator)),
        new IidSpikeDetector(env, args))
        {
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            if (!inputSchema.TryFindColumn(Transformer.InputColumnName, out var col))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", Transformer.InputColumnName);
            if (col.ItemType != NumberType.R4)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", Transformer.InputColumnName, NumberType.R4.ToString(), col.GetTypeString());

            var metadata = new List<SchemaShape.Column>() {
                new SchemaShape.Column(MetadataUtils.Kinds.SlotNames, SchemaShape.Column.VectorKind.Vector, TextType.Instance, false)
            };
            var resultDic = inputSchema.Columns.ToDictionary(x => x.Name);
            resultDic[Transformer.OutputColumnName] = new SchemaShape.Column(
                Transformer.OutputColumnName, SchemaShape.Column.VectorKind.Vector, NumberType.R8, false, new SchemaShape(metadata));

            return new SchemaShape(resultDic.Values);
        }
    }
}
