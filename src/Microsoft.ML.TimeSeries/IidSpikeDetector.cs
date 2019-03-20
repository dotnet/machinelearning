// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.TimeSeries;

[assembly: LoadableClass(IidSpikeDetector.Summary, typeof(IDataTransform), typeof(IidSpikeDetector), typeof(IidSpikeDetector.Options), typeof(SignatureDataTransform),
    IidSpikeDetector.UserName, IidSpikeDetector.LoaderSignature, IidSpikeDetector.ShortName)]

[assembly: LoadableClass(IidSpikeDetector.Summary, typeof(IDataTransform), typeof(IidSpikeDetector), null, typeof(SignatureLoadDataTransform),
    IidSpikeDetector.UserName, IidSpikeDetector.LoaderSignature)]

[assembly: LoadableClass(IidSpikeDetector.Summary, typeof(IidSpikeDetector), null, typeof(SignatureLoadModel),
    IidSpikeDetector.UserName, IidSpikeDetector.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(IidSpikeDetector), null, typeof(SignatureLoadRowMapper),
   IidSpikeDetector.UserName, IidSpikeDetector.LoaderSignature)]

namespace Microsoft.ML.Transforms.TimeSeries
{
    /// <summary>
    /// <see cref="ITransformer"/> produced by fitting the <see cref="IDataView"/> to an <see cref="IidSpikeEstimator" />.
    /// </summary>
    public sealed class IidSpikeDetector : IidAnomalyDetectionBaseWrapper, IStatefulTransformer
    {
        internal const string Summary = "This transform detects the spikes in a i.i.d. sequence using adaptive kernel density estimation.";
        internal const string LoaderSignature = "IidSpikeDetector";
        internal const string UserName = "IID Spike Detection";
        internal const string ShortName = "ispike";

        internal sealed class Options : TransformInputBase
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
            public BaseArguments(Options options)
            {
                Source = options.Source;
                Name = options.Name;
                Side = options.Side;
                WindowSize = options.PvalueHistoryLength;
                AlertThreshold = 1 - options.Confidence / 100;
                AlertOn = AlertingScore.PValueScore;
                Martingale = MartingaleType.None;
            }

            public BaseArguments(IidSpikeDetector transform)
            {
                Source = transform.InternalTransform.InputColumnName;
                Name = transform.InternalTransform.OutputColumnName;
                Side = transform.InternalTransform.Side;
                WindowSize = transform.InternalTransform.WindowSize;
                AlertThreshold = transform.InternalTransform.AlertThreshold;
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
        private static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));

            return new IidSpikeDetector(env, options).MakeDataTransform(input);
        }

        IStatefulTransformer IStatefulTransformer.Clone()
        {
            var clone = (IidSpikeDetector)MemberwiseClone();
            clone.InternalTransform.StateRef = (IidAnomalyDetectionBase.State)clone.InternalTransform.StateRef.Clone();
            clone.InternalTransform.StateRef.InitState(clone.InternalTransform, InternalTransform.Host);
            return clone;
        }

        internal IidSpikeDetector(IHostEnvironment env, Options options)
            : base(new BaseArguments(options), LoaderSignature, env)
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

        internal IidSpikeDetector(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, ctx, LoaderSignature)
        {
            // *** Binary format ***
            // <base>

            InternalTransform.Host.CheckDecode(InternalTransform.ThresholdScore == AlertingScore.PValueScore);
        }

        private IidSpikeDetector(IHostEnvironment env, IidSpikeDetector transform)
           : base(new BaseArguments(transform), LoaderSignature, env)
        {
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            InternalTransform.Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            InternalTransform.Host.Assert(InternalTransform.ThresholdScore == AlertingScore.PValueScore);

            // *** Binary format ***
            // <base>

            base.SaveModel(ctx);
        }

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);
    }

    /// <summary>
    /// The <see cref="IEstimator{ITransformer}"/> for detecting a signal spike on an
    /// <a href="https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables"> independent identically distributed (i.i.d.)</a> time series.
    /// Detection is based on adaptive kernel density estimation.
    /// </summary>
    public sealed class IidSpikeEstimator : TrivialEstimator<IidSpikeDetector>
    {
        /// <summary>
        /// Create a new instance of <see cref="IidSpikeEstimator"/>
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// Column is a vector of type double and size 4. The vector contains Alert, Raw Score, P-Value as first three values.</param>
        /// <param name="confidence">The confidence for spike detection in the range [0, 100].</param>
        /// <param name="pvalueHistoryLength">The size of the sliding window for computing the p-value.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="side">The argument that determines whether to detect positive or negative anomalies, or both.</param>
        internal IidSpikeEstimator(IHostEnvironment env, string outputColumnName, int confidence, int pvalueHistoryLength, string inputColumnName, AnomalySide side = AnomalySide.TwoSided)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(IidSpikeDetector)),
                new IidSpikeDetector(env, new IidSpikeDetector.Options
                {
                    Name = outputColumnName,
                    Source = inputColumnName,
                    Confidence = confidence,
                    PvalueHistoryLength = pvalueHistoryLength,
                    Side = side
                }))
        {
        }

        internal IidSpikeEstimator(IHostEnvironment env, IidSpikeDetector.Options options)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(IidSpikeEstimator)), new IidSpikeDetector(env, options))
        {
        }

        /// <summary>
        /// Schema propagation for transformers.
        /// Returns the output schema of the data, if the input schema is like the one provided.
        /// </summary>
        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            if (!inputSchema.TryFindColumn(Transformer.InternalTransform.InputColumnName, out var col))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", Transformer.InternalTransform.InputColumnName);
            if (col.ItemType != NumberDataViewType.Single)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", Transformer.InternalTransform.InputColumnName, "float", col.GetTypeString());

            var metadata = new List<SchemaShape.Column>() {
                new SchemaShape.Column(AnnotationUtils.Kinds.SlotNames, SchemaShape.Column.VectorKind.Vector, TextDataViewType.Instance, false)
            };
            var resultDic = inputSchema.ToDictionary(x => x.Name);
            resultDic[Transformer.InternalTransform.OutputColumnName] = new SchemaShape.Column(
                Transformer.InternalTransform.OutputColumnName, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Double, false, new SchemaShape(metadata));

            return new SchemaShape(resultDic.Values);
        }
    }
}
