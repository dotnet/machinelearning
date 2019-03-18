// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.TimeSeries;

[assembly: LoadableClass(IidChangePointDetector.Summary, typeof(IDataTransform), typeof(IidChangePointDetector), typeof(IidChangePointDetector.Options), typeof(SignatureDataTransform),
    IidChangePointDetector.UserName, IidChangePointDetector.LoaderSignature, IidChangePointDetector.ShortName)]

[assembly: LoadableClass(IidChangePointDetector.Summary, typeof(IDataTransform), typeof(IidChangePointDetector), null, typeof(SignatureLoadDataTransform),
    IidChangePointDetector.UserName, IidChangePointDetector.LoaderSignature)]

[assembly: LoadableClass(IidChangePointDetector.Summary, typeof(IidChangePointDetector), null, typeof(SignatureLoadModel),
    IidChangePointDetector.UserName, IidChangePointDetector.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(IidChangePointDetector), null, typeof(SignatureLoadRowMapper),
   IidChangePointDetector.UserName, IidChangePointDetector.LoaderSignature)]

namespace Microsoft.ML.Transforms.TimeSeries
{
    /// <summary>
    /// <see cref="ITransformer"/> produced by fitting the <see cref="IDataView"/> to an <see cref="IidChangePointEstimator" />.
    /// </summary>
    public sealed class IidChangePointDetector : IidAnomalyDetectionBaseWrapper, IStatefulTransformer
    {
        internal const string Summary = "This transform detects the change-points in an i.i.d. sequence using adaptive kernel density estimation and martingales.";
        internal const string LoaderSignature = "IidChangePointDetector";
        internal const string UserName = "IID Change Point Detection";
        internal const string ShortName = "ichgpnt";

        internal sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "The name of the source column.", ShortName = "src",
                SortOrder = 1, Purpose = SpecialPurpose.ColumnName)]
            public string Source;

            [Argument(ArgumentType.Required, HelpText = "The name of the new column.",
                SortOrder = 2)]
            public string Name;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The length of the sliding window on p-values for computing the martingale score.", ShortName = "wnd",
                SortOrder = 102)]
            public int ChangeHistoryLength = 20;

            [Argument(ArgumentType.Required, HelpText = "The confidence for change point detection in the range [0, 100].",
                ShortName = "cnf", SortOrder = 3)]
            public double Confidence = 95;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The martingale used for scoring.", ShortName = "mart", SortOrder = 103)]
            public MartingaleType Martingale = MartingaleType.Power;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The epsilon parameter for the Power martingale.",
                ShortName = "eps", SortOrder = 104)]
            public double PowerMartingaleEpsilon = 0.1;
        }

        private sealed class BaseArguments : ArgumentsBase
        {
            public BaseArguments(Options options)
            {
                Source = options.Source;
                Name = options.Name;
                Side = AnomalySide.TwoSided;
                WindowSize = options.ChangeHistoryLength;
                Martingale = options.Martingale;
                PowerMartingaleEpsilon = options.PowerMartingaleEpsilon;
                AlertOn = AlertingScore.MartingaleScore;
            }

            public BaseArguments(IidChangePointDetector transform)
            {
                Source = transform.InternalTransform.InputColumnName;
                Name = transform.InternalTransform.OutputColumnName;
                Side = AnomalySide.TwoSided;
                WindowSize = transform.InternalTransform.WindowSize;
                Martingale = transform.InternalTransform.Martingale;
                PowerMartingaleEpsilon = transform.InternalTransform.PowerMartingaleEpsilon;
                AlertOn = AlertingScore.MartingaleScore;
                AlertThreshold = transform.InternalTransform.AlertThreshold;
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
        private static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));

            return new IidChangePointDetector(env, options).MakeDataTransform(input);
        }

        IStatefulTransformer IStatefulTransformer.Clone()
        {
            var clone = (IidChangePointDetector)MemberwiseClone();
            clone.InternalTransform.StateRef = (IidAnomalyDetectionBase.State)clone.InternalTransform.StateRef.Clone();
            clone.InternalTransform.StateRef.InitState(clone.InternalTransform, InternalTransform.Host);
            return clone;
        }

        internal IidChangePointDetector(IHostEnvironment env, Options options)
            : base(new BaseArguments(options), LoaderSignature, env)
        {
            switch (InternalTransform.Martingale)
            {
                case MartingaleType.None:
                    InternalTransform.AlertThreshold = Double.MaxValue;
                    break;
                case MartingaleType.Power:
                    InternalTransform.AlertThreshold = Math.Exp(InternalTransform.WindowSize * InternalTransform.LogPowerMartigaleBettingFunc(1 - options.Confidence / 100, InternalTransform.PowerMartingaleEpsilon));
                    break;
                case MartingaleType.Mixture:
                    InternalTransform.AlertThreshold = Math.Exp(InternalTransform.WindowSize * InternalTransform.LogMixtureMartigaleBettingFunc(1 - options.Confidence / 100));
                    break;
                default:
                    throw InternalTransform.Host.ExceptParam(nameof(options.Martingale),
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

            InternalTransform.Host.CheckDecode(InternalTransform.ThresholdScore == AlertingScore.MartingaleScore);
            InternalTransform.Host.CheckDecode(InternalTransform.Side == AnomalySide.TwoSided);
        }

        private IidChangePointDetector(IHostEnvironment env, IidChangePointDetector transform)
            : base(new BaseArguments(transform), LoaderSignature, env)
        {
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            InternalTransform.Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            InternalTransform.Host.Assert(InternalTransform.ThresholdScore == AlertingScore.MartingaleScore);
            InternalTransform.Host.Assert(InternalTransform.Side == AnomalySide.TwoSided);

            // *** Binary format ***
            // <base>

            base.SaveModel(ctx);
        }

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);
    }

    /// <summary>
    /// The <see cref="IEstimator{ITransformer}"/> for detecting a signal change on an
    /// <a href="https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables"> independent identically distributed (i.i.d.)</a> time series.
    /// Detection is based on adaptive kernel density estimation and martingales.
    /// </summary>
    public sealed class IidChangePointEstimator : TrivialEstimator<IidChangePointDetector>
    {
        /// <summary>
        /// Create a new instance of <see cref="IidChangePointEstimator"/>
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// Column is a vector of type double and size 4. The vector contains Alert, Raw Score, P-Value and Martingale score as first four values.</param>
        /// <param name="confidence">The confidence for change point detection in the range [0, 100].</param>
        /// <param name="changeHistoryLength">The length of the sliding window on p-values for computing the martingale score.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="martingale">The martingale used for scoring.</param>
        /// <param name="eps">The epsilon parameter for the Power martingale.</param>
        internal IidChangePointEstimator(IHostEnvironment env, string outputColumnName, int confidence,
            int changeHistoryLength, string inputColumnName, MartingaleType martingale = MartingaleType.Power, double eps = 0.1)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(IidChangePointEstimator)),
                new IidChangePointDetector(env, new IidChangePointDetector.Options
                {
                    Name = outputColumnName,
                    Source = inputColumnName ?? outputColumnName,
                    Confidence = confidence,
                    ChangeHistoryLength = changeHistoryLength,
                    Martingale = martingale,
                    PowerMartingaleEpsilon = eps
                }))
        {
        }

        internal IidChangePointEstimator(IHostEnvironment env, IidChangePointDetector.Options options)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(IidChangePointEstimator)),
                new IidChangePointDetector(env, options))
        {
        }

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
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
