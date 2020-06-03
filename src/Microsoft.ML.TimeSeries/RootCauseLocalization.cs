﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.TimeSeries;
using Microsoft.ML.Transforms.TimeSeries;

[assembly: LoadableClass(typeof(RootCauseLocalizationTransformer), null, typeof(SignatureLoadModel),
    RootCauseLocalizationTransformer.UserName, RootCauseLocalizationTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(RootCauseLocalizationTransformer), null, typeof(SignatureLoadRowMapper),
    RootCauseLocalizationTransformer.UserName, RootCauseLocalizationTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms.TimeSeries
{
    /// <summary>
    /// <see cref="ITransformer"/> resulting from fitting an <see cref="RootCauseLocalizationTransformer"/>.
    /// </summary>
    public sealed class RootCauseLocalizationTransformer : OneToOneTransformerBase
    {
        internal const string Summary = "Localize root cause for anomaly.";
        internal const string UserName = "Root Cause Localization Transform";
        internal const string LoaderSignature = "RootCauseTransform";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "ROOTCAUS",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(RootCauseLocalizationTransformer).Assembly.FullName);
        }

        private const string RegistrationName = "RootCauseLocalization";

        internal sealed class Column : OneToOneColumn
        {
            internal static Column Parse(string str)
            {
                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            internal bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                return TryUnparseCore(sb);
            }
        }

        internal class Options : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "The name of the source column.", ShortName = "src", SortOrder = 1, Purpose = SpecialPurpose.ColumnName)]
            public string Source;

            [Argument(ArgumentType.Required, HelpText = "The name of the output column.", SortOrder = 2)]
            public string Output;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Weight for getting the score for the root cause item.", ShortName = "Beta", SortOrder = 2)]
            public double Beta = RootCauseLocalizationEstimator.Defaults.Beta;

        }

        /// <summary>
        /// The input and output column pairs passed to this <see cref="ITransformer"/>.
        /// </summary>
        internal IReadOnlyCollection<(string outputColumnName, string inputColumnName)> Columns => ColumnPairs.AsReadOnly();

        private readonly double _beta;

        /// <summary>
        /// Localization root cause for multi-dimensional anomaly.
        /// </summary>
        /// <param name="env">The estimator's local <see cref="IHostEnvironment"/>.</param>
        /// <param name="beta">Weight for generating score.</param>
        /// <param name="columns">The name of the columns (first item of the tuple), and the name of the resulting output column (second item of the tuple).</param>

        internal RootCauseLocalizationTransformer(IHostEnvironment env, double beta = RootCauseLocalizationEstimator.Defaults.Beta, params (string outputColumnName, string inputColumnName)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), columns)
        {
            Host.CheckUserArg(beta >= 0 && beta <= 1, nameof(Options.Beta), "Must be in [0,1]");

            _beta = beta;
        }

        // Factory method for SignatureLoadModel.
        private static RootCauseLocalizationTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);
            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new RootCauseLocalizationTransformer(host, ctx);
        }

        private RootCauseLocalizationTransformer(IHost host, ModelLoadContext ctx)
            : base(host, ctx)
        {
            // *** Binary format ***
            // <base>
            // double: beta
            _beta = ctx.Reader.ReadDouble();
            Host.CheckDecode(_beta >= 0 && _beta <= 1);
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));

            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>
            base.SaveColumns(ctx);
            // double: beta
            ctx.Writer.Write(_beta);
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private protected override void CheckInputColumn(DataViewSchema inputSchema, int col, int srcCol)
        {
            if (!(inputSchema[srcCol].Type is RootCauseLocalizationInputDataViewType))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[col].inputColumnName, "RootCauseLocalizationInputDataViewType", inputSchema[srcCol].Type.ToString());
        }

        private sealed class Mapper : OneToOneMapperBase
        {
            private readonly RootCauseLocalizationTransformer _parent;

            public Mapper(RootCauseLocalizationTransformer parent, DataViewSchema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new DataViewSchema.DetachedColumn[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    InputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].inputColumnName, out int colIndex);
                    Host.Assert(colIndex >= 0);

                    DataViewType type;
                    type = new RootCauseDataViewType();

                    result[i] = new DataViewSchema.DetachedColumn(_parent.ColumnPairs[i].outputColumnName, type, null);
                }
                return result;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);

                var src = default(RootCauseLocalizationInput);
                var getSrc = input.GetGetter<RootCauseLocalizationInput>(input.Schema[ColMapNewToOld[iinfo]]);

                disposer = null;

                ValueGetter<RootCause> del =
                    (ref RootCause dst) =>
                    {
                        getSrc(ref src);
                        if (src == null)
                            return;

                        CheckRootCauseInput(src, Host);

                        LocalizeRootCauses(src, ref dst);
                    };

                return del;
            }

            private void CheckRootCauseInput(RootCauseLocalizationInput src, IHost host)
            {
                if (src.Slices.Count < 1)
                {
                    throw host.Except($"Length of Slices must be larger than 0");
                }

                bool containsAnomalyTimestamp = false;
                foreach (MetricSlice slice in src.Slices)
                {
                    if (slice.TimeStamp.Equals(src.AnomalyTimestamp))
                    {
                        containsAnomalyTimestamp = true;
                    }
                }
                if (!containsAnomalyTimestamp)
                {
                    throw host.Except($"Has no points in the given anomaly timestamp");
                }
            }

            private void LocalizeRootCauses(RootCauseLocalizationInput src, ref RootCause dst)
            {
                RootCauseAnalyzer analyzer = new RootCauseAnalyzer(src, _parent._beta);
                dst = analyzer.Analyze();
            }
        }
    }

    /// <summary>
    /// <see cref="IEstimator{TTransformer}"/> for the <see cref="RootCauseLocalizationTransformer"/>.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    /// To create this estimator, use
    /// [LocalizeRootCause](xref:Microsoft.ML.TimeSeriesCatalog.LocalizeRootCause(Microsoft.ML.TransformsCatalog,System.String,System.String,System.Double))
    /// ###  Estimator Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Does this estimator need to look at the data to train its parameters? | No |
    /// | Input column data type | <xref: Microsoft.ML.TimeSeries.RootCauseLocalizationInput> |
    /// | Output column data type | <xref:Microsoft.ML.TimeSeries.RootCause> |
    /// | Exportable to ONNX | No |
    ///
    /// [!include[io](~/../docs/samples/docs/api-reference/time-series-root-cause-localization.md)]
    ///
    /// The resulting <xref:Microsoft.ML.Transforms.TimeSeries.RootCauseLocalizationTransformer> creates a new column, named as specified in the output column name parameters, and
    /// localize the root causes which contribute most to the anomaly.
    /// Check the See Also section for links to usage examples.
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="TimeSeriesCatalog.LocalizeRootCause" />
    public sealed class RootCauseLocalizationEstimator : TrivialEstimator<RootCauseLocalizationTransformer>
    {
        internal static class Defaults
        {
            public const double Beta = 0.5;
        }

        /// <summary>
        /// Localize root cause.
        /// </summary>
        /// <param name="env">The estimator's local <see cref="IHostEnvironment"/>.</param>
        /// <param name="outputColumnName">Name of output column to run the root cause localization.</param>
        /// <param name="inputColumnName">Name of input column to run the root cause localization.</param>
        /// <param name="beta">The weight for generating score in output result.</param>
        [BestFriend]
        internal RootCauseLocalizationEstimator(IHostEnvironment env, string outputColumnName, string inputColumnName, double beta = Defaults.Beta)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(RootCauseLocalizationEstimator)), new RootCauseLocalizationTransformer(env, beta, new[] { (outputColumnName, inputColumnName ?? outputColumnName) }))
        {
        }

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            foreach (var colInfo in Transformer.Columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.inputColumnName, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.inputColumnName);
                if (!(col.ItemType is RootCauseLocalizationInputDataViewType) || col.Kind != SchemaShape.Column.VectorKind.Scalar)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.inputColumnName, new RootCauseLocalizationInputDataViewType().ToString(), col.GetTypeString());

                result[colInfo.outputColumnName] = new SchemaShape.Column(colInfo.outputColumnName, SchemaShape.Column.VectorKind.Scalar, new RootCauseDataViewType(), false);
            }

            return new SchemaShape(result.Values);
        }
    }
}
