// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Numeric;

[assembly: LoadableClass(FeatureContributionCalculatingTransformer.Summary, typeof(FeatureContributionCalculatingTransformer), null, typeof(SignatureLoadModel),
    FeatureContributionCalculatingTransformer.FriendlyName, FeatureContributionCalculatingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(FeatureContributionCalculatingTransformer), null, typeof(SignatureLoadRowMapper),
   FeatureContributionCalculatingTransformer.FriendlyName, FeatureContributionCalculatingTransformer.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// Feature Contribution Calculation Transform.
    /// </summary>
    /// <remarks>
    /// The Feature Contribution Calculation Transform scores the model on an input dataset and
    /// computes model-specific contribution scores for each feature. See the sample below for
    /// an example of how to compute feature importance using the Feature Contribution Calculation Transform.
    /// </remarks>
    /// <example>
    /// <format type="text/markdown">
    /// <![CDATA[
    /// [!code-csharp[FCT](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/FeatureContributionCalculationTransform.cs)]
    /// ]]>
    /// </format>
    /// </example>
    public sealed class FeatureContributionCalculatingTransformer : RowToRowTransformerBase
    {
        public sealed class Arguments : ScorerArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of top contributions", SortOrder = 1)]
            public int Top = 10;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of bottom contributions", SortOrder = 2)]
            public int Bottom = 10;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether or not output of Features contribution should be normalized", ShortName = "norm", SortOrder = 3)]
            public bool Normalize = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether or not output of Features contribution in string key-value format", ShortName = "str", SortOrder = 4)]
            public bool Stringify = false;

            // REVIEW: the scorer currently ignores the 'suffix' argument from the base class. It should respect it.
        }

        // Apparently, loader signature is limited in length to 24 characters.
        internal const string Summary = "For each data point, calculates the contribution of individual features to the model prediction.";
        internal const string FriendlyName = "Feature Contribution Transform";
        internal const string LoaderSignature = "FeatureContribution";

        private const int MaxTopBottom = 1000;

        private readonly string _features;
        private readonly FeatureContributionCalculationTransform.BindableMapper _mapper;

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "FCC TRAN",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(FeatureContributionCalculatingTransformer).Assembly.FullName);
        }

        public FeatureContributionCalculatingTransformer(IHostEnvironment env, IPredictor predictor, string featuresColumn, Arguments args)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(FeatureContributionCalculatingTransformer)))
        {
            Host.CheckValue(args, nameof(args));
            Host.CheckValue(predictor, nameof(predictor));
            if (args.Top <= 0 || args.Top > MaxTopBottom)
                throw Host.Except($"Number of top contribution must be in range (0,{MaxTopBottom}]");
            if (args.Bottom <= 0 || args.Bottom > MaxTopBottom)
                throw Host.Except($"Number of bottom contribution must be in range (0,{MaxTopBottom}]");

            var pred = predictor as IFeatureContributionMapper;
            Host.CheckParam(pred != null, nameof(predictor), "Predictor doesn't support getting feature contributions");

            // TODO check that the featues column is not empty.
            _features = featuresColumn;
            _mapper = new FeatureContributionCalculationTransform.BindableMapper(Host, pred, args.Top, args.Bottom, args.Normalize, args.Stringify);
        }

        // Factory method for SignatureLoadModel
        public FeatureContributionCalculatingTransformer(IHostEnvironment env, ModelLoadContext ctx)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(FeatureContributionCalculatingTransformer)))
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // string features
            // BindableMapper mapper

            _features = ctx.LoadNonEmptyString();
            _mapper = new FeatureContributionCalculationTransform.BindableMapper(env, ctx);
        }

        // Factory method for SignatureLoadRowMapper.
        internal static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
            => new FeatureContributionCalculatingTransformer(env, ctx).MakeRowMapper(Schema.Create(inputSchema));

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // string features
            // BindableMapper mapper

            ctx.SaveNonEmptyString(_features);
            _mapper.Save(ctx);
        }

        protected override IRowMapper MakeRowMapper(Schema schema)
            => new Mapper(this, schema);

        private class Mapper : MapperBase
        {
            private readonly FeatureContributionCalculatingTransformer _parent;
            private readonly FeatureContributionCalculationTransform.BindableMapper _bindableMapper;
            private readonly RoleMappedSchema _roleMappedSchema;
            private readonly ISchemaBoundRowMapper _genericRowMapper;
            private readonly Schema _outputGenericSchema;
            private readonly VBuffer<ReadOnlyMemory<char>> _slotNames;

            public Mapper(FeatureContributionCalculatingTransformer parent, Schema schema)
                : base(parent.Host, schema)
            {
                // TODO some checks? get soem of the columns, initialize some stuff
                _parent = parent;
                _bindableMapper = _parent._mapper;

                var roles = new List<KeyValuePair<RoleMappedSchema.ColumnRole, string>>();
                roles.Add(new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RoleMappedSchema.ColumnRole.Feature, _parent._features));
                _roleMappedSchema = new RoleMappedSchema(InputSchema, roles);

                var genericMapper = _bindableMapper.GenericMapper.Bind(Host, _roleMappedSchema);
                _genericRowMapper = genericMapper as ISchemaBoundRowMapper;
                _outputGenericSchema = _genericRowMapper.OutputSchema;

                if (InputSchema.HasSlotNames(_roleMappedSchema.Feature.Index, _roleMappedSchema.Feature.Type.VectorSize))
                    InputSchema.GetMetadata(MetadataUtils.Kinds.SlotNames, _roleMappedSchema.Feature.Index,
                        ref _slotNames);
                else
                    _slotNames = VBufferUtils.CreateEmpty<ReadOnlyMemory<char>>(_roleMappedSchema.Feature.Type.VectorSize);
            }

        /// <summary>
        /// Returns the input columns needed for the requested output columns.
        /// </summary>
        public override Func<int, bool> GetDependencies(Func<int, bool> activeOutput)
            {
                var active = new bool[InputSchema.ColumnCount];
                InputSchema.TryGetColumnIndex(_parent._features, out int featureCol);
                active[featureCol] = true;
                return col => active[col];
            }

            public override void Save(ModelSaveContext ctx)
                => _parent.Save(ctx);

            // The FeatureContributionCalculatingTransformer produces two columns: Score and FeatureContribution.
            // If the argument stringify is true, the type of the FeatureContribution column is string, otherwise it is a vector of float.
            protected override Schema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new List<Schema.DetachedColumn>();

                // Add Score Column.
                foreach (var pair in _outputGenericSchema.GetColumns())
                    result.Add(new Schema.DetachedColumn(pair.column));

                // Add FeatureContributions column.
                var builder = new MetadataBuilder();
                builder.Add(InputSchema[_roleMappedSchema.Feature.Index].Metadata, x => x == MetadataUtils.Kinds.SlotNames);
                if (_bindableMapper.Stringify)
                    result.Add(new Schema.DetachedColumn(DefaultColumnNames.FeatureContributions, TextType.Instance, builder.GetMetadata()));
                else
                    result.Add(new Schema.DetachedColumn(DefaultColumnNames.FeatureContributions, new VectorType(NumberType.R4, _roleMappedSchema.Feature.Type.AsVector), builder.GetMetadata()));

                return result.ToArray();
            }

            protected Delegate GetScoreGetter(Row input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                var genericRow = _genericRowMapper.GetRow(input, col => activeOutput(col), out disposer);
                return RowCursorUtils.GetGetterAsDelegate(genericRow, iinfo);
            }

            protected Delegate GetFeatureContributionGetter(Row input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;

                if (activeOutput(iinfo))
                {
                    return _bindableMapper.Stringify
                        ? _bindableMapper.GetTextContributionGetter(input, _roleMappedSchema.Feature.Index, _slotNames)
                        : _bindableMapper.GetContributionGetter(input, _roleMappedSchema.Feature.Index);
                }
                return null;
            }

            protected override Delegate MakeGetter(Row input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                if (iinfo < _outputGenericSchema.ColumnCount)
                    return GetScoreGetter(input, iinfo, activeOutput, out disposer);
                else
                    return GetFeatureContributionGetter(input, iinfo, activeOutput, out disposer);
            }
        }
    }

    public sealed class FeatureContributionCalculatingEstimator : TrivialEstimator<FeatureContributionCalculatingTransformer>
    {
        private readonly FeatureContributionCalculatingTransformer.Arguments _args;
        private readonly string _features;
        private readonly IPredictor _predictor;

        // TODO comments
        public FeatureContributionCalculatingEstimator(IHostEnvironment env, IPredictor predictor, string featuresColumn, FeatureContributionCalculatingTransformer.Arguments args)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(FeatureContributionCalculatingTransformer)), new FeatureContributionCalculatingTransformer(env, predictor, featuresColumn, args))
        {
            // TODO argcheck?
            _args = args;
            _features = featuresColumn;
            _predictor = predictor;
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.Columns.ToList();

            if (!inputSchema.TryFindColumn(_features, out var col))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", _features);
            var metadata = new List<SchemaShape.Column>();
            if (col.Metadata.TryFindColumn(MetadataUtils.Kinds.SlotNames, out var slotMeta))
                metadata.Add(slotMeta);
            // TODO: check type of feature column.

            // TODO: How do we deal with multiclassScoreColumn? should also contain slotnames
            // Add Score column.
            result.Add(new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata())));

            // Add FeatureContributions column.
            if (_args.Stringify)
            {
                result.Add(new SchemaShape.Column(DefaultColumnNames.FeatureContributions, col.Kind,
                    TextType.Instance, false, new SchemaShape(metadata.ToArray())));
            }
            else
            {
                result.Add(new SchemaShape.Column(DefaultColumnNames.FeatureContributions, col.Kind,
                    col.ItemType, false, new SchemaShape(metadata.ToArray())));
            }

            return new SchemaShape(result.ToArray());
        }
    }
}
