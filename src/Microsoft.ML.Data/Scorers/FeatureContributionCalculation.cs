// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Numeric;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;

[assembly: LoadableClass(typeof(IDataScorerTransform), typeof(FeatureContributionScorer), typeof(FeatureContributionScorer.Arguments),
    typeof(SignatureDataScorer), "Feature Contribution Scorer", "fcc", "wtf", "fct", "FeatureContributionCalculationScorer", AnnotationUtils.Const.ScoreColumnKind.FeatureContribution)]

[assembly: LoadableClass(typeof(ISchemaBindableMapper), typeof(FeatureContributionScorer), typeof(FeatureContributionScorer.Arguments),
    typeof(SignatureBindableMapper), "Feature Contribution Mapper", "fcc", "wtf", "fct", AnnotationUtils.Const.ScoreColumnKind.FeatureContribution)]

[assembly: LoadableClass(typeof(ISchemaBindableMapper), typeof(FeatureContributionScorer), null, typeof(SignatureLoadModel),
    "Feature Contribution Mapper", FeatureContributionScorer.MapperLoaderSignature)]

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Used only by the command line API for scoring and calculation of feature contribution.
    /// </summary>
    internal sealed class FeatureContributionScorer
    {
        // Apparently, loader signature is limited in length to 24 characters.
        internal const string MapperLoaderSignature = "WTFBindable";

        internal sealed class Arguments : ScorerArgumentsBase
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

        // Factory method for SignatureDataScorer.
        private static IDataScorerTransform Create(IHostEnvironment env, Arguments args, IDataView data, ISchemaBoundMapper mapper, RoleMappedSchema trainSchema)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(data, nameof(data));
            env.CheckValue(mapper, nameof(mapper));
            if (args.Top< 0)
                throw env.Except($"Number of top contribution must be non negative");
            if (args.Bottom < 0)
                throw env.Except($"Number of bottom contribution must be non negative");

            var contributionMapper = mapper as RowMapper;
            env.CheckParam(mapper != null, nameof(mapper), "Unexpected mapper");

            var scorer = ScoreUtils.GetScorerComponent(env, contributionMapper);
            var scoredPipe = scorer.CreateComponent(env, data, contributionMapper, trainSchema);
            return scoredPipe;
        }

        // Factory method for SignatureBindableMapper.
        private static ISchemaBindableMapper Create(IHostEnvironment env, Arguments args, IPredictor predictor)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(predictor, nameof(predictor));
            var pred = predictor as IFeatureContributionMapper;
            env.CheckParam(pred != null, nameof(predictor), "Predictor doesn't support getting feature contributions");
            return new BindableMapper(env, pred, args.Top, args.Bottom, args.Normalize, args.Stringify);
        }

        // Factory method for SignatureLoadModel.
        private static ISchemaBindableMapper Create(IHostEnvironment env, ModelLoadContext ctx)
            => new BindableMapper(env, ctx);

        /// <summary>
        /// Holds the definition of the getters for the FeatureContribution column. It also contains the generic mapper that is used to score the Predictor.
        /// This is only used by the command line API.
        /// </summary>
        private sealed class BindableMapper : ISchemaBindableMapper, ICanSaveModel, IPredictor
        {
            private readonly int _topContributionsCount;
            private readonly int _bottomContributionsCount;
            private readonly bool _normalize;
            private readonly IHostEnvironment _env;

            public readonly IFeatureContributionMapper Predictor;
            public readonly ISchemaBindableMapper GenericMapper;
            public readonly bool Stringify;

            private static VersionInfo GetVersionInfo()
            {
                return new VersionInfo(
                    modelSignature: "WTF SCBI",
                    verWrittenCur: 0x00010001, // Initial
                    verReadableCur: 0x00010001,
                    verWeCanReadBack: 0x00010001,
                    loaderSignature: MapperLoaderSignature,
                    loaderAssemblyName: typeof(FeatureContributionScorer).Assembly.FullName);
            }

            public PredictionKind PredictionKind => Predictor.PredictionKind;

            public BindableMapper(IHostEnvironment env, IFeatureContributionMapper predictor, int topContributionsCount, int bottomContributionsCount, bool normalize, bool stringify)
            {
                Contracts.CheckValue(env, nameof(env));
                _env = env;
                _env.CheckValue(predictor, nameof(predictor));
                if (topContributionsCount < 0)
                    throw env.Except($"Number of top contribution must be non negative");
                if (bottomContributionsCount < 0)
                    throw env.Except($"Number of bottom contribution must be non negative");

                _topContributionsCount = topContributionsCount;
                _bottomContributionsCount = bottomContributionsCount;
                _normalize = normalize;
                Stringify = stringify;
                Predictor = predictor;

                GenericMapper = ScoreUtils.GetSchemaBindableMapper(_env, Predictor, null);
            }

            public BindableMapper(IHostEnvironment env, ModelLoadContext ctx)
            {
                Contracts.CheckValue(env, nameof(env));
                _env = env;
                _env.CheckValue(ctx, nameof(ctx));
                ctx.CheckAtModel(GetVersionInfo());

                // *** Binary format ***
                // IFeatureContributionMapper: Predictor
                // int: topContributionsCount
                // int: bottomContributionsCount
                // bool: normalize
                // bool: stringify

                ctx.LoadModel<IFeatureContributionMapper, SignatureLoadModel>(env, out Predictor, ModelFileUtils.DirPredictor);
                GenericMapper = ScoreUtils.GetSchemaBindableMapper(_env, Predictor, null);
                _topContributionsCount = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(0 <= _topContributionsCount);
                _bottomContributionsCount = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(0 <= _bottomContributionsCount);
                _normalize = ctx.Reader.ReadBoolByte();
                Stringify = ctx.Reader.ReadBoolByte();
            }

            public ISchemaBoundMapper Bind(IHostEnvironment env, RoleMappedSchema schema)
            {
                Contracts.CheckValue(env, nameof(env));
                env.CheckValue(schema, nameof(schema));
                CheckSchemaValid(env, schema, Predictor);
                return new RowMapper(env, this, schema);
            }

            void ICanSaveModel.Save(ModelSaveContext ctx)
            {
                Contracts.CheckValue(ctx, nameof(ctx));
                ctx.SetVersionInfo(GetVersionInfo());

                // *** Binary format ***
                // IFeatureContributionMapper: Predictor
                // int: topContributionsCount
                // int: bottomContributionsCount
                // bool: normalize
                // bool: stringify

                ctx.SaveModel(Predictor, ModelFileUtils.DirPredictor);
                Contracts.Assert(0 <= _topContributionsCount);
                ctx.Writer.Write(_topContributionsCount);
                Contracts.Assert(0 <= _bottomContributionsCount);
                ctx.Writer.Write(_bottomContributionsCount);
                ctx.Writer.WriteBoolByte(_normalize);
                ctx.Writer.WriteBoolByte(Stringify);
            }

            public Delegate GetTextContributionGetter(DataViewRow input, int colSrc, VBuffer<ReadOnlyMemory<char>> slotNames)
            {
                Contracts.CheckValue(input, nameof(input));
                Contracts.Check(0 <= colSrc && colSrc < input.Schema.Count);
                var typeSrc = input.Schema[colSrc].Type;

                Func<DataViewRow, int, VBuffer<ReadOnlyMemory<char>>, ValueGetter<ReadOnlyMemory<char>>> del = GetTextValueGetter<int>;
                var meth = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(typeSrc.RawType);
                return (Delegate)meth.Invoke(this, new object[] { input, colSrc, slotNames });
            }

            public Delegate GetContributionGetter(DataViewRow input, int colSrc)
            {
                Contracts.CheckValue(input, nameof(input));
                Contracts.Check(0 <= colSrc && colSrc < input.Schema.Count);

                var typeSrc = input.Schema[colSrc].Type;
                Func<DataViewRow, int, ValueGetter<VBuffer<float>>> del = GetValueGetter<int>;

                // REVIEW: Assuming Feature contributions will be VBuffer<float>.
                // For multiclass LR it needs to be(VBuffer<float>[].
                var meth = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(typeSrc.RawType);
                return (Delegate)meth.Invoke(this, new object[] { input, colSrc });
            }

            private ReadOnlyMemory<char> GetSlotName(int index, VBuffer<ReadOnlyMemory<char>> slotNames)
            {
                _env.Assert(0 <= index && index < slotNames.Length);
                var slotName = slotNames.GetItemOrDefault(index);
                return slotName.IsEmpty
                    ? new ReadOnlyMemory<char>($"f{index}".ToCharArray())
                    : slotName;
            }

            private ValueGetter<ReadOnlyMemory<char>> GetTextValueGetter<TSrc>(DataViewRow input, int colSrc, VBuffer<ReadOnlyMemory<char>> slotNames)
            {
                Contracts.AssertValue(input);
                Contracts.AssertValue(Predictor);

                var featureGetter = input.GetGetter<TSrc>(input.Schema[colSrc]);
                var map = Predictor.GetFeatureContributionMapper<TSrc, VBuffer<float>>(_topContributionsCount, _bottomContributionsCount, _normalize);

                var features = default(TSrc);
                var contributions = default(VBuffer<float>);
                return
                    (ref ReadOnlyMemory<char> dst) =>
                    {
                        featureGetter(ref features);
                        map(in features, ref contributions);
                        var editor = VBufferEditor.CreateFromBuffer(ref contributions);
                        var indices = contributions.IsDense ? Utils.GetIdentityPermutation(contributions.Length) : editor.Indices;
                        var values = editor.Values;
                        var count = values.Length;
                        var sb = new StringBuilder();
                        GenericSpanSortHelper<float>.Sort(values, indices, 0, count);
                        for (var i = 0; i < count; i++)
                        {
                            var val = values[i];
                            var ind = indices[i];
                            var name = GetSlotName(ind, slotNames);
                            sb.AppendFormat("{0}: {1}, ", name, val);
                        }

                        if (sb.Length > 0)
                        {
                            _env.Assert(sb.Length >= 2);
                            sb.Remove(sb.Length - 2, 2);
                        }

                        dst = new ReadOnlyMemory<char>(sb.ToString().ToCharArray());
                    };
            }

            private ValueGetter<VBuffer<float>> GetValueGetter<TSrc>(DataViewRow input, int colSrc)
            {
                Contracts.AssertValue(input);
                Contracts.AssertValue(Predictor);

                var featureGetter = input.GetGetter<TSrc>(input.Schema[colSrc]);

                // REVIEW: Scorer can call Sparsification\Norm routine.

                var map = Predictor.GetFeatureContributionMapper<TSrc, VBuffer<float>>(_topContributionsCount, _bottomContributionsCount, _normalize);
                var features = default(TSrc);
                return
                    (ref VBuffer<float> dst) =>
                    {
                        featureGetter(ref features);
                        map(in features, ref dst);
                    };
            }

            private static void CheckSchemaValid(IExceptionContext ectx, RoleMappedSchema schema,
                IFeatureContributionMapper predictor)
            {
                Contracts.AssertValue(ectx);
                ectx.AssertValue(schema);
                ectx.AssertValue(predictor);

                // REVIEW: Check that Features column is present and is of correct size and item type.
            }
        }

        /// <summary>
        /// Maps a schema from input columns to output columns. Keeps track of the input columns that are needed for the mapping.
        /// </summary>
        private sealed class RowMapper : ISchemaBoundRowMapper
        {
            private readonly IHostEnvironment _env;
            private readonly ISchemaBoundRowMapper _genericRowMapper;
            private readonly BindableMapper _parent;
            private readonly DataViewSchema _outputSchema;
            private readonly DataViewSchema _outputGenericSchema;
            private VBuffer<ReadOnlyMemory<char>> _slotNames;

            public RoleMappedSchema InputRoleMappedSchema { get; }

            public DataViewSchema InputSchema => InputRoleMappedSchema.Schema;
            private DataViewSchema.Column FeatureColumn => InputRoleMappedSchema.Feature.Value;

            public DataViewSchema OutputSchema { get; }

            public ISchemaBindableMapper Bindable => _parent;

            public RowMapper(IHostEnvironment env, BindableMapper parent, RoleMappedSchema schema)
            {
                Contracts.AssertValue(env);
                _env = env;
                _env.AssertValue(schema);
                _env.AssertValue(parent);
                _env.Assert(schema.Feature.HasValue);
                _parent = parent;
                InputRoleMappedSchema = schema;
                var genericMapper = parent.GenericMapper.Bind(_env, schema);
                _genericRowMapper = genericMapper as ISchemaBoundRowMapper;
                var featureSize = FeatureColumn.Type.GetVectorSize();

                if (parent.Stringify)
                {
                    var builder = new DataViewSchema.Builder();
                    builder.AddColumn(DefaultColumnNames.FeatureContributions, TextDataViewType.Instance, null);
                    _outputSchema = builder.ToSchema();
                    if (FeatureColumn.HasSlotNames(featureSize))
                        FeatureColumn.Annotations.GetValue(AnnotationUtils.Kinds.SlotNames, ref _slotNames);
                    else
                        _slotNames = VBufferUtils.CreateEmpty<ReadOnlyMemory<char>>(featureSize);
                }
                else
                {
                    var metadataBuilder = new DataViewSchema.Annotations.Builder();
                    if (InputSchema[FeatureColumn.Index].HasSlotNames(featureSize))
                        metadataBuilder.AddSlotNames(featureSize, (ref VBuffer<ReadOnlyMemory<char>> value) =>
                            FeatureColumn.Annotations.GetValue(AnnotationUtils.Kinds.SlotNames, ref value));

                    var schemaBuilder = new DataViewSchema.Builder();
                    var featureContributionType = new VectorType(NumberDataViewType.Single, FeatureColumn.Type as VectorType);
                    schemaBuilder.AddColumn(DefaultColumnNames.FeatureContributions, featureContributionType, metadataBuilder.ToAnnotations());
                    _outputSchema = schemaBuilder.ToSchema();
                }

                _outputGenericSchema = _genericRowMapper.OutputSchema;
                OutputSchema = new ZipBinding(new DataViewSchema[] { _outputGenericSchema, _outputSchema, }).OutputSchema;
            }

            /// <summary>
            /// Returns the input columns needed for the requested output columns.
            /// </summary>
            IEnumerable<DataViewSchema.Column> ISchemaBoundRowMapper.GetDependenciesForNewColumns(IEnumerable<DataViewSchema.Column> dependingColumns)
            {
                if (dependingColumns.Count() == 0)
                    return Enumerable.Empty<DataViewSchema.Column>();

                return Enumerable.Repeat(FeatureColumn, 1);
            }

            DataViewRow ISchemaBoundRowMapper.GetRow(DataViewRow input, IEnumerable<DataViewSchema.Column> activeColumns)
            {
                Contracts.AssertValue(input);
                Contracts.AssertValue(activeColumns);
                var totalColumnsCount = 1 + _outputGenericSchema.Count;
                var getters = new Delegate[totalColumnsCount];

                if (activeColumns.Select(c => c.Index).Contains(_outputGenericSchema.Count))
                {
                    getters[totalColumnsCount - 1] = _parent.Stringify
                        ? _parent.GetTextContributionGetter(input, FeatureColumn.Index, _slotNames)
                        : _parent.GetContributionGetter(input, FeatureColumn.Index);
                }

                var genericRow = _genericRowMapper.GetRow(input, activeColumns);
                for (var i = 0; i < _outputGenericSchema.Count; i++)
                {
                    if (genericRow.IsColumnActive(genericRow.Schema[i]))
                        getters[i] = RowCursorUtils.GetGetterAsDelegate(genericRow, i);
                }

                return new SimpleRow(OutputSchema, genericRow, getters);
            }

            public IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> GetInputColumnRoles()
            {
                yield return RoleMappedSchema.ColumnRole.Feature.Bind(FeatureColumn.Name);
            }
        }
    }
}
