// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Reflection;
using System.Text;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Numeric;

[assembly: LoadableClass(typeof(IDataScorerTransform), typeof(FeatureContributionCalculationTransform), typeof(FeatureContributionCalculationTransform.Arguments),
    typeof(SignatureDataScorer), "Feature Contribution Transform", "fct", "FeatureContributionCalculationTransform", MetadataUtils.Const.ScoreColumnKind.FeatureContribution)]

[assembly: LoadableClass(typeof(ISchemaBindableMapper), typeof(FeatureContributionCalculationTransform), typeof(FeatureContributionCalculationTransform.Arguments),
    typeof(SignatureBindableMapper), "Feature Contribution Mapper", "fct", MetadataUtils.Const.ScoreColumnKind.FeatureContribution)]

[assembly: LoadableClass(typeof(ISchemaBindableMapper), typeof(FeatureContributionCalculationTransform), null, typeof(SignatureLoadModel),
    "Feature Contribution Mapper", FeatureContributionCalculationTransform.MapperLoaderSignature)]

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
    /// [!code-csharp[FCT](~/../docs/samples/doc/samples/Microsoft.ML.Samples/Dynamic/FeatureContributionCalculationTransform.cs)]
    /// ]]>
    /// </format>
    /// </example>
    public sealed class FeatureContributionCalculationTransform
    {
        // Apparently, loader signature is limited in length to 24 characters.
        internal const string MapperLoaderSignature = "WTFBindable";
        private const int MaxTopBottom = 1000;

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

        public static IDataScorerTransform Create(IHostEnvironment env, Arguments args, IDataView data, ISchemaBoundMapper mapper, RoleMappedSchema trainSchema)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(data, nameof(data));
            env.CheckValue(mapper, nameof(mapper));
            if (args.Top <= 0 || args.Top > MaxTopBottom)
                throw env.Except($"Number of top contribution must be in range (0,{MaxTopBottom}]");
            if (args.Bottom <= 0 || args.Bottom > MaxTopBottom)
                throw env.Except($"Number of bottom contribution must be in range (0,{MaxTopBottom}]");

            var contributionMapper = mapper as RowMapper;
            env.CheckParam(mapper != null, nameof(mapper), "Unexpected mapper");

            var scorer = ScoreUtils.GetScorerComponent(env, contributionMapper);
            var scoredPipe = scorer.CreateComponent(env, data, contributionMapper, trainSchema);
            return scoredPipe;
        }

        public static ISchemaBindableMapper Create(IHostEnvironment env, Arguments args, IPredictor predictor)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(predictor, nameof(predictor));
            if (args.Top <= 0 || args.Top > MaxTopBottom)
                throw env.Except($"Number of top contribution must be in range (0,{MaxTopBottom}]");
            if (args.Bottom <= 0 || args.Bottom > MaxTopBottom)
                throw env.Except($"Number of bottom contribution must be in range (0,{MaxTopBottom}]");

            var pred = predictor as IFeatureContributionMapper;
            env.CheckParam(pred != null, nameof(predictor), "Predictor doesn't support getting feature contributions");
            return new BindableMapper(env, pred, args.Top, args.Bottom, args.Normalize, args.Stringify);
        }

        public static IDataScorerTransform Create(IHostEnvironment env, Arguments args, IDataView data, IPredictor predictor, string features = DefaultColumnNames.Features)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(predictor, nameof(predictor));
            if (args.Top <= 0 || args.Top > MaxTopBottom)
                throw env.Except($"Number of top contribution must be in range (0,{MaxTopBottom}]");
            if (args.Bottom <= 0 || args.Bottom > MaxTopBottom)
                throw env.Except($"Number of bottom contribution must be in range (0,{MaxTopBottom}]");

            var roles = new List<KeyValuePair<RoleMappedSchema.ColumnRole, string>>();
            roles.Add(new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RoleMappedSchema.ColumnRole.Feature, features));
            var schema = new RoleMappedSchema(data.Schema, roles);

            var mapper = Create(env, args, predictor);
            var boundMapper = mapper.Bind(env, schema);
            return Create(env, args, data, boundMapper, null);
        }

        public static ISchemaBindableMapper Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            return new BindableMapper(env, ctx);
        }

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
                    loaderAssemblyName: typeof(FeatureContributionCalculationTransform).Assembly.FullName);
            }

            public PredictionKind PredictionKind => Predictor.PredictionKind;

            public BindableMapper(IHostEnvironment env, IFeatureContributionMapper predictor, int topContributionsCount, int bottomContributionsCount, bool normalize, bool stringify)
            {
                Contracts.CheckValue(env, nameof(env));
                _env = env;
                _env.CheckValue(predictor, nameof(predictor));
                if (topContributionsCount <= 0 || topContributionsCount > MaxTopBottom)
                    throw env.Except($"Number of top contribution must be in range (0,{MaxTopBottom}]");
                if (bottomContributionsCount <= 0 || bottomContributionsCount > MaxTopBottom)
                    throw env.Except($"Number of bottom contribution must be in range (0,{MaxTopBottom}]");

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
                // int: topContributionsCount
                // int: bottomContributionsCount
                // bool: normalize
                // bool: stringify
                ctx.LoadModel<IFeatureContributionMapper, SignatureLoadModel>(env, out Predictor, ModelFileUtils.DirPredictor);
                GenericMapper = ScoreUtils.GetSchemaBindableMapper(_env, Predictor, null);
                _topContributionsCount = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(0 < _topContributionsCount && _topContributionsCount <= MaxTopBottom);
                _bottomContributionsCount = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(0 < _bottomContributionsCount && _bottomContributionsCount <= MaxTopBottom);
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

            public void Save(ModelSaveContext ctx)
            {
                Contracts.CheckValue(ctx, nameof(ctx));
                ctx.SetVersionInfo(GetVersionInfo());

                // *** Binary format ***
                // int: topContributionsCount
                // int: bottomContributionsCount
                // bool: normalize
                // bool: stringify
                ctx.SaveModel(Predictor, ModelFileUtils.DirPredictor);

                Contracts.Assert(0 < _topContributionsCount && _topContributionsCount <= MaxTopBottom);
                ctx.Writer.Write(_topContributionsCount);
                Contracts.Assert(0 < _bottomContributionsCount && _bottomContributionsCount <= MaxTopBottom);
                ctx.Writer.Write(_bottomContributionsCount);
                ctx.Writer.WriteBoolByte(_normalize);
                ctx.Writer.WriteBoolByte(Stringify);
            }

            public Delegate GetTextContributionGetter(IRow input, int colSrc, VBuffer<ReadOnlyMemory<char>> slotNames)
            {
                Contracts.CheckValue(input, nameof(input));
                Contracts.Check(0 <= colSrc && colSrc < input.Schema.ColumnCount);
                var typeSrc = input.Schema.GetColumnType(colSrc);

                Func<IRow, int, VBuffer<ReadOnlyMemory<char>>, ValueGetter<ReadOnlyMemory<char>>> del = GetTextValueGetter<int>;
                var meth = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(typeSrc.RawType);
                return (Delegate)meth.Invoke(this, new object[] { input, colSrc, slotNames });
            }

            public Delegate GetContributionGetter(IRow input, int colSrc)
            {
                Contracts.CheckValue(input, nameof(input));
                Contracts.Check(0 <= colSrc && colSrc < input.Schema.ColumnCount);

                var typeSrc = input.Schema.GetColumnType(colSrc);
                Func<IRow, int, ValueGetter<VBuffer<float>>> del = GetValueGetter<int>;

                // REVIEW: Assuming Feature contributions will be VBuffer<float>.
                // For multiclass LR it needs to be(VBuffer<float>[].
                var meth = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(typeSrc.RawType);
                return (Delegate)meth.Invoke(this, new object[] { input, colSrc });
            }

            private ReadOnlyMemory<char> GetSlotName(int index, VBuffer<ReadOnlyMemory<char>> slotNames)
            {
                var count = slotNames.GetValues().Length;
                _env.Assert(count > index || count == 0 && slotNames.Length > index);
                var slotName = slotNames.GetItemOrDefault(index);
                return slotName.IsEmpty
                    ? new ReadOnlyMemory<char>($"f{index}".ToCharArray())
                    : slotName;
            }

            private ValueGetter<ReadOnlyMemory<char>> GetTextValueGetter<TSrc>(IRow input, int colSrc, VBuffer<ReadOnlyMemory<char>> slotNames)
            {
                Contracts.AssertValue(input);
                Contracts.AssertValue(Predictor);

                var featureGetter = input.GetGetter<TSrc>(colSrc);
                var map = Predictor.GetFeatureContributionMapper<TSrc, VBuffer<float>>(_topContributionsCount, _bottomContributionsCount, _normalize);

                var features = default(TSrc);
                var contributions = default(VBuffer<float>);
                return
                    (ref ReadOnlyMemory<char> dst) =>
                    {
                        featureGetter(ref features);
                        map(in features, ref contributions);
                        var indices = new Span<int>();
                        var values = new Span<float>();
                        if (contributions.IsDense)
                            Utils.GetIdentityPermutation(contributions.Length).AsSpan().CopyTo(indices);
                        else
                            contributions.GetIndices().CopyTo(indices);
                        contributions.GetValues().CopyTo(values);
                        var count = values.Length;
                        var sb = new StringBuilder();
                        GenericSpanSortHelper<int, float>.Sort(indices, values, 0, count);
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

            private ValueGetter<VBuffer<float>> GetValueGetter<TSrc>(IRow input, int colSrc)
            {
                Contracts.AssertValue(input);
                Contracts.AssertValue(Predictor);

                var featureGetter = input.GetGetter<TSrc>(colSrc);

                // REVIEW: Scorer can do call to Sparicification\Norm routine.

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

        private sealed class RowMapper : ISchemaBoundRowMapper
        {
            private readonly IHostEnvironment _env;
            private readonly ISchemaBoundRowMapper _genericRowMapper;
            private readonly BindableMapper _parent;
            private readonly ISchema _outputSchema;
            private readonly ISchema _outputGenericSchema;
            private VBuffer<ReadOnlyMemory<char>> _slotNames;

            public RoleMappedSchema InputRoleMappedSchema { get; }

            public Schema InputSchema => InputRoleMappedSchema.Schema;

            public Schema OutputSchema { get; }

            public ISchemaBindableMapper Bindable => _parent;

            public RowMapper(IHostEnvironment env, BindableMapper parent, RoleMappedSchema schema)
            {
                Contracts.AssertValue(env);
                _env = env;
                _env.AssertValue(schema);
                _env.AssertValue(parent);
                _env.AssertValue(schema.Feature);
                _parent = parent;
                InputRoleMappedSchema = schema;
                var genericMapper = parent.GenericMapper.Bind(_env, schema);
                _genericRowMapper = genericMapper as ISchemaBoundRowMapper;

                if (parent.Stringify)
                {
                    _outputSchema = new SimpleSchema(_env,
                        new KeyValuePair<string, ColumnType>(DefaultColumnNames.FeatureContributions, TextType.Instance));
                    if (InputSchema.HasSlotNames(InputRoleMappedSchema.Feature.Index, InputRoleMappedSchema.Feature.Type.VectorSize))
                        InputSchema.GetMetadata(MetadataUtils.Kinds.SlotNames, InputRoleMappedSchema.Feature.Index,
                            ref _slotNames);
                    else
                        _slotNames = VBufferUtils.CreateEmpty<ReadOnlyMemory<char>>(InputRoleMappedSchema.Feature.Type.VectorSize);
                }
                else
                {
                    _outputSchema = new FeatureContributionSchema(_env, DefaultColumnNames.FeatureContributions,
                        new VectorType(NumberType.R4, schema.Feature.Type.AsVector),
                        InputSchema, InputRoleMappedSchema.Feature.Index);
                }

                _outputGenericSchema = _genericRowMapper.OutputSchema;
                OutputSchema = new CompositeSchema(new ISchema[] { _outputGenericSchema, _outputSchema, }).AsSchema;
            }

            /// <summary>
            /// Returns the input columns needed for the requested output columns.
            /// </summary>
            public Func<int, bool> GetDependencies(Func<int, bool> predicate)
            {
                for (int i = 0; i < OutputSchema.ColumnCount; i++)
                {
                    if (predicate(i))
                        return col => col == InputRoleMappedSchema.Feature.Index;
                }
                return col => false;
            }

            public IRow GetOutputRow(IRow input, Func<int, bool> predicate, out Action disposer)
            {
                Contracts.AssertValue(input);
                Contracts.AssertValue(predicate);
                var totalColumnsCount = 1 + _outputGenericSchema.ColumnCount;
                var getters = new Delegate[totalColumnsCount];

                if (predicate(totalColumnsCount - 1))
                {
                    getters[totalColumnsCount - 1] = _parent.Stringify
                        ? _parent.GetTextContributionGetter(input, InputRoleMappedSchema.Feature.Index, _slotNames)
                        : _parent.GetContributionGetter(input, InputRoleMappedSchema.Feature.Index);
                }

                var genericRow = _genericRowMapper.GetRow(input, GetGenericPredicate(predicate), out disposer);
                for (var i = 0; i < _outputGenericSchema.ColumnCount; i++)
                {
                    if (genericRow.IsColumnActive(i))
                        getters[i] = RowCursorUtils.GetGetterAsDelegate(genericRow, i);
                }

                return new SimpleRow(OutputSchema, input, getters);
            }

            public Func<int, bool> GetGenericPredicate(Func<int, bool> predicate)
            {
                return col => predicate(col);
            }

            public IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> GetInputColumnRoles()
            {
                yield return RoleMappedSchema.ColumnRole.Feature.Bind(InputRoleMappedSchema.Feature.Name);
            }

            public IRow GetRow(IRow input, Func<int, bool> active, out Action disposer)
            {
                return GetOutputRow(input, active, out disposer);
            }
        }

        private sealed class FeatureContributionSchema : ISchema
        {
            private readonly Schema _parentSchema;
            private readonly IExceptionContext _ectx;
            private readonly string[] _names;
            private readonly ColumnType[] _types;
            private readonly Dictionary<string, int> _columnNameMap;
            private readonly int _featureCol;
            private readonly int _featureVectorSize;
            private readonly bool _hasSlotNames;

            public int ColumnCount => _types.Length;

            public FeatureContributionSchema(IExceptionContext ectx, string columnName, ColumnType columnType, Schema parentSchema, int featureCol)
            {
                Contracts.CheckValueOrNull(ectx);
                Contracts.CheckValue(parentSchema, nameof(parentSchema));
                _ectx = ectx;
                _ectx.CheckNonEmpty(columnName, nameof(columnName));
                _parentSchema = parentSchema;
                _featureCol = featureCol;
                _featureVectorSize = _parentSchema.GetColumnType(_featureCol).VectorSize;
                _hasSlotNames = _parentSchema.HasSlotNames(_featureCol, _featureVectorSize);

                _names = new string[] { columnName };
                _types = new ColumnType[] { columnType };
                _columnNameMap = new Dictionary<string, int>() { { columnName, 0 } };
            }

            public bool TryGetColumnIndex(string name, out int col)
            {
                return _columnNameMap.TryGetValue(name, out col);
            }

            public string GetColumnName(int col)
            {
                _ectx.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                return _names[col];
            }

            public ColumnType GetColumnType(int col)
            {
                _ectx.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                return _types[col];
            }

            public IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypes(int col)
            {
                _ectx.CheckParam(col == 0, nameof(col));
                if (_hasSlotNames)
                    yield return MetadataUtils.GetSlotNamesPair(_featureVectorSize);
            }

            public ColumnType GetMetadataTypeOrNull(string kind, int col)
            {
                _ectx.CheckNonEmpty(kind, nameof(kind));
                _ectx.CheckParam(col == 0, nameof(col));
                if (_hasSlotNames)
                    return MetadataUtils.GetNamesType(_featureVectorSize);
                return null;
            }

            public void GetMetadata<TValue>(string kind, int col, ref TValue value)
            {
                _ectx.CheckParam(col == 0, nameof(col));
                if (kind == MetadataUtils.Kinds.SlotNames && _hasSlotNames)
                    _parentSchema.GetMetadata(kind, _featureCol, ref value);
                else
                    throw MetadataUtils.ExceptGetMetadata();
            }
        }
    }
}
