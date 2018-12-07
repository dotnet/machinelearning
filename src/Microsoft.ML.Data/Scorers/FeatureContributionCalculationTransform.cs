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

[assembly: LoadableClass(typeof(IDataScorerTransform), typeof(FeatureContributionCalculatingTransformer.BindableMapper), typeof(FeatureContributionCalculatingTransformer.Arguments),
    typeof(SignatureDataScorer), "Feature Contribution Transform", "fct", "FeatureContributionCalculationTransform", MetadataUtils.Const.ScoreColumnKind.FeatureContribution)]

[assembly: LoadableClass(typeof(ISchemaBindableMapper), typeof(FeatureContributionCalculatingTransformer.BindableMapper), typeof(FeatureContributionCalculatingTransformer.Arguments),
    typeof(SignatureBindableMapper), "Feature Contribution Mapper", "fct", MetadataUtils.Const.ScoreColumnKind.FeatureContribution)]

[assembly: LoadableClass(typeof(ISchemaBindableMapper), typeof(FeatureContributionCalculatingTransformer.BindableMapper), null, typeof(SignatureLoadModel),
    "Feature Contribution Mapper", FeatureContributionCalculatingTransformer.MapperLoaderSignature)]

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

        internal const string MapperLoaderSignature = "WTFBindable";

        private const int MaxTopBottom = 1000;

        private readonly string _features;
        private readonly int _topContributionsCount;
        private readonly int _bottomContributionsCount;
        private readonly bool _normalize;
        private readonly bool _stringify;
        private readonly IFeatureContributionMapper _predictor;
        private readonly BindableMapper _mapper;

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

        // TODO documentation
        public FeatureContributionCalculatingTransformer(IHostEnvironment env, IPredictor predictor, string featuresColumn, Arguments args)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(FeatureContributionCalculatingTransformer)))
        {
            Host.CheckValue(args, nameof(args));
            Host.CheckValue(predictor, nameof(predictor));

            var pred = predictor as IFeatureContributionMapper;
            Host.CheckParam(pred != null, nameof(predictor), "Predictor doesn't support getting feature contributions");

            // TODO check that the featues column is not empty.
            _mapper = new BindableMapper(Host, pred, args.Top, args.Bottom, args.Normalize, args.Stringify);
            _features = featuresColumn;
            _predictor = pred;
            _stringify = args.Stringify;
            _topContributionsCount = args.Top;
            _bottomContributionsCount = args.Bottom;
            _normalize = args.Normalize;
        }

        // Factory method for SignatureLoadModel
        private FeatureContributionCalculatingTransformer(IHostEnvironment env, ModelLoadContext ctx)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(FeatureContributionCalculatingTransformer)))
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // string features
            // BindableMapper mapper

            // TODO use ctx.LoadModel with BindableMapper instead of this.
            _features = ctx.LoadNonEmptyString();
            ctx.LoadModel<IFeatureContributionMapper, SignatureLoadModel>(env, out _predictor, ModelFileUtils.DirPredictor);
            _topContributionsCount = ctx.Reader.ReadInt32();
            Contracts.CheckDecode(0 < _topContributionsCount && _topContributionsCount <= MaxTopBottom);
            _bottomContributionsCount = ctx.Reader.ReadInt32();
            Contracts.CheckDecode(0 < _bottomContributionsCount && _bottomContributionsCount <= MaxTopBottom);
            _normalize = ctx.Reader.ReadBoolByte();
            _stringify = ctx.Reader.ReadBoolByte();

            _mapper = new BindableMapper(env, _predictor, _topContributionsCount, _bottomContributionsCount, _normalize, _stringify);
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
            // TODO use ctx.SaveModel with BindableMapper instead of this.
            ctx.SaveModel(_predictor, ModelFileUtils.DirPredictor);
            Contracts.Assert(0 < _topContributionsCount && _topContributionsCount <= MaxTopBottom);
            ctx.Writer.Write(_topContributionsCount);
            Contracts.Assert(0 < _bottomContributionsCount && _bottomContributionsCount <= MaxTopBottom);
            ctx.Writer.Write(_bottomContributionsCount);
            ctx.Writer.WriteBoolByte(_normalize);
            ctx.Writer.WriteBoolByte(_stringify);
        }

        protected override IRowMapper MakeRowMapper(Schema schema)
            => new Mapper(this, schema);

        private class Mapper : MapperBase
        {
            private readonly FeatureContributionCalculatingTransformer _parent;
            private readonly BindableMapper _bindableMapper;
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

        // TODO documentation
        internal sealed class BindableMapper : ISchemaBindableMapper, ICanSaveModel, IPredictor
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
                    loaderAssemblyName: typeof(FeatureContributionCalculatingTransformer).Assembly.FullName);
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

            // Factory method for SignatureLoadModel.
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

            // Factory method for SignatureDataScorer.
            private static IDataScorerTransform Create(IHostEnvironment env, Arguments args, IDataView data, ISchemaBoundMapper mapper, RoleMappedSchema trainSchema)
            {
                Contracts.CheckValue(env, nameof(env));
                env.CheckValue(data, nameof(data));
                env.CheckValue(mapper, nameof(mapper));

                var contributionMapper = mapper as BoundMapper;
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

            public ISchemaBoundMapper Bind(IHostEnvironment env, RoleMappedSchema schema)
            {
                Contracts.CheckValue(env, nameof(env));
                env.CheckValue(schema, nameof(schema));
                CheckSchemaValid(env, schema, Predictor);
                return new BoundMapper(env, this, schema);
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

            public Delegate GetTextContributionGetter(Row input, int colSrc, VBuffer<ReadOnlyMemory<char>> slotNames)
            {
                Contracts.CheckValue(input, nameof(input));
                Contracts.Check(0 <= colSrc && colSrc < input.Schema.ColumnCount);
                var typeSrc = input.Schema.GetColumnType(colSrc);

                Func<Row, int, VBuffer<ReadOnlyMemory<char>>, ValueGetter<ReadOnlyMemory<char>>> del = GetTextValueGetter<int>;
                var meth = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(typeSrc.RawType);
                return (Delegate)meth.Invoke(this, new object[] { input, colSrc, slotNames });
            }

            public Delegate GetContributionGetter(Row input, int colSrc)
            {
                Contracts.CheckValue(input, nameof(input));
                Contracts.Check(0 <= colSrc && colSrc < input.Schema.ColumnCount);

                var typeSrc = input.Schema.GetColumnType(colSrc);
                Func<Row, int, ValueGetter<VBuffer<float>>> del = GetValueGetter<int>;

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

            private ValueGetter<ReadOnlyMemory<char>> GetTextValueGetter<TSrc>(Row input, int colSrc, VBuffer<ReadOnlyMemory<char>> slotNames)
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
                        GenericSpanSortHelper<int>.Sort(indices, values, 0, count);
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

            private ValueGetter<VBuffer<float>> GetValueGetter<TSrc>(Row input, int colSrc)
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

        private sealed class BoundMapper : ISchemaBoundMapper
        {
            private readonly IHostEnvironment _env;

            public RoleMappedSchema InputRoleMappedSchema { get; }
            public Schema OutputSchema { get; }
            public ISchemaBindableMapper Bindable { get; }

            public BoundMapper(IHostEnvironment env, BindableMapper parent, RoleMappedSchema schema)
            {
                Bindable = parent;
                InputRoleMappedSchema = schema;
                _env = env;

                var inputSchema = schema.Schema;
                var genericRowMapper = parent.GenericMapper.Bind(_env, schema) as ISchemaBoundRowMapper;
                ISchema outputSchema;

                if (parent.Stringify)
                {
                    outputSchema = new SimpleSchema(_env,
                        new KeyValuePair<string, ColumnType>(DefaultColumnNames.FeatureContributions, TextType.Instance));
                }
                else
                {
                    outputSchema = new FeatureContributionSchema(_env, DefaultColumnNames.FeatureContributions,
                        new VectorType(NumberType.R4, schema.Feature.Type.AsVector),
                        inputSchema, InputRoleMappedSchema.Feature.Index);
                }
                ISchema outputGenericSchema = genericRowMapper.OutputSchema;
                OutputSchema = new CompositeSchema(new ISchema[] { outputGenericSchema, outputSchema }).AsSchema;
            }

            public IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> GetInputColumnRoles()
            {
                yield return RoleMappedSchema.ColumnRole.Feature.Bind(InputRoleMappedSchema.Feature.Name);
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

    // TODO DOcumentation
    public sealed class FeatureContributionCalculatingEstimator : TrivialEstimator<FeatureContributionCalculatingTransformer>
    {
        private readonly FeatureContributionCalculatingTransformer.Arguments _args;
        private readonly string _features;
        private readonly IPredictor _predictor;

        // TODO Documentation
        public FeatureContributionCalculatingEstimator(IHostEnvironment env, IPredictor predictor, string featuresColumn, FeatureContributionCalculatingTransformer.Arguments args)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(FeatureContributionCalculatingTransformer)), new FeatureContributionCalculatingTransformer(env, predictor, featuresColumn, args))
        {
            // TODO argcheck
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
