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
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(typeof(IDataScorerTransform), typeof(FeatureContributionCalculatingTransformer.BindableMapper), typeof(FeatureContributionCalculatingTransformer.BindableMapper.Arguments),
    typeof(SignatureDataScorer), "Feature Contribution Transform", "fct", "FeatureContributionCalculationTransform", MetadataUtils.Const.ScoreColumnKind.FeatureContribution)]

[assembly: LoadableClass(typeof(ISchemaBindableMapper), typeof(FeatureContributionCalculatingTransformer.BindableMapper), typeof(FeatureContributionCalculatingTransformer.BindableMapper.Arguments),
    typeof(SignatureBindableMapper), "Feature Contribution Mapper", "fct", MetadataUtils.Const.ScoreColumnKind.FeatureContribution)]

[assembly: LoadableClass(typeof(FeatureContributionCalculatingTransformer.BindableMapper), typeof(FeatureContributionCalculatingTransformer.BindableMapper), null, typeof(SignatureLoadModel),
    "Feature Contribution Mapper", FeatureContributionCalculatingTransformer.BindableMapper.MapperLoaderSignature)]

[assembly: LoadableClass(FeatureContributionCalculatingTransformer.Summary, typeof(FeatureContributionCalculatingTransformer), null, typeof(SignatureLoadModel),
    FeatureContributionCalculatingTransformer.FriendlyName, FeatureContributionCalculatingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(FeatureContributionCalculatingTransformer), null, typeof(SignatureLoadRowMapper),
   FeatureContributionCalculatingTransformer.FriendlyName, FeatureContributionCalculatingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(void), typeof(FeatureContributionEntryPoint), null, typeof(SignatureEntryPointModule), FeatureContributionCalculatingTransformer.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// The FeatureContributionCalculationTransformer scores the model on an input dataset and
    /// computes model-specific contribution scores for each feature.
    /// </summary>
    /// <remarks>
    /// See the sample below for an example of how to compute feature importance using the FeatureContributionCalculatingTransformer.
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
        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "The predictor model to apply to data", SortOrder = 1)]
            public PredictorModel PredictorModel;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Name of feature column", SortOrder = 2)]
            public string FeatureColumn = DefaultColumnNames.Features;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of top contributions", SortOrder = 3)]
            public int Top = FeatureContributionCalculatingEstimator.Defaults.Top;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of bottom contributions", SortOrder = 4)]
            public int Bottom = FeatureContributionCalculatingEstimator.Defaults.Bottom;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether or not output of Features contribution should be normalized", ShortName = "norm", SortOrder = 5)]
            public bool Normalize = FeatureContributionCalculatingEstimator.Defaults.Normalize;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether or not output of Features contribution in string key-value format", ShortName = "str", SortOrder = 6)]
            public bool Stringify = FeatureContributionCalculatingEstimator.Defaults.Stringify;
        }

        // Apparently, loader signature is limited in length to 24 characters.
        internal const string Summary = "For each data point, calculates the contribution of individual features to the model prediction.";
        internal const string FriendlyName = "Feature Contribution Transform";
        internal const string LoaderSignature = "FeatureContribution";

        private readonly string _featureColumn;
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

        /// <summary>
        /// The Feature Contribution Calculation Transform scores the model on an input dataset and
        /// computes model-specific contribution scores for each feature.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="predictor">Trained model parameters that support Feature Contribution Calculation and which will be used for scoring.</param>
        /// <param name="featureColumn">The name of the feature column that will be used as input.</param>
        /// <param name="top">The number of top contributing features for each data sample that will be retained in the FeatureContribution column.</param>
        /// <param name="bottom">The number of least contributing features for each data sample that will be retained in the FeatureContribution column.</param>
        /// <param name="normalize">Whether the feature contributions should be normalized to the [-1, 1] interval.</param>
        /// <param name="stringify">Since the features are converted to numbers before the algorithms use them, if you want the contributions presented as
        /// string(key)-values, set stringify to <langword>true</langword></param>
        public FeatureContributionCalculatingTransformer(IHostEnvironment env, IFeatureContributionMappable predictor,
            string featureColumn = DefaultColumnNames.Features,
            int top = FeatureContributionCalculatingEstimator.Defaults.Top,
            int bottom = FeatureContributionCalculatingEstimator.Defaults.Bottom,
            bool normalize = FeatureContributionCalculatingEstimator.Defaults.Normalize,
            bool stringify = FeatureContributionCalculatingEstimator.Defaults.Stringify)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(FeatureContributionCalculatingTransformer)))
        {
            // Other Parameters are checked in the constructor of the BindableMapper.
            Host.CheckValue(predictor, nameof(predictor));
            Host.CheckNonEmpty(featureColumn, nameof(featureColumn));

            // If a predictor implements IFeatureContributionMappable, it also implements the internal interface IFeatureContributionMapper.
            // This is how we keep the implementation of feature contribution calculation internal.
            IFeatureContributionMapper pred = predictor as IFeatureContributionMapper;
            Host.AssertValue(pred);

            _featureColumn = featureColumn;
            _mapper = new BindableMapper(Host, pred, top, bottom, normalize, stringify);
        }

        // Factory constructor for SignatureLoadModel
        private FeatureContributionCalculatingTransformer(IHostEnvironment env, ModelLoadContext ctx)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(FeatureContributionCalculatingTransformer)))
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // string featureColumn
            // BindableMapper mapper

            _featureColumn = ctx.LoadNonEmptyString();
            ctx.LoadModel<BindableMapper, SignatureLoadModel>(env, out _mapper, ModelFileUtils.DirPredictor);
        }

        // Factory method for SignatureLoadRowMapper.
        internal static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, Schema inputSchema)
            => new FeatureContributionCalculatingTransformer(env, ctx).MakeRowMapper(inputSchema);

        // Used by the entrypoints.
        internal static IDataTransform Create(IHostEnvironment env, IFeatureContributionMappable predictor, Arguments args, IDataView input)
            => new FeatureContributionCalculatingTransformer(env, predictor, args.FeatureColumn, args.Top, args.Bottom, args.Normalize, args.Stringify).MakeDataTransform(input);

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // string featureColumn
            // BindableMapper mapper

            ctx.SaveNonEmptyString(_featureColumn);
            ctx.SaveModel(_mapper, ModelFileUtils.DirPredictor);
        }

        private protected override IRowMapper MakeRowMapper(Schema schema)
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
                _parent = parent;
                _bindableMapper = _parent._mapper;

                // Check that the featureColumn is present and has the expected type.
                if (!schema.TryGetColumnIndex(_parent._featureColumn, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(schema), "input", _parent._featureColumn);
                var colType = schema.GetColumnType(col);
                if ( colType.ItemType != NumberType.R4 || !colType.IsVector)
                    throw Host.ExceptUserArg(nameof(schema), "Column '{0}' does not have compatible type. Expected type is vector of float.", _parent._featureColumn);

                var roles = new List<KeyValuePair<RoleMappedSchema.ColumnRole, string>>();
                roles.Add(new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RoleMappedSchema.ColumnRole.Feature, _parent._featureColumn));
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
            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
            {
                var active = new bool[InputSchema.ColumnCount];
                InputSchema.TryGetColumnIndex(_parent._featureColumn, out int featureCol);
                active[featureCol] = true;
                return col => active[col];
            }

            public override void Save(ModelSaveContext ctx)
                => _parent.Save(ctx);

            // The FeatureContributionCalculatingTransformer produces two sets of columns: the columns obtained from scoring and the FeatureContribution column.
            // If the argument stringify is true, the type of the FeatureContribution column is string, otherwise it is a vector of float.
            protected override Schema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new List<Schema.DetachedColumn>();

                // Add columns obtained by scoring the model. Note that the number and type will vary based on which PredictionKind the predictor belongs to.
                result.AddRange(_outputGenericSchema.GetColumns().Select(pair => new Schema.DetachedColumn(pair.column)));

                // Add FeatureContributions column.
                if (_bindableMapper.Stringify)
                    result.Add(new Schema.DetachedColumn(DefaultColumnNames.FeatureContributions, TextType.Instance));
                else
                {
                    var builder = new MetadataBuilder();
                    builder.Add(InputSchema[_roleMappedSchema.Feature.Index].Metadata, x => x == MetadataUtils.Kinds.SlotNames);
                    result.Add(new Schema.DetachedColumn(DefaultColumnNames.FeatureContributions, new VectorType(NumberType.R4, _roleMappedSchema.Feature.Type.ValueCount), builder.GetMetadata()));
                }
                return result.ToArray();
            }

            protected Delegate GetScoreGetter(Row input, int iinfo, Func<int, bool> active)
            {
                var genericRow = _genericRowMapper.GetRow(input, col => active(col));
                return RowCursorUtils.GetGetterAsDelegate(genericRow, iinfo);
            }

            protected Delegate GetFeatureContributionGetter(Row input, int iinfo, Func<int, bool> active)
            {
                if (active(iinfo))
                {
                    return _bindableMapper.Stringify
                        ? _bindableMapper.GetTextContributionGetter(input, _roleMappedSchema.Feature.Index, _slotNames)
                        : _bindableMapper.GetContributionGetter(input, _roleMappedSchema.Feature.Index);
                }
                return null;
            }

            protected override Delegate MakeGetter(Row input, int iinfo, Func<int, bool> active, out Action disposer)
            {
                disposer = null;
                if (iinfo < _outputGenericSchema.ColumnCount)
                    return GetScoreGetter(input, iinfo, active);
                else
                    return GetFeatureContributionGetter(input, iinfo, active);
            }
        }

        /// <summary>
        /// Holds the definition of the getters for the FeatureContribution column. It also contains the generic mapper that is used to score the Predictor.
        /// </summary>
        internal sealed class BindableMapper : ISchemaBindableMapper, ICanSaveModel, IPredictor
        {
            public sealed class Arguments : ScorerArgumentsBase
            {
                [Argument(ArgumentType.AtMostOnce, HelpText = "Number of top contributions", SortOrder = 1)]
                public int Top = FeatureContributionCalculatingEstimator.Defaults.Top;

                [Argument(ArgumentType.AtMostOnce, HelpText = "Number of bottom contributions", SortOrder = 2)]
                public int Bottom = FeatureContributionCalculatingEstimator.Defaults.Bottom;

                [Argument(ArgumentType.AtMostOnce, HelpText = "Whether or not output of Features contribution should be normalized", ShortName = "norm", SortOrder = 3)]
                public bool Normalize = FeatureContributionCalculatingEstimator.Defaults.Normalize;

                [Argument(ArgumentType.AtMostOnce, HelpText = "Whether or not output of Features contribution in string key-value format", ShortName = "str", SortOrder = 4)]
                public bool Stringify = FeatureContributionCalculatingEstimator.Defaults.Stringify;

                // REVIEW: the scorer currently ignores the 'suffix' argument from the base class. It should respect it.
            }

            private readonly int _topContributionsCount;
            private readonly int _bottomContributionsCount;
            private readonly bool _normalize;
            private readonly IHostEnvironment _env;

            public readonly IFeatureContributionMapper Predictor;
            public readonly ISchemaBindableMapper GenericMapper;
            public readonly bool Stringify;

            internal const string MapperLoaderSignature = "WTFBindable";

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

            // Factory constructor for SignatureLoadModel.
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
                _env.Assert(0 <= index && index < slotNames.Length);
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
                        int[] indices;
                        float[] values = contributions.GetValues().ToArray();
                        if (contributions.IsDense)
                            indices = Utils.GetIdentityPermutation(contributions.Length);
                        else
                            indices = contributions.GetIndices().ToArray();
                        var count = values.Length;
                        var sb = new StringBuilder();
                        Array.Sort(indices, values, 0, count);
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

        /// <summary>
        /// Maps a schema from input columns to output columns. Keeps track of the input columns that are needed for the mapping.
        /// </summary>
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
                Schema outputSchema;

                if (parent.Stringify)
                {
                    var builder = new SchemaBuilder();
                    builder.AddColumn(DefaultColumnNames.FeatureContributions, TextType.Instance, null);
                    outputSchema = builder.GetSchema();
                }
                else
                {
                    outputSchema = Schema.Create(new FeatureContributionSchema(_env, DefaultColumnNames.FeatureContributions,
                        new VectorType(NumberType.R4, schema.Feature.Type.ValueCount),
                        inputSchema, InputRoleMappedSchema.Feature.Index));
                }
                Schema outputGenericSchema = genericRowMapper.OutputSchema;
                OutputSchema = new CompositeSchema(new Schema[] { outputGenericSchema, outputSchema }).AsSchema;
            }

            public IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> GetInputColumnRoles()
            {
                yield return RoleMappedSchema.ColumnRole.Feature.Bind(InputRoleMappedSchema.Feature.Name);
            }
        }

        /// <summary>
        /// Specifies the schema of the FeatureContribution column, needed by the BoundMapper.
        /// </summary>
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

    /// <summary>
    /// Estimator producing a FeatureContributionCalculatingTransformer which scores the model on an input dataset and
    /// computes model-specific contribution scores for each feature.
    /// </summary>
    public sealed class FeatureContributionCalculatingEstimator : TrivialEstimator<FeatureContributionCalculatingTransformer>
    {
        private readonly string _featureColumn;
        private readonly IFeatureContributionMappable _predictor;
        private readonly bool _stringify;

        public static class Defaults
        {
            public const int Top = 10;
            public const int Bottom = 10;
            public const bool Normalize = true;
            public const bool Stringify = false;
        }

        /// <summary>
        /// The Feature Contribution Calculation Transform scores the model on an input dataset and
        /// computes model-specific contribution scores for each feature.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="predictor">Trained model parameters that support Feature Contribution Calculation and which will be used for scoring.</param>
        /// <param name="featureColumn">The name of the feature column that will be used as input.</param>
        /// <param name="top">The number of top contributing features for each data sample that will be retained in the FeatureContribution column.</param>
        /// <param name="bottom">The number of least contributing features for each data sample that will be retained in the FeatureContribution column.</param>
        /// <param name="normalize">Whether the feature contributions should be normalized to the [-1, 1] interval.</param>
        /// <param name="stringify">Since the features are converted to numbers before the algorithms use them, if you want the contributions presented as
        /// string(key)-values, set stringify to <langword>true</langword></param>
        public FeatureContributionCalculatingEstimator(IHostEnvironment env, IFeatureContributionMappable predictor,
            string featureColumn = DefaultColumnNames.Features,
            int top = Defaults.Top,
            int bottom = Defaults.Bottom,
            bool normalize = Defaults.Normalize,
            bool stringify = Defaults.Stringify)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(FeatureContributionCalculatingTransformer)),
                  new FeatureContributionCalculatingTransformer(env, predictor, featureColumn, top, bottom, normalize, stringify))
        {
            _featureColumn = featureColumn;
            _predictor = predictor;
            _stringify = stringify;
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            // Check that the featureColumn is present.
            Host.CheckValue(inputSchema, nameof(inputSchema));
            if (!inputSchema.TryFindColumn(_featureColumn, out var col))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", _featureColumn);
            // Check that the feature column is of the correct type: a vector of float.
            if (col.ItemType != NumberType.R4 || col.Kind != SchemaShape.Column.VectorKind.Vector)
                throw Host.ExceptUserArg(nameof(inputSchema), "Column '{0}' does not have compatible type. Expected type is vector of float.", _featureColumn);

            // Build output schemaShape.
            var result = inputSchema.ToDictionary(x => x.Name);

            // Add columns produced by scorer.
            foreach (var column in ScoringUtils.GetPredictorOutputColumns(_predictor.PredictionKind))
                result[column.Name] = column;

            // REVIEW: We should change the scorers so that they produce consistently probabilities and predicted labels for binary classification.
            // Notice that the generic scorer used here will not produce label column. If the predictor is not IValueMapperDist it does not produce probability either.
            if (!(_predictor is IValueMapperDist))
                result.Remove(DefaultColumnNames.Probability);
            if (_predictor.PredictionKind == PredictionKind.BinaryClassification)
                result.Remove(DefaultColumnNames.PredictedLabel);

            // Add FeatureContributions column.
            if (_stringify)
                result[DefaultColumnNames.FeatureContributions] = new SchemaShape.Column(DefaultColumnNames.FeatureContributions, SchemaShape.Column.VectorKind.Scalar, TextType.Instance, false);
            else
            {
                var featContributionMetadata = new List<SchemaShape.Column>();
                if (col.Metadata.TryFindColumn(MetadataUtils.Kinds.SlotNames, out var slotMeta))
                    featContributionMetadata.Add(slotMeta);
                result[DefaultColumnNames.FeatureContributions] = new SchemaShape.Column(DefaultColumnNames.FeatureContributions, col.Kind, col.ItemType, false, new SchemaShape(featContributionMetadata));
            }
            return new SchemaShape(result.Values);
        }
    }

    internal static class FeatureContributionEntryPoint
    {
        [TlcModule.EntryPoint(Name = "Transforms.FeatureContributionCalculationTransformer",
            Desc = FeatureContributionCalculatingTransformer.Summary,
            UserName = FeatureContributionCalculatingTransformer.FriendlyName)]
        public static CommonOutputs.TransformOutput FeatureContributionCalculation(IHostEnvironment env, FeatureContributionCalculatingTransformer.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(nameof(FeatureContributionCalculatingTransformer));
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var predictor = input.PredictorModel.Predictor as IFeatureContributionMappable;
            if (predictor == null)
                throw host.ExceptUserArg(nameof(predictor), "The provided predictor does not support feature contribution calculation.");

            var xf = FeatureContributionCalculatingTransformer.Create(host, predictor, input, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModelImpl(env, xf, input.Data), OutputData = xf };
        }
    }
}
