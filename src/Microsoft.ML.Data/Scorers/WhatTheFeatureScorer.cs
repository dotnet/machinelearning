// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Reflection;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(typeof(IDataScorerTransform), typeof(WhatTheFeatureScorerTransform), typeof(WhatTheFeatureScorerTransform.Arguments),
    typeof(SignatureDataScorer), "WhatTheFeature Scorer", "wtf", "WhatTheFeatureScorer", MetadataUtils.Const.ScoreColumnKind.WhatTheFeature)]

[assembly: LoadableClass(typeof(ISchemaBindableMapper), typeof(WhatTheFeatureScorerTransform), typeof(WhatTheFeatureScorerTransform.Arguments),
    typeof(SignatureBindableMapper), "WhatTheFeature Mapper", "wtf", MetadataUtils.Const.ScoreColumnKind.WhatTheFeature)]

[assembly: LoadableClass(typeof(ISchemaBindableMapper), typeof(WhatTheFeatureScorerTransform), null, typeof(SignatureLoadModel),
    "WhatTheFeature Mapper", WhatTheFeatureScorerTransform.MapperLoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// The What The Feature scorer is superset of a generic scorer.
    /// It outputs score columns from Generic Scorer plus for given features provides vector of corresponding feature contributions.
    /// </summary>
    public sealed class WhatTheFeatureScorerTransform
    {
        // Apparently, loader signature is limited in length to 24 characters.
        public const string MapperLoaderSignature = "WTFBindable";
        public const string LoaderSignature = "WTFScorer";
        public const int MaxTopBottom = 1000;

        public sealed class Arguments : ScorerArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of top contributions", SortOrder = 1)]
            public int Top = 10;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of bottom contributions", SortOrder = 2)]
            public int Bottom = 10;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether or not output of Features' contribution should be normalized", ShortName = "norm", SortOrder = 3)]
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

            var wtfMapper = mapper as RowMapper;
            env.CheckParam(mapper != null, nameof(mapper), "Unexpected mapper");

            var scorer = ScoreUtils.GetScorerComponent(env, wtfMapper);
            var scoredPipe = scorer.CreateInstance(env, data, wtfMapper, trainSchema);
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

            var pred = predictor as IWhatTheFeatureValueMapper;
            env.CheckParam(pred != null, nameof(predictor), "Predictor doesn't support getting feature contributions");
            return new BindableMapper(env, pred, args.Top, args.Bottom, args.Normalize, args.Stringify);
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

            private static VersionInfo GetVersionInfo()
            {
                return new VersionInfo(
                    modelSignature: "WTF SCBI",
                    verWrittenCur: 0x00010001, // Initial
                    verReadableCur: 0x00010001,
                    verWeCanReadBack: 0x00010001,
                    loaderSignature: MapperLoaderSignature,
                    loaderAssemblyName: typeof(WhatTheFeatureScorerTransform).Assembly.FullName);
            }

            public readonly IWhatTheFeatureValueMapper Predictor;
            public readonly ISchemaBindableMapper GenericMapper;
            public readonly bool Stringify;

            public PredictionKind PredictionKind => Predictor.PredictionKind;

            public BindableMapper(IHostEnvironment env, IWhatTheFeatureValueMapper predictor, int topContributionsCount, int bottomContributionsCount, bool normalize, bool stringify)
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
                ctx.LoadModel<IWhatTheFeatureValueMapper, SignatureLoadModel>(env, out Predictor, ModelFileUtils.DirPredictor);
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

            public Delegate GetTextWtfGetter(IRow input, int colSrc, VBuffer<ReadOnlyMemory<char>> slotNames)
            {
                Contracts.CheckValue(input, nameof(input));
                Contracts.Check(0 <= colSrc && colSrc < input.Schema.ColumnCount);
                var typeSrc = input.Schema.GetColumnType(colSrc);

                Func<IRow, int, VBuffer<ReadOnlyMemory<char>>, ValueGetter<ReadOnlyMemory<char>>> del = GetTextValueGetter<int>;
                var meth = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(typeSrc.RawType);
                return (Delegate)meth.Invoke(this, new object[] { input, colSrc, slotNames });
            }

            public Delegate GetWtfGetter(IRow input, int colSrc)
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
                _env.Assert(slotNames.Count > index || slotNames.Count == 0 && slotNames.Length > index);
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
                var map = Predictor.GetWhatTheFeatureMapper<TSrc, VBuffer<float>>(_topContributionsCount, _bottomContributionsCount, _normalize);

                var features = default(TSrc);
                var contributions = default(VBuffer<float>);
                return
                    (ref ReadOnlyMemory<char> dst) =>
                    {
                        featureGetter(ref features);
                        map(in features, ref contributions);
                        int[] indices = contributions.IsDense ? Utils.GetIdentityPermutation(contributions.Length) : contributions.Indices;
                        var sb = new StringBuilder();
                        Array.Sort(contributions.Values, indices, 0, contributions.Count);
                        for (var i = 0; i < contributions.Count; i++)
                        {
                            var val = contributions.Values[i];
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

                var map = Predictor.GetWhatTheFeatureMapper<TSrc, VBuffer<float>>(_topContributionsCount, _bottomContributionsCount, _normalize);
                var features = default(TSrc);
                return
                    (ref VBuffer<float> dst) =>
                    {
                        featureGetter(ref features);
                        map(in features, ref dst);
                    };
            }

            private static void CheckSchemaValid(IExceptionContext ectx, RoleMappedSchema schema,
                IWhatTheFeatureValueMapper predictor)
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

            public RoleMappedSchema InputSchema { get; }

            public ISchema OutputSchema { get; }

            public ISchemaBindableMapper Bindable => _parent;

            public RowMapper(IHostEnvironment env, BindableMapper parent, RoleMappedSchema schema)
            {
                Contracts.AssertValue(env);
                _env = env;
                _env.AssertValue(schema);
                _env.AssertValue(parent);
                _env.AssertValue(schema.Feature);
                _parent = parent;
                InputSchema = schema;
                var genericMapper = parent.GenericMapper.Bind(_env, schema);
                _genericRowMapper = genericMapper as ISchemaBoundRowMapper;

                if (parent.Stringify)
                {
                    _outputSchema = new SimpleSchema(_env,
                        new KeyValuePair<string, ColumnType>(DefaultColumnNames.FeatureContributions, TextType.Instance));
                    if (InputSchema.Schema.HasSlotNames(InputSchema.Feature.Index, InputSchema.Feature.Type.VectorSize))
                        InputSchema.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, InputSchema.Feature.Index,
                            ref _slotNames);
                    else
                        _slotNames = VBufferUtils.CreateEmpty<ReadOnlyMemory<char>>(InputSchema.Feature.Type.VectorSize);
                }
                else
                {
                    _outputSchema = new WtfSchema(_env, DefaultColumnNames.FeatureContributions,
                        new VectorType(NumberType.R4, schema.Feature.Type.AsVector),
                        InputSchema.Schema, InputSchema.Feature.Index);
                }

                _outputGenericSchema = _genericRowMapper.Schema;
                OutputSchema = new CompositeSchema(new ISchema[] { _outputGenericSchema, _outputSchema, });
            }

            /// <summary>
            /// Returns the input columns needed for the requested output columns.
            /// </summary>
            public Func<int, bool> GetDependencies(Func<int, bool> predicate)
            {
                for (int i = 0; i < OutputSchema.ColumnCount; i++)
                {
                    if (predicate(i))
                        return col => col == InputSchema.Feature.Index;
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
                        ? _parent.GetTextWtfGetter(input, InputSchema.Feature.Index, _slotNames)
                        : _parent.GetWtfGetter(input, InputSchema.Feature.Index);
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
                yield return RoleMappedSchema.ColumnRole.Feature.Bind(InputSchema.Feature.Name);
            }
        }

        private sealed class WtfSchema : ISchema
        {
            private readonly ISchema _parentSchema;
            private readonly IExceptionContext _ectx;
            private readonly string[] _names;
            private readonly ColumnType[] _types;
            private readonly Dictionary<string, int> _columnNameMap;
            private readonly int _featureCol;
            private readonly int _featureVectorSize;
            private readonly bool _hasSlotNames;

            public int ColumnCount => _types.Length;

            public WtfSchema(IExceptionContext ectx, string columnName, ColumnType columnType, ISchema parentSchema, int featureCol)
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
