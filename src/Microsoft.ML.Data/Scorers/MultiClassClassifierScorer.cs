// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections.Generic;
using System.Threading;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Model.Onnx;
using Microsoft.ML.Runtime.Model.Pfa;
using Microsoft.ML.Runtime.Numeric;
using Newtonsoft.Json.Linq;

[assembly: LoadableClass(typeof(MultiClassClassifierScorer),
    typeof(MultiClassClassifierScorer.Arguments), typeof(SignatureDataScorer),
    "Multi-Class Classifier Scorer", "MultiClassClassifierScorer", "MultiClassClassifier",
    "MultiClass", MetadataUtils.Const.ScoreColumnKind.MultiClassClassification)]

[assembly: LoadableClass(typeof(MultiClassClassifierScorer), null, typeof(SignatureLoadDataTransform),
    "Multi-Class Classifier Scorer", MultiClassClassifierScorer.LoaderSignature)]

[assembly: LoadableClass(typeof(ISchemaBindableMapper), typeof(MultiClassClassifierScorer.LabelNameBindableMapper), null, typeof(SignatureLoadModel),
    "Multi-Class Label-Name Mapper", MultiClassClassifierScorer.LabelNameBindableMapper.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    public sealed class MultiClassClassifierScorer : PredictedLabelScorerBase
    {
        // REVIEW: consider outputting probabilities when multi-class classifiers distinguish
        // between scores and probabilities (using IDistributionPredictor)
        public sealed class Arguments : ScorerArgumentsBase
        {
        }

        public const string LoaderSignature = "MultiClassScoreTrans";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "MULCLSCR",
                //verWrittenCur: 0x00010001, // Initial
                //verWrittenCur: 0x00010002, // ISchemaBindableMapper
                verWrittenCur: 0x00010003, // ISchemaBindableMapper update
                verReadableCur: 0x00010003,
                verWeCanReadBack: 0x00010003,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(MultiClassClassifierScorer).Assembly.FullName);
        }

        private const string RegistrationName = "MultiClassClassifierScore";

        /// <summary>
        /// This bindable mapper facilitates the serialization and rebinding of the special bound
        /// mapper that attaches the label metadata to the slot names of the output score column.
        /// </summary>
        // REVIEW: It seems like the attachment of metadata should be solvable in a manner
        // less ridiculously verbose than this.
        public sealed class LabelNameBindableMapper : ISchemaBindableMapper, ICanSaveModel, IBindableCanSavePfa, IBindableCanSaveOnnx
        {
            public const string LoaderSignature = "LabelSlotNameMapper";
            private const string _innerDir = "InnerMapper";
            private readonly ISchemaBindableMapper _bindable;
            private readonly VectorType _type;
            private readonly string _metadataKind;
            // In an ideal world this would be a value getter of the appropriate type. However, it is awkward
            // to have this class be generic due to restrictions on loadable classes, so we instead pay the
            // price of a handful of runtime casts.
            // REVIEW: Worth it to have this be abstract, with a nested generic implementation?
            // That seems like a bit much...
            private readonly Delegate _getter;
            private readonly IHost _host;
            private readonly Func<ISchemaBoundMapper, ColumnType, bool> _canWrap;

            public VectorType Type => _type;
            public bool CanSavePfa => (_bindable as ICanSavePfa)?.CanSavePfa == true;
            public bool CanSaveOnnx(OnnxContext ctx) => (_bindable as ICanSaveOnnx)?.CanSaveOnnx(ctx) == true;
            public ISchemaBindableMapper InnerBindable => _bindable;

            private static VersionInfo GetVersionInfo()
            {
                return new VersionInfo(
                    modelSignature: "LABNAMBM",
                    // verWrittenCur: 0x00010001, // Initial
                    verWrittenCur: 0x00010002, // Added metadataKind
                    verReadableCur: 0x00010002,
                    verWeCanReadBack: 0x00010001,
                    loaderSignature: LoaderSignature,
                    loaderAssemblyName: typeof(LabelNameBindableMapper).Assembly.FullName);
            }

            private const int VersionAddedMetadataKind = 0x00010002;

            private LabelNameBindableMapper(IHostEnvironment env, ISchemaBoundMapper mapper, VectorType type, Delegate getter,
                string metadataKind, Func<ISchemaBoundMapper, ColumnType, bool> canWrap)
                : this(env, mapper.Bindable, type, getter, metadataKind, canWrap)
            {
            }

            private LabelNameBindableMapper(IHostEnvironment env, ISchemaBindableMapper bindable, VectorType type, Delegate getter,
                string metadataKind, Func<ISchemaBoundMapper, ColumnType, bool> canWrap)
            {
                Contracts.AssertValue(env);
                _host = env.Register(LoaderSignature);
                _host.AssertValue(bindable);
                _host.AssertValue(type);
                _host.AssertValue(getter);
                _host.AssertNonEmpty(metadataKind);
                _host.AssertValueOrNull(canWrap);

                _bindable = bindable;
                _type = type;
                _getter = getter;
                _metadataKind = metadataKind;
                _canWrap = canWrap;
            }

            private LabelNameBindableMapper(IHost host, ModelLoadContext ctx)
            {
                Contracts.AssertValue(host);
                _host = host;
                _host.AssertValue(ctx);

                ctx.LoadModel<ISchemaBindableMapper, SignatureLoadModel>(_host, out _bindable, _innerDir);
                BinarySaver saver = new BinarySaver(_host, new BinarySaver.Arguments());
                ColumnType type;
                object value;
                _host.CheckDecode(saver.TryLoadTypeAndValue(ctx.Reader.BaseStream, out type, out value));
                _host.CheckDecode(type.IsVector);
                _host.CheckDecode(value != null);
                _type = type.AsVector;
                _getter = Utils.MarshalInvoke(DecodeInit<int>, _type.ItemType.RawType, value);
                _metadataKind = ctx.Header.ModelVerReadable >= VersionAddedMetadataKind ?
                    ctx.LoadNonEmptyString() : MetadataUtils.Kinds.SlotNames;
            }

            public ISchemaBindableMapper Clone(ISchemaBindableMapper inner)
            {
                return new LabelNameBindableMapper(_host, inner, _type, _getter, _metadataKind, _canWrap);
            }

            private Delegate DecodeInit<T>(object value)
            {
                _host.CheckDecode(value is VBuffer<T>);
                VBuffer<T> buffValue = (VBuffer<T>)value;
                ValueGetter<VBuffer<T>> buffGetter = (ref VBuffer<T> dst) => buffValue.CopyTo(ref dst);
                return buffGetter;
            }

            public static ISchemaBindableMapper Create(IHostEnvironment env, ModelLoadContext ctx)
            {
                Contracts.CheckValue(env, nameof(env));
                var h = env.Register(LoaderSignature);

                h.CheckValue(ctx, nameof(ctx));
                ctx.CheckAtModel(GetVersionInfo());

                // *** Binary format ***
                // byte[]: A chunk of data saving both the type and value of the label names, as saved by the BinarySaver.
                // int: string id of the metadata kind

                return h.Apply("Loading Model", ch => new LabelNameBindableMapper(h, ctx));
            }

            public void Save(ModelSaveContext ctx)
            {
                Contracts.CheckValue(ctx, nameof(ctx));
                ctx.CheckAtModel();
                ctx.SetVersionInfo(GetVersionInfo());

                // *** Binary format ***
                // byte[]: A chunk of data saving both the type and value of the label names, as saved by the BinarySaver.
                // int: string id of the metadata kind

                ctx.SaveModel(_bindable, _innerDir);
                Utils.MarshalActionInvoke(SaveCore<int>, _type.ItemType.RawType, ctx);
                ctx.SaveNonEmptyString(_metadataKind);
            }

            private void SaveCore<T>(ModelSaveContext ctx)
            {
                Contracts.Assert(_type.ItemType.RawType == typeof(T));
                Contracts.Assert(_getter is ValueGetter<VBuffer<T>>);

                var getter = (ValueGetter<VBuffer<T>>)_getter;
                var val = default(VBuffer<T>);
                getter(ref val);

                BinarySaver saver = new BinarySaver(_host, new BinarySaver.Arguments());
                int bytesWritten;
                if (!saver.TryWriteTypeAndValue<VBuffer<T>>(ctx.Writer.BaseStream, _type, ref val, out bytesWritten))
                    throw _host.Except("We do not know how to serialize label names of type '{0}'", _type.ItemType);
            }

            public void SaveAsPfa(BoundPfaContext ctx, RoleMappedSchema schema, string[] outputNames)
            {
                Contracts.CheckValue(ctx, nameof(ctx));
                Contracts.CheckValue(schema, nameof(schema));
                Contracts.Check(CanSavePfa, "Cannot be saved as PFA");
                Contracts.Assert(_bindable is IBindableCanSavePfa);
                ((IBindableCanSavePfa)_bindable).SaveAsPfa(ctx, schema, outputNames);
            }

            public bool SaveAsOnnx(OnnxContext ctx, RoleMappedSchema schema, string[] outputNames)
            {
                Contracts.CheckValue(ctx, nameof(ctx));
                Contracts.CheckValue(schema, nameof(schema));
                Contracts.Check(CanSaveOnnx(ctx), "Cannot be saved as ONNX.");
                Contracts.Assert(_bindable is IBindableCanSaveOnnx);
                return ((IBindableCanSaveOnnx)_bindable).SaveAsOnnx(ctx, schema, outputNames);
            }

            public ISchemaBoundMapper Bind(IHostEnvironment env, RoleMappedSchema schema)
            {
                var innerBound = _bindable.Bind(env, schema);
                if (_canWrap != null && !_canWrap(innerBound, _type))
                    return innerBound;
                Contracts.Assert(innerBound is ISchemaBoundRowMapper);
                return Utils.MarshalInvoke(CreateBound<int>, _type.ItemType.RawType, env, (ISchemaBoundRowMapper)innerBound, _type, _getter, _metadataKind, _canWrap);
            }

            public static ISchemaBoundMapper CreateBound<T>(IHostEnvironment env, ISchemaBoundRowMapper mapper, VectorType type, Delegate getter,
                string metadataKind, Func<ISchemaBoundMapper, ColumnType, bool> canWrap)
            {
                Contracts.AssertValue(env);
                env.AssertValue(mapper);
                env.AssertValue(type);
                env.AssertValue(getter);
                env.Assert(getter is ValueGetter<VBuffer<T>>);
                env.AssertNonEmpty(metadataKind);
                env.AssertValueOrNull(canWrap);

                return new Bound<T>(env, mapper, type, (ValueGetter<VBuffer<T>>)getter, metadataKind, canWrap);
            }

            private sealed class Bound<T> : ISchemaBoundRowMapper
            {
                private readonly IHost _host;
                /// <summary>The mapper we are wrapping.</summary>
                private readonly ISchemaBoundRowMapper _mapper;
                private readonly VectorType _labelNameType;
                private readonly string _metadataKind;
                private readonly ValueGetter<VBuffer<T>> _labelNameGetter;
                private readonly SchemaImpl _outSchema;
                // Lazily initialized by the property.
                private LabelNameBindableMapper _bindable;
                private readonly Func<ISchemaBoundMapper, ColumnType, bool> _canWrap;

                public Schema Schema => _outSchema.AsSchema;

                public RoleMappedSchema InputRoleMappedSchema => _mapper.InputRoleMappedSchema;
                public Schema InputSchema => _mapper.InputSchema;

                public ISchemaBindableMapper Bindable
                {
                    get
                    {
                        if (_bindable == null)
                        {
                            Interlocked.CompareExchange(ref _bindable,
                                new LabelNameBindableMapper(_host, _mapper, _labelNameType, _labelNameGetter, _metadataKind, _canWrap), null);
                        }
                        Contracts.AssertValue(_bindable);
                        return _bindable;
                    }
                }

                /// <summary>
                /// This is the constructor called for the initial wrapping.
                /// </summary>
                public Bound(IHostEnvironment env, ISchemaBoundRowMapper mapper, VectorType type, ValueGetter<VBuffer<T>> getter,
                    string metadataKind, Func<ISchemaBoundMapper, ColumnType, bool> canWrap)
                {
                    Contracts.CheckValue(env, nameof(env));
                    _host = env.Register(LoaderSignature);
                    _host.CheckValue(mapper, nameof(mapper));
                    _host.CheckValue(type, nameof(type));
                    _host.CheckValue(getter, nameof(getter));
                    _host.CheckNonEmpty(metadataKind, nameof(metadataKind));
                    _host.CheckValueOrNull(canWrap);

                    _mapper = mapper;

                    int scoreIdx;
                    bool result = mapper.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out scoreIdx);
                    if (!result)
                        throw env.ExceptParam(nameof(mapper), "Mapper did not have a '{0}' column", MetadataUtils.Const.ScoreValueKind.Score);

                    _labelNameType = type;
                    _labelNameGetter = getter;
                    _metadataKind = metadataKind;

                    _outSchema = new SchemaImpl(mapper.Schema, scoreIdx, _labelNameType, _labelNameGetter, _metadataKind);
                    _canWrap = canWrap;
                }

                public Func<int, bool> GetDependencies(Func<int, bool> predicate)
                {
                    return _mapper.GetDependencies(predicate);
                }

                public IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> GetInputColumnRoles()
                {
                    return _mapper.GetInputColumnRoles();
                }

                public IRow GetRow(IRow input, Func<int, bool> predicate, out Action disposer)
                {
                    var innerRow = _mapper.GetRow(input, predicate, out disposer);
                    return new RowImpl(innerRow, Schema);
                }

                private sealed class SchemaImpl : ISchema
                {
                    private readonly ISchema _parent;
                    private readonly int _scoreCol;
                    private readonly VectorType _labelNameType;
                    private readonly MetadataUtils.MetadataGetter<VBuffer<T>> _labelNameGetter;
                    private readonly string _metadataKind;

                    public Schema AsSchema { get; }

                    public int ColumnCount { get { return _parent.ColumnCount; } }

                    public SchemaImpl(ISchema parent, int col, VectorType type, ValueGetter<VBuffer<T>> getter, string metadataKind)
                    {
                        Contracts.AssertValue(parent);
                        Contracts.Assert(0 <= col && col < parent.ColumnCount);
                        Contracts.AssertValue(type);
                        Contracts.AssertValue(getter);
                        Contracts.Assert(type.ItemType.RawType == typeof(T));
                        Contracts.AssertNonEmpty(metadataKind);
                        Contracts.Assert(parent.GetMetadataTypeOrNull(metadataKind, col) == null);

                        _parent = parent;
                        _scoreCol = col;
                        _labelNameType = type;
                        // We change to this metadata variant of the getter to enable the marshal call to work.
                        _labelNameGetter = (int c, ref VBuffer<T> val) => getter(ref val);
                        _metadataKind = metadataKind;

                        AsSchema = Data.Schema.Create(this);
                    }

                    public bool TryGetColumnIndex(string name, out int col)
                    {
                        return _parent.TryGetColumnIndex(name, out col);
                    }

                    public string GetColumnName(int col)
                    {
                        return _parent.GetColumnName(col);
                    }

                    public ColumnType GetColumnType(int col)
                    {
                        return _parent.GetColumnType(col);
                    }

                    public IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypes(int col)
                    {
                        var result = _parent.GetMetadataTypes(col);
                        if (col == _scoreCol)
                            return result.Prepend(_labelNameType.GetPair(_metadataKind));
                        return result;
                    }

                    public ColumnType GetMetadataTypeOrNull(string kind, int col)
                    {
                        if (col == _scoreCol && kind == _metadataKind)
                            return _labelNameType;
                        return _parent.GetMetadataTypeOrNull(kind, col);
                    }

                    public void GetMetadata<TValue>(string kind, int col, ref TValue value)
                    {
                        if (col == _scoreCol && kind == _metadataKind)
                        {
                            _labelNameGetter.Marshal(0, ref value);
                            return;
                        }
                        _parent.GetMetadata(kind, col, ref value);
                    }
                }

                private sealed class RowImpl : IRow
                {
                    private readonly IRow _row;
                    private readonly Schema _schema;

                    public long Batch { get { return _row.Batch; } }
                    public long Position { get { return _row.Position; } }
                    // The schema is of course the only difference from _row.
                    public Schema Schema => _schema;

                    public RowImpl(IRow row, Schema schema)
                    {
                        Contracts.AssertValue(row);
                        Contracts.AssertValue(schema);

                        _row = row;
                        _schema = schema;
                    }

                    public bool IsColumnActive(int col)
                    {
                        return _row.IsColumnActive(col);
                    }

                    public ValueGetter<TValue> GetGetter<TValue>(int col)
                    {
                        return _row.GetGetter<TValue>(col);
                    }

                    public ValueGetter<UInt128> GetIdGetter()
                    {
                        return _row.GetIdGetter();
                    }
                }
            }
        }

        /// <summary>
        /// This function performs a number of checks on the inputs and, if appropriate and possible, will produce
        /// a mapper with slots names on the output score column properly mapped. If this is not possible for any
        /// reason, it will just return the input bound mapper.
        /// </summary>
        private static ISchemaBoundMapper WrapIfNeeded(IHostEnvironment env, ISchemaBoundMapper mapper, RoleMappedSchema trainSchema)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(mapper, nameof(mapper));
            env.CheckValueOrNull(trainSchema);

            // The idea is that we will take the key values from the train schema label, and present
            // them as slot name metadata. But there are a number of conditions for this to actually
            // happen, so we test those here. If these are not

            if (trainSchema == null || trainSchema.Label == null)
                return mapper; // We don't even have a label identified in a training schema.
            var keyType = trainSchema.Schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.KeyValues, trainSchema.Label.Index);
            if (keyType == null || !CanWrap(mapper, keyType))
                return mapper;

            // Great!! All checks pass.
            return Utils.MarshalInvoke(WrapCore<int>, keyType.ItemType.RawType, env, mapper, trainSchema);
        }

        /// <summary>
        /// This is a utility method used to determine whether <see cref="LabelNameBindableMapper"/>
        /// can or should be used to wrap <paramref name="mapper"/>. This will not throw, since the
        /// desired behavior in the event that it cannot be wrapped, is to just back off to the original
        /// "unwrapped" bound mapper.
        /// </summary>
        /// <param name="mapper">The mapper we are seeing if we can wrap</param>
        /// <param name="labelNameType">The type of the label names from the metadata (either
        /// originating from the key value metadata of the training label column, or deserialized
        /// from the model of a bindable mapper)</param>
        /// <returns>Whether we can call <see cref="LabelNameBindableMapper.CreateBound{T}"/> with
        /// this mapper and expect it to succeed</returns>
        public static bool CanWrap(ISchemaBoundMapper mapper, ColumnType labelNameType)
        {
            Contracts.AssertValue(mapper);
            Contracts.AssertValue(labelNameType);

            ISchemaBoundRowMapper rowMapper = mapper as ISchemaBoundRowMapper;
            if (rowMapper == null)
                return false; // We could cover this case, but it is of no practical worth as far as I see, so I decline to do so.

            ISchema outSchema = mapper.Schema;
            int scoreIdx;
            if (!outSchema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out scoreIdx))
                return false; // The mapper doesn't even publish a score column to attach the metadata to.
            if (outSchema.GetMetadataTypeOrNull(MetadataUtils.Kinds.SlotNames, scoreIdx) != null)
                return false; // The mapper publishes a score column, and already produces its own slot names.
            var scoreType = outSchema.GetColumnType(scoreIdx);

            // Check that the type is vector, and is of compatible size with the score output.
            return labelNameType.IsVector && labelNameType.VectorSize == scoreType.VectorSize;
        }

        public static ISchemaBoundMapper WrapCore<T>(IHostEnvironment env, ISchemaBoundMapper mapper, RoleMappedSchema trainSchema)
        {
            Contracts.AssertValue(env);
            env.AssertValue(mapper);
            env.AssertValue(trainSchema);
            env.Assert(mapper is ISchemaBoundRowMapper);

            // Key values from the training schema label, will map to slot names of the score output.
            var type = trainSchema.Schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.KeyValues, trainSchema.Label.Index);
            env.AssertValue(type);
            env.Assert(type.IsVector);

            // Wrap the fetching of the metadata as a simple getter.
            ValueGetter<VBuffer<T>> getter =
                (ref VBuffer<T> value) =>
                {
                    trainSchema.Schema.GetMetadata(MetadataUtils.Kinds.KeyValues,
                        trainSchema.Label.Index, ref value);
                };

            return LabelNameBindableMapper.CreateBound<T>(env, (ISchemaBoundRowMapper)mapper, type.AsVector, getter, MetadataUtils.Kinds.SlotNames, CanWrap);
        }

        public MultiClassClassifierScorer(IHostEnvironment env, Arguments args, IDataView data, ISchemaBoundMapper mapper, RoleMappedSchema trainSchema)
            : base(args, env, data, WrapIfNeeded(env, mapper, trainSchema), trainSchema, RegistrationName, MetadataUtils.Const.ScoreColumnKind.MultiClassClassification,
                MetadataUtils.Const.ScoreValueKind.Score, OutputTypeMatches, GetPredColType)
        {
        }

        private MultiClassClassifierScorer(IHostEnvironment env, MultiClassClassifierScorer transform, IDataView newSource)
            : base(env, transform, newSource, RegistrationName)
        {
        }

        private MultiClassClassifierScorer(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, OutputTypeMatches, GetPredColType)
        {
            // *** Binary format ***
            // <base info>
        }

        public static MultiClassClassifierScorer Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new MultiClassClassifierScorer(h, ctx, input));
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            Contracts.AssertValue(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base info>

            base.SaveCore(ctx);
        }

        public override IDataTransform ApplyToData(IHostEnvironment env, IDataView newSource)
        {
            Contracts.CheckValue(env, nameof(env));
            Contracts.CheckValue(newSource, nameof(newSource));

            return new MultiClassClassifierScorer(env, this, newSource);
        }

        protected override Delegate GetPredictedLabelGetter(IRow output, out Delegate scoreGetter)
        {
            Host.AssertValue(output);
            Host.Assert(output.Schema == Bindings.RowMapper.Schema);
            Host.Assert(output.IsColumnActive(Bindings.ScoreColumnIndex));

            ValueGetter<VBuffer<Float>> mapperScoreGetter = output.GetGetter<VBuffer<Float>>(Bindings.ScoreColumnIndex);

            long cachedPosition = -1;
            VBuffer<Float> score = default(VBuffer<Float>);
            int scoreLength = Bindings.PredColType.KeyCount;

            ValueGetter<uint> predFn =
                (ref uint dst) =>
                {
                    EnsureCachedPosition(ref cachedPosition, ref score, output, mapperScoreGetter);
                    Host.Check(score.Length == scoreLength);
                    int index = VectorUtils.ArgMax(ref score);
                    if (index < 0)
                        dst = 0;
                    else
                        dst = (uint)index + 1;
                };
            ValueGetter<VBuffer<Float>> scoreFn =
                (ref VBuffer<Float> dst) =>
                {
                    EnsureCachedPosition(ref cachedPosition, ref score, output, mapperScoreGetter);
                    Host.Check(score.Length == scoreLength);
                    score.CopyTo(ref dst);
                };

            scoreGetter = scoreFn;
            return predFn;
        }

        protected override JToken PredictedLabelPfa(string[] mapperOutputs)
        {
            Contracts.Assert(Utils.Size(mapperOutputs) == 1);
            return PfaUtils.Call("a.argmax", mapperOutputs[0]);
        }

        private static ColumnType GetPredColType(ColumnType scoreType, ISchemaBoundRowMapper mapper)
        {
            return new KeyType(DataKind.U4, 0, scoreType.VectorSize);
        }

        private static bool OutputTypeMatches(ColumnType scoreType)
        {
            return scoreType.IsKnownSizeVector && scoreType.ItemType == NumberType.Float;
        }
    }
}
