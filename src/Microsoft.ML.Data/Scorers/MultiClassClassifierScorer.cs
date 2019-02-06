// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Threading;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Model.Onnx;
using Microsoft.ML.Model.Pfa;
using Microsoft.ML.Numeric;
using Newtonsoft.Json.Linq;
using Float = System.Single;

[assembly: LoadableClass(typeof(MultiClassClassifierScorer),
    typeof(MultiClassClassifierScorer.Arguments), typeof(SignatureDataScorer),
    "Multi-Class Classifier Scorer", "MultiClassClassifierScorer", "MultiClassClassifier",
    "MultiClass", MetadataUtils.Const.ScoreColumnKind.MultiClassClassification)]

[assembly: LoadableClass(typeof(MultiClassClassifierScorer), null, typeof(SignatureLoadDataTransform),
    "Multi-Class Classifier Scorer", MultiClassClassifierScorer.LoaderSignature)]

[assembly: LoadableClass(typeof(ISchemaBindableMapper), typeof(MultiClassClassifierScorer.LabelNameBindableMapper), null, typeof(SignatureLoadModel),
    "Multi-Class Label-Name Mapper", MultiClassClassifierScorer.LabelNameBindableMapper.LoaderSignature)]

namespace Microsoft.ML.Data
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

            internal ISchemaBindableMapper Bindable => _bindable;

            public VectorType Type => _type;
            bool ICanSavePfa.CanSavePfa => (_bindable as ICanSavePfa)?.CanSavePfa == true;
            bool ICanSaveOnnx.CanSaveOnnx(OnnxContext ctx) => (_bindable as ICanSaveOnnx)?.CanSaveOnnx(ctx) == true;

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
                _type = type as VectorType;
                _host.CheckDecode(_type != null);
                _host.CheckDecode(value != null);
                _getter = Utils.MarshalInvoke(DecodeInit<int>, _type.ItemType.RawType, value);
                _metadataKind = ctx.Header.ModelVerReadable >= VersionAddedMetadataKind ?
                    ctx.LoadNonEmptyString() : MetadataUtils.Kinds.SlotNames;
            }

            private Delegate DecodeInit<T>(object value)
            {
                _host.CheckDecode(value is VBuffer<T>);
                VBuffer<T> buffValue = (VBuffer<T>)value;
                ValueGetter<VBuffer<T>> buffGetter = (ref VBuffer<T> dst) => buffValue.CopyTo(ref dst);
                return buffGetter;
            }

            /// <summary>
            /// Method corresponding to <see cref="SignatureLoadModel"/>.
            /// </summary>
            private static ISchemaBindableMapper Create(IHostEnvironment env, ModelLoadContext ctx)
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

            internal ISchemaBindableMapper Clone(ISchemaBindableMapper inner)
            {
                return new LabelNameBindableMapper(_host, inner, _type, _getter, _metadataKind, _canWrap);
            }

            void IBindableCanSavePfa.SaveAsPfa(BoundPfaContext ctx, RoleMappedSchema schema, string[] outputNames)
            {
                Contracts.CheckValue(ctx, nameof(ctx));
                Contracts.CheckValue(schema, nameof(schema));
                Contracts.Check(((ICanSavePfa)this).CanSavePfa, "Cannot be saved as PFA");
                Contracts.Assert(_bindable is IBindableCanSavePfa);
                ((IBindableCanSavePfa)_bindable).SaveAsPfa(ctx, schema, outputNames);
            }

            bool IBindableCanSaveOnnx.SaveAsOnnx(OnnxContext ctx, RoleMappedSchema schema, string[] outputNames)
            {
                Contracts.CheckValue(ctx, nameof(ctx));
                Contracts.CheckValue(schema, nameof(schema));
                Contracts.Check(((ICanSaveOnnx)this).CanSaveOnnx(ctx), "Cannot be saved as ONNX.");
                Contracts.Assert(_bindable is IBindableCanSaveOnnx);
                return ((IBindableCanSaveOnnx)_bindable).SaveAsOnnx(ctx, schema, outputNames);
            }

            ISchemaBoundMapper ISchemaBindableMapper.Bind(IHostEnvironment env, RoleMappedSchema schema)
            {
                var innerBound = _bindable.Bind(env, schema);
                if (_canWrap != null && !_canWrap(innerBound, _type))
                    return innerBound;
                Contracts.Assert(innerBound is ISchemaBoundRowMapper);
                return Utils.MarshalInvoke(CreateBound<int>, _type.ItemType.RawType, env, (ISchemaBoundRowMapper)innerBound, _type, _getter, _metadataKind, _canWrap);
            }

            internal static ISchemaBoundMapper CreateBound<T>(IHostEnvironment env, ISchemaBoundRowMapper mapper, VectorType type, Delegate getter,
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
                // Lazily initialized by the property.
                private LabelNameBindableMapper _bindable;
                private readonly Func<ISchemaBoundMapper, ColumnType, bool> _canWrap;

                public RoleMappedSchema InputRoleMappedSchema => _mapper.InputRoleMappedSchema;
                public Schema InputSchema => _mapper.InputSchema;
                public Schema OutputSchema { get; }

                public ISchemaBindableMapper Bindable
                {
                    get
                    {
                        return _bindable ??
                            Interlocked.CompareExchange(ref _bindable,
                                new LabelNameBindableMapper(_host, _mapper, _labelNameType, _labelNameGetter, _metadataKind, _canWrap), null) ??
                            _bindable;
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
                    bool result = mapper.OutputSchema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out scoreIdx);
                    if (!result)
                        throw env.ExceptParam(nameof(mapper), "Mapper did not have a '{0}' column", MetadataUtils.Const.ScoreValueKind.Score);

                    _labelNameType = type;
                    _labelNameGetter = getter;
                    _metadataKind = metadataKind;
                    _canWrap = canWrap;

                    OutputSchema = DecorateOutputSchema(mapper.OutputSchema, scoreIdx, _labelNameType, _labelNameGetter, _metadataKind);
                }

                /// <summary>
                /// Append label names to score column as its metadata.
                /// </summary>
                private Schema DecorateOutputSchema(Schema partialSchema, int scoreColumnIndex, VectorType labelNameType,
                    ValueGetter<VBuffer<T>> labelNameGetter, string labelNameKind)
                {
                    var builder = new SchemaBuilder();
                    // Sequentially add columns so that the order of them is not changed comparing with the schema in the mapper
                    // that computes score column.
                    for (int i = 0; i < partialSchema.Count; ++i)
                    {
                        var meta = new MetadataBuilder();
                        if (i == scoreColumnIndex)
                        {
                            // Add label names for score column.
                            meta.Add(partialSchema[i].Metadata, selector: s => s != labelNameKind);
                            meta.Add(labelNameKind, labelNameType, labelNameGetter);
                        }
                        else
                        {
                            // Copy all existing metadata because this transform only affects score column.
                            meta.Add(partialSchema[i].Metadata, selector: s => true);
                        }
                        // Instead of appending extra metadata to the existing score column, we create new one because
                        // metadata is read-only.
                        builder.AddColumn(partialSchema[i].Name, partialSchema[i].Type, meta.GetMetadata());
                    }
                    return builder.GetSchema();
                }

                public Func<int, bool> GetDependencies(Func<int, bool> predicate) => _mapper.GetDependencies(predicate);

                public IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> GetInputColumnRoles() => _mapper.GetInputColumnRoles();

                public Row GetRow(Row input, Func<int, bool> predicate)
                {
                    var innerRow = _mapper.GetRow(input, predicate);
                    return new RowImpl(innerRow, OutputSchema);
                }

                private sealed class RowImpl : WrappingRow
                {
                    private readonly Schema _schema;

                    // The schema is of course the only difference from _row.
                    public override Schema Schema => _schema;

                    public RowImpl(Row row, Schema schema)
                        : base(row)
                    {
                        Contracts.AssertValue(row);
                        Contracts.AssertValue(schema);

                        _schema = schema;
                    }

                    public override bool IsColumnActive(int col) => Input.IsColumnActive(col);

                    public override ValueGetter<TValue> GetGetter<TValue>(int col) => Input.GetGetter<TValue>(col);
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

            if (trainSchema?.Label == null)
                return mapper; // We don't even have a label identified in a training schema.
            var keyType = trainSchema.Label.Value.Metadata.Schema.GetColumnOrNull(MetadataUtils.Kinds.KeyValues)?.Type as VectorType;
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
        internal static bool CanWrap(ISchemaBoundMapper mapper, ColumnType labelNameType)
        {
            Contracts.AssertValue(mapper);
            Contracts.AssertValue(labelNameType);

            ISchemaBoundRowMapper rowMapper = mapper as ISchemaBoundRowMapper;
            if (rowMapper == null)
                return false; // We could cover this case, but it is of no practical worth as far as I see, so I decline to do so.

            var outSchema = mapper.OutputSchema;
            int scoreIdx;
            var scoreCol = outSchema.GetColumnOrNull(MetadataUtils.Const.ScoreValueKind.Score);
            if (!outSchema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out scoreIdx))
                return false; // The mapper doesn't even publish a score column to attach the metadata to.
            if (outSchema[scoreIdx].Metadata.Schema.GetColumnOrNull(MetadataUtils.Kinds.SlotNames)?.Type != null)
                return false; // The mapper publishes a score column, and already produces its own slot names.
            var scoreType = outSchema[scoreIdx].Type;

            // Check that the type is vector, and is of compatible size with the score output.
            return labelNameType is VectorType vectorType && vectorType.Size == scoreType.GetVectorSize();
        }

        internal static ISchemaBoundMapper WrapCore<T>(IHostEnvironment env, ISchemaBoundMapper mapper, RoleMappedSchema trainSchema)
        {
            Contracts.AssertValue(env);
            env.AssertValue(mapper);
            env.AssertValue(trainSchema);
            env.Assert(mapper is ISchemaBoundRowMapper);

            // Key values from the training schema label, will map to slot names of the score output.
            var type = trainSchema.Label.Value.Metadata.Schema.GetColumnOrNull(MetadataUtils.Kinds.KeyValues)?.Type;
            env.AssertValue(type);
            env.Assert(type is VectorType);

            // Wrap the fetching of the metadata as a simple getter.
            ValueGetter<VBuffer<T>> getter =
                (ref VBuffer<T> value) =>
                {
                    trainSchema.Label.Value.GetKeyValues(ref value);
                };

            return LabelNameBindableMapper.CreateBound<T>(env, (ISchemaBoundRowMapper)mapper, type as VectorType, getter, MetadataUtils.Kinds.SlotNames, CanWrap);
        }

        [BestFriend]
        internal MultiClassClassifierScorer(IHostEnvironment env, Arguments args, IDataView data, ISchemaBoundMapper mapper, RoleMappedSchema trainSchema)
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

        /// <summary>
        /// Corresponds to <see cref="SignatureLoadDataTransform"/>.
        /// </summary>
        private static MultiClassClassifierScorer Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new MultiClassClassifierScorer(h, ctx, input));
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            Contracts.AssertValue(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base info>

            base.SaveCore(ctx);
        }

        private protected override IDataTransform ApplyToDataCore(IHostEnvironment env, IDataView newSource)
        {
            Contracts.CheckValue(env, nameof(env));
            Contracts.CheckValue(newSource, nameof(newSource));

            return new MultiClassClassifierScorer(env, this, newSource);
        }

        protected override Delegate GetPredictedLabelGetter(Row output, out Delegate scoreGetter)
        {
            Host.AssertValue(output);
            Host.Assert(output.Schema == Bindings.RowMapper.OutputSchema);
            Host.Assert(output.IsColumnActive(Bindings.ScoreColumnIndex));

            ValueGetter<VBuffer<Float>> mapperScoreGetter = output.GetGetter<VBuffer<Float>>(Bindings.ScoreColumnIndex);

            long cachedPosition = -1;
            VBuffer<Float> score = default;
            int scoreLength = Bindings.PredColType.GetKeyCountAsInt32(Host);

            ValueGetter<uint> predFn =
                (ref uint dst) =>
                {
                    EnsureCachedPosition(ref cachedPosition, ref score, output, mapperScoreGetter);
                    Host.Check(score.Length == scoreLength);
                    int index = VectorUtils.ArgMax(in score);
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

        private protected override JToken PredictedLabelPfa(string[] mapperOutputs)
        {
            Contracts.Assert(Utils.Size(mapperOutputs) == 1);
            return PfaUtils.Call("a.argmax", mapperOutputs[0]);
        }

        private static ColumnType GetPredColType(ColumnType scoreType, ISchemaBoundRowMapper mapper) => new KeyType(typeof(uint), scoreType.GetVectorSize());

        private static bool OutputTypeMatches(ColumnType scoreType) =>
            scoreType is VectorType vectorType && vectorType.IsKnownSize && vectorType.ItemType == NumberType.Float;
    }
}
