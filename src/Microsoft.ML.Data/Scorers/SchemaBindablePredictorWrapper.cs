// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma warning disable 420 // volatile with Interlocked.CompareExchange

using Float = System.Single;

using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Model.Onnx;
using Microsoft.ML.Runtime.Model.Pfa;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Newtonsoft.Json.Linq;

[assembly: LoadableClass(typeof(SchemaBindablePredictorWrapper), null, typeof(SignatureLoadModel),
    "Bindable Mapper", SchemaBindablePredictorWrapper.LoaderSignature)]

[assembly: LoadableClass(typeof(SchemaBindableQuantileRegressionPredictor), null, typeof(SignatureLoadModel),
    "Regression Bindable Mapper", SchemaBindableQuantileRegressionPredictor.LoaderSignature)]

[assembly: LoadableClass(typeof(SchemaBindableBinaryPredictorWrapper), null, typeof(SignatureLoadModel),
    "Binary Classification Bindable Mapper", SchemaBindableBinaryPredictorWrapper.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    // REVIEW: Consider implementing ICanSaveAs(Code/Text/etc.) for these classes as well.
    /// <summary>
    /// This is a base class for wrapping <see cref="IPredictor"/>s in an <see cref="ISchemaBindableMapper"/>.
    /// </summary>
    public abstract class SchemaBindablePredictorWrapperBase : ISchemaBindableMapper, ICanSaveModel, ICanSaveSummary,
        IBindableCanSavePfa, IBindableCanSaveOnnx
    {
        // The ctor guarantees that Predictor is non-null. It also ensures that either
        // ValueMapper or FloatPredictor is non-null (or both). With these guarantees,
        // the score value type (_scoreType) can be determined.
        protected readonly IPredictor Predictor;
        protected readonly IValueMapper ValueMapper;
        protected readonly ColumnType ScoreType;

        public bool CanSavePfa => (ValueMapper as ICanSavePfa)?.CanSavePfa == true;

        public bool CanSaveOnnx(OnnxContext ctx) => (ValueMapper as ICanSaveOnnx)?.CanSaveOnnx(ctx) == true;

        public SchemaBindablePredictorWrapperBase(IPredictor predictor)
        {
            // REVIEW: Eventually drop support for predictors that don't implement IValueMapper.
            Contracts.CheckValue(predictor, nameof(predictor));
            Predictor = predictor;
            ScoreType = GetScoreType(Predictor, out ValueMapper);
        }

        private static ColumnType GetScoreType(IPredictor predictor, out IValueMapper valueMapper)
        {
            Contracts.AssertValue(predictor);

            valueMapper = predictor as IValueMapper;
            if (valueMapper != null)
                return valueMapper.OutputType;
            throw Contracts.Except(
                "Predictor score type cannot be determined since it doesn't implement IValueMapper");
        }

        protected SchemaBindablePredictorWrapperBase(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.AssertValue(ctx);

            // *** Binary format ***
            // <nothing>

            ctx.LoadModel<IPredictor, SignatureLoadModel>(env, out Predictor, ModelFileUtils.DirPredictor);
            ScoreType = GetScoreType(Predictor, out ValueMapper);
        }

        public virtual void Save(ModelSaveContext ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();

            // *** Binary format ***
            // <nothing>

            ctx.SaveModel(Predictor, ModelFileUtils.DirPredictor);
        }

        public virtual void SaveAsPfa(BoundPfaContext ctx, RoleMappedSchema schema, string[] outputNames)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.Assert(ValueMapper is ISingleCanSavePfa);
            var mapper = (ISingleCanSavePfa)ValueMapper;

            ctx.Hide(outputNames);
        }

        public virtual bool SaveAsOnnx(OnnxContext ctx, RoleMappedSchema schema, string[] outputNames) => false;

        public ISchemaBoundMapper Bind(IHostEnvironment env, RoleMappedSchema schema)
        {
            Contracts.CheckValue(env, nameof(env));

            using (var ch = env.Register("SchemaBindableWrapper").Start("Bind"))
            {
                ch.CheckValue(schema, nameof(schema));
                if (schema.Feature != null)
                {
                    // Ensure that the feature column type is compatible with the needed input type.
                    var type = schema.Feature.Type;
                    var typeIn = ValueMapper != null ? ValueMapper.InputType : new VectorType(NumberType.Float);
                    if (type != typeIn)
                    {
                        if (!type.ItemType.Equals(typeIn.ItemType))
                            throw ch.Except("Incompatible features column type item type: '{0}' vs '{1}'", type.ItemType, typeIn.ItemType);
                        if (type.IsVector != typeIn.IsVector)
                            throw ch.Except("Incompatible features column type: '{0}' vs '{1}'", type, typeIn);
                        // typeIn can legally have unknown size.
                        if (type.VectorSize != typeIn.VectorSize && typeIn.VectorSize > 0)
                            throw ch.Except("Incompatible features column type: '{0}' vs '{1}'", type, typeIn);
                    }
                }
                return BindCore(ch, schema);
            }
        }

        protected abstract ISchemaBoundMapper BindCore(IChannel ch, RoleMappedSchema schema);

        protected virtual Delegate GetPredictionGetter(IRow input, int colSrc)
        {
            Contracts.AssertValue(input);
            Contracts.Assert(0 <= colSrc && colSrc < input.Schema.ColumnCount);

            var typeSrc = input.Schema.GetColumnType(colSrc);
            Func<IRow, int, ValueGetter<int>> del = GetValueGetter<int, int>;
            var meth = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(typeSrc.RawType, ScoreType.RawType);
            return (Delegate)meth.Invoke(this, new object[] { input, colSrc });
        }

        private ValueGetter<TDst> GetValueGetter<TSrc, TDst>(IRow input, int colSrc)
        {
            Contracts.AssertValue(input);
            Contracts.Assert(ValueMapper != null);

            var featureGetter = input.GetGetter<TSrc>(colSrc);
            var map = ValueMapper.GetMapper<TSrc, TDst>();
            var features = default(TSrc);
            return
                (ref TDst dst) =>
                {
                    featureGetter(ref features);
                    map(in features, ref dst);
                };
        }

        public void SaveSummary(TextWriter writer, RoleMappedSchema schema)
        {
            var summarySaver = Predictor as ICanSaveSummary;
            if (summarySaver == null)
                writer.WriteLine("{0} does not support saving summaries", Predictor);
            else
                summarySaver.SaveSummary(writer, schema);
        }

        /// <summary>
        /// The <see cref="ISchemaBoundRowMapper"/> implementation for predictor wrappers that produce a
        /// single output column. Note that the Bindable wrapper should do any input schema validation.
        /// This class doesn't care. It DOES care that the role mapped schema specifies a unique Feature column.
        /// It also requires that the output schema has ColumnCount == 1.
        /// </summary>
        protected sealed class SingleValueRowMapper : ISchemaBoundRowMapper
        {
            private readonly SchemaBindablePredictorWrapperBase _parent;

            public RoleMappedSchema InputRoleMappedSchema { get; }
            public Schema Schema { get; }
            public ISchemaBindableMapper Bindable => _parent;

            public SingleValueRowMapper(RoleMappedSchema schema, SchemaBindablePredictorWrapperBase parent, Schema outputSchema)
            {
                Contracts.AssertValue(schema);
                Contracts.AssertValue(parent);
                Contracts.AssertValue(schema.Feature);
                Contracts.Assert(outputSchema.ColumnCount == 1);

                _parent = parent;
                InputRoleMappedSchema = schema;
                Schema = outputSchema;
            }

            public Func<int, bool> GetDependencies(Func<int, bool> predicate)
            {
                for (int i = 0; i < Schema.ColumnCount; i++)
                {
                    if (predicate(i))
                        return col => col == InputRoleMappedSchema.Feature.Index;
                }
                return col => false;
            }

            public IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> GetInputColumnRoles()
            {
                yield return RoleMappedSchema.ColumnRole.Feature.Bind(InputRoleMappedSchema.Feature.Name);
            }

            public Schema InputSchema => InputRoleMappedSchema.Schema;

            public IRow GetRow(IRow input, Func<int, bool> predicate, out Action disposer)
            {
                Contracts.AssertValue(input);
                Contracts.AssertValue(predicate);

                var getters = new Delegate[1];
                if (predicate(0))
                    getters[0] = _parent.GetPredictionGetter(input, InputRoleMappedSchema.Feature.Index);
                disposer = null;
                return new SimpleRow(Schema, input, getters);
            }
        }
    }

    /// <summary>
    /// This class is a wrapper for all <see cref="IPredictor"/>s except for quantile regression predictors,
    /// and calibrated binary classification predictors.
    /// </summary>
    public sealed class SchemaBindablePredictorWrapper : SchemaBindablePredictorWrapperBase
    {
        public const string LoaderSignature = "SchemaBindableWrapper";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "SCH BIND",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // ISchemaBindableWrapper update
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010002,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(SchemaBindablePredictorWrapper).Assembly.FullName);
        }

        private readonly string _scoreColumnKind;

        public SchemaBindablePredictorWrapper(IPredictor predictor)
            : base(predictor)
        {
            _scoreColumnKind = GetScoreColumnKind(Predictor);
        }

        private SchemaBindablePredictorWrapper(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, ctx)
        {
            _scoreColumnKind = GetScoreColumnKind(Predictor);
        }

        public static SchemaBindablePredictorWrapper Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new SchemaBindablePredictorWrapper(env, ctx);
        }

        public override void Save(ModelSaveContext ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            ctx.SetVersionInfo(GetVersionInfo());
            base.Save(ctx);
        }

        public override void SaveAsPfa(BoundPfaContext ctx, RoleMappedSchema schema, string[] outputNames)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.Assert(ValueMapper is ISingleCanSavePfa);
            Contracts.AssertValue(schema.Feature);
            Contracts.Assert(Utils.Size(outputNames) == 1); // Score.
            var mapper = (ISingleCanSavePfa)ValueMapper;
            // If the features column was not produced, we must hide the outputs.
            var featureToken = ctx.TokenOrNullForName(schema.Feature.Name);
            if (featureToken == null)
                ctx.Hide(outputNames);
            var scoreToken = mapper.SaveAsPfa(ctx, featureToken);
            ctx.DeclareVar(outputNames[0], scoreToken);
        }

        public override bool SaveAsOnnx(OnnxContext ctx, RoleMappedSchema schema, string[] outputNames)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.Assert(ValueMapper is ISingleCanSaveOnnx);
            Contracts.AssertValue(schema.Feature);
            Contracts.Assert(Utils.Size(outputNames) <= 2); // PredictedLabel and/or Score.
            var mapper = (ISingleCanSaveOnnx)ValueMapper;
            if (!ctx.ContainsColumn(schema.Feature.Name))
                return false;

            Contracts.Assert(ctx.ContainsColumn(schema.Feature.Name));

            return mapper.SaveAsOnnx(ctx, outputNames, ctx.GetVariableName(schema.Feature.Name));
        }

        protected override ISchemaBoundMapper BindCore(IChannel ch, RoleMappedSchema schema)
        {
            var outputSchema = Schema.Create(new ScoreMapperSchema(ScoreType, _scoreColumnKind));
            return new SingleValueRowMapper(schema, this, outputSchema);
        }

        private static string GetScoreColumnKind(IPredictor predictor)
        {
            Contracts.AssertValue(predictor);

            switch (predictor.PredictionKind)
            {
                case PredictionKind.BinaryClassification:
                    return MetadataUtils.Const.ScoreColumnKind.BinaryClassification;
                case PredictionKind.MultiClassClassification:
                    return MetadataUtils.Const.ScoreColumnKind.MultiClassClassification;
                case PredictionKind.Regression:
                    return MetadataUtils.Const.ScoreColumnKind.Regression;
                case PredictionKind.MultiOutputRegression:
                    return MetadataUtils.Const.ScoreColumnKind.MultiOutputRegression;
                case PredictionKind.Ranking:
                    return MetadataUtils.Const.ScoreColumnKind.Ranking;
                case PredictionKind.AnomalyDetection:
                    return MetadataUtils.Const.ScoreColumnKind.AnomalyDetection;
                case PredictionKind.Clustering:
                    return MetadataUtils.Const.ScoreColumnKind.Clustering;
                default:
                    throw Contracts.Except("Unknown prediction kind, can't map to score column kind: {0}", predictor.PredictionKind);
            }
        }
    }

    /// <summary>
    /// This is an <see cref="ISchemaBindableMapper"/> wrapper for calibrated binary classification predictors.
    /// They need a separate wrapper because they return two values instead of one: the raw score and the probability.
    /// </summary>
    public sealed class SchemaBindableBinaryPredictorWrapper : SchemaBindablePredictorWrapperBase
    {
        public const string LoaderSignature = "BinarySchemaBindable";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "BINSCHBD",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // ISchemaBindableWrapper update
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010002,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(SchemaBindableBinaryPredictorWrapper).Assembly.FullName);
        }

        private readonly IValueMapperDist _distMapper;

        public SchemaBindableBinaryPredictorWrapper(IPredictor predictor)
            : base(predictor)
        {
            CheckValid(out _distMapper);
        }

        private SchemaBindableBinaryPredictorWrapper(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, ctx)
        {
            CheckValid(out _distMapper);
        }

        public static SchemaBindableBinaryPredictorWrapper Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new SchemaBindableBinaryPredictorWrapper(env, ctx);
        }

        public override void Save(ModelSaveContext ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            ctx.SetVersionInfo(GetVersionInfo());
            base.Save(ctx);
        }

        public override void SaveAsPfa(BoundPfaContext ctx, RoleMappedSchema schema, string[] outputNames)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.Assert(ValueMapper is IDistCanSavePfa);
            Contracts.AssertValue(schema.Feature);
            Contracts.Assert(Utils.Size(outputNames) == 2); // Score and prob.
            var mapper = (IDistCanSavePfa)ValueMapper;
            // If the features column was not produced, we must hide the outputs.
            string featureToken = ctx.TokenOrNullForName(schema.Feature.Name);
            if (featureToken == null)
                ctx.Hide(outputNames);

            JToken scoreToken;
            JToken probToken;
            mapper.SaveAsPfa(ctx, featureToken, outputNames[0], out scoreToken, outputNames[1], out probToken);
            Contracts.Assert(ctx.TokenOrNullForName(outputNames[0]) == scoreToken.ToString());
            Contracts.Assert(ctx.TokenOrNullForName(outputNames[1]) == probToken.ToString());
        }

        public override bool SaveAsOnnx(OnnxContext ctx, RoleMappedSchema schema, string[] outputNames)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            Contracts.CheckValue(schema, nameof(schema));

            var mapper = ValueMapper as ISingleCanSaveOnnx;
            Contracts.CheckValue(mapper, nameof(mapper));
            Contracts.AssertValue(schema.Feature);
            Contracts.Assert(Utils.Size(outputNames) == 3); // Predicted Label, Score and Probablity.

            if (!ctx.ContainsColumn(schema.Feature.Name))
                return false;

            Contracts.Assert(ctx.ContainsColumn(schema.Feature.Name));

            return mapper.SaveAsOnnx(ctx, outputNames, ctx.GetVariableName(schema.Feature.Name));
        }

        private void CheckValid(out IValueMapperDist distMapper)
        {
            Contracts.Check(ScoreType == NumberType.Float, "Expected predictor result type to be Float");

            distMapper = Predictor as IValueMapperDist;
            if (distMapper == null)
                throw Contracts.Except("Predictor does not provide probabilities");

            // REVIEW: In theory the restriction on input type could be relaxed at the expense
            // of more complicated code in CalibratedRowMapper.GetGetters. Not worth it at this point
            // and no good way to test it.
            Contracts.Check(distMapper.InputType.IsVector && distMapper.InputType.ItemType == NumberType.Float,
                "Invalid input type for the IValueMapperDist");
            Contracts.Check(distMapper.DistType == NumberType.Float,
                "Invalid probability type for the IValueMapperDist");
        }

        protected override ISchemaBoundMapper BindCore(IChannel ch, RoleMappedSchema schema)
        {
            if (Predictor.PredictionKind != PredictionKind.BinaryClassification)
                ch.Warning("Scoring predictor of kind '{0}' as '{1}'.", Predictor.PredictionKind, PredictionKind.BinaryClassification);

            // For distribution mappers, produce both score and probability.
            Contracts.AssertValue(_distMapper);
            return new CalibratedRowMapper(schema, this);
        }

        /// <summary>
        /// The <see cref="ISchemaBoundRowMapper"/> implementation for distribution predictor wrappers that produce
        /// two Float-valued output columns. Note that the Bindable wrapper does input schema validation.
        /// </summary>
        private sealed class CalibratedRowMapper : ISchemaBoundRowMapper
        {
            private readonly SchemaBindableBinaryPredictorWrapper _parent;

            public RoleMappedSchema InputRoleMappedSchema { get; }
            public Schema InputSchema => InputRoleMappedSchema.Schema;

            public Schema Schema { get; }
            public ISchemaBindableMapper Bindable => _parent;

            public CalibratedRowMapper(RoleMappedSchema schema, SchemaBindableBinaryPredictorWrapper parent)
            {
                Contracts.AssertValue(parent);
                Contracts.Assert(parent._distMapper != null);
                Contracts.AssertValue(schema);
                Contracts.AssertValueOrNull(schema.Feature);

                _parent = parent;
                InputRoleMappedSchema = schema;
                Schema = Schema.Create(new BinaryClassifierSchema());

                if (schema.Feature != null)
                {
                    var typeSrc = InputRoleMappedSchema.Feature.Type;
                    Contracts.Check(typeSrc.IsKnownSizeVector && typeSrc.ItemType == NumberType.Float,
                        "Invalid feature column type");
                }
            }

            public Func<int, bool> GetDependencies(Func<int, bool> predicate)
            {
                for (int i = 0; i < Schema.ColumnCount; i++)
                {
                    if (predicate(i) && InputRoleMappedSchema.Feature != null)
                        return col => col == InputRoleMappedSchema.Feature.Index;
                }
                return col => false;
            }

            public IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> GetInputColumnRoles()
            {
                yield return RoleMappedSchema.ColumnRole.Feature.Bind(InputRoleMappedSchema.Feature?.Name);
            }

            private Delegate[] CreateGetters(IRow input, bool[] active)
            {
                Contracts.Assert(Utils.Size(active) == 2);
                Contracts.Assert(_parent._distMapper != null);

                var getters = new Delegate[2];
                if (active[0] || active[1])
                {
                    // Put all captured locals at this scope.
                    var featureGetter = InputRoleMappedSchema.Feature != null ? input.GetGetter<VBuffer<Float>>(InputRoleMappedSchema.Feature.Index) : null;
                    Float prob = 0;
                    Float score = 0;
                    long cachedPosition = -1;
                    var features = default(VBuffer<Float>);
                    ValueMapper<VBuffer<Float>, Float, Float> mapper;

                    mapper = _parent._distMapper.GetMapper<VBuffer<Float>, Float, Float>();
                    if (active[0])
                    {
                        ValueGetter<Float> getScore =
                            (ref Float dst) =>
                            {
                                EnsureCachedResultValueMapper(mapper, ref cachedPosition, featureGetter, ref features, ref score, ref prob, input);
                                dst = score;
                            };
                        getters[0] = getScore;
                    }
                    if (active[1])
                    {
                        ValueGetter<Float> getProb =
                            (ref Float dst) =>
                            {
                                EnsureCachedResultValueMapper(mapper, ref cachedPosition, featureGetter, ref features, ref score, ref prob, input);
                                dst = prob;
                            };
                        getters[1] = getProb;
                    }
                }
                return getters;
            }

            private static void EnsureCachedResultValueMapper(ValueMapper<VBuffer<Float>, Float, Float> mapper,
                ref long cachedPosition, ValueGetter<VBuffer<Float>> featureGetter, ref VBuffer<Float> features,
                ref Float score, ref Float prob, IRow input)
            {
                Contracts.AssertValue(mapper);
                if (cachedPosition != input.Position)
                {
                    if (featureGetter != null)
                        featureGetter(ref features);

                    mapper(in features, ref score, ref prob);
                    cachedPosition = input.Position;
                }
            }

            public IRow GetRow(IRow input, Func<int, bool> predicate, out Action disposer)
            {
                Contracts.AssertValue(input);
                var active = Utils.BuildArray(Schema.ColumnCount, predicate);
                var getters = CreateGetters(input, active);
                disposer = null;
                return new SimpleRow(Schema, input, getters);
            }
        }
    }

    /// <summary>
    /// This is an <see cref="ISchemaBindableMapper"/> wrapper for quantile regression predictors. They need a separate
    /// wrapper because they need the quantiles to create the ISchemaBound.
    /// </summary>
    public sealed class SchemaBindableQuantileRegressionPredictor : SchemaBindablePredictorWrapperBase
    {
        public const string LoaderSignature = "QuantileSchemaBindable";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "QRSCHBND",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // ISchemaBindableWrapper update
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010002,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(SchemaBindableQuantileRegressionPredictor).Assembly.FullName);
        }

        private readonly IQuantileValueMapper _qpred;
        private readonly Double[] _quantiles;

        public SchemaBindableQuantileRegressionPredictor(IPredictor predictor, Double[] quantiles)
            : base(predictor)
        {
            var qpred = Predictor as IQuantileValueMapper;
            Contracts.CheckParam(qpred != null, nameof(predictor), "Predictor doesn't implement " + nameof(IQuantileValueMapper));
            _qpred = qpred;
            Contracts.CheckParam(ScoreType == NumberType.Float, nameof(predictor), "Unexpected predictor output type");
            Contracts.CheckParam(ValueMapper != null && ValueMapper.InputType.IsVector
                && ValueMapper.InputType.ItemType == NumberType.Float,
                nameof(predictor), "Unexpected predictor input type");
            Contracts.CheckNonEmpty(quantiles, nameof(quantiles), "Quantiles must not be empty");
            _quantiles = quantiles;
        }

        private SchemaBindableQuantileRegressionPredictor(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, ctx)
        {
            // *** Binary format ***
            // <base info>
            // int: the number of quantiles
            // Double[]: the quantiles

            var qpred = Predictor as IQuantileValueMapper;
            Contracts.CheckDecode(qpred != null);
            _qpred = qpred;
            Contracts.CheckDecode(ScoreType == NumberType.Float);
            Contracts.CheckDecode(ValueMapper != null && ValueMapper.InputType.IsVector
                && ValueMapper.InputType.ItemType == NumberType.Float);
            _quantiles = ctx.Reader.ReadDoubleArray();
            Contracts.CheckDecode(Utils.Size(_quantiles) > 0);
        }

        public override void Save(ModelSaveContext ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base info>
            // int: the number of quantiles
            // Double[]: the quantiles

            base.Save(ctx);
            ctx.Writer.WriteDoubleArray(_quantiles);
        }

        public static SchemaBindableQuantileRegressionPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new SchemaBindableQuantileRegressionPredictor(env, ctx);
        }

        protected override ISchemaBoundMapper BindCore(IChannel ch, RoleMappedSchema schema)
        {
            return new SingleValueRowMapper(schema, this, Schema.Create(new SchemaImpl(ScoreType, _quantiles)));
        }

        protected override Delegate GetPredictionGetter(IRow input, int colSrc)
        {
            Contracts.AssertValue(input);
            Contracts.Assert(0 <= colSrc && colSrc < input.Schema.ColumnCount);

            var typeSrc = input.Schema.GetColumnType(colSrc);
            Contracts.Assert(typeSrc.IsVector && typeSrc.ItemType == NumberType.Float);
            Contracts.Assert(ValueMapper == null ||
                typeSrc.VectorSize == ValueMapper.InputType.VectorSize || ValueMapper.InputType.VectorSize == 0);
            Contracts.Assert(Utils.Size(_quantiles) > 0);

            var featureGetter = input.GetGetter<VBuffer<Float>>(colSrc);
            var featureCount = ValueMapper != null ? ValueMapper.InputType.VectorSize : 0;

            var quantiles = new Float[_quantiles.Length];
            for (int i = 0; i < quantiles.Length; i++)
                quantiles[i] = (Float)_quantiles[i];
            var map = _qpred.GetMapper(quantiles);

            var features = default(VBuffer<Float>);
            ValueGetter<VBuffer<Float>> del =
                (ref VBuffer<Float> value) =>
                {
                    featureGetter(ref features);
                    Contracts.Check(features.Length == featureCount || featureCount == 0);
                    map(in features, ref value);
                };
            return del;
        }

        private sealed class SchemaImpl : ScoreMapperSchemaBase
        {
            private readonly string[] _slotNames;
            private readonly MetadataUtils.MetadataGetter<VBuffer<ReadOnlyMemory<char>>> _getSlotNames;

            public SchemaImpl(ColumnType scoreType, Double[] quantiles)
                : base(scoreType, MetadataUtils.Const.ScoreColumnKind.QuantileRegression)
            {
                Contracts.Assert(Utils.Size(quantiles) > 0);
                _slotNames = new string[quantiles.Length];
                for (int i = 0; i < _slotNames.Length; i++)
                    _slotNames[i] = string.Format("Quantile-{0}", quantiles[i]);
                _getSlotNames = GetSlotNames;
            }

            public override int ColumnCount { get { return 1; } }

            public override IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypes(int col)
            {
                Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                Contracts.Assert(Utils.Size(_slotNames) > 0);
                Contracts.Assert(col == 0);

                var items = base.GetMetadataTypes(col);
                items = items.Prepend(MetadataUtils.GetSlotNamesPair(_slotNames.Length));
                return items;
            }

            public override ColumnType GetMetadataTypeOrNull(string kind, int col)
            {
                Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                Contracts.CheckNonEmpty(kind, nameof(kind));
                Contracts.Assert(Utils.Size(_slotNames) > 0);
                Contracts.Assert(col == 0);

                if (kind == MetadataUtils.Kinds.SlotNames)
                    return MetadataUtils.GetNamesType(_slotNames.Length);
                return base.GetMetadataTypeOrNull(kind, col);
            }

            public override void GetMetadata<TValue>(string kind, int col, ref TValue value)
            {
                Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                Contracts.CheckNonEmpty(kind, nameof(kind));
                Contracts.Assert(Utils.Size(_slotNames) > 0);
                Contracts.Assert(col == 0);
                Contracts.Assert(_getSlotNames != null);

                if (kind == MetadataUtils.Kinds.SlotNames)
                    _getSlotNames.Marshal(col, ref value);
                else
                    base.GetMetadata(kind, col, ref value);
            }

            public override ColumnType GetColumnType(int col)
            {
                Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                Contracts.Assert(col == 0);
                Contracts.Assert(Utils.Size(_slotNames) > 0);
                return new VectorType(NumberType.Float, _slotNames.Length);
            }

            private void GetSlotNames(int iinfo, ref VBuffer<ReadOnlyMemory<char>> dst)
            {
                Contracts.Assert(iinfo == 0);
                Contracts.Assert(Utils.Size(_slotNames) > 0);

                int size = Utils.Size(_slotNames);
                var values = dst.Values;
                if (Utils.Size(values) < size)
                    values = new ReadOnlyMemory<char>[size];
                for (int i = 0; i < _slotNames.Length; i++)
                    values[i] = _slotNames[i].AsMemory();
                dst = new VBuffer<ReadOnlyMemory<char>>(size, values, dst.Indices);
            }
        }
    }
}
