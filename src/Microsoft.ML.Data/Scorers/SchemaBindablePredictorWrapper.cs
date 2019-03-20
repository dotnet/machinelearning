// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Model.Pfa;
using Microsoft.ML.Runtime;
using Newtonsoft.Json.Linq;

[assembly: LoadableClass(typeof(SchemaBindablePredictorWrapper), null, typeof(SignatureLoadModel),
    "Bindable Mapper", SchemaBindablePredictorWrapper.LoaderSignature)]

[assembly: LoadableClass(typeof(SchemaBindableQuantileRegressionPredictor), null, typeof(SignatureLoadModel),
    "Regression Bindable Mapper", SchemaBindableQuantileRegressionPredictor.LoaderSignature)]

[assembly: LoadableClass(typeof(SchemaBindableBinaryPredictorWrapper), null, typeof(SignatureLoadModel),
    "Binary Classification Bindable Mapper", SchemaBindableBinaryPredictorWrapper.LoaderSignature)]

namespace Microsoft.ML.Data
{
    // REVIEW: Consider implementing ICanSaveAs(Code/Text/etc.) for these classes as well.
    /// <summary>
    /// This is a base class for wrapping <see cref="IPredictor"/>s in an <see cref="ISchemaBindableMapper"/>.
    /// </summary>
    internal abstract class SchemaBindablePredictorWrapperBase : ISchemaBindableMapper, ICanSaveModel, ICanSaveSummary,
        IBindableCanSavePfa, IBindableCanSaveOnnx
    {
        // The ctor guarantees that Predictor is non-null. It also ensures that either
        // ValueMapper or FloatPredictor is non-null (or both). With these guarantees,
        // the score value type (_scoreType) can be determined.
        protected readonly IPredictor Predictor;
        private protected readonly IValueMapper ValueMapper;
        protected readonly DataViewType ScoreType;

        bool ICanSavePfa.CanSavePfa => (ValueMapper as ICanSavePfa)?.CanSavePfa == true;

        bool ICanSaveOnnx.CanSaveOnnx(OnnxContext ctx) => (ValueMapper as ICanSaveOnnx)?.CanSaveOnnx(ctx) == true;

        public SchemaBindablePredictorWrapperBase(IPredictor predictor)
        {
            // REVIEW: Eventually drop support for predictors that don't implement IValueMapper.
            Contracts.CheckValue(predictor, nameof(predictor));
            Predictor = predictor;
            ScoreType = GetScoreType(Predictor, out ValueMapper);
        }

        private static DataViewType GetScoreType(IPredictor predictor, out IValueMapper valueMapper)
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

        void ICanSaveModel.Save(ModelSaveContext ctx) => SaveModel(ctx);

        private protected virtual void SaveModel(ModelSaveContext ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();

            // *** Binary format ***
            // <nothing>

            ctx.SaveModel(Predictor, ModelFileUtils.DirPredictor);
        }

        void IBindableCanSavePfa.SaveAsPfa(BoundPfaContext ctx, RoleMappedSchema schema, string[] outputNames)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.Assert(ValueMapper is ISingleCanSavePfa);
            SaveAsPfaCore(ctx, schema, outputNames);
        }

        [BestFriend]
        private protected virtual void SaveAsPfaCore(BoundPfaContext ctx, RoleMappedSchema schema, string[] outputNames)
        {
            ctx.Hide(outputNames);
        }

        bool IBindableCanSaveOnnx.SaveAsOnnx(OnnxContext ctx, RoleMappedSchema schema, string[] outputNames)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.Assert(ValueMapper is ISingleCanSaveOnnx);
            var mapper = (ISingleCanSaveOnnx)ValueMapper;
            return SaveAsOnnxCore(ctx, schema, outputNames);
        }

        [BestFriend]
        private protected virtual bool SaveAsOnnxCore(OnnxContext ctx, RoleMappedSchema schema, string[] outputNames) => false;

        ISchemaBoundMapper ISchemaBindableMapper.Bind(IHostEnvironment env, RoleMappedSchema schema)
        {
            Contracts.CheckValue(env, nameof(env));

            using (var ch = env.Register("SchemaBindableWrapper").Start("Bind"))
            {
                ch.CheckValue(schema, nameof(schema));
                if (schema.Feature?.Type is DataViewType type)
                {
                    // Ensure that the feature column type is compatible with the needed input type.
                    var typeIn = ValueMapper != null ? ValueMapper.InputType : new VectorType(NumberDataViewType.Single);
                    if (type != typeIn)
                    {
                        VectorType typeVectorType = type as VectorType;
                        VectorType typeInVectorType = typeIn as VectorType;

                        DataViewType typeItemType = typeVectorType?.ItemType ?? type;
                        DataViewType typeInItemType = typeInVectorType?.ItemType ?? typeIn;

                        if (!typeItemType.Equals(typeInItemType))
                            throw ch.Except("Incompatible features column type item type: '{0}' vs '{1}'", typeItemType, typeInItemType);
                        if ((typeVectorType != null) != (typeInVectorType != null))
                            throw ch.Except("Incompatible features column type: '{0}' vs '{1}'", type, typeIn);
                        // typeIn can legally have unknown size.
                        int typeVectorSize = typeVectorType?.Size ?? 0;
                        int typeInVectorSize = typeInVectorType?.Size ?? 0;
                        if (typeVectorSize != typeInVectorSize && typeInVectorSize > 0)
                            throw ch.Except("Incompatible features column type: '{0}' vs '{1}'", type, typeIn);
                    }
                }
                return BindCore(ch, schema);
            }
        }

        [BestFriend]
        private protected abstract ISchemaBoundMapper BindCore(IChannel ch, RoleMappedSchema schema);

        protected virtual Delegate GetPredictionGetter(DataViewRow input, int colSrc)
        {
            Contracts.AssertValue(input);
            Contracts.Assert(0 <= colSrc && colSrc < input.Schema.Count);

            var typeSrc = input.Schema[colSrc].Type;
            Func<DataViewRow, int, ValueGetter<int>> del = GetValueGetter<int, int>;
            var meth = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(typeSrc.RawType, ScoreType.RawType);
            return (Delegate)meth.Invoke(this, new object[] { input, colSrc });
        }

        private ValueGetter<TDst> GetValueGetter<TSrc, TDst>(DataViewRow input, int colSrc)
        {
            Contracts.AssertValue(input);
            Contracts.Assert(ValueMapper != null);

            var featureGetter = input.GetGetter<TSrc>(input.Schema[colSrc]);
            var map = ValueMapper.GetMapper<TSrc, TDst>();
            var features = default(TSrc);
            return
                (ref TDst dst) =>
                {
                    featureGetter(ref features);
                    map(in features, ref dst);
                };
        }

        void ICanSaveSummary.SaveSummary(TextWriter writer, RoleMappedSchema schema)
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
            public DataViewSchema OutputSchema { get; }
            public ISchemaBindableMapper Bindable => _parent;

            public SingleValueRowMapper(RoleMappedSchema schema, SchemaBindablePredictorWrapperBase parent, DataViewSchema outputSchema)
            {
                Contracts.AssertValue(schema);
                Contracts.AssertValue(parent);
                Contracts.Assert(schema.Feature.HasValue);
                Contracts.Assert(outputSchema.Count == 1);

                _parent = parent;
                InputRoleMappedSchema = schema;
                OutputSchema = outputSchema;
            }

            /// <summary>
            /// Given a set of columns, return the input columns that are needed to generate those output columns.
            /// </summary>
            IEnumerable<DataViewSchema.Column> ISchemaBoundRowMapper.GetDependenciesForNewColumns(IEnumerable<DataViewSchema.Column> dependingColumns)
            {
                if (!InputRoleMappedSchema.Feature.HasValue || dependingColumns.Count() == 0)
                    return Enumerable.Empty<DataViewSchema.Column>();

                return Enumerable.Repeat(InputRoleMappedSchema.Feature.Value, 1); ;
            }

            public IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> GetInputColumnRoles()
            {
                yield return RoleMappedSchema.ColumnRole.Feature.Bind(InputRoleMappedSchema.Feature.Value.Name);
            }

            public DataViewSchema InputSchema => InputRoleMappedSchema.Schema;

            DataViewRow ISchemaBoundRowMapper.GetRow(DataViewRow input, IEnumerable<DataViewSchema.Column> activeColumns)
            {
                Contracts.AssertValue(input);
                Contracts.AssertValue(activeColumns);

                var getters = new Delegate[1];
                if (activeColumns.Select(c => c.Index).Contains(0))
                    getters[0] = _parent.GetPredictionGetter(input, InputRoleMappedSchema.Feature.Value.Index);
                return new SimpleRow(OutputSchema, input, getters);
            }
        }
    }

    /// <summary>
    /// This class is a wrapper for all <see cref="IPredictor"/>s except for quantile regression predictors,
    /// and calibrated binary classification predictors.
    /// </summary>
    internal sealed class SchemaBindablePredictorWrapper : SchemaBindablePredictorWrapperBase
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

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            ctx.SetVersionInfo(GetVersionInfo());
            base.SaveModel(ctx);
        }

        private protected override void SaveAsPfaCore(BoundPfaContext ctx, RoleMappedSchema schema, string[] outputNames)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.Assert(ValueMapper is ISingleCanSavePfa);
            Contracts.Assert(schema.Feature.HasValue);
            Contracts.Assert(Utils.Size(outputNames) == 1); // Score.
            var mapper = (ISingleCanSavePfa)ValueMapper;
            // If the features column was not produced, we must hide the outputs.
            var featureToken = ctx.TokenOrNullForName(schema.Feature.Value.Name);
            if (featureToken == null)
                ctx.Hide(outputNames);
            var scoreToken = mapper.SaveAsPfa(ctx, featureToken);
            ctx.DeclareVar(outputNames[0], scoreToken);
        }

        private protected override bool SaveAsOnnxCore(OnnxContext ctx, RoleMappedSchema schema, string[] outputNames)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.Assert(ValueMapper is ISingleCanSaveOnnx);
            Contracts.Assert(schema.Feature.HasValue);
            Contracts.Assert(Utils.Size(outputNames) <= 2); // PredictedLabel and/or Score.
            var mapper = (ISingleCanSaveOnnx)ValueMapper;
            string featName = schema.Feature.Value.Name;
            if (!ctx.ContainsColumn(featName))
                return false;
            Contracts.Assert(ctx.ContainsColumn(featName));
            return mapper.SaveAsOnnx(ctx, outputNames, ctx.GetVariableName(featName));
        }

        private protected override ISchemaBoundMapper BindCore(IChannel ch, RoleMappedSchema schema) =>
            new SingleValueRowMapper(schema, this, ScoreSchemaFactory.Create(ScoreType, _scoreColumnKind));

        private static string GetScoreColumnKind(IPredictor predictor)
        {
            Contracts.AssertValue(predictor);

            switch (predictor.PredictionKind)
            {
                case PredictionKind.BinaryClassification:
                    return AnnotationUtils.Const.ScoreColumnKind.BinaryClassification;
                case PredictionKind.MulticlassClassification:
                    return AnnotationUtils.Const.ScoreColumnKind.MulticlassClassification;
                case PredictionKind.Regression:
                    return AnnotationUtils.Const.ScoreColumnKind.Regression;
                case PredictionKind.MultiOutputRegression:
                    return AnnotationUtils.Const.ScoreColumnKind.MultiOutputRegression;
                case PredictionKind.Ranking:
                    return AnnotationUtils.Const.ScoreColumnKind.Ranking;
                case PredictionKind.AnomalyDetection:
                    return AnnotationUtils.Const.ScoreColumnKind.AnomalyDetection;
                case PredictionKind.Clustering:
                    return AnnotationUtils.Const.ScoreColumnKind.Clustering;
                default:
                    throw Contracts.Except("Unknown prediction kind, can't map to score column kind: {0}", predictor.PredictionKind);
            }
        }
    }

    /// <summary>
    /// This is an <see cref="ISchemaBindableMapper"/> wrapper for calibrated binary classification predictors.
    /// They need a separate wrapper because they return two values instead of one: the raw score and the probability.
    /// </summary>
    internal sealed class SchemaBindableBinaryPredictorWrapper : SchemaBindablePredictorWrapperBase
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

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            ctx.SetVersionInfo(GetVersionInfo());
            base.SaveModel(ctx);
        }

        private protected override void SaveAsPfaCore(BoundPfaContext ctx, RoleMappedSchema schema, string[] outputNames)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.Assert(ValueMapper is IDistCanSavePfa);
            Contracts.Assert(schema.Feature.HasValue);
            Contracts.Assert(Utils.Size(outputNames) == 2); // Score and prob.
            var mapper = (IDistCanSavePfa)ValueMapper;
            // If the features column was not produced, we must hide the outputs.
            string featureToken = ctx.TokenOrNullForName(schema.Feature.Value.Name);
            if (featureToken == null)
                ctx.Hide(outputNames);

            JToken scoreToken;
            JToken probToken;
            mapper.SaveAsPfa(ctx, featureToken, outputNames[0], out scoreToken, outputNames[1], out probToken);
            Contracts.Assert(ctx.TokenOrNullForName(outputNames[0]) == scoreToken.ToString());
            Contracts.Assert(ctx.TokenOrNullForName(outputNames[1]) == probToken.ToString());
        }

        private protected override bool SaveAsOnnxCore(OnnxContext ctx, RoleMappedSchema schema, string[] outputNames)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            Contracts.CheckValue(schema, nameof(schema));

            var mapper = ValueMapper as ISingleCanSaveOnnx;
            Contracts.CheckValue(mapper, nameof(mapper));
            Contracts.Assert(schema.Feature.HasValue);
            Contracts.Assert(Utils.Size(outputNames) == 3); // Predicted Label, Score and Probablity.

            var featName = schema.Feature.Value.Name;
            if (!ctx.ContainsColumn(featName))
                return false;
            Contracts.Assert(ctx.ContainsColumn(featName));
            return mapper.SaveAsOnnx(ctx, outputNames, ctx.GetVariableName(featName));
        }

        private void CheckValid(out IValueMapperDist distMapper)
        {
            Contracts.Check(ScoreType == NumberDataViewType.Single, "Expected predictor result type to be float");

            distMapper = Predictor as IValueMapperDist;
            if (distMapper == null)
                throw Contracts.Except("Predictor does not provide probabilities");

            // REVIEW: In theory the restriction on input type could be relaxed at the expense
            // of more complicated code in CalibratedRowMapper.GetGetters. Not worth it at this point
            // and no good way to test it.
            Contracts.Check(distMapper.InputType is VectorType vectorType && vectorType.ItemType == NumberDataViewType.Single,
                "Invalid input type for the IValueMapperDist");
            Contracts.Check(distMapper.DistType == NumberDataViewType.Single,
                "Invalid probability type for the IValueMapperDist");
        }

        private protected override ISchemaBoundMapper BindCore(IChannel ch, RoleMappedSchema schema)
        {
            if (Predictor.PredictionKind != PredictionKind.BinaryClassification)
                ch.Warning("Scoring predictor of kind '{0}' as '{1}'.", Predictor.PredictionKind, PredictionKind.BinaryClassification);

            // For distribution mappers, produce both score and probability.
            Contracts.AssertValue(_distMapper);
            return new CalibratedRowMapper(schema, this);
        }

        /// <summary>
        /// The <see cref="ISchemaBoundRowMapper"/> implementation for distribution predictor wrappers that produce
        /// two float-valued output columns. Note that the Bindable wrapper does input schema validation.
        /// </summary>
        private sealed class CalibratedRowMapper : ISchemaBoundRowMapper
        {
            private readonly SchemaBindableBinaryPredictorWrapper _parent;

            public RoleMappedSchema InputRoleMappedSchema { get; }
            public DataViewSchema InputSchema => InputRoleMappedSchema.Schema;

            public DataViewSchema OutputSchema { get; }

            public ISchemaBindableMapper Bindable => _parent;

            public CalibratedRowMapper(RoleMappedSchema schema, SchemaBindableBinaryPredictorWrapper parent)
            {
                Contracts.AssertValue(parent);
                Contracts.Assert(parent._distMapper != null);
                Contracts.AssertValue(schema);

                _parent = parent;
                InputRoleMappedSchema = schema;
                OutputSchema = ScoreSchemaFactory.CreateBinaryClassificationSchema();

                if (schema.Feature?.Type is DataViewType typeSrc)
                {
                    Contracts.Check(typeSrc is VectorType vectorType
                        && vectorType.IsKnownSize
                        && vectorType.ItemType == NumberDataViewType.Single,
                        "Invalid feature column type");
                }
            }

            /// <summary>
            /// Given a set of columns, return the input columns that are needed to generate those output columns.
            /// </summary>
            IEnumerable<DataViewSchema.Column> ISchemaBoundRowMapper.GetDependenciesForNewColumns(IEnumerable<DataViewSchema.Column> dependingColumns)
            {

                if (dependingColumns.Count() == 0 || !InputRoleMappedSchema.Feature.HasValue)
                    return Enumerable.Empty<DataViewSchema.Column>();

                return Enumerable.Repeat(InputRoleMappedSchema.Feature.Value, 1);
            }

            public IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> GetInputColumnRoles()
            {
                yield return RoleMappedSchema.ColumnRole.Feature.Bind(InputRoleMappedSchema.Feature?.Name);
            }

            private Delegate[] CreateGetters(DataViewRow input, bool[] active)
            {
                Contracts.Assert(Utils.Size(active) == 2);
                Contracts.Assert(_parent._distMapper != null);

                var getters = new Delegate[2];
                if (active[0] || active[1])
                {
                    // Put all captured locals at this scope.
                    var featureGetter = InputRoleMappedSchema.Feature.HasValue ? input.GetGetter<VBuffer<float>>(InputRoleMappedSchema.Feature.Value) : null;
                    float prob = 0;
                    float score = 0;
                    long cachedPosition = -1;
                    var features = default(VBuffer<float>);
                    ValueMapper<VBuffer<float>, float, float> mapper;

                    mapper = _parent._distMapper.GetMapper<VBuffer<float>, float, float>();
                    if (active[0])
                    {
                        ValueGetter<float> getScore =
                            (ref float dst) =>
                            {
                                EnsureCachedResultValueMapper(mapper, ref cachedPosition, featureGetter, ref features, ref score, ref prob, input);
                                dst = score;
                            };
                        getters[0] = getScore;
                    }
                    if (active[1])
                    {
                        ValueGetter<float> getProb =
                            (ref float dst) =>
                            {
                                EnsureCachedResultValueMapper(mapper, ref cachedPosition, featureGetter, ref features, ref score, ref prob, input);
                                dst = prob;
                            };
                        getters[1] = getProb;
                    }
                }
                return getters;
            }

            private static void EnsureCachedResultValueMapper(ValueMapper<VBuffer<float>, float, float> mapper,
                ref long cachedPosition, ValueGetter<VBuffer<float>> featureGetter, ref VBuffer<float> features,
                ref float score, ref float prob, DataViewRow input)
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

            DataViewRow ISchemaBoundRowMapper.GetRow(DataViewRow input, IEnumerable<DataViewSchema.Column> activeColumns)
            {
                Contracts.AssertValue(input);
                var active = Utils.BuildArray(OutputSchema.Count, activeColumns);
                var getters = CreateGetters(input, active);
                return new SimpleRow(OutputSchema, input, getters);
            }
        }
    }

    /// <summary>
    /// This is an <see cref="ISchemaBindableMapper"/> wrapper for quantile regression predictors. They need a separate
    /// wrapper because they need the quantiles to create the <see cref="ISchemaBoundMapper"/>.
    /// </summary>
    [BestFriend]
    internal sealed class SchemaBindableQuantileRegressionPredictor : SchemaBindablePredictorWrapperBase
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
        private readonly double[] _quantiles;

        public SchemaBindableQuantileRegressionPredictor(IPredictor predictor, double[] quantiles)
            : base(predictor)
        {
            var qpred = Predictor as IQuantileValueMapper;
            Contracts.CheckParam(qpred != null, nameof(predictor), "Predictor doesn't implement " + nameof(IQuantileValueMapper));
            _qpred = qpred;
            Contracts.CheckParam(ScoreType == NumberDataViewType.Single, nameof(predictor), "Unexpected predictor output type");
            Contracts.CheckParam(ValueMapper != null && ValueMapper.InputType is VectorType vectorType
                && vectorType.ItemType == NumberDataViewType.Single,
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
            // double[]: the quantiles

            var qpred = Predictor as IQuantileValueMapper;
            Contracts.CheckDecode(qpred != null);
            _qpred = qpred;
            Contracts.CheckDecode(ScoreType == NumberDataViewType.Single);
            Contracts.CheckDecode(ValueMapper != null && ValueMapper.InputType is VectorType vectorType
                && vectorType.ItemType == NumberDataViewType.Single);
            _quantiles = ctx.Reader.ReadDoubleArray();
            Contracts.CheckDecode(Utils.Size(_quantiles) > 0);
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base info>
            // int: the number of quantiles
            // double[]: the quantiles

            base.SaveModel(ctx);
            ctx.Writer.WriteDoubleArray(_quantiles);
        }

        public static SchemaBindableQuantileRegressionPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new SchemaBindableQuantileRegressionPredictor(env, ctx);
        }

        private protected override ISchemaBoundMapper BindCore(IChannel ch, RoleMappedSchema schema) =>
            new SingleValueRowMapper(schema, this, ScoreSchemaFactory.CreateQuantileRegressionSchema(ScoreType, _quantiles));

        protected override Delegate GetPredictionGetter(DataViewRow input, int colSrc)
        {
            Contracts.AssertValue(input);
            Contracts.Assert(0 <= colSrc && colSrc < input.Schema.Count);

            var column = input.Schema[colSrc];
            var typeSrc = column.Type as VectorType;
            Contracts.Assert(typeSrc != null && typeSrc.ItemType == NumberDataViewType.Single);
            Contracts.Assert(ValueMapper == null ||
                typeSrc.Size == ValueMapper.InputType.GetVectorSize() || ValueMapper.InputType.GetVectorSize() == 0);
            Contracts.Assert(Utils.Size(_quantiles) > 0);

            var featureGetter = input.GetGetter<VBuffer<float>>(column);
            var featureCount = ValueMapper != null ? ValueMapper.InputType.GetVectorSize() : 0;

            var quantiles = new float[_quantiles.Length];
            for (int i = 0; i < quantiles.Length; i++)
                quantiles[i] = (float)_quantiles[i];
            var map = _qpred.GetMapper(quantiles);

            var features = default(VBuffer<float>);
            ValueGetter<VBuffer<float>> del =
                (ref VBuffer<float> value) =>
                {
                    featureGetter(ref features);
                    Contracts.Check(features.Length == featureCount || featureCount == 0);
                    map(in features, ref value);
                };
            return del;
        }
    }
}
