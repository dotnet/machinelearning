// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Model.Onnx;
using Microsoft.ML.Runtime.Model.Pfa;
using Microsoft.ML.Runtime.Numeric;
using Newtonsoft.Json.Linq;

// This is for deserialization from a model repository.
[assembly: LoadableClass(typeof(IPredictorProducing<Float>), typeof(LinearBinaryPredictor), null, typeof(SignatureLoadModel),
    "Linear Binary Executor",
    LinearBinaryPredictor.LoaderSignature)]

// This is for deserialization from a model repository.
[assembly: LoadableClass(typeof(LinearRegressionPredictor), null, typeof(SignatureLoadModel),
    "Linear Regression Executor",
    LinearRegressionPredictor.LoaderSignature)]

// This is for deserialization from a model repository.
[assembly: LoadableClass(typeof(PoissonRegressionPredictor), null, typeof(SignatureLoadModel),
    "Poisson Regression Executor",
    PoissonRegressionPredictor.LoaderSignature)]

namespace Microsoft.ML.Runtime.Learners
{
    public abstract class LinearPredictor : PredictorBase<Float>,
        IValueMapper,
        ICanSaveInIniFormat,
        ICanSaveInTextFormat,
        ICanSaveInSourceCode,
        ICanSaveModel,
        ICanGetSummaryAsIRow,
        ICanSaveSummary,
        IPredictorWithFeatureWeights<Float>,
        IWhatTheFeatureValueMapper,
        ISingleCanSavePfa,
        ISingleCanSaveOnnx
    {
        protected readonly VBuffer<Float> Weight;

        // _weightsDense is not persisted and is used for performance when the input instance is sparse.
        private VBuffer<Float> _weightsDense;
        private readonly object _weightsDenseLock;

        private sealed class WeightsCollection : IReadOnlyList<Float>
        {
            private readonly LinearPredictor _pred;

            public int Count => _pred.Weight.Length;

            public Float this[int index] {
                get {
                    Contracts.CheckParam(0 <= index && index < Count, nameof(index), "Out of range");
                    Float value = 0;
                    _pred.Weight.GetItemOrDefault(index, ref value);
                    return value;
                }
            }

            public WeightsCollection(LinearPredictor pred)
            {
                Contracts.AssertValue(pred);
                _pred = pred;
            }

            public IEnumerator<Float> GetEnumerator()
            {
                return _pred.Weight.Items(all: true).Select(iv => iv.Value).GetEnumerator();
            }

            IEnumerator IEnumerable.GetEnumerator()
            {
                return GetEnumerator();
            }
        }

        /// <summary> The predictor's feature weight coefficients.</summary>
        public IReadOnlyList<Float> Weights2 => new WeightsCollection(this);

        /// <summary> The predictor's bias term.</summary>
        public Float Bias { get; protected set; }

        public ColumnType InputType { get; }

        public ColumnType OutputType => NumberType.Float;

        public bool CanSavePfa => true;

        public bool CanSaveOnnx => true;

        /// <summary>
        /// Constructs a new linear predictor.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="name">Component name.</param>
        /// <param name="weights">The weights for the linear predictor. Note that this
        /// will take ownership of the <see cref="VBuffer{T}"/>.</param>
        /// <param name="bias">The bias added to every output score.</param>
        internal LinearPredictor(IHostEnvironment env, string name, ref VBuffer<Float> weights, Float bias)
            : base(env, name)
        {
            Host.CheckParam(FloatUtils.IsFinite(weights.Values, weights.Count), nameof(weights), "Cannot initialize linear predictor with non-finite weights");
            Host.CheckParam(FloatUtils.IsFinite(bias), nameof(bias), "Cannot initialize linear predictor with non-finite bias");

            Weight = weights;
            Bias = bias;
            InputType = new VectorType(NumberType.Float, Weight.Length);

            if (Weight.IsDense)
                _weightsDense = Weight;
            else
                _weightsDenseLock = new object();
        }

        protected LinearPredictor(IHostEnvironment env, string name, ModelLoadContext ctx)
            : base(env, name, ctx)
        {
            // *** Binary format ***
            // Float: bias
            // int: number of features (weights)
            // int: number of indices
            // int[]: indices
            // int: number of weights
            // Float[]: weights
            // bool: has model stats
            // (Conditional) LinearModelStatistics: stats

            Bias = ctx.Reader.ReadFloat();
            Host.CheckDecode(FloatUtils.IsFinite(Bias));

            int len = ctx.Reader.ReadInt32();
            Host.Assert(len > 0);

            int cind = ctx.Reader.ReadInt32();
            Host.CheckDecode(0 <= cind & cind < len);
            var indices = ctx.Reader.ReadIntArray(cind);

            // Verify monotonicity of indices.
            int prev = -1;
            for (int i = 0; i < cind; i++)
            {
                Host.CheckDecode(prev < indices[i]);
                prev = indices[i];
            }
            Host.CheckDecode(prev < len);

            int cwht = ctx.Reader.ReadInt32();
            // Either there are as many weights as there are indices (in the
            // sparse case), or (in the dense case) there are no indices and the
            // number of weights is the length of the vector. Note that for the
            // trivial predictor it is quite legal to have 0 in both counts.
            Host.CheckDecode(cwht == cind || (cind == 0 && cwht == len));

            var weights = ctx.Reader.ReadFloatArray(cwht);
            Host.CheckDecode(Utils.Size(weights) == 0 || weights.All(x => FloatUtils.IsFinite(x)));

            if (cwht == 0)
                Weight = VBufferUtils.CreateEmpty<Float>(len);
            else
                Weight = new VBuffer<Float>(len, Utils.Size(weights), weights, indices);

            InputType = new VectorType(NumberType.Float, Weight.Length);
            WarnOnOldNormalizer(ctx, GetType(), Host);

            if (Weight.IsDense)
                _weightsDense = Weight;
            else
                _weightsDenseLock = new object();
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);

            // *** Binary format ***
            // Float: bias
            // int: number of features (weights)
            // int: number of indices
            // int[]: indices
            // int: number of weights
            // Float[]: weights
            // bool: has model stats
            // (Conditional) LinearModelStatistics: stats

            ctx.Writer.Write(Bias);
            ctx.Writer.Write(Weight.Length);
            ctx.Writer.WriteIntArray(Weight.Indices, Weight.IsDense ? 0 : Weight.Count);
            ctx.Writer.WriteFloatArray(Weight.Values, Weight.Count);
        }

        public JToken SaveAsPfa(BoundPfaContext ctx, JToken input)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.CheckValue(input, nameof(input));

            const string typeName = "LinearPredictor";
            JToken typeDecl = typeName;
            if (ctx.Pfa.RegisterType(typeName))
            {
                JObject type = new JObject();
                type["type"] = "record";
                type["name"] = typeName;
                JArray fields = new JArray();
                JObject jobj = null;
                fields.Add(jobj.AddReturn("name", "coeff").AddReturn("type", PfaUtils.Type.Array(PfaUtils.Type.Double)));
                fields.Add(jobj.AddReturn("name", "const").AddReturn("type", PfaUtils.Type.Double));
                type["fields"] = fields;
                typeDecl = type;
            }
            JObject predictor = new JObject();
            predictor["coeff"] = new JArray(Weight.DenseValues());
            predictor["const"] = Bias;
            var cell = ctx.DeclareCell("LinearPredictor", typeDecl, predictor);
            var cellRef = PfaUtils.Cell(cell);
            return PfaUtils.Call("model.reg.linear", input, cellRef);
        }

        public bool SaveAsOnnx(IOnnxContext ctx, string[] outputs, string featureColumn)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.Check(Utils.Size(outputs) == 1);

            string opType = "LinearRegressor";
            var node = ctx.CreateNode(opType, new[] { featureColumn }, outputs, ctx.GetNodeName(opType));
            // Selection of logit or probit output transform. enum {'NONE', 'LOGIT', 'PROBIT}
            node.AddAttribute("post_transform", 0);
            node.AddAttribute("targets", 1);
            node.AddAttribute("coefficients", Weight.DenseValues());
            node.AddAttribute("intercepts", Bias);
            return true;
        }

        // Generate the score from the given values, assuming they have already been normalized.
        protected virtual Float Score(ref VBuffer<Float> src)
        {
            if (src.IsDense)
            {
                var weights = Weight;
                return Bias + VectorUtils.DotProduct(ref weights, ref src);
            }
            EnsureWeightsDense();
            return Bias + VectorUtils.DotProduct(ref _weightsDense, ref src);
        }

        protected virtual void GetFeatureContributions(ref VBuffer<Float> features, ref VBuffer<Float> contributions, int top, int bottom, bool normalize)
        {
            if (features.Length != Weight.Length)
                throw Contracts.Except("Input is of length {0} does not match expected length  of weights {1}", features.Length, Weight.Length);

            var weights = Weight;
            VBuffer<Float>.Copy(ref features, ref contributions);
            VectorUtils.MulElementWise(ref weights, ref contributions);
            VectorUtils.SparsifyNormalize(ref contributions, top, bottom, normalize);
        }

        private void EnsureWeightsDense()
        {
            if (_weightsDense.Length == 0 && Weight.Length > 0)
            {
                Contracts.AssertValue(_weightsDenseLock);
                lock (_weightsDenseLock)
                {
                    if (_weightsDense.Length == 0 && Weight.Length > 0)
                        Weight.CopyToDense(ref _weightsDense);
                }
            }
        }

        public ValueMapper<TIn, TOut> GetMapper<TIn, TOut>()
        {
            Contracts.Check(typeof(TIn) == typeof(VBuffer<Float>));
            Contracts.Check(typeof(TOut) == typeof(Float));

            ValueMapper<VBuffer<Float>, Float> del =
                (ref VBuffer<Float> src, ref Float dst) =>
                {
                    if (src.Length != Weight.Length)
                        throw Contracts.Except("Input is of length {0}, but predictor expected length {1}", src.Length, Weight.Length);
                    dst = Score(ref src);
                };
            return (ValueMapper<TIn, TOut>)(Delegate)del;
        }

        /// <summary>
        /// Combine a bunch of models into one by averaging parameters
        /// </summary>
        protected void CombineParameters(IList<IParameterMixer<Float>> models, out VBuffer<Float> weights, out Float bias)
        {
            Type type = GetType();

            Contracts.Check(type == models[0].GetType(), "Submodel for parameter mixer has the wrong type");
            var first = (LinearPredictor)models[0];

            weights = default(VBuffer<Float>);
            first.Weight.CopyTo(ref weights);
            bias = first.Bias;

            for (int i = 1; i < models.Count; i++)
            {
                var m = models[i];
                Contracts.Check(type == m.GetType(), "Submodel for parameter mixer has the wrong type");

                var sub = (LinearPredictor)m;
                var subweights = sub.Weight;
                VectorUtils.Add(ref subweights, ref weights);
                bias += sub.Bias;
            }
            VectorUtils.ScaleBy(ref weights, (Float)1 / models.Count);
            bias /= models.Count;
        }

        public void SaveAsText(TextWriter writer, RoleMappedSchema schema)
        {
            Host.CheckValue(writer, nameof(writer));
            Host.CheckValue(schema, nameof(schema));

            SaveSummary(writer, schema);
        }

        public void SaveAsCode(TextWriter writer, RoleMappedSchema schema)
        {
            Host.CheckValue(writer, nameof(writer));
            Host.CheckValue(schema, nameof(schema));

            var weights = Weight;
            LinearPredictorUtils.SaveAsCode(writer, ref weights, Bias, schema);
        }

        public abstract void SaveSummary(TextWriter writer, RoleMappedSchema schema);

        public virtual IRow GetSummaryIRowOrNull(RoleMappedSchema schema)
        {
            var cols = new List<IColumn>();

            var names = default(VBuffer<DvText>);
            MetadataUtils.GetSlotNames(schema, RoleMappedSchema.ColumnRole.Feature, Weight.Length, ref names);
            var slotNamesCol = RowColumnUtils.GetColumn(MetadataUtils.Kinds.SlotNames,
                new VectorType(TextType.Instance, Weight.Length), ref names);
            var slotNamesRow = RowColumnUtils.GetRow(null, slotNamesCol);
            var colType = new VectorType(NumberType.R4, Weight.Length);

            // Add the bias and the weight columns.
            var bias = Bias;
            cols.Add(RowColumnUtils.GetColumn("Bias", NumberType.R4, ref bias));
            var weights = Weight;
            cols.Add(RowColumnUtils.GetColumn("Weights", colType, ref weights, slotNamesRow));
            return RowColumnUtils.GetRow(null, cols.ToArray());
        }

        public virtual IRow GetStatsIRowOrNull(RoleMappedSchema schema)
        {
            return null;
        }

        public abstract void SaveAsIni(TextWriter writer, RoleMappedSchema schema, ICalibrator calibrator = null);

        public virtual void GetFeatureWeights(ref VBuffer<Float> weights)
        {
            Weight.CopyTo(ref weights);
        }

        public ValueMapper<TSrc, VBuffer<Float>> GetWhatTheFeatureMapper<TSrc, TDstContributions>(int top, int bottom, bool normalize)
        {
            Contracts.Check(typeof(TSrc) == typeof(VBuffer<Float>));
            Contracts.Check(typeof(TDstContributions) == typeof(VBuffer<Float>));

            ValueMapper<VBuffer<Float>, VBuffer<Float>> del =
                (ref VBuffer<Float> src, ref VBuffer<Float> dstContributions) =>
                {
                    GetFeatureContributions(ref src, ref dstContributions, top, bottom, normalize);
                };
            return (ValueMapper<TSrc, VBuffer<Float>>)(Delegate)del;
        }
    }

    public sealed partial class LinearBinaryPredictor : LinearPredictor,
        ICanGetSummaryInKeyValuePairs,
        IParameterMixer<Float>
    {
        public const string LoaderSignature = "Linear2CExec";
        public const string RegistrationName = "LinearBinaryPredictor";

        private const string ModelStatsSubModelFilename = "ModelStats";
        private readonly LinearModelStatistics _stats;

        public LinearModelStatistics Statistics { get { return _stats; } }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "LINEAR2C",
                // verWrittenCur: 0x00010001, // Initial
                // verWrittenCur: 0x00020001, // Fixed sparse serialization
                verWrittenCur: 0x00020002, // Added model statistics
                verReadableCur: 0x00020001,
                verWeCanReadBack: 0x00020001,
                loaderSignature: LoaderSignature);
        }

        /// <summary>
        /// Constructs a new linear binary predictor.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="weights">The weights for the linear predictor. Note that this
        /// will take ownership of the <see cref="VBuffer{T}"/>.</param>
        /// <param name="bias">The bias added to every output score.</param>
        /// <param name="stats"></param>
        public LinearBinaryPredictor(IHostEnvironment env, ref VBuffer<Float> weights, Float bias, LinearModelStatistics stats = null)
            : base(env, RegistrationName, ref weights, bias)
        {
            Contracts.AssertValueOrNull(stats);
            _stats = stats;
        }

        private LinearBinaryPredictor(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx)
        {
            // For model version earlier than 0x00020001, there is no model statisitcs.
            if (ctx.Header.ModelVerWritten <= 0x00020001)
                return;

            // *** Binary format ***
            // (Base class)
            // LinearModelStatistics: model statistics (optional, in a separate stream)

            string statsDir = Path.Combine(ctx.Directory ?? "", ModelStatsSubModelFilename);
            using (var statsEntry = ctx.Repository.OpenEntryOrNull(statsDir, ModelLoadContext.ModelStreamName))
            {
                if (statsEntry == null)
                    _stats = null;
                else
                {
                    using (var statsCtx = new ModelLoadContext(ctx.Repository, statsEntry, statsDir))
                        _stats = LinearModelStatistics.Create(Host, statsCtx);
                }
            }
        }

        public static IPredictorProducing<Float> Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            var predictor = new LinearBinaryPredictor(env, ctx);
            ICalibrator calibrator;
            ctx.LoadModelOrNull<ICalibrator, SignatureLoadModel>(env, out calibrator, @"Calibrator");
            if (calibrator == null)
                return predictor;
            if (calibrator is IParameterMixer)
                return new ParameterMixingCalibratedPredictor(env, predictor, calibrator);
            return new SchemaBindableCalibratedPredictor(env, predictor, calibrator);
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            // *** Binary format ***
            // (Base class)
            // LinearModelStatistics: model statistics (optional, in a separate stream)

            base.SaveCore(ctx);
            Contracts.AssertValueOrNull(_stats);
            if (_stats != null)
            {
                using (var statsCtx = new ModelSaveContext(ctx.Repository,
                    Path.Combine(ctx.Directory ?? "", ModelStatsSubModelFilename), ModelLoadContext.ModelStreamName))
                {
                    _stats.Save(statsCtx);
                    statsCtx.Done();
                }
            }

            ctx.SetVersionInfo(GetVersionInfo());
        }

        public override PredictionKind PredictionKind {
            get { return PredictionKind.BinaryClassification; }
        }

        /// <summary>
        /// Combine a bunch of models into one by averaging parameters
        /// </summary>
        public IParameterMixer<Float> CombineParameters(IList<IParameterMixer<Float>> models)
        {
            VBuffer<Float> weights;
            Float bias;
            CombineParameters(models, out weights, out bias);
            return new LinearBinaryPredictor(Host, ref weights, bias);
        }

        public override void SaveSummary(TextWriter writer, RoleMappedSchema schema)
        {
            Host.CheckValue(schema, nameof(schema));

            // REVIEW: Would be nice to have the settings!
            var weights = Weight;
            writer.WriteLine(LinearPredictorUtils.LinearModelAsText("Linear Binary Classification Predictor", null, null,
                ref weights, Bias, schema));

            _stats?.SaveText(writer, this, schema, 20);
        }

        ///<inheritdoc/>
        public IList<KeyValuePair<string, object>> GetSummaryInKeyValuePairs(RoleMappedSchema schema)
        {
            Host.CheckValue(schema, nameof(schema));

            var weights = Weight;
            List<KeyValuePair<string, object>> results = new List<KeyValuePair<string, object>>();
            LinearPredictorUtils.SaveLinearModelWeightsInKeyValuePairs(ref weights, Bias, schema, results);
            _stats?.SaveSummaryInKeyValuePairs(this, schema, int.MaxValue, results);
            return results;
        }

        public override IRow GetStatsIRowOrNull(RoleMappedSchema schema)
        {
            if (_stats == null)
                return null;
            var cols = new List<IColumn>();
            var names = default(VBuffer<DvText>);
            MetadataUtils.GetSlotNames(schema, RoleMappedSchema.ColumnRole.Feature, Weight.Length, ref names);

            // Add the stat columns.
            _stats.AddStatsColumns(cols, this, schema, ref names);
            return RowColumnUtils.GetRow(null, cols.ToArray());
        }

        public override void SaveAsIni(TextWriter writer, RoleMappedSchema schema, ICalibrator calibrator = null)
        {
            Host.CheckValue(writer, nameof(writer));
            Host.CheckValue(schema, nameof(schema));
            Host.CheckValueOrNull(calibrator);

            var weights = Weight;
            writer.Write(LinearPredictorUtils.LinearModelAsIni(ref weights, Bias, this,
                schema, calibrator as PlattCalibrator));
        }
    }

    public abstract class RegressionPredictor : LinearPredictor
    {
        internal RegressionPredictor(IHostEnvironment env, string name, ref VBuffer<Float> weights, Float bias)
            : base(env, name, ref weights, bias)
        {
        }

        protected RegressionPredictor(IHostEnvironment env, string name, ModelLoadContext ctx)
            : base(env, name, ctx)
        {
        }

        public override PredictionKind PredictionKind {
            get { return PredictionKind.Regression; }
        }

        /// <summary>
        /// Output the INI model to a given writer
        /// </summary>
        public override void SaveAsIni(TextWriter writer, RoleMappedSchema schema, ICalibrator calibrator)
        {
            if (calibrator != null)
                throw Host.ExceptNotImpl("Saving calibrators is not implemented yet.");

            Host.CheckValue(writer, nameof(writer));
            Host.CheckValue(schema, nameof(schema));

            // REVIEW: For Poisson should encode the exp operation in the ini as well, bug 2433.
            var weights = Weight;
            writer.Write(LinearPredictorUtils.LinearModelAsIni(ref weights, Bias, this, schema, null));
        }
    }

    public sealed class LinearRegressionPredictor : RegressionPredictor,
        IParameterMixer<Float>,
        ICanGetSummaryInKeyValuePairs
    {
        public const string LoaderSignature = "LinearRegressionExec";
        public const string RegistrationName = "LinearRegressionPredictor";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "LIN RGRS",
                // verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00020001, // Fixed sparse serialization
                verReadableCur: 0x00020001,
                verWeCanReadBack: 0x00020001,
                loaderSignature: LoaderSignature);
        }

        /// <summary>
        /// Constructs a new linear regression predictor.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="weights">The weights for the linear predictor. Note that this
        /// will take ownership of the <see cref="VBuffer{T}"/>.</param>
        /// <param name="bias">The bias added to every output score.</param>
        public LinearRegressionPredictor(IHostEnvironment env, ref VBuffer<Float> weights, Float bias)
            : base(env, RegistrationName, ref weights, bias)
        {
        }

        private LinearRegressionPredictor(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx)
        {
        }

        public static LinearRegressionPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new LinearRegressionPredictor(env, ctx);
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
        }

        public override void SaveSummary(TextWriter writer, RoleMappedSchema schema)
        {
            Host.CheckValue(writer, nameof(writer));
            Host.CheckValue(schema, nameof(schema));

            // REVIEW: Would be nice to have the settings!
            var weights = Weight;
            writer.WriteLine(LinearPredictorUtils.LinearModelAsText("Linear Regression Predictor", null, null,
                ref weights, Bias, schema, null));
        }

        /// <summary>
        /// Combine a bunch of models into one by averaging parameters
        /// </summary>
        public IParameterMixer<Float> CombineParameters(IList<IParameterMixer<Float>> models)
        {
            VBuffer<Float> weights;
            Float bias;
            CombineParameters(models, out weights, out bias);
            return new LinearRegressionPredictor(Host, ref weights, bias);
        }

        ///<inheritdoc/>
        public IList<KeyValuePair<string, object>> GetSummaryInKeyValuePairs(RoleMappedSchema schema)
        {
            Host.CheckValue(schema, nameof(schema));

            var weights = Weight;
            List<KeyValuePair<string, object>> results = new List<KeyValuePair<string, object>>();
            LinearPredictorUtils.SaveLinearModelWeightsInKeyValuePairs(ref weights, Bias, schema, results);

            return results;
        }
    }

    public sealed class PoissonRegressionPredictor : RegressionPredictor, IParameterMixer<Float>
    {
        public const string LoaderSignature = "PoissonRegressionExec";
        public const string RegistrationName = "PoissonRegressionPredictor";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "POI RGRS",
                // verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00020001, // Fixed sparse serialization
                verReadableCur: 0x00020001,
                verWeCanReadBack: 0x00020001,
                loaderSignature: LoaderSignature);
        }

        internal PoissonRegressionPredictor(IHostEnvironment env, ref VBuffer<Float> weights, Float bias)
            : base(env, RegistrationName, ref weights, bias)
        {
        }

        private PoissonRegressionPredictor(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx)
        {
        }

        public static PoissonRegressionPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new PoissonRegressionPredictor(env, ctx);
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
        }

        protected override Float Score(ref VBuffer<Float> src)
        {
            return MathUtils.ExpSlow(base.Score(ref src));
        }

        public override void SaveSummary(TextWriter writer, RoleMappedSchema schema)
        {
            Host.CheckValue(writer, nameof(writer));
            Host.CheckValue(schema, nameof(schema));

            // REVIEW: Would be nice to have the settings!
            var weights = Weight;
            writer.WriteLine(LinearPredictorUtils.LinearModelAsText("Poisson Regression Predictor", null, null,
                ref weights, Bias, schema, null));
        }

        /// <summary>
        /// Combine a bunch of models into one by averaging parameters
        /// </summary>
        public IParameterMixer<Float> CombineParameters(IList<IParameterMixer<Float>> models)
        {
            VBuffer<Float> weights;
            Float bias;
            CombineParameters(models, out weights, out bias);
            return new PoissonRegressionPredictor(Host, ref weights, bias);
        }
    }
}