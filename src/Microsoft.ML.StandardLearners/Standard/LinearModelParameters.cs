// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Calibrator;
using Microsoft.ML.Data;
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
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;

// This is for deserialization from a model repository.
[assembly: LoadableClass(typeof(IPredictorProducing<float>), typeof(LinearBinaryModelParameters), null, typeof(SignatureLoadModel),
    "Linear Binary Executor",
    LinearBinaryModelParameters.LoaderSignature)]

// This is for deserialization from a model repository.
[assembly: LoadableClass(typeof(LinearRegressionModelParameters), null, typeof(SignatureLoadModel),
    "Linear Regression Executor",
    LinearRegressionModelParameters.LoaderSignature)]

// This is for deserialization from a model repository.
[assembly: LoadableClass(typeof(PoissonRegressionModelParameters), null, typeof(SignatureLoadModel),
    "Poisson Regression Executor",
    PoissonRegressionModelParameters.LoaderSignature)]

namespace Microsoft.ML.Runtime.Learners
{
    public abstract class LinearModelParameters : PredictorBase<float>,
        IValueMapper,
        ICanSaveInIniFormat,
        ICanSaveInTextFormat,
        ICanSaveInSourceCode,
        ICanSaveModel,
        ICanGetSummaryAsIRow,
        ICanSaveSummary,
        IPredictorWithFeatureWeights<float>,
        IFeatureContributionMapper,
        ISingleCanSavePfa,
        ISingleCanSaveOnnx
    {
        protected readonly VBuffer<float> Weight;

        // _weightsDense is not persisted and is used for performance when the input instance is sparse.
        private VBuffer<float> _weightsDense;
        private readonly object _weightsDenseLock;

        private sealed class WeightsCollection : IReadOnlyList<float>
        {
            private readonly LinearModelParameters _pred;

            public int Count => _pred.Weight.Length;

            public float this[int index]
            {
                get
                {
                    Contracts.CheckParam(0 <= index && index < Count, nameof(index), "Out of range");
                    float value = 0;
                    _pred.Weight.GetItemOrDefault(index, ref value);
                    return value;
                }
            }

            public WeightsCollection(LinearModelParameters pred)
            {
                Contracts.AssertValue(pred);
                _pred = pred;
            }

            public IEnumerator<float> GetEnumerator()
            {
                return _pred.Weight.Items(all: true).Select(iv => iv.Value).GetEnumerator();
            }

            IEnumerator IEnumerable.GetEnumerator()
            {
                return GetEnumerator();
            }
        }

        /// <summary> The predictor's feature weight coefficients.</summary>
        public IReadOnlyList<float> Weights => new WeightsCollection(this);

        /// <summary> The predictor's bias term.</summary>
        public float Bias { get; protected set; }

        private readonly ColumnType _inputType;

        bool ICanSavePfa.CanSavePfa => true;

        bool ICanSaveOnnx.CanSaveOnnx(OnnxContext ctx) => true;

        /// <summary>
        /// Constructs a new linear predictor.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="name">Component name.</param>
        /// <param name="weights">The weights for the linear predictor. Note that this
        /// will take ownership of the <see cref="VBuffer{T}"/>.</param>
        /// <param name="bias">The bias added to every output score.</param>
        public LinearModelParameters(IHostEnvironment env, string name, in VBuffer<float> weights, float bias)
            : base(env, name)
        {
            Host.CheckParam(FloatUtils.IsFinite(weights.GetValues()), nameof(weights), "Cannot initialize linear predictor with non-finite weights");
            Host.CheckParam(FloatUtils.IsFinite(bias), nameof(bias), "Cannot initialize linear predictor with non-finite bias");

            Weight = weights;
            Bias = bias;
            _inputType = new VectorType(NumberType.Float, Weight.Length);

            if (Weight.IsDense)
                _weightsDense = Weight;
            else
                _weightsDenseLock = new object();
        }

        protected LinearModelParameters(IHostEnvironment env, string name, ModelLoadContext ctx)
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
                Weight = VBufferUtils.CreateEmpty<float>(len);
            else
                Weight = new VBuffer<float>(len, Utils.Size(weights), weights, indices);

            _inputType = new VectorType(NumberType.Float, Weight.Length);
            WarnOnOldNormalizer(ctx, GetType(), Host);

            if (Weight.IsDense)
                _weightsDense = Weight;
            else
                _weightsDenseLock = new object();
        }

        [BestFriend]
        private protected override void SaveCore(ModelSaveContext ctx)
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
            ctx.Writer.WriteIntArray(Weight.GetIndices());
            ctx.Writer.WriteSingleArray(Weight.GetValues());
        }

        JToken ISingleCanSavePfa.SaveAsPfa(BoundPfaContext ctx, JToken input)
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

        bool ISingleCanSaveOnnx.SaveAsOnnx(OnnxContext ctx, string[] outputs, string featureColumn)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.Check(Utils.Size(outputs) == 1);

            string opType = "LinearRegressor";
            var node = ctx.CreateNode(opType, new[] { featureColumn }, outputs, ctx.GetNodeName(opType));
            // Selection of logit or probit output transform. enum {'NONE', 'LOGIT', 'PROBIT}
            node.AddAttribute("post_transform", "NONE");
            node.AddAttribute("targets", 1);
            node.AddAttribute("coefficients", Weight.DenseValues());
            node.AddAttribute("intercepts", new float[] { Bias });
            return true;
        }

        // Generate the score from the given values, assuming they have already been normalized.
        protected virtual float Score(in VBuffer<float> src)
        {
            if (src.IsDense)
            {
                var weights = Weight;
                return Bias + VectorUtils.DotProduct(in weights, in src);
            }
            EnsureWeightsDense();
            return Bias + VectorUtils.DotProduct(in _weightsDense, in src);
        }

        protected virtual void GetFeatureContributions(in VBuffer<float> features, ref VBuffer<float> contributions, int top, int bottom, bool normalize)
        {
            if (features.Length != Weight.Length)
                throw Contracts.Except("Input is of length {0} does not match expected length  of weights {1}", features.Length, Weight.Length);

            var weights = Weight;
            features.CopyTo(ref contributions);
            VectorUtils.MulElementWise(in weights, ref contributions);
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

        ColumnType IValueMapper.InputType
        {
            get { return _inputType; }
        }

        ColumnType IValueMapper.OutputType
        {
            get { return NumberType.Float; }
        }

        ValueMapper<TIn, TOut> IValueMapper.GetMapper<TIn, TOut>()
        {
            Contracts.Check(typeof(TIn) == typeof(VBuffer<float>));
            Contracts.Check(typeof(TOut) == typeof(float));

            ValueMapper<VBuffer<float>, float> del =
                (in VBuffer<float> src, ref float dst) =>
                {
                    if (src.Length != Weight.Length)
                        throw Contracts.Except("Input is of length {0}, but predictor expected length {1}", src.Length, Weight.Length);
                    dst = Score(in src);
                };
            return (ValueMapper<TIn, TOut>)(Delegate)del;
        }

        /// <summary>
        /// Combine a bunch of models into one by averaging parameters
        /// </summary>
        private protected void CombineParameters(IList<IParameterMixer<float>> models, out VBuffer<float> weights, out float bias)
        {
            Type type = GetType();

            Contracts.Check(type == models[0].GetType(), "Submodel for parameter mixer has the wrong type");
            var first = (LinearModelParameters)models[0];

            weights = default(VBuffer<float>);
            first.Weight.CopyTo(ref weights);
            bias = first.Bias;

            for (int i = 1; i < models.Count; i++)
            {
                var m = models[i];
                Contracts.Check(type == m.GetType(), "Submodel for parameter mixer has the wrong type");

                var sub = (LinearModelParameters)m;
                var subweights = sub.Weight;
                VectorUtils.Add(in subweights, ref weights);
                bias += sub.Bias;
            }
            VectorUtils.ScaleBy(ref weights, (float)1 / models.Count);
            bias /= models.Count;
        }

        void ICanSaveInTextFormat.SaveAsText(TextWriter writer, RoleMappedSchema schema)
        {
            Host.CheckValue(writer, nameof(writer));
            Host.CheckValue(schema, nameof(schema));

            SaveSummary(writer, schema);
        }

        void ICanSaveInSourceCode.SaveAsCode(TextWriter writer, RoleMappedSchema schema)
        {
            Host.CheckValue(writer, nameof(writer));
            Host.CheckValue(schema, nameof(schema));

            var weights = Weight;
            LinearPredictorUtils.SaveAsCode(writer, in weights, Bias, schema);
        }

        [BestFriend]
        private protected abstract void SaveSummary(TextWriter writer, RoleMappedSchema schema);

        void ICanSaveSummary.SaveSummary(TextWriter writer, RoleMappedSchema schema) => SaveSummary(writer, schema);

        private protected virtual Row GetSummaryIRowOrNull(RoleMappedSchema schema)
        {
            var names = default(VBuffer<ReadOnlyMemory<char>>);
            MetadataUtils.GetSlotNames(schema, RoleMappedSchema.ColumnRole.Feature, Weight.Length, ref names);
            var subBuilder = new MetadataBuilder();
            subBuilder.AddSlotNames(Weight.Length, (ref VBuffer<ReadOnlyMemory<char>> dst) => names.CopyTo(ref dst));
            var colType = new VectorType(NumberType.R4, Weight.Length);
            var builder = new MetadataBuilder();
            builder.AddPrimitiveValue("Bias", NumberType.R4, Bias);
            builder.Add("Weights", colType, (ref VBuffer<float> dst) => Weight.CopyTo(ref dst), subBuilder.GetMetadata());
            return MetadataUtils.MetadataAsRow(builder.GetMetadata());
        }

        Row ICanGetSummaryAsIRow.GetSummaryIRowOrNull(RoleMappedSchema schema) => GetSummaryIRowOrNull(schema);

        private protected virtual Row GetStatsIRowOrNull(RoleMappedSchema schema) => null;

        Row ICanGetSummaryAsIRow.GetStatsIRowOrNull(RoleMappedSchema schema) => GetStatsIRowOrNull(schema);

        private protected abstract void SaveAsIni(TextWriter writer, RoleMappedSchema schema, ICalibrator calibrator = null);

        void ICanSaveInIniFormat.SaveAsIni(TextWriter writer, RoleMappedSchema schema, ICalibrator calibrator) => SaveAsIni(writer, schema, calibrator);

        public virtual void GetFeatureWeights(ref VBuffer<float> weights)
        {
            Weight.CopyTo(ref weights);
        }

        ValueMapper<TSrc, VBuffer<float>> IFeatureContributionMapper.GetFeatureContributionMapper<TSrc, TDstContributions>(int top, int bottom, bool normalize)
        {
            Contracts.Check(typeof(TSrc) == typeof(VBuffer<float>));
            Contracts.Check(typeof(TDstContributions) == typeof(VBuffer<float>));

            ValueMapper<VBuffer<float>, VBuffer<float>> del =
                (in VBuffer<float> src, ref VBuffer<float> dstContributions) =>
                {
                    GetFeatureContributions(in src, ref dstContributions, top, bottom, normalize);
                };
            return (ValueMapper<TSrc, VBuffer<float>>)(Delegate)del;
        }
    }

    public sealed partial class LinearBinaryModelParameters : LinearModelParameters,
        ICanGetSummaryInKeyValuePairs,
        IParameterMixer<float>
    {
        internal const string LoaderSignature = "Linear2CExec";
        internal const string RegistrationName = "LinearBinaryPredictor";

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
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(LinearBinaryModelParameters).Assembly.FullName);
        }

        /// <summary>
        /// Constructs a new linear binary predictor.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="weights">The weights for the linear predictor. Note that this
        /// will take ownership of the <see cref="VBuffer{T}"/>.</param>
        /// <param name="bias">The bias added to every output score.</param>
        /// <param name="stats"></param>
        public LinearBinaryModelParameters(IHostEnvironment env, in VBuffer<float> weights, float bias, LinearModelStatistics stats = null)
            : base(env, RegistrationName, in weights, bias)
        {
            Contracts.AssertValueOrNull(stats);
            _stats = stats;
        }

        private LinearBinaryModelParameters(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx)
        {
            // For model version earlier than 0x00020001, there is no model statisitcs.
            if (ctx.Header.ModelVerWritten <= 0x00020001)
                return;

            // *** Binary format ***
            // (Base class)
            // LinearModelStatistics: model statistics (optional, in a separate stream)

            ctx.LoadModelOrNull<LinearModelStatistics, SignatureLoadModel>(Host, out _stats, ModelStatsSubModelFilename);
        }

        private static IPredictorProducing<float> Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            var predictor = new LinearBinaryModelParameters(env, ctx);
            ICalibrator calibrator;
            ctx.LoadModelOrNull<ICalibrator, SignatureLoadModel>(env, out calibrator, @"Calibrator");
            if (calibrator == null)
                return predictor;
            if (calibrator is IParameterMixer)
                return new ParameterMixingCalibratedPredictor(env, predictor, calibrator);
            return new SchemaBindableCalibratedPredictor(env, predictor, calibrator);
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            // *** Binary format ***
            // (Base class)
            // LinearModelStatistics: model statistics (optional, in a separate stream)

            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            Contracts.AssertValueOrNull(_stats);
            if (_stats != null)
                ctx.SaveModel(_stats, ModelStatsSubModelFilename);
        }

        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        /// <summary>
        /// Combine a bunch of models into one by averaging parameters
        /// </summary>
        IParameterMixer<float> IParameterMixer<float>.CombineParameters(IList<IParameterMixer<float>> models)
        {
            VBuffer<float> weights;
            float bias;
            CombineParameters(models, out weights, out bias);
            return new LinearBinaryModelParameters(Host, in weights, bias);
        }

        private protected override void SaveSummary(TextWriter writer, RoleMappedSchema schema)
        {
            Host.CheckValue(schema, nameof(schema));

            // REVIEW: Would be nice to have the settings!
            var weights = Weight;
            writer.WriteLine(LinearPredictorUtils.LinearModelAsText("Linear Binary Classification Predictor", null, null,
                in weights, Bias, schema));

            _stats?.SaveText(writer, this, schema, 20);
        }

        ///<inheritdoc/>
        IList<KeyValuePair<string, object>> ICanGetSummaryInKeyValuePairs.GetSummaryInKeyValuePairs(RoleMappedSchema schema)
        {
            Host.CheckValue(schema, nameof(schema));

            var weights = Weight;
            List<KeyValuePair<string, object>> results = new List<KeyValuePair<string, object>>();
            LinearPredictorUtils.SaveLinearModelWeightsInKeyValuePairs(in weights, Bias, schema, results);
            _stats?.SaveSummaryInKeyValuePairs(this, schema, int.MaxValue, results);
            return results;
        }

        private protected override Row GetStatsIRowOrNull(RoleMappedSchema schema)
        {
            if (_stats == null)
                return null;
            var names = default(VBuffer<ReadOnlyMemory<char>>);
            MetadataUtils.GetSlotNames(schema, RoleMappedSchema.ColumnRole.Feature, Weight.Length, ref names);
            var meta = _stats.MakeStatisticsMetadata(this, schema, in names);
            return MetadataUtils.MetadataAsRow(meta);
        }

        private protected override void SaveAsIni(TextWriter writer, RoleMappedSchema schema, ICalibrator calibrator = null)
        {
            Host.CheckValue(writer, nameof(writer));
            Host.CheckValue(schema, nameof(schema));
            Host.CheckValueOrNull(calibrator);

            var weights = Weight;
            writer.Write(LinearPredictorUtils.LinearModelAsIni(in weights, Bias, this,
                schema, calibrator as PlattCalibrator));
        }
    }

    public abstract class RegressionModelParameters : LinearModelParameters
    {
       public RegressionModelParameters(IHostEnvironment env, string name, in VBuffer<float> weights, float bias)
            : base(env, name, in weights, bias)
        {
        }

        protected RegressionModelParameters(IHostEnvironment env, string name, ModelLoadContext ctx)
            : base(env, name, ctx)
        {
        }

        public override PredictionKind PredictionKind
        {
            get { return PredictionKind.Regression; }
        }

        /// <summary>
        /// Output the INI model to a given writer
        /// </summary>
        private protected override void SaveAsIni(TextWriter writer, RoleMappedSchema schema, ICalibrator calibrator)
        {
            if (calibrator != null)
                throw Host.ExceptNotImpl("Saving calibrators is not implemented yet.");

            Host.CheckValue(writer, nameof(writer));
            Host.CheckValue(schema, nameof(schema));

            // REVIEW: For Poisson should encode the exp operation in the ini as well, bug 2433.
            var weights = Weight;
            writer.Write(LinearPredictorUtils.LinearModelAsIni(in weights, Bias, this, schema, null));
        }
    }

    public sealed class LinearRegressionModelParameters : RegressionModelParameters,
        IParameterMixer<float>,
        ICanGetSummaryInKeyValuePairs
    {
        internal const string LoaderSignature = "LinearRegressionExec";
        internal const string RegistrationName = "LinearRegressionPredictor";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "LIN RGRS",
                // verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00020001, // Fixed sparse serialization
                verReadableCur: 0x00020001,
                verWeCanReadBack: 0x00020001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(LinearRegressionModelParameters).Assembly.FullName);
        }

        /// <summary>
        /// Constructs a new linear regression predictor.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="weights">The weights for the linear predictor. Note that this
        /// will take ownership of the <see cref="VBuffer{T}"/>.</param>
        /// <param name="bias">The bias added to every output score.</param>
        public LinearRegressionModelParameters(IHostEnvironment env, in VBuffer<float> weights, float bias)
            : base(env, RegistrationName, in weights, bias)
        {
        }

        private LinearRegressionModelParameters(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx)
        {
        }

        private static LinearRegressionModelParameters Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new LinearRegressionModelParameters(env, ctx);
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
        }

        private protected override void SaveSummary(TextWriter writer, RoleMappedSchema schema)
        {
            Host.CheckValue(writer, nameof(writer));
            Host.CheckValue(schema, nameof(schema));

            // REVIEW: Would be nice to have the settings!
            var weights = Weight;
            writer.WriteLine(LinearPredictorUtils.LinearModelAsText("Linear Regression Predictor", null, null,
                in weights, Bias, schema, null));
        }

        /// <summary>
        /// Combine a bunch of models into one by averaging parameters
        /// </summary>
        IParameterMixer<float> IParameterMixer<float>.CombineParameters(IList<IParameterMixer<float>> models)
        {
            VBuffer<float> weights;
            float bias;
            CombineParameters(models, out weights, out bias);
            return new LinearRegressionModelParameters(Host, in weights, bias);
        }

        ///<inheritdoc/>
        IList<KeyValuePair<string, object>> ICanGetSummaryInKeyValuePairs.GetSummaryInKeyValuePairs(RoleMappedSchema schema)
        {
            Host.CheckValue(schema, nameof(schema));

            var weights = Weight;
            List<KeyValuePair<string, object>> results = new List<KeyValuePair<string, object>>();
            LinearPredictorUtils.SaveLinearModelWeightsInKeyValuePairs(in weights, Bias, schema, results);

            return results;
        }
    }

    public sealed class PoissonRegressionModelParameters : RegressionModelParameters, IParameterMixer<float>
    {
        internal const string LoaderSignature = "PoissonRegressionExec";
        internal const string RegistrationName = "PoissonRegressionPredictor";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "POI RGRS",
                // verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00020001, // Fixed sparse serialization
                verReadableCur: 0x00020001,
                verWeCanReadBack: 0x00020001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(PoissonRegressionModelParameters).Assembly.FullName);
        }

        public PoissonRegressionModelParameters(IHostEnvironment env, in VBuffer<float> weights, float bias)
            : base(env, RegistrationName, in weights, bias)
        {
        }

        private PoissonRegressionModelParameters(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx)
        {
        }

        private static PoissonRegressionModelParameters Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new PoissonRegressionModelParameters(env, ctx);
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
        }

        protected override float Score(in VBuffer<float> src)
        {
            return MathUtils.ExpSlow(base.Score(in src));
        }

        private protected override void SaveSummary(TextWriter writer, RoleMappedSchema schema)
        {
            Host.CheckValue(writer, nameof(writer));
            Host.CheckValue(schema, nameof(schema));

            // REVIEW: Would be nice to have the settings!
            var weights = Weight;
            writer.WriteLine(LinearPredictorUtils.LinearModelAsText("Poisson Regression Predictor", null, null,
                in weights, Bias, schema, null));
        }

        /// <summary>
        /// Combine a bunch of models into one by averaging parameters
        /// </summary>
        IParameterMixer<float> IParameterMixer<float>.CombineParameters(IList<IParameterMixer<float>> models)
        {
            VBuffer<float> weights;
            float bias;
            CombineParameters(models, out weights, out bias);
            return new PoissonRegressionModelParameters(Host, in weights, bias);
        }
    }
}