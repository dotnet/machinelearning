// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Trainers;
using Microsoft.ML.Training;

[assembly: LoadableClass(RandomTrainer.Summary, typeof(RandomTrainer), typeof(RandomTrainer.Options),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer) },
    RandomTrainer.UserNameValue,
    RandomTrainer.LoadNameValue,
    "random")]

[assembly: LoadableClass(RandomTrainer.Summary, typeof(PriorTrainer), typeof(PriorTrainer.Options),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer) },
    PriorTrainer.UserNameValue,
    PriorTrainer.LoadNameValue,
    "prior",
    "constant")]

[assembly: LoadableClass(typeof(RandomModelParameters), null, typeof(SignatureLoadModel),
    "Random predictor", RandomModelParameters.LoaderSignature)]

[assembly: LoadableClass(typeof(PriorModelParameters), null, typeof(SignatureLoadModel),
    "Prior predictor", PriorModelParameters.LoaderSignature)]

namespace Microsoft.ML.Trainers
{
    /// <summary>
    /// A trainer that trains a predictor that returns random values
    /// </summary>
    public sealed class RandomTrainer : TrainerBase<RandomModelParameters>,
        ITrainerEstimator<BinaryPredictionTransformer<RandomModelParameters>, RandomModelParameters>
    {
        internal const string LoadNameValue = "RandomPredictor";
        internal const string UserNameValue = "Random Predictor";
        internal const string Summary = "A toy predictor that returns a random value.";

        internal sealed class Options
        {
        }

        /// <summary> Return the type of prediction task.</summary>
        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        private static readonly TrainerInfo _info = new TrainerInfo(normalization: false, caching: false);

        /// <summary>
        /// Auxiliary information about the trainer in terms of its capabilities
        /// and requirements.
        /// </summary>
        public override TrainerInfo Info => _info;

        /// <summary>
        /// Initializes RandomTrainer object.
        /// </summary>
        internal RandomTrainer(IHostEnvironment env)
            : base(env, LoadNameValue)
        {
        }

        internal RandomTrainer(IHostEnvironment env, Options options)
            : base(env, LoadNameValue)
        {
            Host.CheckValue(options, nameof(options));
        }

        /// <summary>
        /// Trains and returns a <see cref="BinaryPredictionTransformer{RandomModelParameters}"/>.
        /// </summary>
        public BinaryPredictionTransformer<RandomModelParameters> Fit(IDataView input)
        {
            RoleMappedData trainRoles = new RoleMappedData(input);
            var pred = Train(new TrainContext(trainRoles));
            return new BinaryPredictionTransformer<RandomModelParameters>(Host, pred, input.Schema, featureColumn: null);
        }

        private protected override RandomModelParameters Train(TrainContext context)
        {
            Host.CheckValue(context, nameof(context));
            return new RandomModelParameters(Host, Host.Rand.Next());
        }

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            var outColumns = inputSchema.ToDictionary(x => x.Name);

            var newColumns = new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata())),
                new SchemaShape.Column(DefaultColumnNames.Probability, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata(true))),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata()))
            };
            foreach (SchemaShape.Column column in newColumns)
                outColumns[column.Name] = column;

            return new SchemaShape(outColumns.Values);
        }
    }

    /// <summary>
    /// The predictor implements the Predict() interface. The predictor returns a
    ///  uniform random probability and classification assignment.
    /// </summary>
    public sealed class RandomModelParameters :
        ModelParametersBase<float>,
        IDistPredictorProducing<float, float>,
        IValueMapperDist
    {
        internal const string LoaderSignature = "RandomPredictor";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "RANDOMPR", // Unique 8-letter string.
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(RandomModelParameters).Assembly.FullName);
        }

        // Keep all the serializable state here.
        private readonly int _seed;
        private readonly object _instanceLock;
        private readonly Random _random;

        /// <summary> Return the type of prediction task.</summary>
        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        private readonly DataViewType _inputType;
        DataViewType IValueMapper.InputType => _inputType;
        DataViewType IValueMapper.OutputType => NumberDataViewType.Single;
        DataViewType IValueMapperDist.DistType => NumberDataViewType.Single;

        /// <summary>
        /// Instantiate a model that returns a uniform random probability.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="seed">The random seed.</param>
        internal RandomModelParameters(IHostEnvironment env, int seed)
            : base(env, LoaderSignature)
        {
            _seed = seed;

            _instanceLock = new object();
            _random = RandomUtils.Create(_seed);

            _inputType = new VectorType(NumberDataViewType.Single);
        }

        /// <summary>
        /// Load the predictor from the binary format.
        /// </summary>
        private RandomModelParameters(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, LoaderSignature, ctx)
        {
            // *** Binary format ***
            // int: _seed

            _seed = ctx.Reader.ReadInt32();

            _instanceLock = new object();
            _random = RandomUtils.Create(_seed);

            _inputType = new VectorType(NumberDataViewType.Single);
        }

        private static RandomModelParameters Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new RandomModelParameters(env, ctx);
        }

        /// <summary>
        /// Save the predictor in the binary format.
        /// </summary>
        /// <param name="ctx"></param>
        private protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: _seed

            ctx.Writer.Write(_seed);
        }

        ValueMapper<TIn, TOut> IValueMapper.GetMapper<TIn, TOut>()
        {
            Contracts.Check(typeof(TIn) == typeof(VBuffer<float>));
            Contracts.Check(typeof(TOut) == typeof(float));

            ValueMapper<VBuffer<float>, float> del = Map;
            return (ValueMapper<TIn, TOut>)(Delegate)del;
        }

        ValueMapper<TIn, TOut, TDist> IValueMapperDist.GetMapper<TIn, TOut, TDist>()
        {
            Contracts.Check(typeof(TIn) == typeof(VBuffer<float>));
            Contracts.Check(typeof(TOut) == typeof(float));
            Contracts.Check(typeof(TDist) == typeof(float));

            ValueMapper<VBuffer<float>, float, float> del = MapDist;
            return (ValueMapper<TIn, TOut, TDist>)(Delegate)del;
        }

        private float PredictCore()
        {
            // Predict can be called from different threads.
            // Ensure your implementation is thread-safe
            lock (_instanceLock)
            {
                // For binary classification, prediction may be in range [-infinity, infinity].
                //  default class decision boundary is at 0.
                return (_random.NextDouble() * 2 - 1).ToFloat();
            }
        }

        private void Map(in VBuffer<float> src, ref float dst)
        {
            dst = PredictCore();
        }

        private void MapDist(in VBuffer<float> src, ref float score, ref float prob)
        {
            score = PredictCore();
            prob = (score + 1) / 2;
        }
    }

    /// <summary>
    /// Learns the prior distribution for 0/1 class labels and outputs that.
    /// </summary>
    public sealed class PriorTrainer : TrainerBase<PriorModelParameters>,
        ITrainerEstimator<BinaryPredictionTransformer<PriorModelParameters>, PriorModelParameters>
    {
        internal const string LoadNameValue = "PriorPredictor";
        internal const string UserNameValue = "Prior Predictor";

        internal sealed class Options
        {
        }

        private readonly String _labelColumnName;
        private readonly String _weightColumnName;

        /// <summary> Return the type of prediction task.</summary>
        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        private static readonly TrainerInfo _info = new TrainerInfo(normalization: false, caching: false);

        /// <summary>
        /// Auxiliary information about the trainer in terms of its capabilities
        /// and requirements.
        /// </summary>
        public override TrainerInfo Info => _info;

        internal PriorTrainer(IHostEnvironment env, Options options)
            : base(env, LoadNameValue)
        {
            Host.CheckValue(options, nameof(options));
        }

        /// <summary>
        /// Initializes PriorTrainer object.
        /// </summary>
        internal PriorTrainer(IHostEnvironment env, String labelColumn, String weightColunn = null)
            : base(env, LoadNameValue)
        {
            Contracts.CheckValue(labelColumn, nameof(labelColumn));
            Contracts.CheckValueOrNull(weightColunn);

            _labelColumnName = labelColumn;
            _weightColumnName = weightColunn != null ? weightColunn : null;
        }

        /// <summary>
        /// Trains and returns a <see cref="BinaryPredictionTransformer{PriorModelParameters}"/>.
        /// </summary>
        public BinaryPredictionTransformer<PriorModelParameters> Fit(IDataView input)
        {
            RoleMappedData trainRoles = new RoleMappedData(input, feature: null, label: _labelColumnName, weight: _weightColumnName);
            var pred = Train(new TrainContext(trainRoles));
            return new BinaryPredictionTransformer<PriorModelParameters>(Host, pred, input.Schema, featureColumn: null);
        }

        private protected override PriorModelParameters Train(TrainContext context)
        {
            Host.CheckValue(context, nameof(context));
            var data = context.TrainingSet;
            data.CheckBinaryLabel();
            Host.CheckParam(data.Schema.Label.HasValue, nameof(data), "Missing Label column");
            var labelCol = data.Schema.Label.Value;
            Host.CheckParam(labelCol.Type == NumberDataViewType.Single, nameof(data), "Invalid type for Label column");

            double pos = 0;
            double neg = 0;

            int colWeight = -1;
            if (data.Schema.Weight?.Type == NumberDataViewType.Single)
                colWeight = data.Schema.Weight.Value.Index;

            var cols = colWeight > -1 ? new DataViewSchema.Column[] { labelCol, data.Schema.Weight.Value } : new DataViewSchema.Column[] { labelCol };

            using (var cursor = data.Data.GetRowCursor(cols))
            {
                var getLab = cursor.GetLabelFloatGetter(data);
                var getWeight = colWeight >= 0 ? cursor.GetGetter<float>(colWeight) : null;
                float lab = default;
                float weight = 1;
                while (cursor.MoveNext())
                {
                    getLab(ref lab);
                    if (getWeight != null)
                    {
                        getWeight(ref weight);
                        if (!(0 < weight && weight < float.PositiveInfinity))
                            continue;
                    }

                    // Testing both directions effectively ignores NaNs.
                    if (lab > 0)
                        pos += weight;
                    else if (lab <= 0)
                        neg += weight;
                }
            }

            float prob = prob = pos + neg > 0 ? (float)(pos / (pos + neg)) : float.NaN;
            return new PriorModelParameters(Host, prob);
        }

        private static SchemaShape.Column MakeFeatureColumn(string featureColumn)
            => new SchemaShape.Column(featureColumn, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Single, false);

        private static SchemaShape.Column MakeLabelColumn(string labelColumn)
            => new SchemaShape.Column(labelColumn, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false);

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            var outColumns = inputSchema.ToDictionary(x => x.Name);

            var newColumns = new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata())),
                new SchemaShape.Column(DefaultColumnNames.Probability, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata(true))),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata()))
            };
            foreach (SchemaShape.Column column in newColumns)
                outColumns[column.Name] = column;

            return new SchemaShape(outColumns.Values);
        }
    }

    public sealed class PriorModelParameters :
        ModelParametersBase<float>,
        IDistPredictorProducing<float, float>,
        IValueMapperDist
    {
        internal const string LoaderSignature = "PriorPredictor";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "PRIORPRD",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(PriorModelParameters).Assembly.FullName);
        }

        private readonly float _prob;
        private readonly float _raw;

        /// <summary>
        /// Instantiates a model that returns the prior probability of the positive class in the training set.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="prob">The probability of the positive class.</param>
        internal PriorModelParameters(IHostEnvironment env, float prob)
            : base(env, LoaderSignature)
        {
            Host.Check(!float.IsNaN(prob));

            _prob = prob;
            _raw = 2 * _prob - 1;       // This could be other functions -- logodds for instance

            _inputType = new VectorType(NumberDataViewType.Single);
        }

        private PriorModelParameters(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, LoaderSignature, ctx)
        {
            // *** Binary format ***
            // Float: _prob

            _prob = ctx.Reader.ReadFloat();
            Host.CheckDecode(!float.IsNaN(_prob));

            _raw = 2 * _prob - 1;

            _inputType = new VectorType(NumberDataViewType.Single);
        }

        private static PriorModelParameters Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new PriorModelParameters(env, ctx);
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // Float: _prob

            Contracts.Assert(!float.IsNaN(_prob));
            ctx.Writer.Write(_prob);
        }

        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        private readonly DataViewType _inputType;
        DataViewType IValueMapper.InputType => _inputType;
        DataViewType IValueMapper.OutputType => NumberDataViewType.Single;
        DataViewType IValueMapperDist.DistType => NumberDataViewType.Single;

        ValueMapper<TIn, TOut> IValueMapper.GetMapper<TIn, TOut>()
        {
            Contracts.Check(typeof(TIn) == typeof(VBuffer<float>));
            Contracts.Check(typeof(TOut) == typeof(float));

            ValueMapper<VBuffer<float>, float> del = Map;
            return (ValueMapper<TIn, TOut>)(Delegate)del;
        }

        ValueMapper<TIn, TOut, TDist> IValueMapperDist.GetMapper<TIn, TOut, TDist>()
        {
            Contracts.Check(typeof(TIn) == typeof(VBuffer<float>));
            Contracts.Check(typeof(TOut) == typeof(float));
            Contracts.Check(typeof(TDist) == typeof(float));

            ValueMapper<VBuffer<float>, float, float> del = MapDist;
            return (ValueMapper<TIn, TOut, TDist>)(Delegate)del;
        }

        private void Map(in VBuffer<float> src, ref float dst)
        {
            dst = _raw;
        }

        private void MapDist(in VBuffer<float> src, ref float score, ref float prob)
        {
            score = _raw;
            prob = _prob;
        }
    }
}
