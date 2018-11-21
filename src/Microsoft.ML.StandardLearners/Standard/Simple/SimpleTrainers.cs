// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Training;
using System;
using System.Linq;

[assembly: LoadableClass(RandomTrainer.Summary, typeof(RandomTrainer), typeof(RandomTrainer.Arguments),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer) },
    RandomTrainer.UserNameValue,
    RandomTrainer.LoadNameValue,
    "random")]

[assembly: LoadableClass(RandomTrainer.Summary, typeof(PriorTrainer), typeof(PriorTrainer.Arguments),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer) },
    PriorTrainer.UserNameValue,
    PriorTrainer.LoadNameValue,
    "prior",
    "constant")]

[assembly: LoadableClass(typeof(RandomPredictor), null, typeof(SignatureLoadModel),
    "Random predictor", RandomPredictor.LoaderSignature)]

[assembly: LoadableClass(typeof(PriorPredictor), null, typeof(SignatureLoadModel),
    "Prior predictor", PriorPredictor.LoaderSignature)]

namespace Microsoft.ML.Runtime.Learners
{
    /// <summary>
    /// A trainer that trains a predictor that returns random values
    /// </summary>

    public sealed class RandomTrainer : TrainerBase<RandomPredictor>,
        ITrainerEstimator<BinaryPredictionTransformer<RandomPredictor>, RandomPredictor>
    {
        internal const string LoadNameValue = "RandomPredictor";
        internal const string UserNameValue = "Random Predictor";
        internal const string Summary = "A toy predictor that returns a random value.";

        public sealed class Arguments
        {
        }

        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        private static readonly TrainerInfo _info = new TrainerInfo(normalization: false, caching: false);
        public override TrainerInfo Info => _info;

        /// <summary>
        /// Initializes RandomTrainer object.
        /// </summary>
        public RandomTrainer(IHostEnvironment env)
            : base(env, LoadNameValue)
        {
        }

        public RandomTrainer(IHostEnvironment env, Arguments args)
            : base(env, LoadNameValue)
        {
            Host.CheckValue(args, nameof(args));
        }

        public BinaryPredictionTransformer<RandomPredictor> Fit(IDataView input)
        {
            var cachedTrain = Info.WantCaching ? new CacheDataView(Host, input, prefetch: null) : input;

            RoleMappedData trainRoles = new RoleMappedData(cachedTrain);
            var pred = Train(new TrainContext(trainRoles));
            return new BinaryPredictionTransformer<RandomPredictor>(Host, pred, cachedTrain.Schema, featureColumn: null);
        }

        public override RandomPredictor Train(TrainContext context)
        {
            Host.CheckValue(context, nameof(context));
            return new RandomPredictor(Host, Host.Rand.Next());
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            var outColumns = inputSchema.Columns.ToDictionary(x => x.Name);

            var newColumns = new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata())),
                new SchemaShape.Column(DefaultColumnNames.Probability, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata(true))),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BoolType.Instance, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata()))
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
    public sealed class RandomPredictor :
        PredictorBase<float>,
        IDistPredictorProducing<float, float>,
        IValueMapperDist,
        ICanSaveModel
    {
        public const string LoaderSignature = "RandomPredictor";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "RANDOMPR", // Unique 8-letter string.
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(RandomPredictor).Assembly.FullName);
        }

        // Keep all the serializable state here.
        private readonly int _seed;
        private readonly object _instanceLock;
        private readonly IRandom _random;

        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;
        public ColumnType InputType { get; }
        public ColumnType OutputType => NumberType.Float;
        public ColumnType DistType => NumberType.Float;

        public RandomPredictor(IHostEnvironment env, int seed)
            : base(env, LoaderSignature)
        {
            _seed = seed;

            _instanceLock = new object();
            _random = RandomUtils.Create(_seed);

            InputType = new VectorType(NumberType.Float);
        }

        /// <summary>
        /// Load the predictor from the binary format.
        /// </summary>
        private RandomPredictor(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, LoaderSignature, ctx)
        {
            // *** Binary format ***
            // int: _seed

            _seed = ctx.Reader.ReadInt32();

            _instanceLock = new object();
            _random = RandomUtils.Create(_seed);

            InputType = new VectorType(NumberType.Float);
        }

        public static RandomPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new RandomPredictor(env, ctx);
        }

        /// <summary>
        /// Save the predictor in the binary format.
        /// </summary>
        /// <param name="ctx"></param>
        protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: _seed

            ctx.Writer.Write(_seed);
        }

        public ValueMapper<TIn, TOut> GetMapper<TIn, TOut>()
        {
            Contracts.Check(typeof(TIn) == typeof(VBuffer<float>));
            Contracts.Check(typeof(TOut) == typeof(float));

            ValueMapper<VBuffer<float>, float> del = Map;
            return (ValueMapper<TIn, TOut>)(Delegate)del;
        }

        public ValueMapper<TIn, TOut, TDist> GetMapper<TIn, TOut, TDist>()
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
    /// Learns the prior distribution for 0/1 class labels and just outputs that.
    /// </summary>
    public sealed class PriorTrainer : TrainerBase<PriorPredictor>,
        ITrainerEstimator<BinaryPredictionTransformer<PriorPredictor>, PriorPredictor>
    {
        internal const string LoadNameValue = "PriorPredictor";
        internal const string UserNameValue = "Prior Predictor";

        public sealed class Arguments
        {
        }

        private readonly String _labelColumnName;
        private readonly String _weightColumnName;

        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        private static readonly TrainerInfo _info = new TrainerInfo(normalization: false, caching: false);
        public override TrainerInfo Info => _info;

        public PriorTrainer(IHostEnvironment env, Arguments args)
            : base(env, LoadNameValue)
        {
            Host.CheckValue(args, nameof(args));
        }

        /// <summary>
        /// Initializes PriorTrainer object.
        /// </summary>
        public PriorTrainer(IHost host, String labelColumn, String weightColunn = null)
            : base(host, LoadNameValue)
        {
            Contracts.CheckValue(labelColumn, nameof(labelColumn));
            Contracts.CheckValueOrNull(weightColunn);

            _labelColumnName = labelColumn;
            _weightColumnName = weightColunn != null ? weightColunn : null;
        }

        public BinaryPredictionTransformer<PriorPredictor> Fit(IDataView input)
        {
            var cachedTrain = Info.WantCaching ? new CacheDataView(Host, input, prefetch: null) : input;

            RoleMappedData trainRoles = new RoleMappedData(cachedTrain, feature: null, label: _labelColumnName, weight: _weightColumnName);
            var pred = Train(new TrainContext(trainRoles));
            return new BinaryPredictionTransformer<PriorPredictor>(Host, pred, cachedTrain.Schema, featureColumn: null);
        }

        public override PriorPredictor Train(TrainContext context)
        {
            Contracts.CheckValue(context, nameof(context));
            var data = context.TrainingSet;
            data.CheckBinaryLabel();
            Contracts.CheckParam(data.Schema.Label != null, nameof(data), "Missing Label column");
            Contracts.CheckParam(data.Schema.Label.Type == NumberType.Float, nameof(data), "Invalid type for Label column");

            double pos = 0;
            double neg = 0;

            int col = data.Schema.Label.Index;
            int colWeight = -1;
            if (data.Schema.Weight?.Type == NumberType.Float)
                colWeight = data.Schema.Weight.Index;
            using (var cursor = data.Data.GetRowCursor(c => c == col || c == colWeight))
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
            return new PriorPredictor(Host, prob);
        }

        private static SchemaShape.Column MakeFeatureColumn(string featureColumn)
            => new SchemaShape.Column(featureColumn, SchemaShape.Column.VectorKind.Vector, NumberType.R4, false);

        private static SchemaShape.Column MakeLabelColumn(string labelColumn)
            => new SchemaShape.Column(labelColumn, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false);

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            var outColumns = inputSchema.Columns.ToDictionary(x => x.Name);

            var newColumns = new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata())),
                new SchemaShape.Column(DefaultColumnNames.Probability, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata(true))),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BoolType.Instance, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata()))
            };
            foreach (SchemaShape.Column column in newColumns)
                outColumns[column.Name] = column;

            return new SchemaShape(outColumns.Values);
        }
    }

    public sealed class PriorPredictor :
        PredictorBase<float>,
        IDistPredictorProducing<float, float>,
        IValueMapperDist,
        ICanSaveModel
    {
        public const string LoaderSignature = "PriorPredictor";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "PRIORPRD",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(PriorPredictor).Assembly.FullName);
        }

        private readonly float _prob;
        private readonly float _raw;

        public PriorPredictor(IHostEnvironment env, float prob)
            : base(env, LoaderSignature)
        {
            Host.Check(!float.IsNaN(prob));

            _prob = prob;
            _raw = 2 * _prob - 1;       // This could be other functions -- logodds for instance

            InputType = new VectorType(NumberType.Float);
        }

        private PriorPredictor(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, LoaderSignature, ctx)
        {
            // *** Binary format ***
            // Float: _prob

            _prob = ctx.Reader.ReadFloat();
            Host.CheckDecode(!float.IsNaN(_prob));

            _raw = 2 * _prob - 1;

            InputType = new VectorType(NumberType.Float);
        }

        public static PriorPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new PriorPredictor(env, ctx);
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // Float: _prob

            Contracts.Assert(!float.IsNaN(_prob));
            ctx.Writer.Write(_prob);
        }

        public override PredictionKind PredictionKind
        { get { return PredictionKind.BinaryClassification; } }
        public ColumnType InputType { get; }
        public ColumnType OutputType => NumberType.Float;
        public ColumnType DistType => NumberType.Float;

        public ValueMapper<TIn, TOut> GetMapper<TIn, TOut>()
        {
            Contracts.Check(typeof(TIn) == typeof(VBuffer<float>));
            Contracts.Check(typeof(TOut) == typeof(float));

            ValueMapper<VBuffer<float>, float> del = Map;
            return (ValueMapper<TIn, TOut>)(Delegate)del;
        }

        public ValueMapper<TIn, TOut, TDist> GetMapper<TIn, TOut, TDist>()
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
