// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Core.Data;

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

    public sealed class RandomTrainer : TrainerEstimatorBase<BinaryPredictionTransformer<RandomPredictor>, RandomPredictor>
    {
        internal const string LoadNameValue = "RandomPredictor";
        internal const string UserNameValue = "Random Predictor";
        internal const string Summary = "A toy predictor that returns a random value.";

        public class Arguments
        {
        }

        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        private static readonly TrainerInfo _info = new TrainerInfo(normalization: false, caching: false);
        public override TrainerInfo Info => _info;

        protected override SchemaShape.Column[] OutputColumns => throw new NotImplementedException();

        public RandomTrainer(IHost host, SchemaShape.Column feature, SchemaShape.Column label, SchemaShape.Column weight)
            : base(host, feature, label, weight)
        {
        }

        protected override RandomPredictor TrainModelCore(TrainContext trainContext)
        {
            Host.CheckValue(trainContext, nameof(trainContext));
            return new RandomPredictor(Host, Host.Rand.Next());
        }

        protected override BinaryPredictionTransformer<RandomPredictor> MakeTransformer(RandomPredictor model, ISchema trainSchema)
            => new BinaryPredictionTransformer<RandomPredictor>(Host, model, trainSchema, FeatureColumn.Name);
    }

    /// <summary>
    /// The predictor implements the Predict() interface. The predictor returns a
    ///  uniform random probability and classification assignment.
    /// </summary>
    public sealed class RandomPredictor :
        PredictorBase<Float>,
        IDistPredictorProducing<Float, Float>,
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
                loaderSignature: LoaderSignature);
        }

        // Keep all the serializable state here.
        private readonly int _seed;
        private readonly object _instanceLock;
        private readonly Random _random;

        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;
        public ColumnType InputType { get; }
        public ColumnType OutputType => NumberType.Float;
        public ColumnType DistType => NumberType.Float;

        public RandomPredictor(IHostEnvironment env, int seed)
            : base(env, LoaderSignature)
        {
            _seed = seed;

            _instanceLock = new object();
            _random = new Random(_seed);

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
            _random = new Random(_seed);
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
            Contracts.Check(typeof(TIn) == typeof(VBuffer<Float>));
            Contracts.Check(typeof(TOut) == typeof(Float));

            ValueMapper<VBuffer<Float>, Float> del = Map;
            return (ValueMapper<TIn, TOut>)(Delegate)del;
        }

        public ValueMapper<TIn, TOut, TDist> GetMapper<TIn, TOut, TDist>()
        {
            Contracts.Check(typeof(TIn) == typeof(VBuffer<Float>));
            Contracts.Check(typeof(TOut) == typeof(Float));
            Contracts.Check(typeof(TDist) == typeof(Float));

            ValueMapper<VBuffer<Float>, Float, Float> del = MapDist;
            return (ValueMapper<TIn, TOut, TDist>)(Delegate)del;
        }

        private Float PredictCore()
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

        private void Map(ref VBuffer<Float> src, ref Float dst)
        {
            dst = PredictCore();
        }

        private void MapDist(ref VBuffer<Float> src, ref Float score, ref Float prob)
        {
            score = PredictCore();
            prob = (score + 1) / 2;
        }
    }

    // Learns the prior distribution for 0/1 class labels and just outputs that.
    public sealed class PriorTrainer : TrainerEstimatorBase<BinaryPredictionTransformer<PriorPredictor>, PriorPredictor>
    {
        internal const string LoadNameValue = "PriorPredictor";
        internal const string UserNameValue = "Prior Predictor";

        public sealed class Arguments
        {
        }

        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        private static readonly TrainerInfo _info = new TrainerInfo(normalization: false, caching: false);
        public override TrainerInfo Info => _info;

        protected override SchemaShape.Column[] OutputColumns { get; }

        public PriorTrainer(IHost host, SchemaShape.Column feature, SchemaShape.Column label, SchemaShape.Column weight)
            : base(host, feature, label, weight)
        {
        }

        protected override PriorPredictor TrainModelCore(TrainContext context)
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
                var getWeight = colWeight >= 0 ? cursor.GetGetter<Float>(colWeight) : null;
                Float lab = default(Float);
                Float weight = 1;
                while (cursor.MoveNext())
                {
                    getLab(ref lab);
                    if (getWeight != null)
                    {
                        getWeight(ref weight);
                        if (!(0 < weight && weight < Float.PositiveInfinity))
                            continue;
                    }

                    // Testing both directions effectively ignores NaNs.
                    if (lab > 0)
                        pos += weight;
                    else if (lab <= 0)
                        neg += weight;
                }
            }

            Float prob = prob = pos + neg > 0 ? (Float)(pos / (pos + neg)) : Float.NaN;
            return new PriorPredictor(Host, prob);
        }

        protected override BinaryPredictionTransformer<PriorPredictor> MakeTransformer(PriorPredictor model, ISchema trainSchema)
             => new BinaryPredictionTransformer<PriorPredictor>(Host, model, trainSchema, FeatureColumn.Name);

    }

    public sealed class PriorPredictor :
        PredictorBase<Float>,
        IDistPredictorProducing<Float, Float>,
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
                loaderSignature: LoaderSignature);
        }

        private readonly Float _prob;
        private readonly Float _raw;

        public PriorPredictor(IHostEnvironment env, Float prob)
            : base(env, LoaderSignature)
        {
            Host.Check(!Float.IsNaN(prob));

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
            Host.CheckDecode(!Float.IsNaN(_prob));

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

            Contracts.Assert(!Float.IsNaN(_prob));
            ctx.Writer.Write(_prob);
        }

        public override PredictionKind PredictionKind
        { get { return PredictionKind.BinaryClassification; } }
        public ColumnType InputType { get; }
        public ColumnType OutputType => NumberType.Float;
        public ColumnType DistType => NumberType.Float;

        public ValueMapper<TIn, TOut> GetMapper<TIn, TOut>()
        {
            Contracts.Check(typeof(TIn) == typeof(VBuffer<Float>));
            Contracts.Check(typeof(TOut) == typeof(Float));

            ValueMapper<VBuffer<Float>, Float> del = Map;
            return (ValueMapper<TIn, TOut>)(Delegate)del;
        }

        public ValueMapper<TIn, TOut, TDist> GetMapper<TIn, TOut, TDist>()
        {
            Contracts.Check(typeof(TIn) == typeof(VBuffer<Float>));
            Contracts.Check(typeof(TOut) == typeof(Float));
            Contracts.Check(typeof(TDist) == typeof(Float));

            ValueMapper<VBuffer<Float>, Float, Float> del = MapDist;
            return (ValueMapper<TIn, TOut, TDist>)(Delegate)del;
        }

        private void Map(ref VBuffer<Float> src, ref Float dst)
        {
            dst = _raw;
        }

        private void MapDist(ref VBuffer<Float> src, ref Float score, ref Float prob)
        {
            score = _raw;
            prob = _prob;
        }
    }
}
