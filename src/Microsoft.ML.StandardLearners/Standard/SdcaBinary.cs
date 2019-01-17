// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Data.Conversion;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Calibration;
using Microsoft.ML.Internal.CpuMath;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Learners;
using Microsoft.ML.Numeric;
using Microsoft.ML.Trainers;
using Microsoft.ML.Training;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(typeof(SdcaBinaryTrainer), typeof(SdcaBinaryTrainer.Arguments),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer), typeof(SignatureFeatureScorerTrainer) },
    SdcaBinaryTrainer.UserNameValue,
    SdcaBinaryTrainer.LoadNameValue,
    "LinearClassifier",
    "lc",
    "sasdca")]

[assembly: LoadableClass(typeof(StochasticGradientDescentClassificationTrainer), typeof(StochasticGradientDescentClassificationTrainer.Arguments),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer), typeof(SignatureFeatureScorerTrainer) },
    StochasticGradientDescentClassificationTrainer.UserNameValue,
    StochasticGradientDescentClassificationTrainer.LoadNameValue,
    "sgd")]

[assembly: LoadableClass(typeof(void), typeof(Sdca), null, typeof(SignatureEntryPointModule), "SDCA")]
[assembly: LoadableClass(typeof(void), typeof(StochasticGradientDescentClassificationTrainer), null, typeof(SignatureEntryPointModule), StochasticGradientDescentClassificationTrainer.ShortName)]

namespace Microsoft.ML.Trainers
{
    using ConditionalAttribute = System.Diagnostics.ConditionalAttribute;
    using Stopwatch = System.Diagnostics.Stopwatch;
    using TScalarPredictor = IPredictorWithFeatureWeights<float>;

    public abstract class LinearTrainerBase<TTransformer, TModel> : TrainerEstimatorBase<TTransformer, TModel>
        where TTransformer : ISingleFeaturePredictionTransformer<TModel>
        where TModel : IPredictor
    {
        private const string RegisterName = nameof(LinearTrainerBase<TTransformer, TModel>);

        protected bool NeedShuffle;

        private static readonly TrainerInfo _info = new TrainerInfo();
        public override TrainerInfo Info => _info;

        /// <summary>
        /// Whether data is to be shuffled every epoch.
        /// </summary>
        protected abstract bool ShuffleData { get; }

        private protected LinearTrainerBase(IHostEnvironment env, string featureColumn, SchemaShape.Column labelColumn,
            string weightColumn = null)
            : base(Contracts.CheckRef(env, nameof(env)).Register(RegisterName), TrainerUtils.MakeR4VecFeature(featureColumn),
                          labelColumn, TrainerUtils.MakeR4ScalarWeightColumn(weightColumn))
        {
        }

        private protected override TModel TrainModelCore(TrainContext context)
        {
            Host.CheckValue(context, nameof(context));
            using (var ch = Host.Start("Training"))
            {
                var preparedData = PrepareDataFromTrainingExamples(ch, context.TrainingSet, out int weightSetCount);
                var initPred = context.InitialPredictor;
                var linInitPred = (initPred as CalibratedPredictorBase)?.SubPredictor as LinearModelParameters;
                linInitPred = linInitPred ?? initPred as LinearModelParameters;
                Host.CheckParam(context.InitialPredictor == null || linInitPred != null, nameof(context),
                    "Initial predictor was not a linear predictor.");
                return TrainCore(ch, preparedData, linInitPred, weightSetCount);
            }
        }

        private protected abstract TModel TrainCore(IChannel ch, RoleMappedData data, LinearModelParameters predictor, int weightSetCount);

        /// <summary>
        /// This method ensures that the data meets the requirements of this trainer and its
        /// subclasses, injects necessary transforms, and throws if it couldn't meet them.
        /// </summary>
        /// <param name="ch">The channel</param>
        /// <param name="examples">The training examples</param>
        /// <param name="weightSetCount">Gets the length of weights and bias array. For binary classification and regression,
        /// this is 1. For multi-class classification, this equals the number of classes on the label.</param>
        /// <returns>A potentially modified version of <paramref name="examples"/></returns>
        private protected RoleMappedData PrepareDataFromTrainingExamples(IChannel ch, RoleMappedData examples, out int weightSetCount)
        {
            ch.AssertValue(examples);
            CheckLabel(examples, out weightSetCount);
            examples.CheckFeatureFloatVector();
            var idvToShuffle = examples.Data;
            IDataView idvToFeedTrain;
            if (idvToShuffle.CanShuffle)
                idvToFeedTrain = idvToShuffle;
            else
            {
                var shuffleArgs = new RowShufflingTransformer.Arguments
                {
                    PoolOnly = false,
                    ForceShuffle = ShuffleData
                };
                idvToFeedTrain = new RowShufflingTransformer(Host, shuffleArgs, idvToShuffle);
            }

            ch.Assert(idvToFeedTrain.CanShuffle);

            var roles = examples.Schema.GetColumnRoleNames();
            var examplesToFeedTrain = new RoleMappedData(idvToFeedTrain, roles);

            ch.Assert(examplesToFeedTrain.Schema.Label.HasValue);
            ch.Assert(examplesToFeedTrain.Schema.Feature.HasValue);
            if (examples.Schema.Weight.HasValue)
                ch.Assert(examplesToFeedTrain.Schema.Weight.HasValue);

            ch.Check(examplesToFeedTrain.Schema.Feature.Value.Type is VectorType vecType && vecType.Size > 0, "Training set has no features, aborting training.");
            return examplesToFeedTrain;
        }

        private protected abstract void CheckLabel(RoleMappedData examples, out int weightSetCount);

        protected float WDot(in VBuffer<float> features, in VBuffer<float> weights, float bias)
        {
            return VectorUtils.DotProduct(in weights, in features) + bias;
        }

        protected float WScaledDot(in VBuffer<float> features, Double scaling, in VBuffer<float> weights, float bias)
        {
            return VectorUtils.DotProduct(in weights, in features) * (float)scaling + bias;
        }

        private protected virtual int ComputeNumThreads(FloatLabelCursor.Factory cursorFactory)
        {
            int maxThreads = Math.Min(8, Math.Max(1, Environment.ProcessorCount / 2));
            if (0 < Host.ConcurrencyFactor && Host.ConcurrencyFactor < maxThreads)
                maxThreads = Host.ConcurrencyFactor;

            return maxThreads;
        }
    }

    public abstract class SdcaTrainerBase<TArgs, TTransformer, TModel> : StochasticTrainerBase<TTransformer, TModel>
        where TTransformer : ISingleFeaturePredictionTransformer<TModel>
        where TModel : IPredictor
        where TArgs : SdcaTrainerBase<TArgs, TTransformer, TModel>.ArgumentsBase, new()
    {
        // REVIEW: Making it even faster and more accurate:
        // 1. Train with not-too-many threads. nt = 2 or 4 seems to be good enough. Didn't seem additional benefit over more threads.
        // 2. Make tol smaller. 0.1 seems to be too large. 0.01 is more ideal.
        // 3. Don't "guess" the iteration to converge. It is very data-set dependent and hard to control. Always check for at least once to ensure convergence.
        // 4. Use dual variable updates to infer whether a full iteration of convergence checking is necessary. Convergence checking iteration is time-consuming.

        public abstract class ArgumentsBase : LearnerInputBaseWithLabel
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "L2 regularizer constant. By default the l2 constant is automatically inferred based on data set.", NullName = "<Auto>", ShortName = "l2", SortOrder = 1)]
            [TGUI(Label = "L2 Regularizer Constant", SuggestedSweeps = "<Auto>,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2")]
            [TlcModule.SweepableDiscreteParam("L2Const", new object[] { "<Auto>", 1e-7f, 1e-6f, 1e-5f, 1e-4f, 1e-3f, 1e-2f })]
            public float? L2Const;

            // REVIEW: make the default positive when we know how to consume a sparse model
            [Argument(ArgumentType.AtMostOnce, HelpText = "L1 soft threshold (L1/L2). Note that it is easier to control and sweep using the threshold parameter than the raw L1-regularizer constant. By default the l1 threshold is automatically inferred based on data set.", NullName = "<Auto>", ShortName = "l1", SortOrder = 2)]
            [TGUI(Label = "L1 Soft Threshold", SuggestedSweeps = "<Auto>,0,0.25,0.5,0.75,1")]
            [TlcModule.SweepableDiscreteParam("L1Threshold", new object[] { "<Auto>", 0f, 0.25f, 0.5f, 0.75f, 1f })]
            public float? L1Threshold;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Degree of lock-free parallelism. Defaults to automatic. Determinism not guaranteed.", NullName = "<Auto>", ShortName = "nt,t,threads", SortOrder = 50)]
            [TGUI(Label = "Number of threads", SuggestedSweeps = "<Auto>,1,2,4")]
            public int? NumThreads;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The tolerance for the ratio between duality gap and primal loss for convergence checking.", ShortName = "tol")]
            [TGUI(SuggestedSweeps = "0.001, 0.01, 0.1, 0.2")]
            [TlcModule.SweepableDiscreteParam("ConvergenceTolerance", new object[] { 0.001f, 0.01f, 0.1f, 0.2f })]
            public float ConvergenceTolerance = 0.1f;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum number of iterations; set to 1 to simulate online learning. Defaults to automatic.", NullName = "<Auto>", ShortName = "iter")]
            [TGUI(Label = "Max number of iterations", SuggestedSweeps = "<Auto>,10,20,100")]
            [TlcModule.SweepableDiscreteParam("MaxIterations", new object[] { "<Auto>", 10, 20, 100 })]
            public int? MaxIterations;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Shuffle data every epoch?", ShortName = "shuf")]
            [TlcModule.SweepableDiscreteParam("Shuffle", null, isBool: true)]
            public bool Shuffle = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Convergence check frequency (in terms of number of iterations). Set as negative or zero for not checking at all. If left blank, it defaults to check after every 'numThreads' iterations.", NullName = "<Auto>", ShortName = "checkFreq")]
            public int? CheckFrequency;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The learning rate for adjusting bias from being regularized.", ShortName = "blr")]
            [TGUI(SuggestedSweeps = "0, 0.01, 0.1, 1")]
            [TlcModule.SweepableDiscreteParam("BiasLearningRate", new object[] { 0.0f, 0.01f, 0.1f, 1f })]
            public float BiasLearningRate = 0;

            internal virtual void Check(IHostEnvironment env)
            {
                Contracts.AssertValue(env);
                env.CheckUserArg(L2Const == null || L2Const >= 0, nameof(L2Const), "L2 constant must be non-negative.");
                env.CheckUserArg(L1Threshold == null || L1Threshold >= 0, nameof(L1Threshold), "L1 threshold must be non-negative.");
                env.CheckUserArg(MaxIterations == null || MaxIterations > 0, nameof(MaxIterations), "Max number of iterations must be positive.");
                env.CheckUserArg(ConvergenceTolerance > 0 && ConvergenceTolerance <= 1, nameof(ConvergenceTolerance), "Convergence tolerance must be positive and no larger than 1.");

                if (L2Const < L2LowerBound)
                {
                    using (var ch = env.Start("SDCA arguments checking"))
                    {
                        ch.Warning($"The L2 regularization constant must be at least {L2LowerBound}. In SDCA, the dual formulation " +
                            $"is only valid with a positive constant, and values below {L2LowerBound} cause very slow convergence. " +
                            $"The original {nameof(L2Const)} = {L2Const}, was replaced with {nameof(L2Const)} = {L2LowerBound}.");
                        L2Const = L2LowerBound;
                    }
                }
            }
        }

        // The order of these matter, since they are used as indices into arrays.
        protected enum MetricKind
        {
            Loss,
            DualLoss,
            DualityGap,
            L1Sparsity,
            BiasUnreg,
            BiasReg
        }

        // The maximum number of dual variables SDCA intends to support.
        // Actual bound of training dataset size may depend on hardware limit.
        // Note that currently the maximum dimension linear learners can support is about 2 billion,
        // it is not clear if training a linear learner with more than 10^15 examples provides
        // substantial additional benefits in terms of accuracy.
        private const long MaxDualTableSize = 1L << 50;
        private const float L2LowerBound = 1e-09f;
        protected readonly TArgs Args;
        protected ISupportSdcaLoss Loss;

        protected override bool ShuffleData => Args.Shuffle;

        private const string RegisterName = nameof(SdcaTrainerBase<TArgs, TTransformer, TModel>);

        private static TArgs ArgsInit(string featureColumn, SchemaShape.Column labelColumn, Action<TArgs> advancedSettings = null)
        {
            var args = new TArgs();

            // Apply the advanced args, if the user supplied any.
            advancedSettings?.Invoke(args);
            args.FeatureColumn = featureColumn;
            args.LabelColumn = labelColumn.Name;
            return args;
        }

        internal SdcaTrainerBase(IHostEnvironment env, string featureColumn, SchemaShape.Column labelColumn,
           SchemaShape.Column weight = default, Action<TArgs> advancedSettings = null, float? l2Const = null,
            float? l1Threshold = null, int? maxIterations = null)
          : this(env, ArgsInit(featureColumn, labelColumn, advancedSettings), labelColumn, weight, l2Const, l1Threshold, maxIterations)
        {
        }

        internal SdcaTrainerBase(IHostEnvironment env, TArgs args, SchemaShape.Column label, SchemaShape.Column weight = default,
            float? l2Const = null, float? l1Threshold = null, int? maxIterations = null)
            : base(Contracts.CheckRef(env, nameof(env)).Register(RegisterName), TrainerUtils.MakeR4VecFeature(args.FeatureColumn), label, weight)
        {
            Args = args;
            Args.L2Const = l2Const ?? args.L2Const;
            Args.L1Threshold = l1Threshold ?? args.L1Threshold;
            Args.MaxIterations = maxIterations ?? args.MaxIterations;
            Args.Check(env);
        }

        protected float WDot(in VBuffer<float> features, in VBuffer<float> weights, float bias)
        {
            return VectorUtils.DotProduct(in weights, in features) + bias;
        }

        private protected sealed override TModel TrainCore(IChannel ch, RoleMappedData data, LinearModelParameters predictor, int weightSetCount)
        {
            Contracts.Assert(predictor == null, "SDCA based trainers don't support continuous training.");
            Contracts.Assert(weightSetCount >= 1);

            int numFeatures = data.Schema.Feature.Value.Type.GetVectorSize();
            long maxTrainingExamples = MaxDualTableSize / weightSetCount;
            var cursorFactory = new FloatLabelCursor.Factory(data, CursOpt.Label | CursOpt.Features | CursOpt.Weight | CursOpt.Id);
            int numThreads;
            if (Args.NumThreads.HasValue)
            {
                numThreads = Args.NumThreads.Value;
                Host.CheckUserArg(numThreads > 0, nameof(ArgumentsBase.NumThreads), "The number of threads must be either null or a positive integer.");
                if (0 < Host.ConcurrencyFactor && Host.ConcurrencyFactor < numThreads)
                {
                    numThreads = Host.ConcurrencyFactor;
                    ch.Warning("The number of threads specified in trainer arguments is larger than the concurrency factor "
                        + "setting of the environment. Using {0} training thread(s) instead.", numThreads);
                }
            }
            else
                numThreads = ComputeNumThreads(cursorFactory);

            ch.Assert(numThreads > 0);
            if (numThreads == 1)
                ch.Info("Using 1 thread to train.");
            else
                ch.Info("Using {0} threads to train.", numThreads);

            int checkFrequency = 0;
            if (Args.CheckFrequency.HasValue)
                checkFrequency = Args.CheckFrequency.Value;
            else
            {
                checkFrequency = numThreads;
                ch.Info("Automatically choosing a check frequency of {0}.", checkFrequency);
            }

            if (checkFrequency <= 0)
                checkFrequency = int.MaxValue;

            ch.Assert(checkFrequency > 0);

            var pOptions = new ParallelOptions { MaxDegreeOfParallelism = numThreads };
            bool converged = false;

            // Getting the total count of rows in data. Ignore rows with bad label and feature values.
            long count = 0;

            // The maximum value of cursor.Id.Low.
            ulong idLoMax = 0;

            // Counting the number of bad training examples.
            long numBad = 0;

            // Indicates whether a lookup structure is needed.
            // Typicall we need a lookup structure if the id of training examples are not contiguous.
            bool needLookup = false;
            IdToIdxLookup idToIdx = null;

            using (var cursor = cursorFactory.Create())
            using (var pch = Host.StartProgressChannel("SDCA preprocessing"))
            {
                pch.SetHeader(new ProgressHeader("examples"), e => e.SetProgress(0, count));
                while (cursor.MoveNext())
                {
                    RowId id = cursor.Id;
                    if (id.High > 0 || id.Low >= (ulong)maxTrainingExamples)
                    {
                        needLookup = true;
                        break;
                    }
                    else
                    {
                        Contracts.Assert(id.High == 0);
                        if (id.Low > idLoMax)
                            idLoMax = id.Low;
                    }

                    count++;
                }

                Contracts.Assert(count <= maxTrainingExamples);
                if (!needLookup)
                {
                    Contracts.Assert((long)idLoMax < maxTrainingExamples);
                    numBad = cursor.SkippedRowCount;
                    pch.Checkpoint(count, count);

                    // REVIEW: Is 1024 a good lower bound to enforce sparsity?
                    if (1024 < count && count < (long)idLoMax / 5)
                    {
                        // The distribution of id.Low is sparse in [0, idLoMax].
                        // Building a lookup table is more memory efficient.
                        needLookup = true;
                    }
                }
            }

            if (needLookup)
            {
                // Note: At this point, 'count' may be less than the actual count of training examples.
                // We initialize the hash table with this partial size to avoid unnecessary rehashing.
                // However, it does not mean there are exactly 'count' many trainining examples.
                // Necessary rehashing will still occur as the hash table grows.
                idToIdx = new IdToIdxLookup(count);
                // Resetting 'count' to zero.
                count = 0;

                using (var cursor = cursorFactory.Create())
                using (var pch = Host.StartProgressChannel("SDCA preprocessing with lookup"))
                {
                    pch.SetHeader(new ProgressHeader("examples"), e => e.SetProgress(0, count));
                    while (cursor.MoveNext())
                    {
                        idToIdx.Add(cursor.Id);
                        count++;
                    }

                    numBad = cursor.SkippedRowCount;
                    pch.Checkpoint(count, count);
                }

                Contracts.Assert(idToIdx.Count == count);
            }

            if (numBad > 0)
                ch.Warning("Skipped {0} instances with missing features/label during training", numBad);

            ch.Check(count > 0, "Training set has 0 instances, aborting training.");
            // Tune the default hyperparameters based on dataset size.
            if (Args.MaxIterations == null)
                Args.MaxIterations = TuneDefaultMaxIterations(ch, count, numThreads);

            Contracts.Assert(Args.MaxIterations.HasValue);
            if (Args.L2Const == null)
                Args.L2Const = TuneDefaultL2(ch, Args.MaxIterations.Value, count, numThreads);

            Contracts.Assert(Args.L2Const.HasValue);
            if (Args.L1Threshold == null)
                Args.L1Threshold = TuneDefaultL1(ch, numFeatures);

            ch.Assert(Args.L1Threshold.HasValue);
            var l1Threshold = Args.L1Threshold.Value;
            var l1ThresholdZero = l1Threshold == 0;
            var weights = new VBuffer<float>[weightSetCount];
            var bestWeights = new VBuffer<float>[weightSetCount];
            var l1IntermediateWeights = l1ThresholdZero ? null : new VBuffer<float>[weightSetCount];
            var biasReg = new float[weightSetCount];
            var bestBiasReg = new float[weightSetCount];
            var biasUnreg = new float[weightSetCount];
            var bestBiasUnreg = new float[weightSetCount];
            var l1IntermediateBias = l1ThresholdZero ? null : new float[weightSetCount];

            for (int i = 0; i < weightSetCount; i++)
            {
                weights[i] = VBufferUtils.CreateDense<float>(numFeatures);
                bestWeights[i] = VBufferUtils.CreateDense<float>(numFeatures);
                biasReg[i] = 0;
                bestBiasReg[i] = 0;
                biasUnreg[i] = 0;
                bestBiasUnreg[i] = 0;

                if (!l1ThresholdZero)
                {
                    l1IntermediateWeights[i] = VBufferUtils.CreateDense<float>(numFeatures);
                    l1IntermediateBias[i] = 0;
                }
            }

            int bestIter = 0;
            var bestPrimalLoss = double.PositiveInfinity;
            ch.Assert(Args.L2Const.HasValue);
            var l2Const = Args.L2Const.Value;
            float lambdaNInv = 1 / (l2Const * count);

            DualsTableBase duals = null;
            float[] invariants = null;
            float[] featureNormSquared = null;

            if (idToIdx == null)
            {
                Contracts.Assert(!needLookup);
                long dualsLength = ((long)idLoMax + 1) * weightSetCount;
                if (dualsLength <= Utils.ArrayMaxSize)
                {
                    // The dual variables fit into a standard float[].
                    // Also storing invariants in a starndard float[].
                    duals = new StandardArrayDualsTable((int)dualsLength);
                    int invariantsLength = (int)idLoMax + 1;
                    Contracts.Assert(invariantsLength <= Utils.ArrayMaxSize);
                    invariants = new float[invariantsLength];
                }
                else
                {
                    // The dual variables do not fit into standard float[].
                    // Using BigArray<Float> instead.
                    // Storing the invariants gives rise to too large memory consumption,
                    // so we favor re-computing the invariants instead of storing them.
                    Contracts.Assert(dualsLength <= MaxDualTableSize);
                    duals = new BigArrayDualsTable(dualsLength);
                }
            }
            else
            {
                // Similar logic as above when using the id-to-index lookup.
                Contracts.Assert(needLookup);
                long dualsLength = count * weightSetCount;
                if (dualsLength <= Utils.ArrayMaxSize)
                {
                    duals = new StandardArrayDualsTable((int)dualsLength);
                    Contracts.Assert(count <= Utils.ArrayMaxSize);
                    invariants = new float[count];
                }
                else
                {
                    Contracts.Assert(dualsLength <= MaxDualTableSize);
                    duals = new BigArrayDualsTable(count);
                }
            }

            if (invariants != null)
            {
                Contracts.Assert(Utils.Size(invariants) > 0);
                featureNormSquared = InitializeFeatureNormSquared(invariants.Length);
            }

            Contracts.AssertValue(duals);
            Contracts.AssertValueOrNull(invariants);

            string[] metricNames;
            Double[] metrics;
            InitializeConvergenceMetrics(out metricNames, out metrics);
            ch.AssertValue(metricNames);
            ch.AssertValue(metrics);
            ch.Assert(metricNames.Length == metrics.Length);
            ch.Assert(Args.MaxIterations.HasValue);
            var maxIterations = Args.MaxIterations.Value;

            var rands = new Random[maxIterations];
            for (int i = 0; i < maxIterations; i++)
                rands[i] = RandomUtils.Create(Host.Rand.Next());

            // If we favor storing the invariants, precompute the invariants now.
            if (invariants != null)
            {
                Contracts.Assert((idToIdx == null & ((long)idLoMax + 1) * weightSetCount <= Utils.ArrayMaxSize) | (idToIdx != null & count * weightSetCount <= Utils.ArrayMaxSize));
                Func<RowId, long, long> getIndexFromIdAndRow = GetIndexFromIdAndRowGetter(idToIdx, biasReg.Length);
                int invariantCoeff = weightSetCount == 1 ? 1 : 2;
                using (var cursor = cursorFactory.Create())
                using (var pch = Host.StartProgressChannel("SDCA invariants initialization"))
                {
                    long row = 0;
                    pch.SetHeader(new ProgressHeader("examples"), e => e.SetProgress(0, row, count));
                    while (cursor.MoveNext())
                    {
                        long longIdx = getIndexFromIdAndRow(cursor.Id, row);
                        Contracts.Assert(0 <= longIdx & longIdx < invariants.Length, $"longIdx={longIdx}, invariants.Length={invariants.Length}");
                        int idx = (int)longIdx;
                        var features = cursor.Features;
                        var normSquared = VectorUtils.NormSquared(features);
                        if (Args.BiasLearningRate == 0)
                            normSquared += 1;

                        if (featureNormSquared != null)
                            featureNormSquared[idx] = normSquared;

                        // REVIEW: For log-loss, the default loss function for binary classifiation, a large number
                        // of the invariants are 1. Maybe worth to consider a more efficient way to store the invariants
                        // for log-loss.
                        invariants[idx] = Loss.ComputeDualUpdateInvariant(invariantCoeff * normSquared * lambdaNInv * GetInstanceWeight(cursor));
                        row++;
                    }
                }
            }

            // Start training.
            using (var pch = Host.StartProgressChannel("SDCA training"))
            {
                int iter = 0;
                pch.SetHeader(new ProgressHeader(metricNames, new[] { "iterations" }), e => e.SetProgress(0, iter, maxIterations));

                // Separate logic is needed for single-thread execution to ensure the result is deterministic.
                // Note that P.Invoke does not ensure that the actions executes in order even if maximum number of threads is set to 1.
                if (numThreads == 1)
                {
                    // The synchorized SDCA procedure.
                    for (iter = 0; iter < maxIterations; iter++)
                    {
                        if (converged)
                            break;

                        TrainWithoutLock(pch, cursorFactory, rands[iter], idToIdx, numThreads, duals, biasReg, invariants, lambdaNInv, weights, biasUnreg, l1IntermediateWeights, l1IntermediateBias, featureNormSquared);
                        if ((iter + 1) % checkFrequency == 0)
                            converged = CheckConvergence(pch, iter + 1, cursorFactory, duals, idToIdx, weights, bestWeights, biasUnreg, bestBiasUnreg, biasReg, bestBiasReg, count, metrics, ref bestPrimalLoss, ref bestIter);
                    }
                }
                else
                {
                    int numCycles = maxIterations / checkFrequency;
                    int residue = maxIterations % checkFrequency;
                    var convergenceChecked = false;

                    for (int i = 0; i < numCycles; i++)
                    {
                        iter = i * checkFrequency;
                        if (i > 0)
                        {
                            converged = CheckConvergence(pch, iter, cursorFactory, duals, idToIdx, weights, bestWeights, biasUnreg, bestBiasUnreg, biasReg, bestBiasReg, count, metrics, ref bestPrimalLoss, ref bestIter);
                            convergenceChecked = true;
                        }

                        if (converged)
                            break;

                        Parallel.For(0, checkFrequency, pOptions,
                                j => TrainWithoutLock(j == 0 ? pch : null, cursorFactory, rands[iter + j], idToIdx, numThreads, duals, biasReg, invariants, lambdaNInv, weights, biasUnreg, l1IntermediateWeights, l1IntermediateBias, featureNormSquared));
                        // Don't sync: https://www.cs.utexas.edu/~cjhsieh/dcd_parallel_final.main.pdf
                    }

                    iter = numCycles * checkFrequency;
                    if ((residue != 0) && (!converged))
                    {
                        if (numCycles > 0)
                        {
                            converged = CheckConvergence(pch, iter, cursorFactory, duals, idToIdx, weights, bestWeights, biasUnreg, bestBiasUnreg, biasReg, bestBiasReg, count, metrics, ref bestPrimalLoss, ref bestIter);
                            convergenceChecked = true;
                        }

                        Parallel.For(0, residue, pOptions,
                                j => TrainWithoutLock(j == 0 ? pch : null, cursorFactory, rands[iter + j], idToIdx, numThreads, duals, biasReg, invariants, lambdaNInv, weights, biasUnreg, l1IntermediateWeights, l1IntermediateBias, featureNormSquared));
                        // Don't sync: https://www.cs.utexas.edu/~cjhsieh/dcd_parallel_final.main.pdf
                    }

                    if (convergenceChecked && !converged)
                        converged = CheckConvergence(pch, maxIterations, cursorFactory, duals, idToIdx, weights, bestWeights, biasUnreg, bestBiasUnreg, biasReg, bestBiasReg, count, metrics, ref bestPrimalLoss, ref bestIter);
                }
            }

            var bias = new float[weightSetCount];
            if (bestIter > 0)
            {
                ch.Info("Using best model from iteration {0}.", bestIter);
                weights = bestWeights;
                for (int i = 0; i < weightSetCount; i++)
                    bias[i] = bestBiasReg[i] + bestBiasUnreg[i];
            }
            else
            {
                ch.Info("Using model from last iteration.");
                for (int i = 0; i < weightSetCount; i++)
                    bias[i] = biasReg[i] + biasUnreg[i];
            }
            return CreatePredictor(weights, bias);
        }

        protected abstract TModel CreatePredictor(VBuffer<float>[] weights, float[] bias);

        // Assign an upper bound for number of iterations based on data set size first.
        // This ensures SDCA will not run forever...
        // Based on empirical estimation of max iterations needed.
        private static int TuneDefaultMaxIterations(IChannel ch, long count, int numThreads)
        {
            Contracts.AssertValue(ch);
            Contracts.Assert(count > 0);
            Contracts.Assert(numThreads > 0);

            int maxIterations = (int)Math.Max(1, 500000 / count) * 3;
            // Round toward an integral multiple of numThread.
            maxIterations = Math.Max(2, Math.Max(1, maxIterations / numThreads) * numThreads);
            ch.Info("Auto-tuning parameters: maxIterations = {0}.", maxIterations);
            return maxIterations;
        }

        // Tune default for l2.
        protected virtual float TuneDefaultL2(IChannel ch, int maxIterations, long rowCount, int numThreads)
        {
            Contracts.AssertValue(ch);
            Contracts.Assert(maxIterations > 0);
            Contracts.Assert(rowCount > 0);
            Contracts.Assert(numThreads > 0);

            // Based on empirical estimation of expected iterations needed.
            long expectedIterations = Math.Max(2, maxIterations / 2 / numThreads) * numThreads;
            Contracts.Assert(expectedIterations > 0);
            // Suggest the empirically best l2 for good AUC and fast convergence.
            float l2 = Math.Max(1e-06f, 20f / (expectedIterations * rowCount));
            ch.Info("Auto-tuning parameters: L2 = {0}.", l2);
            // REVIEW: Info this line when getting an accurate estimate.
            // ch.Info("Expected to converge in approximately {0} iterations.", expectedIterations * Math.Max(1, -Math.Log10(_args.convergenceTolerance)));
            return l2;
        }

        // Tune default for l1Threshold.
        private static float TuneDefaultL1(IChannel ch, int numFeatures)
        {
            Contracts.Assert(numFeatures > 0);
            float l1Threshold;
            if (numFeatures < 100)
                l1Threshold = 0;
            else if (numFeatures < 1000)
                l1Threshold = 0.25f;
            else if (numFeatures < 10000)
                l1Threshold = 0.5f;
            else
                l1Threshold = 1;

            ch.Info("Auto-tuning parameters: L1Threshold (L1/L2) = {0}.", l1Threshold);
            return l1Threshold;
        }

        /// <summary>
        /// Returns the names of the metrics reported by <see cref="CheckConvergence"/>, as well as the initial values.
        /// </summary>
        private void InitializeConvergenceMetrics(out string[] names, out Double[] initialValues)
        {
            names = new[] { "Loss", "Dual Loss", "Duality Gap", "Sparsity (L1)", "Unregularized Bias", "Regularized Bias" };
            initialValues = new Double[] { Double.PositiveInfinity, 0, Double.PositiveInfinity, 0, 0, 0 };
        }

        /// <summary>
        /// Train the SDCA optimizer with one iteration over the entire training examples.
        /// </summary>
        /// <param name="progress">The progress reporting channel.</param>
        /// <param name="cursorFactory">The cursor factory to create cursors over the training examples.</param>
        /// <param name="rand">
        /// The random number generator to generate random numbers for randomized shuffling of the training examples.
        /// It may be null. When it is null, the training examples are not shuffled and are cursored in its original order.
        /// </param>
        /// <param name="idToIdx">
        /// The id to index mapping. May be null. If it is null, the index is given by the
        /// corresponding lower bits of the id.
        /// </param>
        /// <param name="numThreads">The number of threads used in parallel training. It is used in computing the dual update.</param>
        /// <param name="duals">
        /// The dual variables. For binary classification and regression, there is one dual variable per row.
        /// For multiclass classification, there is one dual variable per class per row.
        /// </param>
        /// <param name="biasReg">The array containing regularized bias terms. For binary classification or regression,
        /// it contains only a single value. For multiclass classification its size equals the number of classes.</param>
        /// <param name="invariants">
        /// The dual updates invariants. It may be null. If not null, it holds an array of pre-computed numerical quantities
        /// that depend on the training example label and features, not the value of dual variables.
        /// </param>
        /// <param name="lambdaNInv">The precomputed numerical quantity 1 / (l2Const * (count of training examples)).</param>
        /// <param name="weights">
        /// The weights array. For binary classification or regression, it consists of only one VBuffer.
        /// For multiclass classification, its size equals the number of classes.
        /// </param>
        /// <param name="biasUnreg">
        /// The array containing unregularized bias terms. For binary classification or regression,
        /// it contains only a single value. For multiclass classification its size equals the number of classes.
        /// </param>
        /// <param name="l1IntermediateWeights">
        /// The array holding the intermediate weights prior to making L1 shrinkage adjustment. It is null iff l1Threshold is zero.
        /// Otherwise, for binary classification or regression, it consists of only one VBuffer;
        /// for multiclass classification, its size equals the number of classes.
        /// </param>
        /// <param name="l1IntermediateBias">
        /// The array holding the intermediate bias prior to making L1 shrinkage adjustment. It is null iff l1Threshold is zero.
        /// Otherwise, for binary classification or regression, it consists of only one value;
        /// for multiclass classification, its size equals the number of classes.
        /// </param>
        /// <param name="featureNormSquared">
        /// The array holding the pre-computed squared L2-norm of features for each training example. It may be null. It is always null for
        /// binary classification and regression because this quantity is not needed.
        /// </param>
        private protected virtual void TrainWithoutLock(IProgressChannelProvider progress, FloatLabelCursor.Factory cursorFactory, Random rand,
            IdToIdxLookup idToIdx, int numThreads, DualsTableBase duals, float[] biasReg, float[] invariants, float lambdaNInv,
            VBuffer<float>[] weights, float[] biasUnreg, VBuffer<float>[] l1IntermediateWeights, float[] l1IntermediateBias, float[] featureNormSquared)
        {
            Contracts.AssertValueOrNull(progress);
            Contracts.Assert(Args.L1Threshold.HasValue);
            Contracts.AssertValueOrNull(idToIdx);
            Contracts.AssertValueOrNull(invariants);
            Contracts.AssertValueOrNull(featureNormSquared);
            int maxUpdateTrials = 2 * numThreads;
            var l1Threshold = Args.L1Threshold.Value;
            bool l1ThresholdZero = l1Threshold == 0;
            var lr = Args.BiasLearningRate * Args.L2Const.Value;
            var pch = progress != null ? progress.StartProgressChannel("Dual update") : null;
            using (pch)
            using (var cursor = Args.Shuffle ? cursorFactory.Create(rand) : cursorFactory.Create())
            {
                long rowCount = 0;
                if (pch != null)
                    pch.SetHeader(new ProgressHeader("examples"), e => e.SetProgress(0, rowCount));

                Func<RowId, long> getIndexFromId = GetIndexFromIdGetter(idToIdx, biasReg.Length);
                while (cursor.MoveNext())
                {
                    long idx = getIndexFromId(cursor.Id);
                    VBuffer<float> features = cursor.Features;
                    var label = cursor.Label;
                    float invariant;
                    if (invariants != null)
                        invariant = invariants[idx];
                    else
                    {
                        Contracts.Assert(featureNormSquared == null);
                        var featuresNormSquared = VectorUtils.NormSquared(features);
                        if (Args.BiasLearningRate == 0)
                            featuresNormSquared += 1;

                        invariant = Loss.ComputeDualUpdateInvariant(featuresNormSquared * lambdaNInv * GetInstanceWeight(cursor));
                    }

                    var weightsEditor = VBufferEditor.CreateFromBuffer(ref weights[0]);
                    var l1IntermediateWeightsEditor =
                        !l1ThresholdZero ? VBufferEditor.CreateFromBuffer(ref l1IntermediateWeights[0]) :
                        default;

                    for (int numTrials = 0; numTrials < maxUpdateTrials; numTrials++)
                    {
                        var dual = duals[idx];
                        var output = WDot(in features, in weights[0], biasReg[0] + biasUnreg[0]);
                        var dualUpdate = Loss.DualUpdate(output, label, dual, invariant, numThreads);

                        // The successive over-relaxation apporach to adjust the sum of dual variables (biasReg) to zero.
                        // Reference to details: http://stat.rutgers.edu/home/tzhang/papers/ml02_dual.pdf pp. 16-17.
                        var adjustment = l1ThresholdZero ? lr * biasReg[0] : lr * l1IntermediateBias[0];
                        dualUpdate -= adjustment;
                        bool success = false;
                        duals.ApplyAt(idx, (long index, ref float value) =>
                        {
                            success = Interlocked.CompareExchange(ref value, dual + dualUpdate, dual) == dual;
                        });

                        if (success)
                        {
                            // Note: dualConstraint = lambdaNInv * (sum of duals)
                            var instanceWeight = GetInstanceWeight(cursor);
                            var primalUpdate = dualUpdate * lambdaNInv * instanceWeight;
                            biasUnreg[0] += adjustment * lambdaNInv * instanceWeight;

                            if (l1ThresholdZero)
                            {
                                VectorUtils.AddMult(in features, weightsEditor.Values, primalUpdate);
                                biasReg[0] += primalUpdate;
                            }
                            else
                            {
                                //Iterative shrinkage-thresholding (aka. soft-thresholding)
                                //Update v=denseWeights as if there's no L1
                                //Thresholding: if |v[j]| < threshold, turn off weights[j]
                                //If not, shrink: w[j] = v[i] - sign(v[j]) * threshold
                                l1IntermediateBias[0] += primalUpdate;
                                if (Args.BiasLearningRate == 0)
                                {
                                    biasReg[0] = Math.Abs(l1IntermediateBias[0]) - l1Threshold > 0.0
                                    ? l1IntermediateBias[0] - Math.Sign(l1IntermediateBias[0]) * l1Threshold
                                        : 0;
                                }

                                var featureValues = features.GetValues();
                                if (features.IsDense)
                                    CpuMathUtils.SdcaL1UpdateDense(primalUpdate, featureValues.Length, featureValues, l1Threshold, l1IntermediateWeightsEditor.Values, weightsEditor.Values);
                                else if (featureValues.Length > 0)
                                    CpuMathUtils.SdcaL1UpdateSparse(primalUpdate, featureValues.Length, featureValues, features.GetIndices(), l1Threshold, l1IntermediateWeightsEditor.Values, weightsEditor.Values);
                            }

                            break;
                        }
                    }
                    rowCount++;
                }
            }
        }

        /// <summary>
        ///  Returns whether the algorithm converged, and also populates the <paramref name="metrics"/>
        /// (which is expected to be parallel to the names returned by <see cref="InitializeConvergenceMetrics"/>).
        /// When called, the <paramref name="metrics"/> is expected to hold the previously reported values.
        /// </summary>
        /// <param name="pch">The progress reporting channel.</param>
        /// <param name="iter">The iteration number, zero based.</param>
        /// <param name="cursorFactory">The cursor factory to create cursors over the training data.</param>
        /// <param name="duals">
        /// The dual variables. For binary classification and regression, there is one dual variable per row.
        /// For multiclass classification, there is one dual variable per class per row.
        /// </param>
        /// <param name="idToIdx">
        /// The id to index mapping. May be null. If it is null, the index is given by the
        /// corresponding lower bits of the id.
        /// </param>
        /// <param name="weights">
        /// The weights array. For binary classification or regression, it consists of only one VBuffer.
        /// For multiclass classification, its size equals the number of classes.
        /// </param>
        /// <param name="bestWeights">
        /// The weights array that corresponds to the best model obtained from the training iterations thus far.
        /// </param>
        /// <param name="biasUnreg">
        /// The array containing unregularized bias terms. For binary classification or regression,
        /// it contains only a single value. For multiclass classification its size equals the number of classes.
        /// </param>
        /// <param name="bestBiasUnreg">
        /// The array containing unregularized bias terms corresponding to the best model obtained from the training iterations thus far.
        /// For binary classification or regression, it contains only a single value.
        /// For multiclass classification its size equals the number of classes.
        /// </param>
        /// <param name="biasReg">
        /// The array containing regularized bias terms. For binary classification or regression,
        /// it contains only a single value. For multiclass classification its size equals the number of classes.
        /// </param>
        /// <param name="bestBiasReg">
        /// The array containing regularized bias terms corresponding to the best model obtained from the training iterations thus far.
        /// For binary classification or regression, it contains only a single value.
        /// For multiclass classification its size equals the number of classes.
        /// </param>
        /// <param name="count">
        /// The count of (valid) training examples. Bad training examples are excluded from this count.
        /// </param>
        /// <param name="metrics">
        /// The array of metrics for progress reporting.
        /// </param>
        /// <param name="bestPrimalLoss">
        /// The primal loss function value corresponding to the best model obtained thus far.
        /// </param>
        /// <param name="bestIter">The iteration number when the best model is obtained.</param>
        /// <returns>Whether the optimization has converged.</returns>
        private protected virtual bool CheckConvergence(
            IProgressChannel pch,
            int iter,
            FloatLabelCursor.Factory cursorFactory,
            DualsTableBase duals,
            IdToIdxLookup idToIdx,
            VBuffer<float>[] weights,
            VBuffer<float>[] bestWeights,
            float[] biasUnreg,
            float[] bestBiasUnreg,
            float[] biasReg,
            float[] bestBiasReg,
            long count,
            Double[] metrics,
            ref Double bestPrimalLoss,
            ref int bestIter)
        {
            Contracts.AssertValueOrNull(idToIdx);
            Contracts.Assert(Utils.Size(metrics) == 6);
            var reportedValues = new Double?[metrics.Length + 1];
            reportedValues[metrics.Length] = iter;
            var lossSum = new CompensatedSum();
            var dualLossSum = new CompensatedSum();
            var biasTotal = biasReg[0] + biasUnreg[0];
            VBuffer<float> firstWeights = weights[0];

            using (var cursor = cursorFactory.Create())
            {
                long row = 0;
                Func<RowId, long, long> getIndexFromIdAndRow = GetIndexFromIdAndRowGetter(idToIdx, biasReg.Length);
                // Iterates through data to compute loss function.
                while (cursor.MoveNext())
                {
                    var instanceWeight = GetInstanceWeight(cursor);
                    var features = cursor.Features;
                    var output = WDot(in features, in weights[0], biasTotal);
                    Double subLoss = Loss.Loss(output, cursor.Label);
                    long idx = getIndexFromIdAndRow(cursor.Id, row);
                    Double subDualLoss = Loss.DualLoss(cursor.Label, duals[idx]);
                    lossSum.Add(subLoss * instanceWeight);
                    dualLossSum.Add(subDualLoss * instanceWeight);
                    row++;
                }
                Host.Assert(idToIdx == null || row == duals.Length);
            }

            Contracts.Assert(Args.L2Const.HasValue);
            Contracts.Assert(Args.L1Threshold.HasValue);
            Double l2Const = Args.L2Const.Value;
            Double l1Threshold = Args.L1Threshold.Value;
            Double l1Regularizer = l1Threshold * l2Const * (VectorUtils.L1Norm(in weights[0]) + Math.Abs(biasReg[0]));
            var l2Regularizer = l2Const * (VectorUtils.NormSquared(weights[0]) + biasReg[0] * biasReg[0]) * 0.5;
            var newLoss = lossSum.Sum / count + l2Regularizer + l1Regularizer;
            var newDualLoss = dualLossSum.Sum / count - l2Regularizer - l2Const * biasUnreg[0] * biasReg[0];
            var loss = metrics[(int)MetricKind.Loss];

            metrics[(int)MetricKind.Loss] = newLoss;
            metrics[(int)MetricKind.DualLoss] = newDualLoss;
            var dualityGap = metrics[(int)MetricKind.DualityGap] = newLoss - newDualLoss;
            metrics[(int)MetricKind.BiasUnreg] = biasUnreg[0];
            metrics[(int)MetricKind.BiasReg] = biasReg[0];
            metrics[(int)MetricKind.L1Sparsity] = Args.L1Threshold == 0 ? 1 : (Double)firstWeights.GetValues().Count(w => w != 0) / weights.Length;

            bool converged = dualityGap / newLoss < Args.ConvergenceTolerance;

            if (metrics[(int)MetricKind.Loss] < bestPrimalLoss)
            {
                // Maintain a copy of weights and bias with best primal loss thus far.
                // This is some extra work and uses extra memory, but it seems worth doing it.
                // REVIEW: Sparsify bestWeights?
                firstWeights.CopyTo(ref bestWeights[0]);
                bestBiasReg[0] = biasReg[0];
                bestBiasUnreg[0] = biasUnreg[0];
                bestPrimalLoss = metrics[(int)MetricKind.Loss];
                bestIter = iter;
            }

            for (int i = 0; i < metrics.Length; i++)
                reportedValues[i] = metrics[i];
            if (pch != null)
                pch.Checkpoint(reportedValues);

            return converged;
        }

        protected virtual float[] InitializeFeatureNormSquared(int length)
        {
            return null;
        }

        private protected abstract float GetInstanceWeight(FloatLabelCursor cursor);

        private protected delegate void Visitor(long index, ref float value);

        /// <summary>
        /// Encapsulates the common functionality of storing and
        /// retrieving the dual variables.
        /// </summary>
        private protected abstract class DualsTableBase
        {
            public abstract float this[long index] { get; set; }
            public abstract long Length { get; }
            public abstract void ApplyAt(long index, Visitor manip);
        }

        /// <summary>
        /// Implementation of <see cref="DualsTableBase"/> using a standard array.
        /// </summary>
        private sealed class StandardArrayDualsTable : DualsTableBase
        {
            private float[] _duals;

            public override long Length => _duals.Length;

            public StandardArrayDualsTable(int length)
            {
                Contracts.Assert(length <= Utils.ArrayMaxSize);
                _duals = new float[length];
            }

            public override float this[long index]
            {
                get => _duals[(int)index];
                set => _duals[(int)index] = value;
            }

            public override void ApplyAt(long index, Visitor manip)
            {
                manip(index, ref _duals[(int)index]);
            }
        }

        /// <summary>
        /// Implementation of <see cref="DualsTableBase"/> using a big array.
        /// </summary>
        private sealed class BigArrayDualsTable : DualsTableBase
        {
            private BigArray<float> _duals;

            public override long Length => _duals.Length;

            public BigArrayDualsTable(long length)
            {
                Contracts.Assert(length <= 1L << 50);
                _duals = new BigArray<float>(length);
            }

            public override float this[long index]
            {
                get => _duals[index];
                set => _duals[index] = value;
            }

            public override void ApplyAt(long index, Visitor manip)
            {
                BigArray<float>.Visitor manip2 = (long idx, ref float value) => manip(idx, ref value);
                _duals.ApplyAt(index, manip2);
            }
        }

        /// <summary>
        /// Returns a function delegate to retrieve index from id.
        /// This is to avoid redundant conditional branches in the tight loop of training.
        /// </summary>
        protected Func<RowId, long> GetIndexFromIdGetter(IdToIdxLookup idToIdx, int biasLength)
        {
            Contracts.AssertValueOrNull(idToIdx);
            long maxTrainingExamples = MaxDualTableSize / biasLength;
            if (idToIdx == null)
            {
                return (RowId id) =>
                {
                    Contracts.Assert(id.High == 0);
                    Contracts.Assert((long)id.Low < maxTrainingExamples);
                    return (long)id.Low;
                };
            }
            else
            {
                return (RowId id) =>
                {
                    long idx;
                    bool found = idToIdx.TryGetIndex(id, out idx);
                    Contracts.Assert(found);
                    Contracts.Assert(0 <= idx && idx < idToIdx.Count);
                    return (long)idx;
                };
            }
        }

        /// <summary>
        /// Returns a function delegate to retrieve index from id and row.
        /// Only works if the cursor is not shuffled.
        /// This is to avoid redundant conditional branches in the tight loop of training.
        /// </summary>
        protected Func<RowId, long, long> GetIndexFromIdAndRowGetter(IdToIdxLookup idToIdx, int biasLength)
        {
            Contracts.AssertValueOrNull(idToIdx);
            long maxTrainingExamples = MaxDualTableSize / biasLength;
            if (idToIdx == null)
            {
                return (RowId id, long row) =>
                {
                    Contracts.Assert(id.High == 0);
                    Contracts.Assert((long)id.Low < maxTrainingExamples);
                    return (long)id.Low;
                };
            }
            else
            {
                return (RowId id, long row) =>
                {
#if DEBUG
                    long idx;
                    bool found = idToIdx.TryGetIndex(id, out idx);
                    Contracts.Assert(found);
                    Contracts.Assert(0 <= idx && idx < idToIdx.Count);
                    Contracts.Assert(idx == row);
#endif
                    return row;
                };
            }
        }

        // REVIEW: This data structure is an extension of HashArray. It may have general
        // purpose of usage to store Id. Should consider lifting this class in the future.
        // This class can also be made to accommodate generic type, as long as the type implements a
        // good 64-bit hash function.
        /// <summary>
        /// A hash table data structure to store Id of type <see cref="T:Microsoft.ML.Data.RowId"/>,
        /// and accommodates size larger than 2 billion. This class is an extension based on BCL.
        /// Two operations are supported: adding and retrieving an id with asymptotically constant complexity.
        /// The bucket size are prime numbers, starting from 3 and grows to the next prime larger than
        /// double the current size until it reaches the maximum possible size. When a table growth is triggered,
        /// the table growing operation initializes a new larger bucket and rehash the existing entries to
        /// the new bucket. Such operation has an expected complexity proportional to the size.
        /// </summary>
        protected sealed class IdToIdxLookup
        {
            // Utilizing this struct gives better cache behavior than using parallel arrays.
            private readonly struct Entry
            {
                public readonly long ItNext;
                public readonly RowId Value;

                public Entry(long itNext, RowId value)
                {
                    ItNext = itNext;
                    Value = value;
                }
            }

            // Buckets of prime size.
            private BigArray<long> _rgit;

            // Count of Id stored.
            private long _count;

            // The entries.
            private BigArray<Entry> _entries;

            /// <summary>
            /// Gets the count of id entries.
            /// </summary>
            public long Count => _count;

            /// <summary>
            /// Initializes an instance of the <see cref="IdToIdxLookup"/> class with the specified size.
            /// </summary>
            public IdToIdxLookup(long size = 0)
            {
                Contracts.Assert(size >= 0);
                long prime = HashHelpers.GetPrime(size);
                _rgit = new BigArray<long>(prime);
                _rgit.FillRange(0, _rgit.Length, -1);
                _entries = new BigArray<Entry>();
                AssertValid();
            }

            /// <summary>
            /// Make sure the given id is in this lookup table and return the index of the id.
            /// </summary>
            public long Add(RowId id)
            {
                long iit = GetIit(Get64BitHashCode(id));
                long index = GetIndexCore(id, iit);
                if (index >= 0)
                    return index;

                return AddCore(id, iit);
            }

            /// <summary>
            /// Find the index of the given id.
            /// Returns a bool representing if id is present.
            /// Index outputs the index that the id, -1 otherwise.
            /// </summary>
            public bool TryGetIndex(RowId id, out long index)
            {
                AssertValid();
                index = GetIndexCore(id, GetIit(Get64BitHashCode(id)));
                return index >= 0;
            }

            private long GetIit(long hash)
            {
                return (long)((ulong)hash % (ulong)_rgit.Length);
            }

            /// <summary>
            /// Return the index of value, -1 if it is not present.
            /// </summary>
            private long GetIndexCore(RowId val, long iit)
            {
                Contracts.Assert(0 <= iit & iit < _rgit.Length);
                long it = _rgit[iit];
                while (it >= 0)
                {
                    Contracts.Assert(it < _count);
                    Entry entry = _entries[it];
                    if (entry.Value.Equals(val))
                        return it;
                    // Get the next item in the bucket.
                    it = entry.ItNext;
                }
                Contracts.Assert(it == -1);
                return -1;
            }

            /// <summary>
            /// Adds the value as a TItem. Does not check whether the TItem is already present.
            /// Returns the index of the added value.
            /// </summary>
            private long AddCore(RowId val, long iit)
            {
                AssertValid();
                Contracts.Assert(0 <= iit && iit < _rgit.Length);

                if (_count >= _entries.Length)
                {
                    Contracts.Assert(_count == _entries.Length);
                    _entries.Resize(_count + 1);
                }

                Contracts.Assert(_count < _entries.Length);
                _entries.ApplyAt(_count, (long index, ref Entry entry) => { entry = new Entry(_rgit[iit], val); });
                _rgit[iit] = _count;

                if (++_count >= _rgit.Length)
                    GrowTable();

                AssertValid();

                // Return the index of the added value.
                return _count - 1;
            }

            private void GrowTable()
            {
                AssertValid();

                long size = HashHelpers.ExpandPrime(_count);
                Contracts.Assert(size >= _rgit.Length);
                if (size <= _rgit.Length)
                    return;

                // Populate new buckets.
                DumpStats();
                _rgit = new BigArray<long>(size);
                FillTable();
                DumpStats();

                AssertValid();
            }

            private void FillTable()
            {
                _rgit.ApplyRange(0, _rgit.Length, (long index, ref long value) => { value = -1; });
                _entries.ApplyRange(0, _count,
                    (long it, ref Entry entry) =>
                    {
                        RowId value = entry.Value;
                        long iit = GetIit(Get64BitHashCode(entry.Value));
                        entry = new Entry(_rgit[iit], value);
                        _rgit[iit] = it;
                    });
            }

            [Conditional("DEBUG")]
            private void AssertValid()
            {
                Contracts.AssertValue(_rgit);
                Contracts.Assert(_rgit.Length > 0);
                Contracts.Assert(0 <= _count & _count <= _entries.Length);

                // The number of buckets should be at least the number of items, unless we're reached the
                // biggest number of buckets allowed.
                Contracts.Assert(_rgit.Length >= _count | _rgit.Length == HashHelpers.MaxPrime);
            }

            [Conditional("DUMP_STATS")]
            private void DumpStats()
            {
                int c = 0;
                _rgit.ApplyRange(0, _rgit.Length, (long i, ref long value) => { if (value >= 0) c++; });
                Console.WriteLine("Table: {0} out of {1}", c, _rgit.Length);
            }

            private static long Get64BitHashCode(RowId value)
            {
                // REVIEW: Is this a good way to compute hash?
                ulong lo = value.Low;
                ulong hi = value.High;
                return (long)(lo ^ (lo >> 32) ^ (hi << 7) ^ (hi >> 57) ^ (hi >> (57 - 32)));
            }

            private static class HashHelpers
            {
                // Note: This HashHelpers class was adapted from the BCL code base, and extended to support 64-bit hash function.

                // This is the maximum prime smaller than long.MaxValue, which is 2^63 - 25.
                // Refer to https://primes.utm.edu/lists/2small/0bit.html for a list of primes
                // just less than powers of two.
                public const long MaxPrime = 0x7FFFFFFFFFFFFFE7;

                // Table of prime numbers to use as hash table sizes.
                // Each subsequent prime, except the last in the list, ensures that the table will at least double in size
                // upon each growth in order to improve the efficiency of the hash table.
                // See https://oeis.org/A065545 for the sequence with a[1] = 3, a[k] = next_prime(2 * a[k - 1]).
                public static readonly long[] Primes =
                {
                    3, 7, 17, 37, 79, 163, 331, 673, 1361, 2729, 5471, 10949, 21911, 43853, 87719, 175447, 350899,
                    701819, 1403641, 2807303, 5614657, 11229331, 22458671, 44917381, 89834777, 179669557, 359339171,
                    718678369, 1437356741, 2874713497, 5749427029, 11498854069, 22997708177, 45995416409, 91990832831,
                    183981665689, 367963331389, 735926662813, 1471853325643, 2943706651297, 5887413302609, 11774826605231,
                    23549653210463, 47099306420939, 94198612841897, 188397225683869, 376794451367743, 753588902735509,
                    1507177805471059, 3014355610942127, 6028711221884317, 12057422443768697, 24114844887537407, 48229689775074839,
                    96459379550149709, 192918759100299439, 385837518200598889, 771675036401197787, 1543350072802395601, 3086700145604791213,
                    6173400291209582429, MaxPrime
                };

                // Returns size of hashtable to grow to.
                public static long ExpandPrime(long oldSize)
                {
                    long newSize = 2 * oldSize;

                    // Note that this check works even when _items.Length overflowed thanks to the (ulong) cast .
                    if ((ulong)newSize >= MaxPrime)
                        return MaxPrime;

                    return GetPrime(newSize);
                }

                public static long GetPrime(long min)
                {
                    Contracts.Assert(0 <= min && min < MaxPrime);

                    for (int i = 0; i < Primes.Length; i++)
                    {
                        long prime = Primes[i];
                        if (prime >= min)
                            return prime;
                    }

                    Contracts.Assert(false);
                    return min + 1;
                }
            }
        }
    }

    /// <summary>
    /// Sum with underflow compensation for better numerical stability.
    /// </summary>
    internal sealed class CompensatedSum
    {
        private Double _roundOffError;
        private Double _sum;

        public Double Sum => _sum;

        public void Add(Double summand)
        {
            var compensated = summand - _roundOffError;
            var sum = _sum + compensated;
            _roundOffError = FloatUtils.IsFinite(sum) ? (sum - _sum) - compensated : 0;
            _sum = sum;
        }
    }

    public sealed class SdcaBinaryTrainer : SdcaTrainerBase<SdcaBinaryTrainer.Arguments, BinaryPredictionTransformer<TScalarPredictor>, TScalarPredictor>
    {
        public const string LoadNameValue = "SDCA";
        internal const string UserNameValue = "Fast Linear (SA-SDCA)";

        public sealed class Arguments : ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "Loss Function", ShortName = "loss", SortOrder = 50)]
            public ISupportSdcaClassificationLossFactory LossFunction = new LogLossFactory();

            [Argument(ArgumentType.AtMostOnce, HelpText = "Apply weight to the positive class, for imbalanced data", ShortName = "piw")]
            public float PositiveInstanceWeight = 1;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The calibrator kind to apply to the predictor. Specify null for no calibration", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            public ICalibratorTrainerFactory Calibrator = new PlattCalibratorTrainerFactory();

            [Argument(ArgumentType.AtMostOnce, HelpText = "The maximum number of examples to use when training the calibrator", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            public int MaxCalibrationExamples = 1000000;

            internal override void Check(IHostEnvironment env)
            {
                base.Check(env);
                env.CheckUserArg(PositiveInstanceWeight > 0, nameof(PositiveInstanceWeight), "Weight for positive instances must be positive");
            }
        }

        private readonly ISupportSdcaClassificationLoss _loss;
        private readonly float _positiveInstanceWeight;

        protected override bool ShuffleData => Args.Shuffle;

        private readonly SchemaShape.Column[] _outputColumns;

        protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema) => _outputColumns;

        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        public override TrainerInfo Info { get; }

        /// <summary>
        /// Initializes a new instance of <see cref="SdcaBinaryTrainer"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="labelColumn">The label, or dependent variable.</param>
        /// <param name="featureColumn">The features, or independent variables.</param>
        /// <param name="loss">The custom loss.</param>
        /// <param name="weightColumn">The optional example weights.</param>
        /// <param name="l2Const">The L2 regularization hyperparameter.</param>
        /// <param name="l1Threshold">The L1 regularization hyperparameter. Higher values will tend to lead to more sparse model.</param>
        /// <param name="maxIterations">The maximum number of passes to perform over the data.</param>
        /// <param name="advancedSettings">A delegate to set more settings.
        /// The settings here will override the ones provided in the direct method signature,
        /// if both are present and have different values.
        /// The columns names, however need to be provided directly, not through the <paramref name="advancedSettings"/>.</param>
        public SdcaBinaryTrainer(IHostEnvironment env,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weightColumn = null,
            ISupportSdcaClassificationLoss loss = null,
            float? l2Const = null,
            float? l1Threshold = null,
            int? maxIterations = null,
            Action<Arguments> advancedSettings = null)
             : base(env, featureColumn, TrainerUtils.MakeBoolScalarLabel(labelColumn), TrainerUtils.MakeR4ScalarWeightColumn(weightColumn), advancedSettings,
                  l2Const, l1Threshold, maxIterations)
        {
            Host.CheckNonEmpty(featureColumn, nameof(featureColumn));
            Host.CheckNonEmpty(labelColumn, nameof(labelColumn));
            _loss = loss ?? Args.LossFunction.CreateComponent(env);
            Loss = _loss;
            Info = new TrainerInfo(calibration: !(_loss is LogLoss));
            _positiveInstanceWeight = Args.PositiveInstanceWeight;

            var outCols = new List<SchemaShape.Column>()
            {
                    new SchemaShape.Column(
                        DefaultColumnNames.Score,
                        SchemaShape.Column.VectorKind.Scalar,
                        NumberType.R4,
                        false,
                        new SchemaShape(MetadataUtils.GetTrainerOutputMetadata())
                    ),
                    new SchemaShape.Column(
                        DefaultColumnNames.PredictedLabel,
                        SchemaShape.Column.VectorKind.Scalar,
                        BoolType.Instance,
                        false,
                        new SchemaShape(MetadataUtils.GetTrainerOutputMetadata()))

            };

            if (!Info.NeedCalibration)
            {
                outCols.Insert(1, new SchemaShape.Column(
                    DefaultColumnNames.Probability,
                    SchemaShape.Column.VectorKind.Scalar,
                    NumberType.R4,
                    false,
                    new SchemaShape(MetadataUtils.GetTrainerOutputMetadata(true))));
            };

            _outputColumns = outCols.ToArray();
        }

        internal SdcaBinaryTrainer(IHostEnvironment env, Arguments args,
            string featureColumn, string labelColumn, string weightColumn = null)
            : base(env, args, TrainerUtils.MakeBoolScalarLabel(labelColumn), TrainerUtils.MakeR4ScalarWeightColumn(weightColumn))
        {
            _loss = args.LossFunction.CreateComponent(env);
            Loss = _loss;
            Info = new TrainerInfo(calibration: !(_loss is LogLoss));
            _positiveInstanceWeight = Args.PositiveInstanceWeight;

            var outCols = new List<SchemaShape.Column>()
            {
                    new SchemaShape.Column(
                        DefaultColumnNames.Score,
                        SchemaShape.Column.VectorKind.Scalar,
                        NumberType.R4,
                        false,
                        new SchemaShape(MetadataUtils.GetTrainerOutputMetadata())
                    ),
                    new SchemaShape.Column(
                        DefaultColumnNames.PredictedLabel,
                        SchemaShape.Column.VectorKind.Scalar,
                        BoolType.Instance,
                        false,
                        new SchemaShape(MetadataUtils.GetTrainerOutputMetadata()))

            };

            if (!Info.NeedCalibration)
            {
                outCols.Insert(1, new SchemaShape.Column(
                    DefaultColumnNames.Probability,
                    SchemaShape.Column.VectorKind.Scalar,
                    NumberType.R4,
                    false,
                    new SchemaShape(MetadataUtils.GetTrainerOutputMetadata(true))));
            };

            _outputColumns = outCols.ToArray();

        }

        public SdcaBinaryTrainer(IHostEnvironment env, Arguments args)
            : this(env, args, args.FeatureColumn, args.LabelColumn)
        {
        }

        protected override void CheckLabelCompatible(SchemaShape.Column labelCol)
        {
            Contracts.Assert(labelCol.IsValid);

            Action error =
                () => throw Host.ExceptSchemaMismatch(nameof(labelCol), RoleMappedSchema.ColumnRole.Label.Value, labelCol.Name, "BL, R8, R4 or a Key", labelCol.GetTypeString());

            if (labelCol.Kind != SchemaShape.Column.VectorKind.Scalar)
                error();

            if (!labelCol.IsKey && labelCol.ItemType != NumberType.R4 && labelCol.ItemType != NumberType.R8 && !(labelCol.ItemType is BoolType))
                error();
        }

        protected override TScalarPredictor CreatePredictor(VBuffer<float>[] weights, float[] bias)
        {
            Host.CheckParam(Utils.Size(weights) == 1, nameof(weights));
            Host.CheckParam(Utils.Size(bias) == 1, nameof(bias));
            Host.CheckParam(weights[0].Length > 0, nameof(weights));

            VBuffer<float> maybeSparseWeights = default;
            // below should be `in weights[0]`, but can't because of https://github.com/dotnet/roslyn/issues/29371
            VBufferUtils.CreateMaybeSparseCopy(weights[0], ref maybeSparseWeights,
                Conversions.Instance.GetIsDefaultPredicate<float>(NumberType.Float));

            var predictor = new LinearBinaryModelParameters(Host, in maybeSparseWeights, bias[0]);
            if (Info.NeedCalibration)
                return predictor;
            return new ParameterMixingCalibratedPredictor(Host, predictor, new PlattCalibrator(Host, -1, 0));
        }

        private protected override float GetInstanceWeight(FloatLabelCursor cursor)
        {
            return cursor.Label > 0 ? cursor.Weight * _positiveInstanceWeight : cursor.Weight;
        }

        private protected override void CheckLabel(RoleMappedData examples, out int weightSetCount)
        {
            examples.CheckBinaryLabel();
            weightSetCount = 1;
        }

        protected override BinaryPredictionTransformer<TScalarPredictor> MakeTransformer(TScalarPredictor model, Schema trainSchema)
            => new BinaryPredictionTransformer<TScalarPredictor>(Host, model, trainSchema, FeatureColumn.Name);
    }

    public sealed class StochasticGradientDescentClassificationTrainer :
        LinearTrainerBase<BinaryPredictionTransformer<TScalarPredictor>, TScalarPredictor>
    {
        internal const string LoadNameValue = "BinarySGD";
        internal const string UserNameValue = "Hogwild SGD (binary)";
        internal const string ShortName = "HogwildSGD";

        public sealed class Arguments : LearnerInputBaseWithWeight
        {
            [Argument(ArgumentType.Multiple, HelpText = "Loss Function", ShortName = "loss", SortOrder = 50)]
            public ISupportClassificationLossFactory LossFunction = new LogLossFactory();

            [Argument(ArgumentType.AtMostOnce, HelpText = "L2 Regularization constant", ShortName = "l2", SortOrder = 50)]
            [TGUI(Label = "L2 Regularization Constant", SuggestedSweeps = "1e-7,5e-7,1e-6,5e-6,1e-5")]
            [TlcModule.SweepableDiscreteParam("L2Const", new object[] { 1e-7f, 5e-7f, 1e-6f, 5e-6f, 1e-5f })]
            public float L2Weight = Defaults.L2Weight;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Degree of lock-free parallelism. Defaults to automatic depending on data sparseness. Determinism not guaranteed.", ShortName = "nt,t,threads", SortOrder = 50)]
            [TGUI(Label = "Number of threads", SuggestedSweeps = "1,2,4")]
            public int? NumThreads;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Exponential moving averaged improvement tolerance for convergence", ShortName = "tol")]
            [TGUI(SuggestedSweeps = "1e-2,1e-3,1e-4,1e-5")]
            [TlcModule.SweepableDiscreteParam("ConvergenceTolerance", new object[] { 1e-2f, 1e-3f, 1e-4f, 1e-5f })]
            public double ConvergenceTolerance = 1e-4;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum number of iterations; set to 1 to simulate online learning.", ShortName = "iter")]
            [TGUI(Label = "Max number of iterations", SuggestedSweeps = "1,5,10,20")]
            [TlcModule.SweepableDiscreteParam("MaxIterations", new object[] { 1, 5, 10, 20 })]
            public int MaxIterations = Defaults.MaxIterations;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Initial learning rate (only used by SGD)", ShortName = "ilr,lr")]
            [TGUI(Label = "Initial Learning Rate (for SGD)")]
            public double InitLearningRate = Defaults.InitLearningRate;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Shuffle data every epoch?", ShortName = "shuf")]
            [TlcModule.SweepableDiscreteParam("Shuffle", null, isBool: true)]
            public bool Shuffle = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Apply weight to the positive class, for imbalanced data", ShortName = "piw")]
            public float PositiveInstanceWeight = 1;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Convergence check frequency (in terms of number of iterations). Default equals number of threads", ShortName = "checkFreq")]
            public int? CheckFrequency;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The calibrator kind to apply to the predictor. Specify null for no calibration", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            public ICalibratorTrainerFactory Calibrator = new PlattCalibratorTrainerFactory();

            [Argument(ArgumentType.AtMostOnce, HelpText = "The maximum number of examples to use when training the calibrator", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            public int MaxCalibrationExamples = 1000000;

            internal void Check(IHostEnvironment env)
            {
                Contracts.CheckValue(env, nameof(env));
                env.CheckUserArg(L2Weight >= 0, nameof(L2Weight), "Must be non-negative.");
                env.CheckUserArg(InitLearningRate > 0, nameof(InitLearningRate), "Must be positive.");
                env.CheckUserArg(MaxIterations > 0, nameof(MaxIterations), "Must be positive.");
                env.CheckUserArg(PositiveInstanceWeight > 0, nameof(PositiveInstanceWeight), "Must be positive");

                if (InitLearningRate * L2Weight >= 1)
                {
                    using (var ch = env.Start("Argument Adjustment"))
                    {
                        ch.Warning("{0} {1} set too high; reducing to {1}", nameof(InitLearningRate),
                            InitLearningRate, InitLearningRate = (float)0.5 / L2Weight);
                    }
                }

                if (ConvergenceTolerance <= 0)
                    ConvergenceTolerance = float.Epsilon;
            }

            [BestFriend]
            internal static class Defaults
            {
                public const float L2Weight = 1e-6f;
                public const int MaxIterations = 20;
                public const double InitLearningRate = 0.01;
            }
        }

        private readonly IClassificationLoss _loss;
        private readonly Arguments _args;

        protected override bool ShuffleData => _args.Shuffle;

        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        public override TrainerInfo Info { get; }

        /// <summary>
        /// Initializes a new instance of <see cref="StochasticGradientDescentClassificationTrainer"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="featureColumn">The name of the feature column.</param>
        /// <param name="labelColumn">The name of the label column.</param>
        /// <param name="weightColumn">The name for the example weight column.</param>
        /// <param name="maxIterations">The maximum number of iterations; set to 1 to simulate online learning.</param>
        /// <param name="initLearningRate">The initial learning rate used by SGD.</param>
        /// <param name="l2Weight">The L2 regularizer constant.</param>
        /// <param name="loss">The loss function to use.</param>
        /// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
        public StochasticGradientDescentClassificationTrainer(IHostEnvironment env,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weightColumn = null,
            int maxIterations = Arguments.Defaults.MaxIterations,
            double initLearningRate = Arguments.Defaults.InitLearningRate,
            float l2Weight = Arguments.Defaults.L2Weight,
            ISupportClassificationLossFactory loss = null,
            Action<Arguments> advancedSettings = null)
            : base(env, featureColumn, TrainerUtils.MakeBoolScalarLabel(labelColumn), weightColumn)
        {
            Host.CheckNonEmpty(featureColumn, nameof(featureColumn));
            Host.CheckNonEmpty(labelColumn, nameof(labelColumn));

            _args = new Arguments();
            _args.MaxIterations = maxIterations;
            _args.InitLearningRate = initLearningRate;
            _args.L2Weight = l2Weight;

            // Apply the advanced args, if the user supplied any.
            advancedSettings?.Invoke(_args);

            _args.FeatureColumn = featureColumn;
            _args.LabelColumn = labelColumn;
            _args.WeightColumn = weightColumn;

            if (loss != null)
                _args.LossFunction = loss;
            _args.Check(env);

            _loss = _args.LossFunction.CreateComponent(env);
            Info = new TrainerInfo(calibration: !(_loss is LogLoss), supportIncrementalTrain: true);
            NeedShuffle = _args.Shuffle;
        }

        /// <summary>
        /// Initializes a new instance of <see cref="StochasticGradientDescentClassificationTrainer"/>
        /// </summary>
        internal StochasticGradientDescentClassificationTrainer(IHostEnvironment env, Arguments args)
            : base(env, args.FeatureColumn, TrainerUtils.MakeBoolScalarLabel(args.LabelColumn), args.WeightColumn)
        {
            args.Check(env);
            _loss = args.LossFunction.CreateComponent(env);
            Info = new TrainerInfo(calibration: !(_loss is LogLoss), supportIncrementalTrain: true);
            NeedShuffle = args.Shuffle;
            _args = args;
        }

        protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata())),
                new SchemaShape.Column(DefaultColumnNames.Probability, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata(true))),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BoolType.Instance, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata()))
            };
        }

        protected override BinaryPredictionTransformer<TScalarPredictor> MakeTransformer(TScalarPredictor model, Schema trainSchema)
            => new BinaryPredictionTransformer<TScalarPredictor>(Host, model, trainSchema, FeatureColumn.Name);

        public BinaryPredictionTransformer<TScalarPredictor> Train(IDataView trainData, IPredictor initialPredictor = null)
            => TrainTransformer(trainData, initPredictor: initialPredictor);

        //For complexity analysis, we assume that
        // - The number of features is N
        // - Average number of non-zero per instance is k
        private protected override TScalarPredictor TrainCore(IChannel ch, RoleMappedData data, LinearModelParameters predictor, int weightSetCount)
        {
            Contracts.AssertValue(data);
            Contracts.Assert(weightSetCount == 1);
            Contracts.AssertValueOrNull(predictor);
            Contracts.Assert(data.Schema.Feature.HasValue);

            int numFeatures = data.Schema.Feature.Value.Type.GetVectorSize();
            var cursorFactory = new FloatLabelCursor.Factory(data, CursOpt.Label | CursOpt.Features | CursOpt.Weight);

            int numThreads;
            if (_args.NumThreads.HasValue)
            {
                numThreads = _args.NumThreads.Value;
                ch.CheckUserArg(numThreads > 0, nameof(_args.NumThreads), "The number of threads must be either null or a positive integer.");
            }
            else
                numThreads = ComputeNumThreads(cursorFactory);

            ch.Assert(numThreads > 0);
            int checkFrequency = _args.CheckFrequency ?? numThreads;
            if (checkFrequency <= 0)
                checkFrequency = int.MaxValue;
            var l2Weight = _args.L2Weight;
            var lossFunc = _loss;
            var pOptions = new ParallelOptions { MaxDegreeOfParallelism = numThreads };
            var positiveInstanceWeight = _args.PositiveInstanceWeight;
            var weights = default(VBuffer<float>);
            float bias = 0.0f;
            if (predictor != null)
            {
                predictor.GetFeatureWeights(ref weights);
                VBufferUtils.Densify(ref weights);
                bias = predictor.Bias;
            }
            else
                weights = VBufferUtils.CreateDense<float>(numFeatures);

            var weightsSync = new object();
            double weightScaling = 1;
            Double loss = Math.Log(2); // for log loss, this is exact; for Hinge, it's a reasonable estimate
            Double improvement = 0; // exponentially weighted moving average of improvements
            bool converged = false;
            var watch = new Stopwatch();

            // REVIEW: Investigate using parallel row cursor set instead of getting cursor independently. The convergence of SDCA need to be verified.
            Action<int, IProgressChannel> checkConvergence = (e, pch) =>
            {
                if (e % checkFrequency == 0 && e != _args.MaxIterations)
                {
                    Double trainTime = watch.Elapsed.TotalSeconds;
                    var lossSum = new CompensatedSum();
                    long count = 0;
                    using (var cursor = cursorFactory.Create())
                    {
                        // Iterates through data to compute loss function.
                        while (cursor.MoveNext())
                        {
                            count++;
                            var instanceWeight = cursor.Weight;
                            var features = cursor.Features;
                            Double subLoss = lossFunc.Loss(WScaledDot(in features, weightScaling, in weights, bias), cursor.Label);

                            if (cursor.Label > 0)
                                lossSum.Add(subLoss * instanceWeight * positiveInstanceWeight);
                            else
                                lossSum.Add(subLoss * instanceWeight);
                        }
                    }

                    var newLoss = lossSum.Sum / count + l2Weight * VectorUtils.NormSquared(weights) * 0.5;
                    improvement = improvement == 0 ? loss - newLoss : 0.5 * (loss - newLoss + improvement);
                    loss = newLoss;

                    pch.Checkpoint(loss, improvement, e, _args.MaxIterations);
                    converged = improvement < _args.ConvergenceTolerance;
                }
            };

            watch.Start();

            //Reference: Leon Bottou. Stochastic Gradient Descent Tricks.
            //https://research.microsoft.com/pubs/192769/tricks-2012.pdf

            var trainingTasks = new Action<Random, IProgressChannel>[_args.MaxIterations];
            var rands = new Random[_args.MaxIterations];
            var ilr = _args.InitLearningRate;
            long t = 0;
            for (int epoch = 1; epoch <= _args.MaxIterations; epoch++)
            {
                int e = epoch; //localize the modified closure
                rands[e - 1] = RandomUtils.Create(Host.Rand.Next());
                trainingTasks[e - 1] = (rand, pch) =>
                {
                    using (var cursor = _args.Shuffle ? cursorFactory.Create(rand) : cursorFactory.Create())
                    {
                        var weightsEditor = VBufferEditor.CreateFromBuffer(ref weights);
                        while (cursor.MoveNext())
                        {
                            VBuffer<float> features = cursor.Features;
                            float label = cursor.Label;
                            float derivative = cursor.Weight * lossFunc.Derivative(WScaledDot(in features, weightScaling, in weights, bias), label); // complexity: O(k)

                            //Note that multiplying the gradient by a weight h is not equivalent to doing h updates
                            //on the same instance. A potentially better way to do weighted update is described in
                            //https://dslpitt.org/uai/papers/11/p392-karampatziakis.pdf
                            if (label > 0)
                                derivative *= positiveInstanceWeight;

                            Double rate = ilr / (1 + ilr * l2Weight * (t++));
                            Double step = -derivative * rate;
                            weightScaling *= 1 - rate * l2Weight;
                            VectorUtils.AddMult(in features, weightsEditor.Values, (float)(step / weightScaling));
                            bias += (float)step;
                        }
                        if (e == 1)
                        {
                            if (cursor.BadFeaturesRowCount > 0)
                                ch.Warning("Skipped {0} instances with missing features during training", cursor.BadFeaturesRowCount);
                            if (cursor.BadLabelCount > 0)
                                ch.Warning("Skipped {0} instances with missing label during training", cursor.BadLabelCount);
                            if (cursor.BadWeightCount > 0)
                                ch.Warning("Skipped {0} instances with missing weight during training", cursor.BadWeightCount);
                        }
                    }

                    // in case scale gets too small, we need to renormalize to avoid precision loss
                    if (weightScaling < 1e-8)
                    {
                        lock (weightsSync)
                        {
                            VectorUtils.ScaleBy(ref weights, (float)weightScaling); // normalize the weights
                            weightScaling = 1;
                        }
                    }

                    if (e % checkFrequency == 0)
                        checkConvergence(e, pch);
                };
            }

            using (var pch = Host.StartProgressChannel("SGD Training"))
            {
                // Separate logic is needed for single-thread execution to ensure the result is deterministic.
                // Note that P.Invoke does not ensure that the actions executes in order even if maximum number of threads is set to 1.
                if (numThreads == 1)
                {
                    int iter = 0;
                    pch.SetHeader(new ProgressHeader(new[] { "Loss", "Improvement" }, new[] { "iterations" }),
                        entry => entry.SetProgress(0, iter, _args.MaxIterations));
                    // Synchorized SGD.
                    for (int i = 0; i < _args.MaxIterations; i++)
                    {
                        iter = i;
                        trainingTasks[i](rands[i], pch);
                    }
                }
                else
                {
                    // Parallel SGD.
                    pch.SetHeader(new ProgressHeader(new[] { "Loss", "Improvement" }, new[] { "iterations" }),
                        entry =>
                        {
                            // Do nothing. All iterations are running in parallel.
                            // The tasks would still occasionally send checkpoints.

                            // REVIEW: technically, we could keep track of how many iterations have started,
                            // but this needs more synchronization than Parallel.For allows.
                        });
                    Parallel.For(0, _args.MaxIterations, pOptions, i => trainingTasks[i](rands[i], pch));
                    //note that P.Invoke will wait until all tasks finish
                }
            }

            VectorUtils.ScaleBy(ref weights, (float)weightScaling); // restore the true weights

            VBuffer<float> maybeSparseWeights = default;
            VBufferUtils.CreateMaybeSparseCopy(in weights, ref maybeSparseWeights, Conversions.Instance.GetIsDefaultPredicate<float>(NumberType.Float));
            var pred = new LinearBinaryModelParameters(Host, in maybeSparseWeights, bias);
            if (!(_loss is LogLoss))
                return pred;
            return new ParameterMixingCalibratedPredictor(Host, pred, new PlattCalibrator(Host, -1, 0));
        }

        private protected override void CheckLabel(RoleMappedData examples, out int weightSetCount)
        {
            examples.CheckBinaryLabel();
            weightSetCount = 1;
        }

        [TlcModule.EntryPoint(Name = "Trainers.StochasticGradientDescentBinaryClassifier", Desc = "Train an Hogwild SGD binary model.", UserName = UserNameValue, ShortName = ShortName)]
        public static CommonOutputs.BinaryClassificationOutput TrainBinary(IHostEnvironment env, Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainHogwildSGD");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<Arguments, CommonOutputs.BinaryClassificationOutput>(host, input,
                () => new StochasticGradientDescentClassificationTrainer(host, input),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.WeightColumn),
                calibrator: input.Calibrator, maxCalibrationExamples: input.MaxCalibrationExamples);

        }
    }

    /// <summary>
    /// A component to train an SDCA model.
    /// </summary>
    public static partial class Sdca
    {
        [TlcModule.EntryPoint(Name = "Trainers.StochasticDualCoordinateAscentBinaryClassifier",
            Desc = "Train an SDCA binary model.",
            UserName = SdcaBinaryTrainer.UserNameValue,
            ShortName = SdcaBinaryTrainer.LoadNameValue,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.StandardLearners/Standard/doc.xml' path='doc/members/member[@name=""SDCA""]/*' />",
                                 @"<include file='../Microsoft.ML.StandardLearners/Standard/doc.xml' path='doc/members/example[@name=""StochasticDualCoordinateAscentBinaryClassifier""]/*'/>" })]
        public static CommonOutputs.BinaryClassificationOutput TrainBinary(IHostEnvironment env, SdcaBinaryTrainer.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainSDCA");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<SdcaBinaryTrainer.Arguments, CommonOutputs.BinaryClassificationOutput>(host, input,
                () => new SdcaBinaryTrainer(host, input),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn),
                calibrator: input.Calibrator, maxCalibrationExamples: input.MaxCalibrationExamples);
        }
    }
}
