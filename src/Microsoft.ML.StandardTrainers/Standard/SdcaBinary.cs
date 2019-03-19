// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Data.Conversion;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.CpuMath;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Numeric;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(typeof(LegacySdcaBinaryTrainer), typeof(LegacySdcaBinaryTrainer.Options),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer), typeof(SignatureFeatureScorerTrainer) },
    LegacySdcaBinaryTrainer.UserNameValue,
    LegacySdcaBinaryTrainer.LoadNameValue,
    "LinearClassifier",
    "lc",
    "sasdca")]

[assembly: LoadableClass(typeof(LegacySgdBinaryTrainer), typeof(LegacySgdBinaryTrainer.Options),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer), typeof(SignatureFeatureScorerTrainer) },
    LegacySgdBinaryTrainer.UserNameValue,
    LegacySgdBinaryTrainer.LoadNameValue,
    "sgd")]

[assembly: LoadableClass(typeof(void), typeof(Sdca), null, typeof(SignatureEntryPointModule), "SDCA")]
[assembly: LoadableClass(typeof(void), typeof(LegacySgdBinaryTrainer), null, typeof(SignatureEntryPointModule), LegacySgdBinaryTrainer.ShortName)]

namespace Microsoft.ML.Trainers
{
    using ConditionalAttribute = System.Diagnostics.ConditionalAttribute;
    using Stopwatch = System.Diagnostics.Stopwatch;

    public abstract class LinearTrainerBase<TTransformer, TModel> : TrainerEstimatorBase<TTransformer, TModel>
        where TTransformer : ISingleFeaturePredictionTransformer<TModel>
        where TModel : class
    {
        private const string RegisterName = nameof(LinearTrainerBase<TTransformer, TModel>);

        private static readonly TrainerInfo _info = new TrainerInfo();
        public override TrainerInfo Info => _info;

        /// <summary>
        /// Whether data is to be shuffled every epoch.
        /// </summary>
        private protected abstract bool ShuffleData { get; }

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
                var linInitPred = (initPred as IWeaklyTypedCalibratedModelParameters)?.WeaklyTypedSubModel as LinearModelParameters;
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
                var shuffleArgs = new RowShufflingTransformer.Options
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

        private protected float WDot(in VBuffer<float> features, in VBuffer<float> weights, float bias)
        {
            return VectorUtils.DotProduct(in weights, in features) + bias;
        }

        private protected float WScaledDot(in VBuffer<float> features, Double scaling, in VBuffer<float> weights, float bias)
        {
            return VectorUtils.DotProduct(in weights, in features) * (float)scaling + bias;
        }

        private protected virtual int ComputeNumThreads(FloatLabelCursor.Factory cursorFactory)
            =>  Math.Min(8, Math.Max(1, Environment.ProcessorCount / 2));
    }

    public abstract class SdcaTrainerBase<TOptions, TTransformer, TModel> : StochasticTrainerBase<TTransformer, TModel>
        where TTransformer : ISingleFeaturePredictionTransformer<TModel>
        where TModel : class
        where TOptions : SdcaTrainerBase<TOptions, TTransformer, TModel>.OptionsBase, new()
    {
        // REVIEW: Making it even faster and more accurate:
        // 1. Train with not-too-many threads. nt = 2 or 4 seems to be good enough. Didn't seem additional benefit over more threads.
        // 2. Make tol smaller. 0.1 seems to be too large. 0.01 is more ideal.
        // 3. Don't "guess" the iteration to converge. It is very data-set dependent and hard to control. Always check for at least once to ensure convergence.
        // 4. Use dual variable updates to infer whether a full iteration of convergence checking is necessary. Convergence checking iteration is time-consuming.

        /// <summary>
        /// Options for the SDCA-based trainers.
        /// </summary>
        public abstract class OptionsBase : TrainerInputBaseWithLabel
        {
            /// <summary>
            /// The L2 <a href='tmpurl_regularization'>regularization</a> hyperparameter.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "L2 regularizer constant. By default the l2 constant is automatically inferred based on data set.", NullName = "<Auto>", ShortName = "l2, L2Const", SortOrder = 1)]
            [TGUI(Label = "L2 Regularizer Constant", SuggestedSweeps = "<Auto>,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2")]
            [TlcModule.SweepableDiscreteParam("L2Const", new object[] { "<Auto>", 1e-7f, 1e-6f, 1e-5f, 1e-4f, 1e-3f, 1e-2f })]
            public float? L2Regularization;

            // REVIEW: make the default positive when we know how to consume a sparse model
            /// <summary>
            /// The L1 <a href='tmpurl_regularization'>regularization</a> hyperparameter.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "L1 soft threshold (L1/L2). Note that it is easier to control and sweep using the threshold parameter than the raw L1-regularizer constant. By default the l1 threshold is automatically inferred based on data set.", NullName = "<Auto>", ShortName = "l1", SortOrder = 2)]
            [TGUI(Label = "L1 Soft Threshold", SuggestedSweeps = "<Auto>,0,0.25,0.5,0.75,1")]
            [TlcModule.SweepableDiscreteParam("L1Threshold", new object[] { "<Auto>", 0f, 0.25f, 0.5f, 0.75f, 1f })]
            public float? L1Threshold;

            /// <summary>
            /// The degree of lock-free parallelism.
            /// </summary>
            /// <value>
            /// Defaults to automatic depending on data sparseness. Determinism is not guaranteed.
            /// </value>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Degree of lock-free parallelism. Defaults to automatic. Determinism not guaranteed.", NullName = "<Auto>", ShortName = "nt,t,threads, NumThreads", SortOrder = 50)]
            [TGUI(Label = "Number of threads", SuggestedSweeps = "<Auto>,1,2,4")]
            public int? NumberOfThreads;

            /// <summary>
            /// The tolerance for the ratio between duality gap and primal loss for convergence checking.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "The tolerance for the ratio between duality gap and primal loss for convergence checking.", ShortName = "tol")]
            [TGUI(SuggestedSweeps = "0.001, 0.01, 0.1, 0.2")]
            [TlcModule.SweepableDiscreteParam("ConvergenceTolerance", new object[] { 0.001f, 0.01f, 0.1f, 0.2f })]
            public float ConvergenceTolerance = 0.1f;

            /// <summary>
            /// The maximum number of passes to perform over the data.
            /// </summary>
            /// <value>
            /// Set to 1 to simulate online learning. Defaults to automatic.
            /// </value>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum number of iterations; set to 1 to simulate online learning. Defaults to automatic.", NullName = "<Auto>", ShortName = "iter, MaxIterations, NumberOfIterations")]
            [TGUI(Label = "Max number of iterations", SuggestedSweeps = "<Auto>,10,20,100")]
            [TlcModule.SweepableDiscreteParam("MaxIterations", new object[] { "<Auto>", 10, 20, 100 })]
            public int? MaximumNumberOfIterations;

            /// <summary>
            /// Determines whether to shuffle data for each training iteration.
            /// </summary>
            /// <value>
            /// <see langword="true" /> to shuffle data for each training iteration; otherwise, <see langword="false" />.
            /// Default is <see langword="true" />.
            /// </value>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Shuffle data every epoch?", ShortName = "shuf")]
            [TlcModule.SweepableDiscreteParam("Shuffle", null, isBool: true)]
            public bool Shuffle = true;

            /// <summary>
            /// Determines the frequency of checking for convergence in terms of number of iterations.
            /// </summary>
            /// <value>
            /// Set to zero or negative value to disable checking. If <see langword="null"/>, it defaults to <see cref="NumberOfThreads"/>."
            /// </value>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Convergence check frequency (in terms of number of iterations). Set as negative or zero for not checking at all. If left blank, it defaults to check after every 'numThreads' iterations.", NullName = "<Auto>", ShortName = "checkFreq, CheckFrequency")]
            public int? ConvergenceCheckFrequency;

            /// <summary>
            /// The learning rate for adjusting bias from being regularized.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "The learning rate for adjusting bias from being regularized.", ShortName = "blr")]
            [TGUI(SuggestedSweeps = "0, 0.01, 0.1, 1")]
            [TlcModule.SweepableDiscreteParam("BiasLearningRate", new object[] { 0.0f, 0.01f, 0.1f, 1f })]
            public float BiasLearningRate = 0;

            internal virtual void Check(IHostEnvironment env)
            {
                Contracts.AssertValue(env);
                env.CheckUserArg(L2Regularization == null || L2Regularization >= 0, nameof(L2Regularization), "L2 constant must be non-negative.");
                env.CheckUserArg(L1Threshold == null || L1Threshold >= 0, nameof(L1Threshold), "L1 threshold must be non-negative.");
                env.CheckUserArg(MaximumNumberOfIterations == null || MaximumNumberOfIterations > 0, nameof(MaximumNumberOfIterations), "Max number of iterations must be positive.");
                env.CheckUserArg(ConvergenceTolerance > 0 && ConvergenceTolerance <= 1, nameof(ConvergenceTolerance), "Convergence tolerance must be positive and no larger than 1.");

                if (L2Regularization < L2LowerBound)
                {
                    using (var ch = env.Start("SDCA arguments checking"))
                    {
                        ch.Warning($"The L2 regularization constant must be at least {L2LowerBound}. In SDCA, the dual formulation " +
                            $"is only valid with a positive constant, and values below {L2LowerBound} cause very slow convergence. " +
                            $"The original {nameof(L2Regularization)} = {L2Regularization}, was replaced with {nameof(L2Regularization)} = {L2LowerBound}.");
                        L2Regularization = L2LowerBound;
                    }
                }
            }

            private protected OptionsBase()
            {
            }
        }

        // The order of these matter, since they are used as indices into arrays.
        private protected enum MetricKind
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
        private protected readonly TOptions SdcaTrainerOptions;
        private protected ISupportSdcaLoss Loss;

        private protected override bool ShuffleData => SdcaTrainerOptions.Shuffle;

        private const string RegisterName = nameof(SdcaTrainerBase<TOptions, TTransformer, TModel>);

        private static TOptions ArgsInit(string featureColumnName, SchemaShape.Column labelColumn)
        {
            var args = new TOptions();

            args.FeatureColumnName = featureColumnName;
            args.LabelColumnName = labelColumn.Name;
            return args;
        }

        internal SdcaTrainerBase(IHostEnvironment env, string featureColumnName, SchemaShape.Column labelColumn,
           SchemaShape.Column weight = default, float? l2Const = null,
            float? l1Threshold = null, int? maxIterations = null)
          : this(env, ArgsInit(featureColumnName, labelColumn), labelColumn, weight, l2Const, l1Threshold, maxIterations)
        {
        }

        internal SdcaTrainerBase(IHostEnvironment env, TOptions options, SchemaShape.Column label, SchemaShape.Column weight = default,
            float? l2Const = null, float? l1Threshold = null, int? maxIterations = null)
            : base(Contracts.CheckRef(env, nameof(env)).Register(RegisterName), TrainerUtils.MakeR4VecFeature(options.FeatureColumnName), label, weight)
        {
            SdcaTrainerOptions = options;
            SdcaTrainerOptions.L2Regularization = l2Const ?? options.L2Regularization;
            SdcaTrainerOptions.L1Threshold = l1Threshold ?? options.L1Threshold;
            SdcaTrainerOptions.MaximumNumberOfIterations = maxIterations ?? options.MaximumNumberOfIterations;
            SdcaTrainerOptions.Check(env);
        }

        private protected float WDot(in VBuffer<float> features, in VBuffer<float> weights, float bias)
        {
            return VectorUtils.DotProduct(in weights, in features) + bias;
        }

        private protected sealed override TModel TrainCore(IChannel ch, RoleMappedData data, LinearModelParameters predictor, int weightSetCount)
        {
            Contracts.Assert(predictor == null, "SDCA based trainers don't support continuous training.");
            Contracts.Assert(weightSetCount >= 1);

            int numFeatures = data.Schema.Feature.Value.Type.GetVectorSize();
            long maxTrainingExamples = MaxDualTableSize / weightSetCount;

            CursOpt cursorOpt = CursOpt.Label | CursOpt.Features | CursOpt.Id;
            if (data.Schema.Weight.HasValue)
                cursorOpt |= CursOpt.Weight;

            var cursorFactory = new FloatLabelCursor.Factory(data, cursorOpt);
            int numThreads;
            if (SdcaTrainerOptions.NumberOfThreads.HasValue)
            {
                numThreads = SdcaTrainerOptions.NumberOfThreads.Value;
                Host.CheckUserArg(numThreads > 0, nameof(OptionsBase.NumberOfThreads), "The number of threads must be either null or a positive integer.");
            }
            else
                numThreads = ComputeNumThreads(cursorFactory);

            ch.Assert(numThreads > 0);
            if (numThreads == 1)
                ch.Info("Using 1 thread to train.");
            else
                ch.Info("Using {0} threads to train.", numThreads);

            int checkFrequency = 0;
            if (SdcaTrainerOptions.ConvergenceCheckFrequency.HasValue)
                checkFrequency = SdcaTrainerOptions.ConvergenceCheckFrequency.Value;
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
                    DataViewRowId id = cursor.Id;
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
            if (SdcaTrainerOptions.MaximumNumberOfIterations == null)
                SdcaTrainerOptions.MaximumNumberOfIterations = TuneDefaultMaxIterations(ch, count, numThreads);

            Contracts.Assert(SdcaTrainerOptions.MaximumNumberOfIterations.HasValue);
            if (SdcaTrainerOptions.L2Regularization == null)
                SdcaTrainerOptions.L2Regularization = TuneDefaultL2(ch, SdcaTrainerOptions.MaximumNumberOfIterations.Value, count, numThreads);

            Contracts.Assert(SdcaTrainerOptions.L2Regularization.HasValue);
            if (SdcaTrainerOptions.L1Threshold == null)
                SdcaTrainerOptions.L1Threshold = TuneDefaultL1(ch, numFeatures);

            ch.Assert(SdcaTrainerOptions.L1Threshold.HasValue);
            var l1Threshold = SdcaTrainerOptions.L1Threshold.Value;
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
            ch.Assert(SdcaTrainerOptions.L2Regularization.HasValue);
            var l2Const = SdcaTrainerOptions.L2Regularization.Value;
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
            ch.Assert(SdcaTrainerOptions.MaximumNumberOfIterations.HasValue);
            var maxIterations = SdcaTrainerOptions.MaximumNumberOfIterations.Value;

            var rands = new Random[maxIterations];
            for (int i = 0; i < maxIterations; i++)
                rands[i] = RandomUtils.Create(Host.Rand.Next());

            // If we favor storing the invariants, precompute the invariants now.
            if (invariants != null)
            {
                Contracts.Assert((idToIdx == null & ((long)idLoMax + 1) * weightSetCount <= Utils.ArrayMaxSize) | (idToIdx != null & count * weightSetCount <= Utils.ArrayMaxSize));
                Func<DataViewRowId, long, long> getIndexFromIdAndRow = GetIndexFromIdAndRowGetter(idToIdx, biasReg.Length);
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
                        if (SdcaTrainerOptions.BiasLearningRate == 0)
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

        private protected abstract TModel CreatePredictor(VBuffer<float>[] weights, float[] bias);

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
        private protected virtual float TuneDefaultL2(IChannel ch, int maxIterations, long rowCount, int numThreads)
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
            Contracts.Assert(SdcaTrainerOptions.L1Threshold.HasValue);
            Contracts.AssertValueOrNull(idToIdx);
            Contracts.AssertValueOrNull(invariants);
            Contracts.AssertValueOrNull(featureNormSquared);
            int maxUpdateTrials = 2 * numThreads;
            var l1Threshold = SdcaTrainerOptions.L1Threshold.Value;
            bool l1ThresholdZero = l1Threshold == 0;
            var lr = SdcaTrainerOptions.BiasLearningRate * SdcaTrainerOptions.L2Regularization.Value;
            var pch = progress != null ? progress.StartProgressChannel("Dual update") : null;
            using (pch)
            using (var cursor = SdcaTrainerOptions.Shuffle ? cursorFactory.Create(rand) : cursorFactory.Create())
            {
                long rowCount = 0;
                if (pch != null)
                    pch.SetHeader(new ProgressHeader("examples"), e => e.SetProgress(0, rowCount));

                Func<DataViewRowId, long> getIndexFromId = GetIndexFromIdGetter(idToIdx, biasReg.Length);
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
                        if (SdcaTrainerOptions.BiasLearningRate == 0)
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
                                if (SdcaTrainerOptions.BiasLearningRate == 0)
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
                Func<DataViewRowId, long, long> getIndexFromIdAndRow = GetIndexFromIdAndRowGetter(idToIdx, biasReg.Length);
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

            Contracts.Assert(SdcaTrainerOptions.L2Regularization.HasValue);
            Contracts.Assert(SdcaTrainerOptions.L1Threshold.HasValue);
            Double l2Const = SdcaTrainerOptions.L2Regularization.Value;
            Double l1Threshold = SdcaTrainerOptions.L1Threshold.Value;
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
            metrics[(int)MetricKind.L1Sparsity] = SdcaTrainerOptions.L1Threshold == 0 ? 1 : (Double)firstWeights.GetValues().Count(w => w != 0) / weights.Length;

            bool converged = dualityGap / newLoss < SdcaTrainerOptions.ConvergenceTolerance;

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

        private protected virtual float[] InitializeFeatureNormSquared(int length)
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
        private protected Func<DataViewRowId, long> GetIndexFromIdGetter(IdToIdxLookup idToIdx, int biasLength)
        {
            Contracts.AssertValueOrNull(idToIdx);
            long maxTrainingExamples = MaxDualTableSize / biasLength;
            if (idToIdx == null)
            {
                return (DataViewRowId id) =>
                {
                    Contracts.Assert(id.High == 0);
                    Contracts.Assert((long)id.Low < maxTrainingExamples);
                    return (long)id.Low;
                };
            }
            else
            {
                return (DataViewRowId id) =>
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
        private protected Func<DataViewRowId, long, long> GetIndexFromIdAndRowGetter(IdToIdxLookup idToIdx, int biasLength)
        {
            Contracts.AssertValueOrNull(idToIdx);
            long maxTrainingExamples = MaxDualTableSize / biasLength;
            if (idToIdx == null)
            {
                return (DataViewRowId id, long row) =>
                {
                    Contracts.Assert(id.High == 0);
                    Contracts.Assert((long)id.Low < maxTrainingExamples);
                    return (long)id.Low;
                };
            }
            else
            {
                return (DataViewRowId id, long row) =>
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
        /// A hash table data structure to store Id of type <see cref="T:Microsoft.ML.Data.DataViewRowId"/>,
        /// and accommodates size larger than 2 billion. This class is an extension based on BCL.
        /// Two operations are supported: adding and retrieving an id with asymptotically constant complexity.
        /// The bucket size are prime numbers, starting from 3 and grows to the next prime larger than
        /// double the current size until it reaches the maximum possible size. When a table growth is triggered,
        /// the table growing operation initializes a new larger bucket and rehash the existing entries to
        /// the new bucket. Such operation has an expected complexity proportional to the size.
        /// </summary>
        private protected sealed class IdToIdxLookup
        {
            // Utilizing this struct gives better cache behavior than using parallel arrays.
            private readonly struct Entry
            {
                public readonly long ItNext;
                public readonly DataViewRowId Value;

                public Entry(long itNext, DataViewRowId value)
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
            public long Add(DataViewRowId id)
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
            public bool TryGetIndex(DataViewRowId id, out long index)
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
            private long GetIndexCore(DataViewRowId val, long iit)
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
            private long AddCore(DataViewRowId val, long iit)
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
                        DataViewRowId value = entry.Value;
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

            private static long Get64BitHashCode(DataViewRowId value)
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

    /// <summary>
    /// SDCA is a general training algorithm for (generalized) linear models such as support vector machine, linear regression, logistic regression,
    /// and so on. SDCA binary classification trainer family includes several sealed members:
    /// (1) <see cref="SdcaNonCalibratedBinaryTrainer"/> supports general loss functions and returns <see cref="LinearBinaryModelParameters"/>.
    /// (2) <see cref="SdcaCalibratedBinaryTrainer"/> essentially trains a regularized logistic regression model. Because logistic regression
    /// naturally provide probability output, this generated model's type is <see cref="CalibratedModelParametersBase{TSubModel, TCalibrator}"/>.
    /// where <see langword="TSubModel"/> is <see cref="LinearBinaryModelParameters"/> and <see langword="TCalibrator "/> is <see cref="PlattCalibrator"/>.
    /// </summary>
    public abstract class SdcaBinaryTrainerBase<TModelParameters> :
        SdcaTrainerBase<SdcaBinaryTrainerBase<TModelParameters>.BinaryOptionsBase, BinaryPredictionTransformer<TModelParameters>, TModelParameters>
        where TModelParameters : class
    {
        private readonly ISupportSdcaClassificationLoss _loss;
        private readonly float _positiveInstanceWeight;

        private protected override bool ShuffleData => SdcaTrainerOptions.Shuffle;

        private readonly SchemaShape.Column[] _outputColumns;

        private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema) => _outputColumns;

        private protected override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        public override TrainerInfo Info { get; }

        /// <summary>
        /// Options base class for binary SDCA trainers.
        /// </summary>
        public class BinaryOptionsBase : OptionsBase
        {
            /// <summary>
            /// The weight to be applied to the positive class. This is useful for training with imbalanced data.
            /// </summary>
            /// <value>
            /// Default value is 1, which means no extra weight.
            /// </value>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Apply weight to the positive class, for imbalanced data", ShortName = "piw")]
            public float PositiveInstanceWeight = 1;

            internal override void Check(IHostEnvironment env)
            {
                base.Check(env);
                env.CheckUserArg(PositiveInstanceWeight > 0, nameof(PositiveInstanceWeight), "Weight for positive instances must be positive");
            }
        }

        /// <summary>
        /// Initializes a new instance of <see cref="SdcaBinaryTrainerBase{TModelParameters}"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="labelColumnName">The label, or dependent variable.</param>
        /// <param name="featureColumnName">The features, or independent variables.</param>
        /// <param name="loss">The custom loss.</param>
        /// <param name="weightColumnName">The optional example weights.</param>
        /// <param name="l2Const">The L2 regularization hyperparameter.</param>
        /// <param name="l1Threshold">The L1 regularization hyperparameter. Higher values will tend to lead to more sparse model.</param>
        /// <param name="maxIterations">The maximum number of passes to perform over the data.</param>
        private protected SdcaBinaryTrainerBase(IHostEnvironment env,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string weightColumnName = null,
            ISupportSdcaClassificationLoss loss = null,
            float? l2Const = null,
            float? l1Threshold = null,
            int? maxIterations = null)
             : base(env, featureColumnName, TrainerUtils.MakeBoolScalarLabel(labelColumnName), TrainerUtils.MakeR4ScalarWeightColumn(weightColumnName),
                   l2Const, l1Threshold, maxIterations)
        {
            Host.CheckNonEmpty(featureColumnName, nameof(featureColumnName));
            Host.CheckNonEmpty(labelColumnName, nameof(labelColumnName));
            _loss = loss ?? new LogLossFactory().CreateComponent(env);
            Loss = _loss;
            Info = new TrainerInfo(calibration: false);
            _positiveInstanceWeight = SdcaTrainerOptions.PositiveInstanceWeight;
            _outputColumns = ComputeSdcaBinaryClassifierSchemaShape();
        }

        private protected SdcaBinaryTrainerBase(IHostEnvironment env, BinaryOptionsBase options, ISupportSdcaClassificationLoss loss = null, bool doCalibration = false)
            : base(env, options, TrainerUtils.MakeBoolScalarLabel(options.LabelColumnName))
        {
            _loss = loss ?? new LogLossFactory().CreateComponent(env);
            Loss = _loss;
            Info = new TrainerInfo(calibration: doCalibration);
            _positiveInstanceWeight = SdcaTrainerOptions.PositiveInstanceWeight;
            _outputColumns = ComputeSdcaBinaryClassifierSchemaShape();
        }

        private protected abstract SchemaShape.Column[] ComputeSdcaBinaryClassifierSchemaShape();

        private protected LinearBinaryModelParameters CreateLinearBinaryModelParameters(VBuffer<float>[] weights, float[] bias)
        {
            Host.CheckParam(Utils.Size(weights) == 1, nameof(weights));
            Host.CheckParam(Utils.Size(bias) == 1, nameof(bias));
            Host.CheckParam(weights[0].Length > 0, nameof(weights));

            VBuffer<float> maybeSparseWeights = default;
            // below should be `in weights[0]`, but can't because of https://github.com/dotnet/roslyn/issues/29371
            VBufferUtils.CreateMaybeSparseCopy(weights[0], ref maybeSparseWeights,
                Conversions.Instance.GetIsDefaultPredicate<float>(NumberDataViewType.Single));

            return new LinearBinaryModelParameters(Host, in maybeSparseWeights, bias[0]);
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

        private protected override BinaryPredictionTransformer<TModelParameters> MakeTransformer(TModelParameters model, DataViewSchema trainSchema)
            => new BinaryPredictionTransformer<TModelParameters>(Host, model, trainSchema, FeatureColumn.Name);
    }

    /// <summary>
    /// The <see cref="IEstimator{TTransformer}"/> for training a binary logistic regression classification model using the stochastic dual coordinate ascent method.
    /// The trained model is <a href='tmpurl_calib'>calibrated</a> and can produce probability by feeding the output value of the
    /// linear function to a <see cref="PlattCalibrator"/>.
    /// </summary>
    /// <include file='doc.xml' path='doc/members/member[@name="SDCA_remarks"]/*' />
    public sealed class SdcaCalibratedBinaryTrainer :
        SdcaBinaryTrainerBase<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>
    {
        /// <summary>
        /// Options for the <see cref="SdcaCalibratedBinaryTrainer"/>.
        /// </summary>
        public sealed class Options : BinaryOptionsBase
        {
        }

        internal SdcaCalibratedBinaryTrainer(IHostEnvironment env,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string weightColumnName = null,
            float? l2Const = null,
            float? l1Threshold = null,
            int? maxIterations = null)
             : base(env, labelColumnName, featureColumnName, weightColumnName, new LogLoss(), l2Const, l1Threshold, maxIterations)
        {
        }

        internal SdcaCalibratedBinaryTrainer(IHostEnvironment env, Options options)
            : base(env, options, new LogLoss())
        {
        }

        private protected override CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator> CreatePredictor(VBuffer<float>[] weights, float[] bias)
        {
            var linearModel = CreateLinearBinaryModelParameters(weights, bias);
            var calibrator = new PlattCalibrator(Host, -1, 0);
            return new ParameterMixingCalibratedModelParameters<LinearBinaryModelParameters, PlattCalibrator>(Host, linearModel, calibrator);
        }

        private protected override SchemaShape.Column[] ComputeSdcaBinaryClassifierSchemaShape()
        {
            return new SchemaShape.Column[]
            {
                    new SchemaShape.Column(
                        DefaultColumnNames.Score,
                        SchemaShape.Column.VectorKind.Scalar,
                        NumberDataViewType.Single,
                        false,
                        new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation())),
                    new SchemaShape.Column(
                        DefaultColumnNames.Probability,
                        SchemaShape.Column.VectorKind.Scalar,
                        NumberDataViewType.Single,
                        false,
                        new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation(true))),
                    new SchemaShape.Column(
                        DefaultColumnNames.PredictedLabel,
                        SchemaShape.Column.VectorKind.Scalar,
                        BooleanDataViewType.Instance,
                        false,
                        new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation()))

            };
        }
    }

    /// <summary>
    /// The <see cref="IEstimator{TTransformer}"/> for training a binary logistic regression classification model using the stochastic dual coordinate ascent method.
    /// </summary>
    /// <include file='doc.xml' path='doc/members/member[@name="SDCA_remarks"]/*' />
    public sealed class SdcaNonCalibratedBinaryTrainer : SdcaBinaryTrainerBase<LinearBinaryModelParameters>
    {
        /// <summary>
        /// Options for the <see cref="SdcaNonCalibratedBinaryTrainer"/>.
        /// </summary>
        public sealed class Options : BinaryOptionsBase
        {
            /// <summary>
            /// The custom <a href="tmpurl_loss">loss</a>.
            /// </summary>
            /// <value>
            /// If unspecified, <see cref="LogLoss"/> will be used.
            /// </value>
            [Argument(ArgumentType.Multiple, Name = "LossFunction", HelpText = "Loss Function", ShortName = "loss", SortOrder = 50)]
            internal ISupportSdcaClassificationLossFactory LossFunctionFactory = new LogLossFactory();

            /// <summary>
            /// The custom <a href="tmpurl_loss">loss</a>.
            /// </summary>
            /// <value>
            /// If unspecified, <see cref="LogLoss"/> will be used.
            /// </value>
            public ISupportSdcaClassificationLoss LossFunction { get; set; }
        }

        internal SdcaNonCalibratedBinaryTrainer(IHostEnvironment env,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string weightColumnName = null,
            ISupportSdcaClassificationLoss loss = null,
            float? l2Const = null,
            float? l1Threshold = null,
            int? maxIterations = null)
             : base(env, labelColumnName, featureColumnName, weightColumnName, loss, l2Const, l1Threshold, maxIterations)
        {
        }

        internal SdcaNonCalibratedBinaryTrainer(IHostEnvironment env, Options options)
            : base(env, options, options.LossFunction ?? options.LossFunctionFactory.CreateComponent(env))
        {
        }

        private protected override SchemaShape.Column[] ComputeSdcaBinaryClassifierSchemaShape()
        {
            return new SchemaShape.Column[]
            {
                    new SchemaShape.Column(
                        DefaultColumnNames.Score,
                        SchemaShape.Column.VectorKind.Scalar,
                        NumberDataViewType.Single,
                        false,
                        new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation())
                    ),
                    new SchemaShape.Column(
                        DefaultColumnNames.PredictedLabel,
                        SchemaShape.Column.VectorKind.Scalar,
                        BooleanDataViewType.Instance,
                        false,
                        new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation()))
            };
        }

        /// <summary>
        /// Comparing with <see cref="SdcaCalibratedBinaryTrainer.CreatePredictor(VBuffer{float}[], float[])"/>,
        /// <see cref="CreatePredictor"/> directly outputs a <see cref="LinearBinaryModelParameters"/> built from
        /// the learned weights and bias without calibration.
        /// </summary>
        private protected override LinearBinaryModelParameters CreatePredictor(VBuffer<float>[] weights, float[] bias)
            => CreateLinearBinaryModelParameters(weights, bias);
    }

    /// <summary>
    /// <see cref="LegacySdcaBinaryTrainer"/> is used to support classical command line tools where model is weakly-typed to
    /// <see cref="IPredictorWithFeatureWeights{TResult}"/>. Please do NOT use it whenever possible.
    /// </summary>
    internal sealed class LegacySdcaBinaryTrainer : SdcaBinaryTrainerBase<IPredictorWithFeatureWeights<float>>
    {
        internal const string LoadNameValue = "SDCA";
        internal const string UserNameValue = "Fast Linear (SA-SDCA)";

        /// <summary>
        /// Legacy configuration to SDCA in legacy framework.
        /// </summary>
        public sealed class Options : BinaryOptionsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "Loss Function", ShortName = "loss", SortOrder = 50)]
            public ISupportSdcaClassificationLossFactory LossFunction = new LogLossFactory();

            [Argument(ArgumentType.AtMostOnce, HelpText = "The calibrator kind to apply to the predictor. Specify null for no calibration", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            internal ICalibratorTrainerFactory Calibrator = new PlattCalibratorTrainerFactory();

            [Argument(ArgumentType.AtMostOnce, HelpText = "The maximum number of examples to use when training the calibrator", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            public int MaxCalibrationExamples = 1000000;
        }

        internal LegacySdcaBinaryTrainer(IHostEnvironment env, Options options)
            : base(env, options, options.LossFunction.CreateComponent(env), !(options.LossFunction is LogLossFactory))
        {
        }

        private protected override SchemaShape.Column[] ComputeSdcaBinaryClassifierSchemaShape()
        {
            var outCols = new List<SchemaShape.Column>()
            {
                    new SchemaShape.Column(
                        DefaultColumnNames.Score,
                        SchemaShape.Column.VectorKind.Scalar,
                        NumberDataViewType.Single,
                        false,
                        new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation())
                    ),
                    new SchemaShape.Column(
                        DefaultColumnNames.PredictedLabel,
                        SchemaShape.Column.VectorKind.Scalar,
                        BooleanDataViewType.Instance,
                        false,
                        new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation()))

            };

            if (!Info.NeedCalibration)
            {
                outCols.Insert(1, new SchemaShape.Column(
                    DefaultColumnNames.Probability,
                    SchemaShape.Column.VectorKind.Scalar,
                    NumberDataViewType.Single,
                    false,
                    new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation(true))));
            };

            return outCols.ToArray();
        }

        /// <summary>
        /// Weekly-typed function to create calibrated or uncalibrated predictors.
        /// </summary>
        private protected override IPredictorWithFeatureWeights<float> CreatePredictor(VBuffer<float>[] weights, float[] bias)
        {
            Host.CheckParam(Utils.Size(weights) == 1, nameof(weights));
            Host.CheckParam(Utils.Size(bias) == 1, nameof(bias));
            Host.CheckParam(weights[0].Length > 0, nameof(weights));

            VBuffer<float> maybeSparseWeights = default;
            // below should be `in weights[0]`, but can't because of https://github.com/dotnet/roslyn/issues/29371
            VBufferUtils.CreateMaybeSparseCopy(weights[0], ref maybeSparseWeights,
                Conversions.Instance.GetIsDefaultPredicate<float>(NumberDataViewType.Single));

            var predictor = new LinearBinaryModelParameters(Host, in maybeSparseWeights, bias[0]);
            if (Info.NeedCalibration)
                return predictor;
            return new ParameterMixingCalibratedModelParameters<LinearBinaryModelParameters, PlattCalibrator>(Host, predictor, new PlattCalibrator(Host, -1, 0));
        }
    }

    public abstract class SgdBinaryTrainerBase<TModel> :
        LinearTrainerBase<BinaryPredictionTransformer<TModel>, TModel>
        where TModel : class
    {
        public class OptionsBase : TrainerInputBaseWithWeight
        {
            /// <summary>
            /// The L2 weight for <a href='tmpurl_regularization'>regularization</a>.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "L2 Regularization constant", ShortName = "l2, L2Weight", SortOrder = 50)]
            [TGUI(Label = "L2 Regularization Constant", SuggestedSweeps = "1e-7,5e-7,1e-6,5e-6,1e-5")]
            [TlcModule.SweepableDiscreteParam("L2Const", new object[] { 1e-7f, 5e-7f, 1e-6f, 5e-6f, 1e-5f })]
            public float L2Regularization = Defaults.L2Regularization;

            /// <summary>
            /// The degree of lock-free parallelism used by SGD.
            /// </summary>
            /// <value>
            /// Defaults to automatic depending on data sparseness. Determinism is not guaranteed.
            /// </value>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Degree of lock-free parallelism. Defaults to automatic depending on data sparseness. Determinism not guaranteed.", ShortName = "nt,t,threads, NumThreads", SortOrder = 50)]
            [TGUI(Label = "Number of threads", SuggestedSweeps = "1,2,4")]
            public int? NumberOfThreads;

            /// <summary>
            /// The convergence tolerance. If the exponential moving average of loss reductions falls below this tolerance,
            /// the algorithm is deemed to have converged and will stop.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Exponential moving averaged improvement tolerance for convergence", ShortName = "tol")]
            [TGUI(SuggestedSweeps = "1e-2,1e-3,1e-4,1e-5")]
            [TlcModule.SweepableDiscreteParam("ConvergenceTolerance", new object[] { 1e-2f, 1e-3f, 1e-4f, 1e-5f })]
            public double ConvergenceTolerance = 1e-4;

            /// <summary>
            /// The maximum number of passes through the training dataset.
            /// </summary>
            /// <value>
            /// Set to 1 to simulate online learning.
            /// </value>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum number of iterations; set to 1 to simulate online learning.", ShortName = "iter, MaxIterations")]
            [TGUI(Label = "Max number of iterations", SuggestedSweeps = "1,5,10,20")]
            [TlcModule.SweepableDiscreteParam("MaxIterations", new object[] { 1, 5, 10, 20 })]
            public int NumberOfIterations = Defaults.NumberOfIterations;

            /// <summary>
            /// The initial <a href="tmpurl_lr">learning rate</a> used by SGD.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Initial learning rate (only used by SGD)", ShortName = "ilr,lr, InitLearningRate")]
            [TGUI(Label = "Initial Learning Rate (for SGD)")]
            public double InitialLearningRate = Defaults.InitialLearningRate;

            /// <summary>
            /// Determines whether to shuffle data for each training iteration.
            /// </summary>
            /// <value>
            /// <see langword="true" /> to shuffle data for each training iteration; otherwise, <see langword="false" />.
            /// Default is <see langword="true" />.
            /// </value>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Shuffle data every epoch?", ShortName = "shuf")]
            [TlcModule.SweepableDiscreteParam("Shuffle", null, isBool: true)]
            public bool Shuffle = true;

            /// <summary>
            /// The weight to be applied to the positive class. This is useful for training with imbalanced data.
            /// </summary>
            /// <value>
            /// Default value is 1, which means no extra weight.
            /// </value>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Apply weight to the positive class, for imbalanced data", ShortName = "piw")]
            public float PositiveInstanceWeight = 1;

            /// <summary>
            /// Determines the frequency of checking for convergence in terms of number of iterations.
            /// </summary>
            /// <value>
            /// Default equals <see cref="NumberOfThreads"/>."
            /// </value>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Convergence check frequency (in terms of number of iterations). Default equals number of threads", ShortName = "checkFreq")]
            public int? CheckFrequency;

            internal void Check(IHostEnvironment env)
            {
                Contracts.CheckValue(env, nameof(env));
                env.CheckUserArg(L2Regularization >= 0, nameof(L2Regularization), "Must be non-negative.");
                env.CheckUserArg(InitialLearningRate > 0, nameof(InitialLearningRate), "Must be positive.");
                env.CheckUserArg(NumberOfIterations > 0, nameof(NumberOfIterations), "Must be positive.");
                env.CheckUserArg(PositiveInstanceWeight > 0, nameof(PositiveInstanceWeight), "Must be positive");

                if (InitialLearningRate * L2Regularization >= 1)
                {
                    using (var ch = env.Start("Argument Adjustment"))
                    {
                        ch.Warning("{0} {1} set too high; reducing to {1}", nameof(InitialLearningRate),
                            InitialLearningRate, InitialLearningRate = (float)0.5 / L2Regularization);
                    }
                }

                if (ConvergenceTolerance <= 0)
                    ConvergenceTolerance = float.Epsilon;
            }

            [BestFriend]
            internal static class Defaults
            {
                public const float L2Regularization = 1e-6f;
                public const int NumberOfIterations = 20;
                public const double InitialLearningRate = 0.01;
            }
        }

        private readonly OptionsBase _options;

        private protected IClassificationLoss Loss { get; }

        private protected override bool ShuffleData => _options.Shuffle;

        private protected override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        public override TrainerInfo Info { get; }

        /// <summary>
        /// Initializes a new instance of <see cref="SgdBinaryTrainerBase{TModelParameters}"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="featureColumn">The name of the feature column.</param>
        /// <param name="labelColumn">The name of the label column.</param>
        /// <param name="weightColumn">The name of the example weight column.</param>
        /// <param name="maxIterations">The maximum number of iterations; set to 1 to simulate online learning.</param>
        /// <param name="initLearningRate">The initial learning rate used by SGD.</param>
        /// <param name="l2Weight">The L2 regularizer constant.</param>
        /// <param name="loss">The loss function to use.</param>
        internal SgdBinaryTrainerBase(IHostEnvironment env,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weightColumn = null,
            IClassificationLoss loss = null,
            int maxIterations = OptionsBase.Defaults.NumberOfIterations,
            double initLearningRate = OptionsBase.Defaults.InitialLearningRate,
            float l2Weight = OptionsBase.Defaults.L2Regularization)
            : base(env, featureColumn, TrainerUtils.MakeBoolScalarLabel(labelColumn), weightColumn)
        {
            Host.CheckNonEmpty(featureColumn, nameof(featureColumn));
            Host.CheckNonEmpty(labelColumn, nameof(labelColumn));

            _options = new OptionsBase();
            _options.NumberOfIterations = maxIterations;
            _options.InitialLearningRate = initLearningRate;
            _options.L2Regularization = l2Weight;

            _options.FeatureColumnName = featureColumn;
            _options.LabelColumnName = labelColumn;
            _options.ExampleWeightColumnName = weightColumn;
            Loss = loss ?? new LogLoss();
            Info = new TrainerInfo(calibration: false, supportIncrementalTrain: true);
        }

        /// <summary>
        /// Initializes a new instance of <see cref="SgdBinaryTrainerBase{TModelParameters}"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="options">Advanced arguments to the algorithm.</param>
        /// <param name="loss">Loss function would be minimized.</param>
        /// <param name="doCalibration">Set to true if a calibration step should be happen after training. Use false otherwise.</param>
        internal SgdBinaryTrainerBase(IHostEnvironment env, OptionsBase options, IClassificationLoss loss = null, bool doCalibration = false)
            : base(env, options.FeatureColumnName, TrainerUtils.MakeBoolScalarLabel(options.LabelColumnName), options.ExampleWeightColumnName)
        {
            options.Check(env);
            Loss = loss;
            Info = new TrainerInfo(calibration: doCalibration, supportIncrementalTrain: true);
            _options = options;
        }

        private protected override BinaryPredictionTransformer<TModel> MakeTransformer(TModel model, DataViewSchema trainSchema)
            => new BinaryPredictionTransformer<TModel>(Host, model, trainSchema, FeatureColumn.Name);

        /// <summary>
        /// Continues the training of a <see cref="SdcaCalibratedBinaryTrainer"/> using an already trained <paramref name="modelParameters"/> and returns a <see cref="BinaryPredictionTransformer"/>.
        /// </summary>
        public BinaryPredictionTransformer<TModel> Fit(IDataView trainData, LinearModelParameters modelParameters)
            => TrainTransformer(trainData, initPredictor: modelParameters);

        //For complexity analysis, we assume that
        // - The number of features is N
        // - Average number of non-zero per instance is k
        private protected override TModel TrainCore(IChannel ch, RoleMappedData data, LinearModelParameters predictor, int weightSetCount)
        {
            Contracts.AssertValue(data);
            Contracts.Assert(weightSetCount == 1);
            Contracts.AssertValueOrNull(predictor);
            Contracts.Assert(data.Schema.Feature.HasValue);

            int numFeatures = data.Schema.Feature.Value.Type.GetVectorSize();

            CursOpt cursorOpt = CursOpt.Label | CursOpt.Features;
            if (data.Schema.Weight.HasValue)
                cursorOpt |= CursOpt.Weight;

            var cursorFactory = new FloatLabelCursor.Factory(data, cursorOpt);

            int numThreads;
            if (_options.NumberOfThreads.HasValue)
            {
                numThreads = _options.NumberOfThreads.Value;
                ch.CheckUserArg(numThreads > 0, nameof(_options.NumberOfThreads), "The number of threads must be either null or a positive integer.");
            }
            else
                numThreads = ComputeNumThreads(cursorFactory);

            ch.Assert(numThreads > 0);
            int checkFrequency = _options.CheckFrequency ?? numThreads;
            if (checkFrequency <= 0)
                checkFrequency = int.MaxValue;
            var l2Weight = _options.L2Regularization;
            var lossFunc = Loss;
            var pOptions = new ParallelOptions { MaxDegreeOfParallelism = numThreads };
            var positiveInstanceWeight = _options.PositiveInstanceWeight;
            var weights = default(VBuffer<float>);
            float bias = 0.0f;
            if (predictor != null)
            {
                ((IHaveFeatureWeights)predictor).GetFeatureWeights(ref weights);
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
                if (e % checkFrequency == 0 && e != _options.NumberOfIterations)
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

                    pch.Checkpoint(loss, improvement, e, _options.NumberOfIterations);
                    converged = improvement < _options.ConvergenceTolerance;
                }
            };

            watch.Start();

            //Reference: Leon Bottou. Stochastic Gradient Descent Tricks.
            //https://research.microsoft.com/pubs/192769/tricks-2012.pdf

            var trainingTasks = new Action<Random, IProgressChannel>[_options.NumberOfIterations];
            var rands = new Random[_options.NumberOfIterations];
            var ilr = _options.InitialLearningRate;
            long t = 0;
            for (int epoch = 1; epoch <= _options.NumberOfIterations; epoch++)
            {
                int e = epoch; //localize the modified closure
                rands[e - 1] = RandomUtils.Create(Host.Rand.Next());
                trainingTasks[e - 1] = (rand, pch) =>
                {
                    using (var cursor = _options.Shuffle ? cursorFactory.Create(rand) : cursorFactory.Create())
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
                        entry => entry.SetProgress(0, iter, _options.NumberOfIterations));
                    // Synchorized SGD.
                    for (int i = 0; i < _options.NumberOfIterations; i++)
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
                    Parallel.For(0, _options.NumberOfIterations, pOptions, i => trainingTasks[i](rands[i], pch));
                    //note that P.Invoke will wait until all tasks finish
                }
            }

            VectorUtils.ScaleBy(ref weights, (float)weightScaling); // restore the true weights

            return CreateModel(weights, bias);
        }

        /// <summary>
        /// <see cref="CreateModel(VBuffer{float}, float)"/> implements a mechanism to craft a typed model out from linear weights and a bias.
        /// It's used at the end of <see cref="TrainCore(IChannel, RoleMappedData, LinearModelParameters, int)"/> to finalize the trained model.
        /// Derived classes should implement <see cref="CreateModel(VBuffer{float}, float)"/> because different trainers may produce different
        /// types of models.
        /// </summary>
        /// <param name="weights">Weights of linear model.</param>
        /// <param name="bias">Bias of linear model.</param>
        /// <returns>A model built upon weights and bias. It can be as simple as a <see cref="LinearBinaryModelParameters"/>.</returns>
        private protected abstract TModel CreateModel(VBuffer<float> weights, float bias);

        /// <summary>
        /// A helper function used to create <see cref="LinearBinaryModelParameters"/> in implementations of <see cref="CreateModel(VBuffer{float}, float)"/>.
        /// </summary>
        private protected LinearBinaryModelParameters CreateLinearBinaryModelParameters(VBuffer<float> weights, float bias)
        {
            Host.CheckParam(weights.Length > 0, nameof(weights));

            VBuffer<float> maybeSparseWeights = default;
            VBufferUtils.CreateMaybeSparseCopy(weights, ref maybeSparseWeights,
                Conversions.Instance.GetIsDefaultPredicate<float>(NumberDataViewType.Single));

            return new LinearBinaryModelParameters(Host, in maybeSparseWeights, bias);
        }

        private protected override void CheckLabel(RoleMappedData examples, out int weightSetCount)
        {
            examples.CheckBinaryLabel();
            weightSetCount = 1;
        }
    }

    /// <summary>
    /// The <see cref="IEstimator{TTransformer}"/> for training logistic regression using a parallel stochastic gradient method.
    /// The trained model is <a href='tmpurl_calib'>calibrated</a> and can produce probability by feeding the output value of the
    /// linear function to a <see cref="PlattCalibrator"/>.
    /// </summary>
    /// <remarks>
    /// The Stochastic Gradient Descent (SGD) is one of the popular stochastic optimization procedures that can be integrated
    /// into several machine learning tasks to achieve state-of-the-art performance. This trainer implements the Hogwild SGD for binary classification
    /// that supports multi-threading without any locking. If the associated optimization problem is sparse, Hogwild SGD achieves a nearly optimal
    /// rate of convergence. For more details about Hogwild SGD, please refer to http://arxiv.org/pdf/1106.5730v2.pdf.
    /// </remarks>
    public sealed class SgdCalibratedTrainer :
        SgdBinaryTrainerBase<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>
    {
        /// <summary>
        /// Options for the <see cref="SgdCalibratedTrainer"/>.
        /// </summary>
        public sealed class Options : OptionsBase
        {
        }

        internal SgdCalibratedTrainer(IHostEnvironment env,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weightColumn = null,
            int maxIterations = Options.Defaults.NumberOfIterations,
            double initLearningRate = Options.Defaults.InitialLearningRate,
            float l2Weight = Options.Defaults.L2Regularization)
            : base(env, labelColumn, featureColumn, weightColumn, new LogLoss(), maxIterations, initLearningRate, l2Weight)
        {
        }

        /// <summary>
        /// Initializes a new instance of <see cref="SgdBinaryTrainerBase{TModelParameters}"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="options">Advanced arguments to the algorithm.</param>
        internal SgdCalibratedTrainer(IHostEnvironment env, Options options)
            : base(env, options, loss: new LogLoss(), doCalibration: false)
        {
        }

        /// <summary>
        /// Logistic regression's output can naturally be interpreted as probability, so this model has three output columns.
        /// </summary>
        private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation())),
                new SchemaShape.Column(DefaultColumnNames.Probability, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation(true))),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation()))
            };
        }

        /// <summary>
        /// Given weights and bias trained in <see cref="SgdBinaryTrainerBase{TModelParameters}.TrainCore(IChannel, RoleMappedData, LinearModelParameters, int)"/>,
        /// <see cref="CreateModel(VBuffer{float}, float)"/> produces the final calibrated linear model.
        /// </summary>
        private protected override CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator> CreateModel(VBuffer<float> weights, float bias)
        {
            // SubModel, which is a linear function.
            var subModel = CreateLinearBinaryModelParameters(weights, bias);
            // Calibrator used to adjust the sub-model's output.
            var calibrator = new PlattCalibrator(Host, -1, 0);
            // Combine sub-model and calibrator as a whole.
            return new ParameterMixingCalibratedModelParameters<LinearBinaryModelParameters, PlattCalibrator>(Host, subModel, calibrator);
        }
    }

    /// <summary>
    /// <see cref="SgdNonCalibratedTrainer"/> can train a linear classification model by minimizing any loss function
    /// which implements <see cref="IClassificationLoss"/>.
    /// </summary>
    public sealed class SgdNonCalibratedTrainer :
        SgdBinaryTrainerBase<LinearBinaryModelParameters>
    {
        public sealed class Options : OptionsBase
        {
            /// <summary>
            /// The loss function to use. Default is <see cref="LogLoss"/>.
            /// </summary>
            [Argument(ArgumentType.Multiple, HelpText = "Loss Function", ShortName = "loss", SortOrder = 50)]
            public IClassificationLoss Loss = new LogLoss();
        }

        internal SgdNonCalibratedTrainer(IHostEnvironment env,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weightColumn = null,
            int maxIterations = Options.Defaults.NumberOfIterations,
            double initLearningRate = Options.Defaults.InitialLearningRate,
            float l2Weight = Options.Defaults.L2Regularization,
            IClassificationLoss loss = null)
            : base(env, labelColumn, featureColumn, weightColumn, loss, maxIterations, initLearningRate, l2Weight)
        {
        }

        /// <summary>
        /// Initializes a new instance of <see cref="SgdNonCalibratedTrainer"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="options">Advanced arguments to the algorithm.</param>
        internal SgdNonCalibratedTrainer(IHostEnvironment env, Options options)
            : base(env, options, loss: options.Loss, doCalibration: false)
        {
        }

        private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation())),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation()))
            };
        }

        private protected override LinearBinaryModelParameters CreateModel(VBuffer<float> weights, float bias)
            => CreateLinearBinaryModelParameters(weights, bias);
    }

    /// <summary>
    /// <see cref="LegacySgdBinaryTrainer"/> is used to support classical command line
    /// tools where model is weakly-typed to <see cref="IPredictorWithFeatureWeights{TResult}"/>. Please do NOT use it
    /// whenever possible.
    /// </summary>
    internal sealed class LegacySgdBinaryTrainer :
        SgdBinaryTrainerBase<IPredictorWithFeatureWeights<float>>
    {
        internal const string LoadNameValue = "BinarySGD";
        internal const string UserNameValue = "Hogwild SGD (binary)";
        internal const string ShortName = "HogwildSGD";

        public sealed class Options : OptionsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "Loss Function", ShortName = "loss", SortOrder = 50)]
            public ISupportClassificationLossFactory LossFunction = new LogLossFactory();

            [Argument(ArgumentType.AtMostOnce, HelpText = "The calibrator kind to apply to the predictor. Specify null for no calibration", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            internal ICalibratorTrainerFactory Calibrator = new PlattCalibratorTrainerFactory();

            [Argument(ArgumentType.AtMostOnce, HelpText = "The maximum number of examples to use when training the calibrator", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            internal int MaxCalibrationExamples = 1000000;
        }

        /// <summary>
        /// Initializes a new instance of <see cref="SgdBinaryTrainerBase{TModelParameters}"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="options">Advanced arguments to the algorithm.</param>
        internal LegacySgdBinaryTrainer(IHostEnvironment env, Options options)
            : base(env, options, options.LossFunction.CreateComponent(env), !(options.LossFunction is LogLossFactory))
        {
        }

        /// <summary>
        /// <see cref="LogLoss"/> leads to logistic regression which naturally supports probablity output. For other loss functions,
        /// a calibrator would be added after <see cref="SgdBinaryTrainerBase{TModelParameters}.TrainCore(IChannel, RoleMappedData, LinearModelParameters, int)"/>
        /// finishing its job. Therefore, we always have three output columns in the legacy world.
        /// </summary>
        /// <param name="inputSchema"></param>
        /// <returns></returns>
        private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation())),
                new SchemaShape.Column(DefaultColumnNames.Probability, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation(true))),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation()))
            };
        }

        private protected override IPredictorWithFeatureWeights<float> CreateModel(VBuffer<float> weights, float bias)
        {
            if (!(Loss is LogLoss))
                return CreateLinearBinaryModelParameters(weights, bias);

            // SubModel, which is a linear function.
            var subModel = CreateLinearBinaryModelParameters(weights, bias);

            // Calibrator used to adjust the sub-model's output.
            var calibrator = new PlattCalibrator(Host, -1, 0);
            // Combine sub-model and calibrator as a whole.
            return new ParameterMixingCalibratedModelParameters<LinearBinaryModelParameters, PlattCalibrator>(Host, subModel, calibrator);
        }

        [TlcModule.EntryPoint(Name = "Trainers.StochasticGradientDescentBinaryClassifier", Desc = "Train an Hogwild SGD binary model.", UserName = UserNameValue, ShortName = ShortName)]
        internal static CommonOutputs.BinaryClassificationOutput TrainBinary(IHostEnvironment env, Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainHogwildSGD");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return TrainerEntryPointsUtils.Train<Options, CommonOutputs.BinaryClassificationOutput>(host, input,
                () => new LegacySgdBinaryTrainer(host, input),
                () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumnName),
                () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.ExampleWeightColumnName),
                calibrator: input.Calibrator, maxCalibrationExamples: input.MaxCalibrationExamples);
        }
    }

    /// <summary>
    /// A component to train an SDCA model.
    /// </summary>
    internal static partial class Sdca
    {
        [TlcModule.EntryPoint(Name = "Trainers.StochasticDualCoordinateAscentBinaryClassifier",
            Desc = "Train an SDCA binary model.",
            UserName = LegacySdcaBinaryTrainer.UserNameValue,
            ShortName = LegacySdcaBinaryTrainer.LoadNameValue)]
        internal static CommonOutputs.BinaryClassificationOutput TrainBinary(IHostEnvironment env, LegacySdcaBinaryTrainer.Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainSDCA");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return TrainerEntryPointsUtils.Train<LegacySdcaBinaryTrainer.Options, CommonOutputs.BinaryClassificationOutput>(host, input,
                () => new LegacySdcaBinaryTrainer(host, input),
                () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumnName),
                calibrator: input.Calibrator, maxCalibrationExamples: input.MaxCalibrationExamples);
        }
    }
}
