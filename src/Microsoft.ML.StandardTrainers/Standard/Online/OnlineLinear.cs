// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Globalization;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Numeric;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Trainers
{
    /// <summary>
    /// Arguments class for online linear trainers.
    /// </summary>
    public abstract class OnlineLinearOptions : TrainerInputBaseWithLabel
    {
        /// <summary>
        /// Number of passes through the training dataset.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Number of iterations", ShortName = "iter,numIterations", SortOrder = 50)]
        [TGUI(Label = "Number of Iterations", Description = "Number of training iterations through data", SuggestedSweeps = "1,10,100")]
        [TlcModule.SweepableLongParamAttribute("NumIterations", 1, 100, stepSize: 10, isLogScale: true)]
        public int NumberOfIterations = OnlineDefault.NumberOfIterations;

        /// <summary>
        /// Initial weights and bias, comma-separated.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Initial Weights and bias, comma-separated", ShortName = "initweights")]
        [TGUI(NoSweep = true)]
        internal string InitialWeights;

        /// <summary>
        /// Initial weights and bias scale.
        /// </summary>
        /// <value>
        /// This property is only used if the provided value is positive and <see cref="InitialWeights"/> is not specified.
        /// The weights and bias will be randomly selected from InitialWeights * [-0.5,0.5] interval with uniform distribution.
        /// </value>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Init weights diameter", ShortName = "initwts,initWtsDiameter", SortOrder = 140)]
        [TGUI(Label = "Initial Weights Scale", SuggestedSweeps = "0,0.1,0.5,1")]
        [TlcModule.SweepableFloatParamAttribute("InitWtsDiameter", 0.0f, 1.0f, numSteps: 5)]
        public float InitialWeightsDiameter = 0;

        /// <summary>
        /// Determines whether to shuffle data for each training iteration.
        /// </summary>
        /// <value>
        /// <see langword="true" /> to shuffle data for each training iteration; otherwise, <see langword="false" />.
        /// Default is <see langword="true" />.
        /// </value>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to shuffle for each training iteration", ShortName = "shuf")]
        [TlcModule.SweepableDiscreteParamAttribute("Shuffle", new object[] { false, true })]
        public bool Shuffle = true;

        [BestFriend]
        internal class OnlineDefault
        {
            public const int NumberOfIterations = 1;
        }
    }

    public abstract class OnlineLinearTrainer<TTransformer, TModel> : TrainerEstimatorBase<TTransformer, TModel>
        where TTransformer : ISingleFeaturePredictionTransformer<TModel>
        where TModel : class
    {
        private protected readonly OnlineLinearOptions OnlineLinearTrainerOptions;
        private protected readonly string Name;

        /// <summary>
        /// An object to hold the mutable updatable state for the online linear trainers. Specific algorithms should subclass
        /// this, and return the instance via <see cref="MakeState(IChannel, int, LinearModelParameters)"/>.
        /// </summary>
        private protected abstract class TrainStateBase
        {
            // Current iteration state.

            /// <summary>
            /// The number of iterations. Incremented by <see cref="BeginIteration(IChannel)"/>.
            /// </summary>
            public int Iteration;

            /// <summary>
            /// The number of examples in the current iteration. Incremented by <see cref="ProcessDataInstance(IChannel, in VBuffer{float}, float, float)"/>,
            /// and reset by <see cref="BeginIteration(IChannel)"/>.
            /// </summary>
            public long NumIterExamples;

            // Current weights and bias. The weights vector is considered to be scaled by
            // weightsScale. Storing this separately allows us to avoid the overhead of
            // an explicit scaling, which many learning algorithms will attempt to do on
            // each update. Bias is not subject to the weights scale.

            /// <summary>
            /// Current weights. The weights vector is considered to be scaled by <see cref="WeightsScale"/>. Storing this separately
            /// allows us to avoid the overhead of an explicit scaling, which some algorithms will attempt to do on each example's update.
            /// </summary>
            public VBuffer<float> Weights;

            /// <summary>
            /// The implicit scaling factor for <see cref="Weights"/>. Note that this does not affect <see cref="Bias"/>.
            /// </summary>
            public float WeightsScale;

            /// <summary>
            /// The intercept term.
            /// </summary>
            public float Bias;

            protected readonly IHost ParentHost;

            protected TrainStateBase(IChannel ch, int numFeatures, LinearModelParameters predictor, OnlineLinearTrainer<TTransformer, TModel> parent)
            {
                Contracts.CheckValue(ch, nameof(ch));
                ch.Check(numFeatures > 0, "Cannot train with zero features!");
                ch.AssertValueOrNull(predictor);
                ch.AssertValue(parent);
                ch.Assert(Iteration == 0);
                ch.Assert(Bias == 0);

                ParentHost = parent.Host;

                ch.Trace("{0} Initializing {1} on {2} features", DateTime.UtcNow, parent.Name, numFeatures);

                // We want a dense vector, to prevent memory creation during training
                // unless we have a lot of features.
                if (predictor != null)
                {
                    ((IHaveFeatureWeights)predictor).GetFeatureWeights(ref Weights);
                    VBufferUtils.Densify(ref Weights);
                    Bias = predictor.Bias;
                }
                else if (!string.IsNullOrWhiteSpace(parent.OnlineLinearTrainerOptions.InitialWeights))
                {
                    ch.Info("Initializing weights and bias to " + parent.OnlineLinearTrainerOptions.InitialWeights);
                    string[] weightStr = parent.OnlineLinearTrainerOptions.InitialWeights.Split(',');
                    if (weightStr.Length != numFeatures + 1)
                    {
                        throw ch.Except(
                            "Could not initialize weights from 'initialWeights': expecting {0} values to initialize {1} weights and the intercept",
                            numFeatures + 1, numFeatures);
                    }

                    var weightValues = new float[numFeatures];
                    for (int i = 0; i < numFeatures; i++)
                        weightValues[i] = float.Parse(weightStr[i], CultureInfo.InvariantCulture);
                    Weights = new VBuffer<float>(numFeatures, weightValues);
                    Bias = float.Parse(weightStr[numFeatures], CultureInfo.InvariantCulture);
                }
                else if (parent.OnlineLinearTrainerOptions.InitialWeightsDiameter > 0)
                {
                    var weightValues = new float[numFeatures];
                    for (int i = 0; i < numFeatures; i++)
                        weightValues[i] = parent.OnlineLinearTrainerOptions.InitialWeightsDiameter * (parent.Host.Rand.NextSingle() - (float)0.5);
                    Weights = new VBuffer<float>(numFeatures, weightValues);
                    Bias = parent.OnlineLinearTrainerOptions.InitialWeightsDiameter * (parent.Host.Rand.NextSingle() - (float)0.5);
                }
                else if (numFeatures <= 1000)
                    Weights = VBufferUtils.CreateDense<float>(numFeatures);
                else
                    Weights = VBufferUtils.CreateEmpty<float>(numFeatures);
                WeightsScale = 1;
            }

            /// <summary>
            /// Propagates the <see cref="WeightsScale"/> to the <see cref="Weights"/> vector.
            /// </summary>
            private void ScaleWeights()
            {
                if (WeightsScale != 1)
                {
                    VectorUtils.ScaleBy(ref Weights, WeightsScale);
                    WeightsScale = 1;
                }
            }

            /// <summary>
            /// Conditionally propagates the <see cref="WeightsScale"/> to the <see cref="Weights"/> vector
            /// when it reaches a scale where additions to weights would start dropping too much precision.
            /// ("Too much" is mostly empirically defined.)
            /// </summary>
            public void ScaleWeightsIfNeeded()
            {
                float absWeightsScale = Math.Abs(WeightsScale);
                if (absWeightsScale < _minWeightScale || absWeightsScale > _maxWeightScale)
                    ScaleWeights();
            }

            /// <summary>
            /// Called by <see cref="TrainCore(IChannel, RoleMappedData, TrainStateBase)"/> at the start of a pass over the dataset.
            /// </summary>
            public virtual void BeginIteration(IChannel ch)
            {
                Iteration++;
                NumIterExamples = 0;

                ch.Trace("{0} Starting training iteration {1}", DateTime.UtcNow, Iteration);
            }

            /// <summary>
            /// Called by <see cref="TrainCore(IChannel, RoleMappedData, TrainStateBase)"/> after a pass over the dataset.
            /// </summary>
            public virtual void FinishIteration(IChannel ch)
            {
                Contracts.Check(NumIterExamples > 0, NoTrainingInstancesMessage);

                ch.Trace("{0} Finished training iteration {1}; iterated over {2} examples.",
                    DateTime.UtcNow, Iteration, NumIterExamples);

                ScaleWeights();
            }

            /// <summary>
            /// This should be overridden by derived classes. This implementation simply increments <see cref="NumIterExamples"/>.
            /// </summary>
            public virtual void ProcessDataInstance(IChannel ch, in VBuffer<float> feat, float label, float weight)
            {
                ch.Assert(FloatUtils.IsFinite(feat.GetValues()));
                ++NumIterExamples;
            }

            /// <summary>
            /// Return the raw margin from the decision hyperplane
            /// </summary>
            public float CurrentMargin(in VBuffer<float> feat)
                => Bias + VectorUtils.DotProduct(in feat, in Weights) * WeightsScale;

            /// <summary>
            /// The default implementation just calls <see cref="CurrentMargin(in VBuffer{float})"/>.
            /// </summary>
            /// <param name="feat"></param>
            /// <returns></returns>
            public virtual float Margin(in VBuffer<float> feat)
                => CurrentMargin(in feat);

            public abstract TModel CreatePredictor();
        }

        // Our tolerance for the error induced by the weight scale may depend on our precision.
        private const float _maxWeightScale = 1 << 10; // Exponent ranges 127 to -128, tolerate 10 being cut off that.
        private const float _minWeightScale = 1 / _maxWeightScale;

        private protected const string UserErrorPositive = "must be positive";
        private protected const string UserErrorNonNegative = "must be non-negative";

        public override TrainerInfo Info { get; }

        private protected virtual bool NeedCalibration => false;

        private protected OnlineLinearTrainer(OnlineLinearOptions options, IHostEnvironment env, string name, SchemaShape.Column label)
            : base(Contracts.CheckRef(env, nameof(env)).Register(name), TrainerUtils.MakeR4VecFeature(options.FeatureColumnName), label, TrainerUtils.MakeR4ScalarWeightColumn(options.InitialWeights))
        {
            Contracts.CheckValue(options, nameof(options));
            Contracts.CheckUserArg(options.NumberOfIterations > 0, nameof(options.NumberOfIterations), UserErrorPositive);
            Contracts.CheckUserArg(options.InitialWeightsDiameter >= 0, nameof(options.InitialWeightsDiameter), UserErrorNonNegative);

            OnlineLinearTrainerOptions = options;
            Name = name;
            // REVIEW: Caching could be false for one iteration, if we got around the whole shuffling issue.
            Info = new TrainerInfo(calibration: NeedCalibration, supportIncrementalTrain: true);
        }

        private protected sealed override TModel TrainModelCore(TrainContext context)
        {
            Host.CheckValue(context, nameof(context));
            var initPredictor = context.InitialPredictor;

            if (initPredictor is LinearModelParameters initLinearPred)
                initLinearPred = (LinearModelParameters)initPredictor;
            else
            {
                initLinearPred = null;
                Host.CheckParam(initPredictor == null || initLinearPred != null, nameof(context),
                    "Initial predictor was not a linear predictor.");
            }

            var data = context.TrainingSet;
            data.CheckFeatureFloatVector(out int numFeatures);
            CheckLabels(data);

            using (var ch = Host.Start("Training"))
            {
                var state = MakeState(ch, numFeatures, initLinearPred);
                TrainCore(ch, data, state);

                ch.Assert(state.WeightsScale == 1);
                float maxNorm = Math.Max(VectorUtils.MaxNorm(in state.Weights), Math.Abs(state.Bias));
                ch.Check(FloatUtils.IsFinite(maxNorm),
                    "The weights/bias contain invalid values (NaN or Infinite). Potential causes: high learning rates, no normalization, high initial weights, etc.");
                return state.CreatePredictor();
            }
        }

        /// <summary>
        /// Continues the training of a <see cref="OnlineLinearTrainer{TTransformer, TModel}"/> using an already trained <paramref name="modelParameters"/> and returns a <see cref="ITransformer"/>.
        /// </summary>
        public TTransformer Fit(IDataView trainData, LinearModelParameters modelParameters)
            => TrainTransformer(trainData, initPredictor: modelParameters);

        private protected abstract void CheckLabels(RoleMappedData data);

        private void TrainCore(IChannel ch, RoleMappedData data, TrainStateBase state)
        {
            bool shuffle = OnlineLinearTrainerOptions.Shuffle;
            if (shuffle && !data.Data.CanShuffle)
            {
                ch.Warning("Training data does not support shuffling, so ignoring request to shuffle");
                shuffle = false;
            }

            var rand = shuffle ? Host.Rand : null;
            CursOpt cursorOpt = CursOpt.Label | CursOpt.Features;
            if (data.Schema.Weight.HasValue)
                cursorOpt |= CursOpt.Weight;

            var cursorFactory = new FloatLabelCursor.Factory(data, cursorOpt);
            long numBad = 0;
            while (state.Iteration < OnlineLinearTrainerOptions.NumberOfIterations)
            {
                state.BeginIteration(ch);

                using (var cursor = cursorFactory.Create(rand))
                {
                    while (cursor.MoveNext())
                        state.ProcessDataInstance(ch, in cursor.Features, cursor.Label, cursor.Weight);
                    numBad += cursor.BadFeaturesRowCount;
                }

                state.FinishIteration(ch);
            }

            if (numBad > 0)
            {
                ch.Warning(
                    "Skipped {0} instances with missing features during training (over {1} iterations; {2} inst/iter)",
                    numBad, OnlineLinearTrainerOptions.NumberOfIterations, numBad / OnlineLinearTrainerOptions.NumberOfIterations);
            }
        }

        private protected abstract TrainStateBase MakeState(IChannel ch, int numFeatures, LinearModelParameters predictor);
    }
}
