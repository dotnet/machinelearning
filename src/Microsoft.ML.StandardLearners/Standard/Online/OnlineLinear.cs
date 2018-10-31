// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Numeric;
using Microsoft.ML.Runtime.Training;
using System;
using System.Globalization;
using Float = System.Single;

namespace Microsoft.ML.Trainers.Online
{

    public abstract class OnlineLinearArguments : LearnerInputBaseWithLabel
    {
        [Argument(ArgumentType.AtMostOnce, HelpText = "Number of iterations", ShortName = "iter", SortOrder = 50)]
        [TGUI(Label = "Number of Iterations", Description = "Number of training iterations through data", SuggestedSweeps = "1,10,100")]
        [TlcModule.SweepableLongParamAttribute("NumIterations", 1, 100, stepSize: 10, isLogScale: true)]
        public int NumIterations = OnlineDefaultArgs.NumIterations;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Initial Weights and bias, comma-separated", ShortName = "initweights")]
        [TGUI(NoSweep = true)]
        public string InitialWeights;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Init weights diameter", ShortName = "initwts", SortOrder = 140)]
        [TGUI(Label = "Initial Weights Scale", SuggestedSweeps = "0,0.1,0.5,1")]
        [TlcModule.SweepableFloatParamAttribute("InitWtsDiameter", 0.0f, 1.0f, numSteps: 5)]
        public Float InitWtsDiameter = 0;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to shuffle for each training iteration", ShortName = "shuf")]
        [TlcModule.SweepableDiscreteParamAttribute("Shuffle", new object[] { false, true })]
        public bool Shuffle = true;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Size of cache when trained in Scope", ShortName = "cache")]
        public int StreamingCacheSize = 1000000;

        internal class OnlineDefaultArgs
        {
            internal const int NumIterations = 1;
        }
    }

    public abstract class OnlineLinearTrainer<TTransformer, TModel> : TrainerEstimatorBase<TTransformer, TModel>
        where TTransformer : ISingleFeaturePredictionTransformer<TModel>
        where TModel : IPredictor
    {
        protected readonly OnlineLinearArguments Args;
        protected readonly string Name;

        /// <summary>
        /// An object to hold the mutable updatable state for the online linear trainers. Specific algorithms should subclass
        /// this, and return the instance via <see cref="MakeState(IChannel, int, LinearPredictor)"/>.
        /// </summary>
        private protected abstract class TrainStateBase
        {
            // Current iteration state.

            /// <summary>
            /// The number of iterations. Incremented by <see cref="BeginIteration(IChannel)"/>.
            /// </summary>
            public int Iteration;

            /// <summary>
            /// The number of examples in the current iteration. Incremented by <see cref="ProcessDataInstance(IChannel, ref VBuffer{Float}, Float, Float)"/>,
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
            public VBuffer<Float> Weights;

            /// <summary>
            /// The implicit scaling factor for <see cref="Weights"/>. Note that this does not affect <see cref="Bias"/>.
            /// </summary>
            public Float WeightsScale;

            /// <summary>
            /// The intercept term.
            /// </summary>
            public Float Bias;

            protected readonly IHost ParentHost;

            protected TrainStateBase(IChannel ch, int numFeatures, LinearPredictor predictor, OnlineLinearTrainer<TTransformer, TModel> parent)
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
                    predictor.GetFeatureWeights(ref Weights);
                    VBufferUtils.Densify(ref Weights);
                    Bias = predictor.Bias;
                }
                else if (!string.IsNullOrWhiteSpace(parent.Args.InitialWeights))
                {
                    ch.Info("Initializing weights and bias to " + parent.Args.InitialWeights);
                    string[] weightStr = parent.Args.InitialWeights.Split(',');
                    if (weightStr.Length != numFeatures + 1)
                    {
                        throw ch.Except(
                            "Could not initialize weights from 'initialWeights': expecting {0} values to initialize {1} weights and the intercept",
                            numFeatures + 1, numFeatures);
                    }

                    Weights = VBufferUtils.CreateDense<Float>(numFeatures);
                    for (int i = 0; i < numFeatures; i++)
                        Weights.Values[i] = Float.Parse(weightStr[i], CultureInfo.InvariantCulture);
                    Bias = Float.Parse(weightStr[numFeatures], CultureInfo.InvariantCulture);
                }
                else if (parent.Args.InitWtsDiameter > 0)
                {
                    Weights = VBufferUtils.CreateDense<Float>(numFeatures);
                    for (int i = 0; i < numFeatures; i++)
                        Weights.Values[i] = parent.Args.InitWtsDiameter * (parent.Host.Rand.NextSingle() - (Float)0.5);
                    Bias = parent.Args.InitWtsDiameter * (parent.Host.Rand.NextSingle() - (Float)0.5);
                }
                else if (numFeatures <= 1000)
                    Weights = VBufferUtils.CreateDense<Float>(numFeatures);
                else
                    Weights = VBufferUtils.CreateEmpty<Float>(numFeatures);
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
                Float absWeightsScale = Math.Abs(WeightsScale);
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
            public virtual void ProcessDataInstance(IChannel ch, ref VBuffer<Float> feat, Float label, Float weight)
            {
                ch.Assert(FloatUtils.IsFinite(feat.Values, feat.Count));
                ++NumIterExamples;
            }

            /// <summary>
            /// Return the raw margin from the decision hyperplane
            /// </summary>
            public Float CurrentMargin(ref VBuffer<Float> feat)
                => Bias + VectorUtils.DotProduct(ref feat, ref Weights) * WeightsScale;

            /// <summary>
            /// The default implementation just calls <see cref="CurrentMargin(ref VBuffer{Float})"/>.
            /// </summary>
            /// <param name="feat"></param>
            /// <returns></returns>
            public virtual Float Margin(ref VBuffer<Float> feat)
                => CurrentMargin(ref feat);

            public abstract TModel CreatePredictor();
        }

        // Our tolerance for the error induced by the weight scale may depend on our precision.
        private const Float _maxWeightScale = 1 << 10; // Exponent ranges 127 to -128, tolerate 10 being cut off that.
        private const Float _minWeightScale = 1 / _maxWeightScale;

        protected const string UserErrorPositive = "must be positive";
        protected const string UserErrorNonNegative = "must be non-negative";

        public override TrainerInfo Info { get; }

        protected virtual bool NeedCalibration => false;

<<<<<<< HEAD
        protected OnlineLinearTrainer(OnlineLinearArguments args, IHostEnvironment env, string name, SchemaShape.Column label)
=======
        private protected OnlineLinearTrainer(OnlineLinearArguments args, IHostEnvironment env, string name, SchemaShape.Column label)
>>>>>>> master
            : base(Contracts.CheckRef(env, nameof(env)).Register(name), TrainerUtils.MakeR4VecFeature(args.FeatureColumn), label, TrainerUtils.MakeR4ScalarWeightColumn(args.InitialWeights))
        {
            Contracts.CheckValue(args, nameof(args));
            Contracts.CheckUserArg(args.NumIterations > 0, nameof(args.NumIterations), UserErrorPositive);
            Contracts.CheckUserArg(args.InitWtsDiameter >= 0, nameof(args.InitWtsDiameter), UserErrorNonNegative);
            Contracts.CheckUserArg(args.StreamingCacheSize > 0, nameof(args.StreamingCacheSize), UserErrorPositive);

            Args = args;
            Name = name;
            // REVIEW: Caching could be false for one iteration, if we got around the whole shuffling issue.
            Info = new TrainerInfo(calibration: NeedCalibration, supportIncrementalTrain: true);
        }

        private protected static TArgs InvokeAdvanced<TArgs>(Action<TArgs> advancedSettings, TArgs args)
        {
            advancedSettings?.Invoke(args);
            return args;
        }

        protected sealed override TModel TrainModelCore(TrainContext context)
        {
            Host.CheckValue(context, nameof(context));
            var initPredictor = context.InitialPredictor;
            var initLinearPred = initPredictor as LinearPredictor ?? (initPredictor as CalibratedPredictorBase)?.SubPredictor as LinearPredictor;
            Host.CheckParam(initPredictor == null || initLinearPred != null, nameof(context), "Not a linear predictor.");
            var data = context.TrainingSet;

            data.CheckFeatureFloatVector(out int numFeatures);
            CheckLabels(data);

            using (var ch = Host.Start("Training"))
            {
                var state = MakeState(ch, numFeatures, initLinearPred);
                TrainCore(ch, data, state);

                ch.Assert(state.WeightsScale == 1);
                Float maxNorm = Math.Max(VectorUtils.MaxNorm(ref state.Weights), Math.Abs(state.Bias));
                ch.Check(FloatUtils.IsFinite(maxNorm),
                    "The weights/bias contain invalid values (NaN or Infinite). Potential causes: high learning rates, no normalization, high initial weights, etc.");
                return state.CreatePredictor();
            }
<<<<<<< HEAD

            return CreatePredictor();
        }

        protected abstract TModel CreatePredictor();

        protected abstract void CheckLabel(RoleMappedData data);

        protected virtual void TrainCore(IChannel ch, RoleMappedData data)
=======
        }

        protected abstract void CheckLabels(RoleMappedData data);

        private void TrainCore(IChannel ch, RoleMappedData data, TrainStateBase state)
>>>>>>> master
        {
            bool shuffle = Args.Shuffle;
            if (shuffle && !data.Data.CanShuffle)
            {
                ch.Warning("Training data does not support shuffling, so ignoring request to shuffle");
                shuffle = false;
            }

            var rand = shuffle ? Host.Rand : null;
            var cursorFactory = new FloatLabelCursor.Factory(data, CursOpt.Label | CursOpt.Features | CursOpt.Weight);
            long numBad = 0;
            while (state.Iteration < Args.NumIterations)
            {
                state.BeginIteration(ch);

                using (var cursor = cursorFactory.Create(rand))
                {
                    while (cursor.MoveNext())
                        state.ProcessDataInstance(ch, ref cursor.Features, cursor.Label, cursor.Weight);
                    numBad += cursor.BadFeaturesRowCount;
                }

                state.FinishIteration(ch);
            }

            if (numBad > 0)
            {
                ch.Warning(
                    "Skipped {0} instances with missing features during training (over {1} iterations; {2} inst/iter)",
                    numBad, Args.NumIterations, numBad / Args.NumIterations);
            }
        }

        private protected abstract TrainStateBase MakeState(IChannel ch, int numFeatures, LinearPredictor predictor);
    }
}