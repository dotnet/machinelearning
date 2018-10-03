// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Globalization;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Numeric;
using Microsoft.ML.Runtime.Training;

namespace Microsoft.ML.Runtime.Learners
{

    public abstract class OnlineLinearArguments : LearnerInputBaseWithLabel
    {
        [Argument(ArgumentType.AtMostOnce, HelpText = "Number of iterations", ShortName = "iter", SortOrder = 50)]
        [TGUI(Label = "Number of Iterations", Description = "Number of training iterations through data", SuggestedSweeps = "1,10,100")]
        [TlcModule.SweepableLongParamAttribute("NumIterations", 1, 100, stepSize: 10, isLogScale: true)]
        public int NumIterations = OnlineDefaultArgs.NumIterations;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Initial Weights and bias, comma-separated", ShortName = "initweights")]
        [TGUI(NoSweep = true)]
        public string InitialWeights = null;

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

        // Initialized by InitCore
        protected int NumFeatures;

        // Current iteration state
        protected int Iteration;
        protected long NumIterExamples;
        protected long NumBad;

        // Current weights and bias. The weights vector is considered to be scaled by
        // weightsScale. Storing this separately allows us to avoid the overhead of
        // an explicit scaling, which many learning algorithms will attempt to do on
        // each update. Bias is not subject to the weights scale.
        protected VBuffer<Float> Weights;
        protected Float WeightsScale;
        protected Float Bias;

        // Our tolerance for the error induced by the weight scale may depend on our precision.
        private const Float _maxWeightScale = 1 << 10; // Exponent ranges 127 to -128, tolerate 10 being cut off that.
        private const Float _minWeightScale = 1 / _maxWeightScale;

        protected const string UserErrorPositive = "must be positive";
        protected const string UserErrorNonNegative = "must be non-negative";

        public override TrainerInfo Info { get; }

        protected virtual bool NeedCalibration => false;

        protected OnlineLinearTrainer(OnlineLinearArguments args, IHostEnvironment env, string name, SchemaShape.Column label)
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

        /// <summary>
        /// Propagates the <see cref="WeightsScale"/> to the <see cref="Weights"/> vector.
        /// </summary>
        protected void ScaleWeights()
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
        protected void ScaleWeightsIfNeeded()
        {
            Float absWeightsScale = Math.Abs(WeightsScale);
            if (absWeightsScale < _minWeightScale || absWeightsScale > _maxWeightScale)
                ScaleWeights();
        }

        protected override TModel TrainModelCore(TrainContext context)
        {
            Host.CheckValue(context, nameof(context));
            var initPredictor = context.InitialPredictor;
            var initLinearPred = initPredictor as LinearPredictor ?? (initPredictor as CalibratedPredictorBase)?.SubPredictor as LinearPredictor;
            Host.CheckParam(initPredictor == null || initLinearPred != null, nameof(context), "Not a linear predictor.");
            var data = context.TrainingSet;

            data.CheckFeatureFloatVector(out int numFeatures);
            CheckLabel(data);

            using (var ch = Host.Start("Training"))
            {
                InitCore(ch, numFeatures, initLinearPred);
                // InitCore should set the number of features field.
                Contracts.Assert(NumFeatures > 0);

                TrainCore(ch, data);

                if (NumBad > 0)
                {
                    ch.Warning(
                        "Skipped {0} instances with missing features during training (over {1} iterations; {2} inst/iter)",
                        NumBad, Args.NumIterations, NumBad / Args.NumIterations);
                }

                Contracts.Assert(WeightsScale == 1);
                Float maxNorm = Math.Max(VectorUtils.MaxNorm(ref Weights), Math.Abs(Bias));
                Contracts.Check(FloatUtils.IsFinite(maxNorm),
                    "The weights/bias contain invalid values (NaN or Infinite). Potential causes: high learning rates, no normalization, high initial weights, etc.");

                ch.Done();
            }

            return CreatePredictor();
        }

        protected abstract TModel CreatePredictor();

        protected abstract void CheckLabel(RoleMappedData data);

        protected virtual void TrainCore(IChannel ch, RoleMappedData data)
        {
            bool shuffle = Args.Shuffle;
            if (shuffle && !data.Data.CanShuffle)
            {
                ch.Warning("Training data does not support shuffling, so ignoring request to shuffle");
                shuffle = false;
            }

            var rand = shuffle ? Host.Rand : null;
            var cursorFactory = new FloatLabelCursor.Factory(data, CursOpt.Label | CursOpt.Features | CursOpt.Weight);
            while (Iteration < Args.NumIterations)
            {
                BeginIteration(ch);

                using (var cursor = cursorFactory.Create(rand))
                {
                    while (cursor.MoveNext())
                        ProcessDataInstance(ch, ref cursor.Features, cursor.Label, cursor.Weight);
                    NumBad += cursor.BadFeaturesRowCount;
                }

                FinishIteration(ch);
            }
            // #if OLD_TRACING // REVIEW: How should this be ported?
            Console.WriteLine();
            // #endif
        }

        protected virtual void InitCore(IChannel ch, int numFeatures, LinearPredictor predictor)
        {
            Contracts.Check(numFeatures > 0, "Can't train with zero features!");
            Contracts.Check(NumFeatures == 0, "Can't re-use trainer!");
            Contracts.Assert(Iteration == 0);
            Contracts.Assert(Bias == 0);

            ch.Trace("{0} Initializing {1} on {2} features", DateTime.UtcNow, Name, numFeatures);
            NumFeatures = numFeatures;

            // We want a dense vector, to prevent memory creation during training
            // unless we have a lot of features.
            // REVIEW: make a setting
            if (predictor != null)
            {
                predictor.GetFeatureWeights(ref Weights);
                VBufferUtils.Densify(ref Weights);
                Bias = predictor.Bias;
            }
            else if (!string.IsNullOrWhiteSpace(Args.InitialWeights))
            {
                ch.Info("Initializing weights and bias to " + Args.InitialWeights);
                string[] weightStr = Args.InitialWeights.Split(',');
                if (weightStr.Length != NumFeatures + 1)
                {
                    throw Contracts.Except(
                        "Could not initialize weights from 'initialWeights': expecting {0} values to initialize {1} weights and the intercept",
                        NumFeatures + 1, NumFeatures);
                }

                Weights = VBufferUtils.CreateDense<Float>(NumFeatures);
                for (int i = 0; i < NumFeatures; i++)
                    Weights.Values[i] = Float.Parse(weightStr[i], CultureInfo.InvariantCulture);
                Bias = Float.Parse(weightStr[NumFeatures], CultureInfo.InvariantCulture);
            }
            else if (Args.InitWtsDiameter > 0)
            {
                Weights = VBufferUtils.CreateDense<Float>(NumFeatures);
                for (int i = 0; i < NumFeatures; i++)
                    Weights.Values[i] = Args.InitWtsDiameter * (Host.Rand.NextSingle() - (Float)0.5);
                Bias = Args.InitWtsDiameter * (Host.Rand.NextSingle() - (Float)0.5);
            }
            else if (NumFeatures <= 1000)
                Weights = VBufferUtils.CreateDense<Float>(NumFeatures);
            else
                Weights = VBufferUtils.CreateEmpty<Float>(NumFeatures);
            WeightsScale = 1;
        }

        protected virtual void BeginIteration(IChannel ch)
        {
            Iteration++;
            NumIterExamples = 0;

            ch.Trace("{0} Starting training iteration {1}", DateTime.UtcNow, Iteration);
            // #if OLD_TRACING // REVIEW: How should this be ported?
            if (Iteration % 20 == 0)
            {
                Console.Write('.');
                if (Iteration % 1000 == 0)
                    Console.WriteLine(" {0}  \t{1}", Iteration, DateTime.UtcNow);
            }
            // #endif
        }

        protected virtual void FinishIteration(IChannel ch)
        {
            Contracts.Check(NumIterExamples > 0, NoTrainingInstancesMessage);

            ch.Trace("{0} Finished training iteration {1}; iterated over {2} examples.",
                DateTime.UtcNow, Iteration, NumIterExamples);

            ScaleWeights();
#if OLD_TRACING // REVIEW: How should this be ported?
            if (DebugLevel > 3)
                PrintWeightsHistogram();
#endif
        }

#if OLD_TRACING // REVIEW: How should this be ported?
        protected virtual void PrintWeightsHistogram()
        {
            // Weights is scaled by weightsScale, but bias is not. Also, the scale term
            // in the histogram function is the inverse.
            PrintWeightsHistogram(ref _weights, _bias * _weightsScale, 1 / _weightsScale);
        }

        /// <summary>
        /// print the weights as an ASCII histogram
        /// </summary>
        protected void PrintWeightsHistogram(ref VBuffer<Float> weightVector, Float bias, Float scale)
        {
            Float min = Float.MaxValue;
            Float max = Float.MinValue;
            foreach (var iv in weightVector.Items())
            {
                var v = iv.Value;
                if (v != 0)
                {
                    if (min > v)
                        min = v;
                    if (max < v)
                        max = v;
                }
            }
            if (min > bias)
                min = bias;
            if (max < bias)
                max = bias;
            int numTicks = 50;
            Float tick = (max - min) / numTicks;

            if (scale != 1)
            {
                min /= scale;
                max /= scale;
                tick /= scale;
            }

            Host.StdOut.WriteLine("    WEIGHTS HISTOGRAM");
            Host.StdOut.Write("\t\t\t" + @"  {0:G2} ", min);
            for (int i = 0; i < numTicks; i = i + 5)
                Host.StdOut.Write(@" {0:G2} ", min + i * tick);
            Host.StdOut.WriteLine();

            foreach (var iv in weightVector.Items())
            {
                if (iv.Value == 0)
                    continue;
                Host.StdOut.Write("  " + iv.Key + "\t");
                Float weight = iv.Value / scale;
                Host.StdOut.Write(@" {0,5:G3} " + "\t|", weight);
                for (int j = 0; j < (weight - min) / tick; j++)
                    Host.StdOut.Write("=");
                Host.StdOut.WriteLine();
            }

            bias /= scale;
            Host.StdOut.Write("  BIAS\t\t\t\t" + @" {0:G3} " + "\t|", bias);
            for (int i = 0; i < (bias - min) / tick; i++)
                Host.StdOut.Write("=");
            Host.StdOut.WriteLine();
        }
#endif

        /// <summary>
        /// This should be overridden by derived classes. This implementation simply increments
        /// _numIterExamples and dumps debug information to the console.
        /// </summary>
        protected virtual void ProcessDataInstance(IChannel ch, ref VBuffer<Float> feat, Float label, Float weight)
        {
            Contracts.Assert(FloatUtils.IsFinite(feat.Values, feat.Count));

            ++NumIterExamples;
#if OLD_TRACING // REVIEW: How should this be ported?
            if (DebugLevel > 2)
            {
                Vector features = instance.Features;
                Host.StdOut.Write("Instance has label {0} and {1} features:", instance.Label, features.Length);
                for (int i = 0; i < features.Length; i++)
                {
                    Host.StdOut.Write('\t');
                    Host.StdOut.Write(features[i]);
                }
                Host.StdOut.WriteLine();
            }

            if (DebugLevel > 1)
            {
                if (_numIterExamples % 5000 == 0)
                {
                    Host.StdOut.Write('.');
                    if (_numIterExamples % 500000 == 0)
                    {
                        Host.StdOut.Write(" ");
                        Host.StdOut.Write(_numIterExamples);
                        if (_numIterExamples % 5000000 == 0)
                        {
                            Host.StdOut.Write(" ");
                            Host.StdOut.Write(DateTime.UtcNow);
                        }
                        Host.StdOut.WriteLine();
                    }
                }
            }
#endif
        }

        /// <summary>
        /// Return the raw margin from the decision hyperplane
        /// </summary>
        protected Float CurrentMargin(ref VBuffer<Float> feat)
        {
            return Bias + VectorUtils.DotProduct(ref feat, ref Weights) * WeightsScale;
        }

        protected virtual Float Margin(ref VBuffer<Float> feat)
        {
            return CurrentMargin(ref feat);
        }
    }
}