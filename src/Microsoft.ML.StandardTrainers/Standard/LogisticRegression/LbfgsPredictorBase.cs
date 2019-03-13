// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Numeric;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Trainers
{
    public abstract class LbfgsTrainerBase<TOptions, TTransformer, TModel> : TrainerEstimatorBase<TTransformer, TModel>
      where TTransformer : ISingleFeaturePredictionTransformer<TModel>
      where TModel : class
      where TOptions : LbfgsTrainerBase<TOptions, TTransformer, TModel>.OptionsBase, new()
    {
        public abstract class OptionsBase : TrainerInputBaseWithWeight
        {
            /// <summary>
            /// L2 regularization weight.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "L2 regularization weight", ShortName = "l2, L2Weight", SortOrder = 50)]
            [TGUI(Label = "L2 Weight", Description = "Weight of L2 regularizer term", SuggestedSweeps = "0,0.1,1")]
            [TlcModule.SweepableFloatParamAttribute(0.0f, 1.0f, numSteps: 4)]
            public float L2Regularization = Defaults.L2Regularization;

            /// <summary>
            /// L1 regularization weight.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "L1 regularization weight", ShortName = "l1, L1Weight", SortOrder = 50)]
            [TGUI(Label = "L1 Weight", Description = "Weight of L1 regularizer term", SuggestedSweeps = "0,0.1,1")]
            [TlcModule.SweepableFloatParamAttribute(0.0f, 1.0f, numSteps: 4)]
            public float L1Regularization = Defaults.L1Regularization;

            /// <summary>
            /// Tolerance parameter for optimization convergence. (Low = slower, more accurate).
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Tolerance parameter for optimization convergence. Low = slower, more accurate",
                ShortName = "ot, OptTol", SortOrder = 50)]
            [TGUI(Label = "Optimization Tolerance", Description = "Threshold for optimizer convergence", SuggestedSweeps = "1e-4,1e-7")]
            [TlcModule.SweepableDiscreteParamAttribute(new object[] { 1e-4f, 1e-7f })]
            public float OptmizationTolerance = Defaults.OptimizationTolerance;

            /// <summary>
            /// Number of previous iterations to remember for estimate of Hessian.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Memory size for L-BFGS. Low=faster, less accurate",
                ShortName = "m, MemorySize", SortOrder = 50)]
            [TGUI(Description = "Memory size for L-BFGS", SuggestedSweeps = "5,20,50")]
            [TlcModule.SweepableDiscreteParamAttribute("MemorySize", new object[] { 5, 20, 50 })]
            public int HistorySize = Defaults.HistorySize;

            /// <summary>
            /// Number of iterations.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum iterations.", ShortName = "maxiter, MaxIterations, NumberOfIterations")]
            [TGUI(Label = "Max Number of Iterations")]
            [TlcModule.SweepableLongParamAttribute("MaxIterations", 1, int.MaxValue)]
            public int MaximumNumberOfIterations = Defaults.MaximumNumberOfIterations;

            /// <summary>
            /// Run SGD to initialize LR weights, converging to this tolerance.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Run SGD to initialize LR weights, converging to this tolerance",
                ShortName = "sgd, SgdInitializationTolerance")]
            public float StochasticGradientDescentInitilaizationTolerance = 0;

            /// <summary>
            /// Features must occur in at least this many instances to be included
            /// </summary>
            /// <remarks>If greater than 1, forces an initialization pass over the data</remarks>
            //AP removed from now. This requires data transformatio. Perhaps we should handle it as seprate (non-learner) dependant
            //Similarly how normalization is done
            //public int CountThreshold { get { return _countThreshold; } set { _countThreshold = value; } }

            [Argument(ArgumentType.AtMostOnce, HelpText = "If set to true, produce no output during training.",
                ShortName = "q")]
            public bool Quiet = false;

            /// <summary>
            /// Init Weights Diameter
            /// </summary>
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Init weights diameter", ShortName = "initwts, InitWtsDiameter", SortOrder = 140)]
            [TGUI(Label = "Initial Weights Scale", SuggestedSweeps = "0,0.1,0.5,1")]
            [TlcModule.SweepableFloatParamAttribute("InitWtsDiameter", 0.0f, 1.0f, numSteps: 5)]
            public float InitialWeightsDiameter = 0;

            // Deprecated
            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether or not to use threads. Default is true",
                ShortName = "t", Hide = true)]
            internal bool UseThreads = true;

            /// <summary>
            /// Number of threads. Null means use the number of processors.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of threads", ShortName = "nt, NumThreads")]
            public int? NumberOfThreads;

            /// <summary>
            /// Force densification of the internal optimization vectors. Default is false.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Force densification of the internal optimization vectors", ShortName = "do")]
            [TlcModule.SweepableDiscreteParamAttribute("DenseOptimizer", new object[] { false, true })]
            public bool DenseOptimizer = false;

            /// <summary>
            /// Enforce non-negative weights. Default is false.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Enforce non-negative weights", ShortName = "nn", SortOrder = 90)]
            public bool EnforceNonNegativity = Defaults.EnforceNonNegativity;

            [BestFriend]
            internal static class Defaults
            {
                public const float L2Regularization = 1;
                public const float L1Regularization = 1;
                public const float OptimizationTolerance = 1e-7f;
                public const int HistorySize = 20;
                public const int MaximumNumberOfIterations = int.MaxValue;
                public const bool EnforceNonNegativity = false;
            }
        }

        private const string RegisterName = nameof(LbfgsTrainerBase<TOptions, TTransformer, TModel>);

        private protected int NumFeatures;
        private protected VBuffer<float> CurrentWeights;
        private protected long NumGoodRows;
        private protected Double WeightSum;
        private protected bool ShowTrainingStats;

        private IPredictor _srcPredictor;

        private protected readonly TOptions LbfgsTrainerOptions;
        private protected readonly float L2Weight;
        private protected readonly float L1Weight;
        private protected readonly float OptTol;
        private protected readonly int MemorySize;
        private protected readonly int MaxIterations;
        private protected readonly float SgdInitializationTolerance;
        private protected readonly bool Quiet;
        private protected readonly float InitWtsDiameter;
        private protected readonly bool UseThreads;
        private protected readonly int? NumThreads;
        private protected readonly bool DenseOptimizer;
        private protected readonly long MaxNormalizationExamples;
        private protected readonly bool EnforceNonNegativity;

        // The training data, when NOT using multiple threads.
        private RoleMappedData _data;
        private FloatLabelCursor.Factory _cursorFactory;

        // Used for the multi-threading case, null otherwise. These three arrays are parallel. _weights may be null.
        private VBuffer<float>[] _features;
        private float[] _labels;
        private float[] _weights;

        // Stores the bounds of the chunk to be used by each thread. The 0th slot is 0. The length
        // is one more than the number of threads to use.
        private int _numChunks;
        private int[] _ranges;

        // Working buffers allocated lazily.
        private VBuffer<float>[] _localGradients;
        private float[] _localLosses;

        // REVIEW: It's pointless to request caching when we're going to load everything into
        // memory, that is, when using multiple threads. So should caching not be requested?
        private static readonly TrainerInfo _info = new TrainerInfo(caching: true, supportIncrementalTrain: true);
        public override TrainerInfo Info => _info;

        internal LbfgsTrainerBase(IHostEnvironment env,
            string featureColumn,
            SchemaShape.Column labelColumn,
            string weightColumn,
            float l1Weight,
            float l2Weight,
            float optimizationTolerance,
            int memorySize,
            bool enforceNoNegativity)
            : this(env, new TOptions
                        {
                            FeatureColumnName = featureColumn,
                            LabelColumnName = labelColumn.Name,
                            ExampleWeightColumnName = weightColumn,
                            L1Regularization = l1Weight,
                            L2Regularization = l2Weight,
                            OptmizationTolerance = optimizationTolerance,
                            HistorySize = memorySize,
                            EnforceNonNegativity = enforceNoNegativity
                        },
                  labelColumn)
        {
        }

        internal LbfgsTrainerBase(IHostEnvironment env,
            TOptions options,
            SchemaShape.Column labelColumn,
            Action<TOptions> advancedSettings = null)
            : base(Contracts.CheckRef(env, nameof(env)).Register(RegisterName), TrainerUtils.MakeR4VecFeature(options.FeatureColumnName),
                  labelColumn, TrainerUtils.MakeR4ScalarWeightColumn(options.ExampleWeightColumnName))
        {
            Host.CheckValue(options, nameof(options));
            LbfgsTrainerOptions = options;

            // Apply the advanced args, if the user supplied any.
            advancedSettings?.Invoke(options);

            options.FeatureColumnName = FeatureColumn.Name;
            options.LabelColumnName = LabelColumn.Name;
            options.ExampleWeightColumnName = WeightColumn.Name;
            Host.CheckUserArg(!LbfgsTrainerOptions.UseThreads || LbfgsTrainerOptions.NumberOfThreads > 0 || LbfgsTrainerOptions.NumberOfThreads == null,
              nameof(LbfgsTrainerOptions.NumberOfThreads), "Must be positive (or empty for default)");
            Host.CheckUserArg(LbfgsTrainerOptions.L2Regularization >= 0, nameof(LbfgsTrainerOptions.L2Regularization), "Must be non-negative");
            Host.CheckUserArg(LbfgsTrainerOptions.L1Regularization >= 0, nameof(LbfgsTrainerOptions.L1Regularization), "Must be non-negative");
            Host.CheckUserArg(LbfgsTrainerOptions.OptmizationTolerance > 0, nameof(LbfgsTrainerOptions.OptmizationTolerance), "Must be positive");
            Host.CheckUserArg(LbfgsTrainerOptions.HistorySize > 0, nameof(LbfgsTrainerOptions.HistorySize), "Must be positive");
            Host.CheckUserArg(LbfgsTrainerOptions.MaximumNumberOfIterations > 0, nameof(LbfgsTrainerOptions.MaximumNumberOfIterations), "Must be positive");
            Host.CheckUserArg(LbfgsTrainerOptions.StochasticGradientDescentInitilaizationTolerance >= 0, nameof(LbfgsTrainerOptions.StochasticGradientDescentInitilaizationTolerance), "Must be non-negative");
            Host.CheckUserArg(LbfgsTrainerOptions.NumberOfThreads == null || LbfgsTrainerOptions.NumberOfThreads.Value >= 0, nameof(LbfgsTrainerOptions.NumberOfThreads), "Must be non-negative");

            Host.CheckParam(!(LbfgsTrainerOptions.L2Regularization < 0), nameof(LbfgsTrainerOptions.L2Regularization), "Must be non-negative, if provided.");
            Host.CheckParam(!(LbfgsTrainerOptions.L1Regularization < 0), nameof(LbfgsTrainerOptions.L1Regularization), "Must be non-negative, if provided");
            Host.CheckParam(!(LbfgsTrainerOptions.OptmizationTolerance <= 0), nameof(LbfgsTrainerOptions.OptmizationTolerance), "Must be positive, if provided.");
            Host.CheckParam(!(LbfgsTrainerOptions.HistorySize <= 0), nameof(LbfgsTrainerOptions.HistorySize), "Must be positive, if provided.");

            L2Weight = LbfgsTrainerOptions.L2Regularization;
            L1Weight = LbfgsTrainerOptions.L1Regularization;
            OptTol = LbfgsTrainerOptions.OptmizationTolerance;
            MemorySize =LbfgsTrainerOptions.HistorySize;
            MaxIterations = LbfgsTrainerOptions.MaximumNumberOfIterations;
            SgdInitializationTolerance = LbfgsTrainerOptions.StochasticGradientDescentInitilaizationTolerance;
            Quiet = LbfgsTrainerOptions.Quiet;
            InitWtsDiameter = LbfgsTrainerOptions.InitialWeightsDiameter;
            UseThreads = LbfgsTrainerOptions.UseThreads;
            NumThreads = LbfgsTrainerOptions.NumberOfThreads;
            DenseOptimizer = LbfgsTrainerOptions.DenseOptimizer;
            EnforceNonNegativity = LbfgsTrainerOptions.EnforceNonNegativity;

            if (EnforceNonNegativity && ShowTrainingStats)
            {
                ShowTrainingStats = false;
                using (var ch = Host.Start("Initialization"))
                {
                    ch.Warning("The training statistics cannot be computed with non-negativity constraint.");
                }
            }

            ShowTrainingStats = false;
            _srcPredictor = default;
        }

        private static TOptions ArgsInit(string featureColumn, SchemaShape.Column labelColumn,
                string weightColumn,
                float l1Weight,
                float l2Weight,
                float optimizationTolerance,
                int memorySize,
                bool enforceNoNegativity)
        {
            var args = new TOptions
            {
                FeatureColumnName = featureColumn,
                LabelColumnName = labelColumn.Name,
                ExampleWeightColumnName = weightColumn,
                L1Regularization = l1Weight,
                L2Regularization = l2Weight,
                OptmizationTolerance = optimizationTolerance,
                HistorySize = memorySize,
                EnforceNonNegativity = enforceNoNegativity
            };

            args.ExampleWeightColumnName = weightColumn;
            return args;
        }

        private protected virtual int ClassCount => 1;
        private protected int BiasCount => ClassCount;
        private protected int WeightCount => ClassCount * NumFeatures;
        private protected virtual Optimizer InitializeOptimizer(IChannel ch, FloatLabelCursor.Factory cursorFactory,
            out VBuffer<float> init, out ITerminationCriterion terminationCriterion)
        {
            // MeanRelativeImprovementCriterion:
            //   Stops optimization when the average objective improvement over the last
            //   n iterations, normalized by the function value, is small enough.
            terminationCriterion = new MeanRelativeImprovementCriterion(OptTol, 5, MaxIterations);

            Optimizer opt = (L1Weight > 0)
                ? new L1Optimizer(Host, BiasCount, L1Weight / NumGoodRows, MemorySize, DenseOptimizer, null, EnforceNonNegativity)
                : new Optimizer(Host, MemorySize, DenseOptimizer, null, EnforceNonNegativity);
            opt.Quiet = Quiet;

            if (_srcPredictor != null)
                init = InitializeWeightsFromPredictor(_srcPredictor);
            else if (InitWtsDiameter > 0)
            {
                float[] initWeights = new float[BiasCount + WeightCount];
                for (int j = 0; j < initWeights.Length; j++)
                    initWeights[j] = InitWtsDiameter * (Host.Rand.NextSingle() - 0.5f);
                init = new VBuffer<float>(initWeights.Length, initWeights);
            }
            else if (SgdInitializationTolerance > 0)
                init = InitializeWeightsSgd(ch, cursorFactory);
            else
                init = VBufferUtils.CreateEmpty<float>(BiasCount + WeightCount);

            return opt;
        }

        /// <summary>
        /// Initialize weights by running SGD up to specified tolerance.
        /// </summary>
        private protected virtual VBuffer<float> InitializeWeightsSgd(IChannel ch, FloatLabelCursor.Factory cursorFactory)
        {
            if (!Quiet)
                ch.Info("Running SGD initialization with tolerance {0}", SgdInitializationTolerance);

            int numExamples = 0;
            var oldWeights = VBufferUtils.CreateEmpty<float>(BiasCount + WeightCount);
            DTerminate terminateSgd =
                (in VBuffer<float> x) =>
                {
                    if (++numExamples % 1000 != 0)
                        return false;
                    VectorUtils.AddMult(in x, -1, ref oldWeights);
                    float normDiff = VectorUtils.Norm(oldWeights);
                    x.CopyTo(ref oldWeights);
                    // #if OLD_TRACING // REVIEW: How should this be ported?
                    if (!Quiet)
                    {
                        Console.Write(".");
                        if (numExamples % 50000 == 0)
                            Console.WriteLine("\t{0}\t{1}", numExamples, normDiff);
                    }
                    // #endif
                    return normDiff < SgdInitializationTolerance;
                };

            VBuffer<float> result = default(VBuffer<float>);
            FloatLabelCursor cursor = null;
            try
            {
                float[] scratch = null;

                SgdOptimizer.DStochasticGradient lossSgd =
                    (in VBuffer<float> x, ref VBuffer<float> grad) =>
                    {
                        // Zero out the gradient by sparsifying.
                        VBufferUtils.Resize(ref grad, grad.Length, 0);
                        EnsureBiases(ref grad);

                        if (cursor == null || !cursor.MoveNext())
                        {
                            if (cursor != null)
                                cursor.Dispose();
                            cursor = cursorFactory.Create();
                            if (!cursor.MoveNext())
                                return;
                        }
                        AccumulateOneGradient(in cursor.Features, cursor.Label, cursor.Weight, in x, ref grad, ref scratch);
                    };

                VBuffer<float> sgdWeights;
                if (DenseOptimizer)
                    sgdWeights = VBufferUtils.CreateDense<float>(BiasCount + WeightCount);
                else
                    sgdWeights = VBufferUtils.CreateEmpty<float>(BiasCount + WeightCount);
                SgdOptimizer sgdo = new SgdOptimizer(terminateSgd);
                sgdo.Minimize(lossSgd, ref sgdWeights, ref result);
                // #if OLD_TRACING // REVIEW: How should this be ported?
                if (!Quiet)
                    Console.WriteLine();
                // #endif
                ch.Info("SGD initialization done in {0} rounds", numExamples);
            }
            finally
            {
                if (cursor != null)
                    cursor.Dispose();
            }

            return result;
        }

        private protected abstract VBuffer<float> InitializeWeightsFromPredictor(IPredictor srcPredictor);

        private protected abstract void CheckLabel(RoleMappedData data);

        private protected virtual void PreTrainingProcessInstance(float label, in VBuffer<float> feat, float weight)
        {
        }

        private protected abstract TModel CreatePredictor();

        /// <summary>
        /// The basic training calls the optimizer
        /// </summary>
        private protected override TModel TrainModelCore(TrainContext context)
        {
            Contracts.CheckValue(context, nameof(context));
            Host.CheckParam(context.InitialPredictor == null || context.InitialPredictor is IPredictor, nameof(context.InitialPredictor));

            if (context.InitialPredictor != null)
                _srcPredictor = context.InitialPredictor;

            var data = context.TrainingSet;
            data.CheckFeatureFloatVector(out NumFeatures);
            CheckLabel(data);
            data.CheckOptFloatWeight();

            if (NumFeatures >= Utils.ArrayMaxSize / ClassCount)
            {
                throw Contracts.ExceptParam(nameof(data),
                    "The number of model parameters which is equal to ('# of features' + 1) * '# of classes' should be less than or equal to {0}.", Utils.ArrayMaxSize);
            }

            using (var ch = Host.Start("Training"))
            {
                TrainCore(ch, data);
                return CreatePredictor();
            }
        }

        private protected virtual void TrainCore(IChannel ch, RoleMappedData data)
        {
            Host.AssertValue(ch);
            ch.AssertValue(data);

            // Compute the number of threads to use. The ctor should have verified that this will
            // produce a positive value.
            int numThreads = !UseThreads ? 1 : (NumThreads ?? Environment.ProcessorCount);

            ch.Assert(numThreads > 0);

            NumGoodRows = 0;
            WeightSum = 0;

            _features = null;
            _labels = null;
            _weights = null;
            if (numThreads > 1)
            {
                ch.Info("LBFGS multi-threading will attempt to load dataset into memory. In case of out-of-memory " +
                        "issues, add 'numThreads=1' to the trainer arguments and 'cache=-' to the command line " +
                        "arguments to turn off multi-threading.");
                _features = new VBuffer<float>[1000];
                _labels = new float[1000];
                if (data.Schema.Weight != null)
                    _weights = new float[1000];
            }

            CursOpt cursorOpt = CursOpt.Label | CursOpt.Features;
            if (data.Schema.Weight.HasValue)
                cursorOpt |= CursOpt.Weight;

            var cursorFactory = new FloatLabelCursor.Factory(data, cursorOpt);

            long numBad;
            // REVIEW: This pass seems overly expensive for the benefit when multi-threading is off....
            using (var cursor = cursorFactory.Create())
            using (var pch = Host.StartProgressChannel("LBFGS data prep"))
            {
                // REVIEW: maybe it makes sense for the factory to capture the good row count after
                // the first successful cursoring?
                Double totalCount = data.Data.GetRowCount() ?? Double.NaN;

                long exCount = 0;
                pch.SetHeader(new ProgressHeader(null, new[] { "examples" }),
                    e => e.SetProgress(0, exCount, totalCount));
                while (cursor.MoveNext())
                {
                    WeightSum += cursor.Weight;
                    if (ShowTrainingStats)
                        ProcessPriorDistribution(cursor.Label, cursor.Weight);

                    PreTrainingProcessInstance(cursor.Label, in cursor.Features, cursor.Weight);
                    exCount++;
                    if (_features != null)
                    {
                        ch.Assert(cursor.KeptRowCount <= int.MaxValue);
                        int index = (int)cursor.KeptRowCount - 1;
                        Utils.EnsureSize(ref _features, index + 1);
                        Utils.EnsureSize(ref _labels, index + 1);
                        if (_weights != null)
                        {
                            Utils.EnsureSize(ref _weights, index + 1);
                            _weights[index] = cursor.Weight;
                        }
                        Utils.Swap(ref _features[index], ref cursor.Features);
                        _labels[index] = cursor.Label;

                        if (cursor.KeptRowCount >= int.MaxValue)
                        {
                            ch.Warning("Limiting data size for multi-threading");
                            break;
                        }
                    }
                }
                NumGoodRows = cursor.KeptRowCount;
                numBad = cursor.SkippedRowCount;
            }
            ch.Check(NumGoodRows > 0, NoTrainingInstancesMessage);
            if (numBad > 0)
                ch.Warning("Skipped {0} instances with missing features/label/weight during training", numBad);

            if (_features != null)
            {
                ch.Assert(numThreads > 1);

                // If there are so many threads that each only gets a small number (less than 10) of instances, trim
                // the number of threads so each gets a more reasonable number (100 or so). These numbers are pretty arbitrary,
                // but avoid the possibility of having no instances on some threads.
                if (numThreads > 1 && NumGoodRows / numThreads < 10)
                {
                    int numNew = Math.Max(1, (int)NumGoodRows / 100);
                    ch.Warning("Too few instances to use {0} threads, decreasing to {1} thread(s)", numThreads, numNew);
                    numThreads = numNew;
                }
                ch.Assert(numThreads > 0);

                // Divide up the instances among the threads.
                _numChunks = numThreads;
                _ranges = new int[_numChunks + 1];
                int cinstTot = (int)NumGoodRows;
                for (int ichk = 0, iinstMin = 0; ichk < numThreads; ichk++)
                {
                    int cchkLeft = numThreads - ichk; // Number of chunks left to fill.
                    ch.Assert(0 < cchkLeft && cchkLeft <= numThreads);
                    int cinstThis = (cinstTot - iinstMin + cchkLeft - 1) / cchkLeft; // Size of this chunk.
                    ch.Assert(0 < cinstThis && cinstThis <= cinstTot - iinstMin);
                    iinstMin += cinstThis;
                    _ranges[ichk + 1] = iinstMin;
                }

                _localLosses = new float[numThreads];
                _localGradients = new VBuffer<float>[numThreads - 1];
                int size = BiasCount + WeightCount;
                for (int i = 0; i < _localGradients.Length; i++)
                    _localGradients[i] = VBufferUtils.CreateEmpty<float>(size);

                ch.Assert(_numChunks > 0 && _data == null);
            }
            else
            {
                // Streaming, single-threaded case.
                _data = data;
                _cursorFactory = cursorFactory;
                ch.Assert(_numChunks == 0 && _data != null);
            }

            VBuffer<float> initWeights;
            ITerminationCriterion terminationCriterion;
            Optimizer opt = InitializeOptimizer(ch, cursorFactory, out initWeights, out terminationCriterion);
            opt.Quiet = Quiet;

            float loss;
            try
            {
                opt.Minimize(DifferentiableFunction, ref initWeights, terminationCriterion, ref CurrentWeights, out loss);
            }
            catch (Optimizer.PrematureConvergenceException e)
            {
                if (!Quiet)
                    ch.Warning("Premature convergence occurred. The OptimizationTolerance may be set too small. {0}", e.Message);
                CurrentWeights = e.State.X;
                loss = e.State.Value;
            }

            ch.Assert(CurrentWeights.Length == BiasCount + WeightCount);

            int numParams = BiasCount;
            if ((L1Weight > 0 && !Quiet) || ShowTrainingStats)
            {
                VBufferUtils.ForEachDefined(in CurrentWeights, (index, value) => { if (index >= BiasCount && value != 0) numParams++; });
                if (L1Weight > 0 && !Quiet)
                    ch.Info("L1 regularization selected {0} of {1} weights.", numParams, BiasCount + WeightCount);
            }

            if (ShowTrainingStats)
                ComputeTrainingStatistics(ch, cursorFactory, loss, numParams);
        }

        // Ensure that the bias portion of vec is represented in vec.
        // REVIEW: Is this really necessary?
        private protected void EnsureBiases(ref VBuffer<float> vec)
        {
            // REVIEW: Consider promoting this "densify first n entries" to a general purpose utility,
            // if we ever encounter other situations where this becomes useful.
            Contracts.Assert(vec.Length == BiasCount + WeightCount);
            VBufferUtils.DensifyFirst(ref vec, BiasCount);
        }

        private protected abstract float AccumulateOneGradient(in VBuffer<float> feat, float label, float weight,
            in VBuffer<float> xDense, ref VBuffer<float> grad, ref float[] scratch);

        private protected abstract void ComputeTrainingStatistics(IChannel ch, FloatLabelCursor.Factory cursorFactory, float loss, int numParams);

        private protected abstract void ProcessPriorDistribution(float label, float weight);
        /// <summary>
        /// The gradient being used by the optimizer
        /// </summary>
        private protected virtual float DifferentiableFunction(in VBuffer<float> x, ref VBuffer<float> gradient,
            IProgressChannelProvider progress)
        {
            Contracts.Assert((_numChunks == 0) != (_data == null));
            Contracts.Assert((_cursorFactory == null) == (_data == null));
            Contracts.Assert(x.Length == BiasCount + WeightCount);
            Contracts.Assert(gradient.Length == BiasCount + WeightCount);
            // REVIEW: if/when LBFGS test code is removed, the progress provider needs to become required.
            Contracts.AssertValueOrNull(progress);

            float scaleFactor = 1 / (float)WeightSum;
            VBuffer<float> xDense = default;
            if (x.IsDense)
                xDense = x;
            else
            {
                VBuffer<float> xDenseTemp = default;
                x.CopyToDense(ref xDenseTemp);
                xDense = xDenseTemp;
            }

            IProgressChannel pch = progress != null ? progress.StartProgressChannel("Gradient") : null;
            float loss;
            using (pch)
            {
                loss = _data == null
                    ? DifferentiableFunctionMultithreaded(in xDense, ref gradient, pch)
                    : DifferentiableFunctionStream(_cursorFactory, in xDense, ref gradient, pch);
            }
            float regLoss = 0;
            if (L2Weight > 0)
            {
                Contracts.Assert(xDense.IsDense);
                var values = xDense.GetValues();
                Double r = 0;
                for (int i = BiasCount; i < values.Length; i++)
                {
                    var xx = values[i];
                    r += xx * xx;
                }
                regLoss = (float)(r * L2Weight * 0.5);

                // Here we probably want to use sparse x
                VBufferUtils.ApplyWithEitherDefined(in x, ref gradient,
                    (int ind, float v1, ref float v2) => { if (ind >= BiasCount) v2 += L2Weight * v1; });
            }
            VectorUtils.ScaleBy(ref gradient, scaleFactor);

            // REVIEW: The regularization component of the loss is being scaled as well,
            // but it's unclear that it should be scaled.
            return (loss + regLoss) * scaleFactor;
        }

        /// <summary>
        /// Batch-parallel optimizer
        /// </summary>
        /// <remarks>
        /// REVIEW: consider getting rid of multithread-targeted members
        /// Using TPL, the distinction between Multithreaded and Sequential implementations is unnecessary
        /// </remarks>
        private protected virtual float DifferentiableFunctionMultithreaded(in VBuffer<float> xDense, ref VBuffer<float> gradient, IProgressChannel pch)
        {
            Contracts.Assert(_data == null);
            Contracts.Assert(_cursorFactory == null);
            Contracts.Assert(_numChunks > 0);
            Contracts.Assert(Utils.Size(_ranges) == _numChunks + 1);
            Contracts.Assert(Utils.Size(_localLosses) == _numChunks);
            Contracts.Assert(Utils.Size(_localGradients) + 1 == _numChunks);
            Contracts.AssertValueOrNull(pch);

            // Declare a local variable, since the lambda cannot capture the xDense. The gradient
            // calculation will modify the local gradients, but not this xx value.
            var xx = xDense;
            var gg = gradient;
            Parallel.For(0, _numChunks,
                ichk =>
                {
                    if (ichk == 0)
                        _localLosses[ichk] = DifferentiableFunctionComputeChunk(ichk, in xx, ref gg, pch);
                    else
                        _localLosses[ichk] = DifferentiableFunctionComputeChunk(ichk, in xx, ref _localGradients[ichk - 1], null);
                });
            gradient = gg;
            float loss = _localLosses[0];
            for (int i = 1; i < _numChunks; i++)
            {
                VectorUtils.Add(in _localGradients[i - 1], ref gradient);
                loss += _localLosses[i];
            }
            return loss;
        }

        private protected float DifferentiableFunctionComputeChunk(int ichk, in VBuffer<float> xDense, ref VBuffer<float> grad, IProgressChannel pch)
        {
            Contracts.Assert(0 <= ichk && ichk < _numChunks);
            Contracts.AssertValueOrNull(pch);

            VBufferUtils.Clear(ref grad);
            VBufferUtils.Densify(ref grad);

            float[] scratch = null;
            double loss = 0;
            int ivMin = _ranges[ichk];
            int ivLim = _ranges[ichk + 1];
            int iv = ivMin;
            if (pch != null)
                pch.SetHeader(new ProgressHeader(null, new[] { "examples" }), e => e.SetProgress(0, iv - ivMin, ivLim - ivMin));
            for (iv = ivMin; iv < ivLim; iv++)
            {
                float weight = _weights != null ? _weights[iv] : 1;
                loss += AccumulateOneGradient(in _features[iv], _labels[iv], weight, in xDense, ref grad, ref scratch);
            }
            // we need use double type to accumulate loss to avoid roundoff error
            // please see http://mathworld.wolfram.com/RoundoffError.html for roundoff error definition
            // finally we need to convert double type to float for function definition
            return (float)loss;
        }

        private protected float DifferentiableFunctionStream(FloatLabelCursor.Factory cursorFactory, in VBuffer<float> xDense, ref VBuffer<float> grad, IProgressChannel pch)
        {
            Contracts.AssertValue(cursorFactory);

            VBufferUtils.Clear(ref grad);
            VBufferUtils.Densify(ref grad);

            float[] scratch = null;
            double loss = 0;
            long count = 0;
            if (pch != null)
                pch.SetHeader(new ProgressHeader(null, new[] { "examples" }), e => e.SetProgress(0, count));
            using (var cursor = cursorFactory.Create())
            {
                while (cursor.MoveNext())
                {
                    loss += AccumulateOneGradient(in cursor.Features, cursor.Label, cursor.Weight,
                        in xDense, ref grad, ref scratch);
                    count++;
                }
            }

            // we need use double type to accumulate loss to avoid roundoff error
            // please see http://mathworld.wolfram.com/RoundoffError.html for roundoff error definition
            // finally we need to convert double type to float for function definition
            return (float)loss;
        }

        private protected VBuffer<float> InitializeWeights(IEnumerable<float> weights, IEnumerable<float> biases)
        {
            Contracts.AssertValue(biases);
            Contracts.AssertValue(weights);

            // REVIEW: Support initializing the weights of a superset of features
            var initWeights = new float[BiasCount + WeightCount];

            int i = 0;
            const string classError = "The initialization predictor has different number of classes than the training data.";
            foreach (var bias in biases)
            {
                Contracts.Check(i < BiasCount, classError);
                initWeights[i++] = bias;
            }
            Contracts.Check(i == BiasCount, classError);

            const string featError = "The initialization predictor has different number of features than the training data.";
            foreach (var weight in weights)
            {
                Contracts.Check(i < initWeights.Length, featError);
                initWeights[i++] = weight;
            }
            Contracts.Check(i == initWeights.Length, featError);

            return new VBuffer<float>(initWeights.Length, initWeights);
        }
    }
}
