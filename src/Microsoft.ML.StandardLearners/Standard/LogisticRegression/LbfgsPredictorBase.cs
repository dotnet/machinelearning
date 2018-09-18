// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Numeric;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML.Runtime.Internal.Internallearn;

namespace Microsoft.ML.Runtime.Learners
{
    public abstract class LbfgsTrainerBase<TOutput, TPredictor> : TrainerBase<TPredictor>
        where TPredictor : class, IPredictorProducing<TOutput>
    {
        public abstract class ArgumentsBase : LearnerInputBaseWithWeight
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "L2 regularization weight", ShortName = "l2", SortOrder = 50)]
            [TGUI(Label = "L2 Weight", Description = "Weight of L2 regularizer term", SuggestedSweeps = "0,0.1,1")]
            [TlcModule.SweepableFloatParamAttribute(0.0f, 1.0f, numSteps:4)]
            public Float L2Weight = 1;

            [Argument(ArgumentType.AtMostOnce, HelpText = "L1 regularization weight", ShortName = "l1", SortOrder = 50)]
            [TGUI(Label = "L1 Weight", Description = "Weight of L1 regularizer term", SuggestedSweeps = "0,0.1,1")]
            [TlcModule.SweepableFloatParamAttribute(0.0f, 1.0f, numSteps: 4)]
            public Float L1Weight = 1;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Tolerance parameter for optimization convergence. Lower = slower, more accurate",
                ShortName = "ot", SortOrder = 50)]
            [TGUI(Label = "Optimization Tolerance", Description = "Threshold for optimizer convergence", SuggestedSweeps = "1e-4,1e-7")]
            [TlcModule.SweepableDiscreteParamAttribute(new object[] {1e-4f, 1e-7f})]
            public Float OptTol = (Float)1e-7;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Memory size for L-BFGS. Lower=faster, less accurate",
                ShortName = "m", SortOrder = 50)]
            [TGUI(Description = "Memory size for L-BFGS", SuggestedSweeps = "5,20,50")]
            [TlcModule.SweepableDiscreteParamAttribute("MemorySize", new object[] { 5, 20, 50 })]
            public int MemorySize = 20;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum iterations.", ShortName = "maxiter")]
            [TGUI(Label = "Max Number of Iterations")]
            [TlcModule.SweepableLongParamAttribute("MaxIterations", 1, int.MaxValue)]
            public int MaxIterations = int.MaxValue;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Run SGD to initialize LR weights, converging to this tolerance",
                ShortName = "sgd")]
            public Float SgdInitializationTolerance = 0;

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
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Init weights diameter", ShortName = "initwts", SortOrder = 140)]
            [TGUI(Label = "Initial Weights Scale", SuggestedSweeps = "0,0.1,0.5,1")]
            [TlcModule.SweepableFloatParamAttribute("InitWtsDiameter", 0.0f, 1.0f, numSteps:5)]
            public Float InitWtsDiameter = 0;

            // Deprecated
            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether or not to use threads. Default is true",
                ShortName = "t", Hide = true)]
            public bool UseThreads = true;

            /// <summary>
            /// Number of threads. Null means use the number of processors.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of threads", ShortName = "nt")]
            public int? NumThreads;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Force densification of the internal optimization vectors", ShortName = "do")]
            [TlcModule.SweepableDiscreteParamAttribute("DenseOptimizer", new object[] { false, true })]
            public bool DenseOptimizer = false;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Enforce non-negative weights", ShortName = "nn", SortOrder = 90)]
            public bool EnforceNonNegativity = false;
        }

        protected int NumFeatures;
        protected VBuffer<Float> CurrentWeights;
        protected long NumGoodRows;
        protected Double WeightSum;

        private TPredictor _srcPredictor;

        protected readonly Float L2Weight;
        protected readonly Float L1Weight;
        protected readonly Float OptTol;
        protected readonly int MemorySize;
        protected readonly int MaxIterations;
        protected readonly Float SgdInitializationTolerance;
        protected readonly bool Quiet;
        protected readonly Float InitWtsDiameter;
        protected readonly bool UseThreads;
        protected readonly int? NumThreads;
        protected readonly bool DenseOptimizer;
        protected readonly long MaxNormalizationExamples;
        protected readonly bool ShowTrainingStats;
        protected readonly bool EnforceNonNegativity;

        // The training data, when NOT using multiple threads.
        private RoleMappedData _data;
        private FloatLabelCursor.Factory _cursorFactory;

        // Used for the multi-threading case, null otherwise. These three arrays are parallel. _weights may be null.
        private VBuffer<Float>[] _features;
        private Float[] _labels;
        private Float[] _weights;

        // Stores the bounds of the chunk to be used by each thread. The 0th slot is 0. The length
        // is one more than the number of threads to use.
        private int _numChunks;
        private int[] _ranges;

        // Working buffers allocated lazily.
        private VBuffer<Float>[] _localGradients;
        private Float[] _localLosses;

        // REVIEW: It's pointless to request caching when we're going to load everything into
        // memory, that is, when using multiple threads. So should caching not be requested?
        private static readonly TrainerInfo _info = new TrainerInfo(caching: true, supportIncrementalTrain: true);
        public override TrainerInfo Info => _info;

        internal LbfgsTrainerBase(ArgumentsBase args, IHostEnvironment env, string name, bool showTrainingStats = false)
            : base(env, name)
        {
            Contracts.CheckUserArg(!args.UseThreads || args.NumThreads > 0 || args.NumThreads == null,
                nameof(args.NumThreads), "numThreads must be positive (or empty for default)");

            Contracts.CheckUserArg(args.L2Weight >= 0, nameof(args.L2Weight), "Must be non-negative");
            L2Weight = args.L2Weight;
            Contracts.CheckUserArg(args.L1Weight >= 0, nameof(args.L1Weight), "Must be non-negative");
            L1Weight = args.L1Weight;
            Contracts.CheckUserArg(args.OptTol > 0, nameof(args.OptTol), "Must be positive");
            OptTol = args.OptTol;
            Contracts.CheckUserArg(args.MemorySize > 0, nameof(args.MemorySize), "Must be positive");
            MemorySize = args.MemorySize;
            Contracts.CheckUserArg(args.MaxIterations > 0, nameof(args.MaxIterations), "Must be positive");
            MaxIterations = args.MaxIterations;
            Contracts.CheckUserArg(args.SgdInitializationTolerance >= 0, nameof(args.SgdInitializationTolerance), "Must be non-negative");
            SgdInitializationTolerance = args.SgdInitializationTolerance;
            Quiet = args.Quiet;
            InitWtsDiameter = args.InitWtsDiameter;
            UseThreads = args.UseThreads;
            Contracts.CheckUserArg(args.NumThreads == null || args.NumThreads.Value >= 0, nameof(args.NumThreads), "Must be non-negative");
            NumThreads = args.NumThreads;
            DenseOptimizer = args.DenseOptimizer;
            ShowTrainingStats = showTrainingStats;
            EnforceNonNegativity = args.EnforceNonNegativity;

            if (EnforceNonNegativity && ShowTrainingStats)
            {
                ShowTrainingStats = false;
                using (var ch = Host.Start("Initialization"))
                {
                    ch.Warning("The training statistics cannot be computed with non-negativity constraint.");
                    ch.Done();
                }
            }
        }

        protected virtual int ClassCount => 1;
        protected int BiasCount => ClassCount;
        protected int WeightCount => ClassCount * NumFeatures;
        protected virtual Optimizer InitializeOptimizer(IChannel ch, FloatLabelCursor.Factory cursorFactory,
            out VBuffer<Float> init, out ITerminationCriterion terminationCriterion)
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
                Float[] initWeights = new Float[BiasCount + WeightCount];
                for (int j = 0; j < initWeights.Length; j++)
                    initWeights[j] = InitWtsDiameter * (Host.Rand.NextSingle() - (Float)0.5);
                init = new VBuffer<Float>(initWeights.Length, initWeights);
            }
            else if (SgdInitializationTolerance > 0)
                init = InitializeWeightsSgd(ch, cursorFactory);
            else
                init = VBufferUtils.CreateEmpty<Float>(BiasCount + WeightCount);

            return opt;
        }

        /// <summary>
        /// Initialize weights by running SGD up to specified tolerance.
        /// </summary>
        protected virtual VBuffer<Float> InitializeWeightsSgd(IChannel ch, FloatLabelCursor.Factory cursorFactory)
        {
            if (!Quiet)
                ch.Info("Running SGD initialization with tolerance {0}", SgdInitializationTolerance);

            int numExamples = 0;
            var oldWeights = VBufferUtils.CreateEmpty<Float>(BiasCount + WeightCount);
            DTerminate terminateSgd =
                (ref VBuffer<Float> x) =>
                {
                    if (++numExamples % 1000 != 0)
                        return false;
                    VectorUtils.AddMult(ref x, -1, ref oldWeights);
                    Float normDiff = VectorUtils.Norm(oldWeights);
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

            VBuffer<Float> result = default(VBuffer<Float>);
            FloatLabelCursor cursor = null;
            try
            {
                Float[] scratch = null;

                SgdOptimizer.DStochasticGradient lossSgd =
                    (ref VBuffer<Float> x, ref VBuffer<Float> grad) =>
                    {
                        // Zero out the gradient by sparsifying.
                        grad = new VBuffer<Float>(grad.Length, 0, grad.Values, grad.Indices);
                        EnsureBiases(ref grad);

                        if (cursor == null || !cursor.MoveNext())
                        {
                            if (cursor != null)
                                cursor.Dispose();
                            cursor = cursorFactory.Create();
                            if (!cursor.MoveNext())
                                return;
                        }
                        AccumulateOneGradient(ref cursor.Features, cursor.Label, cursor.Weight, ref x, ref grad, ref scratch);
                    };

                VBuffer<Float> sgdWeights;
                if (DenseOptimizer)
                    sgdWeights = VBufferUtils.CreateDense<Float>(BiasCount + WeightCount);
                else
                    sgdWeights = VBufferUtils.CreateEmpty<Float>(BiasCount + WeightCount);
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

        protected abstract VBuffer<Float> InitializeWeightsFromPredictor(TPredictor srcPredictor);

        protected abstract void CheckLabel(RoleMappedData data);

        protected virtual void PreTrainingProcessInstance(Float label, ref VBuffer<Float> feat, Float weight)
        {
        }

        protected abstract TPredictor CreatePredictor();

        /// <summary>
        /// The basic training calls the optimizer
        /// </summary>
        public override TPredictor Train(TrainContext context)
        {
            Contracts.CheckValue(context, nameof(context));

            var data = context.TrainingSet;
            _srcPredictor = context.TrainingSet as TPredictor;
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
                var pred = CreatePredictor();
                ch.Done();
                return pred;
            }
        }

        private void TrainCore(IChannel ch, RoleMappedData data)
        {
            Host.AssertValue(ch);
            ch.AssertValue(data);

            // Compute the number of threads to use. The ctor should have verified that this will
            // produce a positive value.
            int numThreads = !UseThreads ? 1 : (NumThreads ?? Environment.ProcessorCount);
            if (Host.ConcurrencyFactor > 0 && numThreads > Host.ConcurrencyFactor)
            {
                numThreads = Host.ConcurrencyFactor;
                ch.Warning("The number of threads specified in trainer arguments is larger than the concurrency factor "
                        + "setting of the environment. Using {0} training threads instead.", numThreads);
            }

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
                _features = new VBuffer<Float>[1000];
                _labels = new Float[1000];
                if (data.Schema.Weight != null)
                    _weights = new Float[1000];
            }

            var cursorFactory = new FloatLabelCursor.Factory(data, CursOpt.Features | CursOpt.Label | CursOpt.Weight);

            long numBad;
            // REVIEW: This pass seems overly expensive for the benefit when multi-threading is off....
            using (var cursor = cursorFactory.Create())
            using (var pch = Host.StartProgressChannel("LBFGS data prep"))
            {
                // REVIEW: maybe it makes sense for the factory to capture the good row count after
                // the first successful cursoring?
                Double totalCount = data.Data.GetRowCount(true) ?? Double.NaN;

                long exCount = 0;
                pch.SetHeader(new ProgressHeader(null, new[] { "examples" }),
                    e => e.SetProgress(0, exCount, totalCount));
                while (cursor.MoveNext())
                {
                    WeightSum += cursor.Weight;
                    if (ShowTrainingStats)
                        ProcessPriorDistribution(cursor.Label, cursor.Weight);

                    PreTrainingProcessInstance(cursor.Label, ref cursor.Features, cursor.Weight);
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

                _localLosses = new Float[numThreads];
                _localGradients = new VBuffer<Float>[numThreads - 1];
                int size = BiasCount + WeightCount;
                for (int i = 0; i < _localGradients.Length; i++)
                    _localGradients[i] = VBufferUtils.CreateEmpty<Float>(size);

                ch.Assert(_numChunks > 0 && _data == null);
            }
            else
            {
                // Streaming, single-threaded case.
                _data = data;
                _cursorFactory = cursorFactory;
                ch.Assert(_numChunks == 0 && _data != null);
            }

            VBuffer<Float> initWeights;
            ITerminationCriterion terminationCriterion;
            Optimizer opt = InitializeOptimizer(ch, cursorFactory, out initWeights, out terminationCriterion);
            opt.Quiet = Quiet;

            Float loss;
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
                VBufferUtils.ForEachDefined(ref CurrentWeights, (index, value) => { if (index >= BiasCount && value != 0) numParams++; });
                if (L1Weight > 0 && !Quiet)
                    ch.Info("L1 regularization selected {0} of {1} weights.", numParams, BiasCount + WeightCount);
            }

            if (ShowTrainingStats)
                ComputeTrainingStatistics(ch, cursorFactory, loss, numParams);
        }

        // Ensure that the bias portion of vec is represented in vec.
        // REVIEW: Is this really necessary?
        protected void EnsureBiases(ref VBuffer<Float> vec)
        {
            // REVIEW: Consider promoting this "densify first n entries" to a general purpose utility,
            // if we ever encounter other situations where this becomes useful.
            Contracts.Assert(vec.Length == BiasCount + WeightCount);
            VBufferUtils.DensifyFirst(ref vec, BiasCount);
        }

        protected abstract Float AccumulateOneGradient(ref VBuffer<Float> feat, Float label, Float weight,
            ref VBuffer<Float> xDense, ref VBuffer<Float> grad, ref Float[] scratch);

        protected abstract void ComputeTrainingStatistics(IChannel ch, FloatLabelCursor.Factory cursorFactory, Float loss, int numParams);

        protected abstract void ProcessPriorDistribution(Float label, Float weight);
        /// <summary>
        /// The gradient being used by the optimizer
        /// </summary>
        protected virtual Float DifferentiableFunction(ref VBuffer<Float> x, ref VBuffer<Float> gradient,
            IProgressChannelProvider progress)
        {
            Contracts.Assert((_numChunks == 0) != (_data == null));
            Contracts.Assert((_cursorFactory == null) == (_data == null));
            Contracts.Assert(x.Length == BiasCount + WeightCount);
            Contracts.Assert(gradient.Length == BiasCount + WeightCount);
            // REVIEW: if/when LBFGS test code is removed, the progress provider needs to become required.
            Contracts.AssertValueOrNull(progress);

            Float scaleFactor = 1 / (Float)WeightSum;
            VBuffer<Float> xDense = default(VBuffer<Float>);
            if (x.IsDense)
                xDense = x;
            else
                x.CopyToDense(ref xDense);

            IProgressChannel pch = progress != null ? progress.StartProgressChannel("Gradient") : null;
            Float loss;
            using (pch)
            {
                loss = _data == null
                    ? DifferentiableFunctionMultithreaded(ref xDense, ref gradient, pch)
                    : DifferentiableFunctionStream(_cursorFactory, ref xDense, ref gradient, pch);
            }
            Float regLoss = 0;
            if (L2Weight > 0)
            {
                Contracts.Assert(xDense.IsDense);
                var values = xDense.Values;
                Double r = 0;
                for (int i = BiasCount; i < values.Length; i++)
                {
                    var xx = values[i];
                    r += xx * xx;
                }
                regLoss = (Float)(r * L2Weight * 0.5);

                // Here we probably want to use sparse x
                VBufferUtils.ApplyWithEitherDefined(ref x, ref gradient,
                    (int ind, Float v1, ref Float v2) => { if (ind >= BiasCount) v2 += L2Weight * v1; });
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
        protected virtual Float DifferentiableFunctionMultithreaded(ref VBuffer<Float> xDense, ref VBuffer<Float> gradient, IProgressChannel pch)
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
                        _localLosses[ichk] = DifferentiableFunctionComputeChunk(ichk, ref xx, ref gg, pch);
                    else
                        _localLosses[ichk] = DifferentiableFunctionComputeChunk(ichk, ref xx, ref _localGradients[ichk - 1], null);
                });
            gradient = gg;
            Float loss = _localLosses[0];
            for (int i = 1; i < _numChunks; i++)
            {
                VectorUtils.Add(ref _localGradients[i - 1], ref gradient);
                loss += _localLosses[i];
            }
            return loss;
        }

        protected Float DifferentiableFunctionComputeChunk(int ichk, ref VBuffer<Float> xDense, ref VBuffer<Float> grad, IProgressChannel pch)
        {
            Contracts.Assert(0 <= ichk && ichk < _numChunks);
            Contracts.AssertValueOrNull(pch);

            VBufferUtils.Clear(ref grad);
            VBufferUtils.Densify(ref grad);

            Float[] scratch = null;
            double loss = 0;
            int ivMin = _ranges[ichk];
            int ivLim = _ranges[ichk + 1];
            int iv = ivMin;
            if (pch != null)
                pch.SetHeader(new ProgressHeader(null, new[] { "examples" }), e => e.SetProgress(0, iv - ivMin, ivLim - ivMin));
            for (iv = ivMin; iv < ivLim; iv++)
            {
                Float weight = _weights != null ? _weights[iv] : 1;
                loss += AccumulateOneGradient(ref _features[iv], _labels[iv], weight, ref xDense, ref grad, ref scratch);
            }
            // we need use double type to accumulate loss to avoid roundoff error
            // please see http://mathworld.wolfram.com/RoundoffError.html for roundoff error definition
            // finally we need to convert double type to float for function definition
            return (Float)loss;
        }

        protected Float DifferentiableFunctionStream(FloatLabelCursor.Factory cursorFactory, ref VBuffer<Float> xDense, ref VBuffer<Float> grad, IProgressChannel pch)
        {
            Contracts.AssertValue(cursorFactory);

            VBufferUtils.Clear(ref grad);
            VBufferUtils.Densify(ref grad);

            Float[] scratch = null;
            double loss = 0;
            long count = 0;
            if (pch != null)
                pch.SetHeader(new ProgressHeader(null, new[] { "examples" }), e => e.SetProgress(0, count));
            using (var cursor = cursorFactory.Create())
            {
                while (cursor.MoveNext())
                {
                    loss += AccumulateOneGradient(ref cursor.Features, cursor.Label, cursor.Weight,
                        ref xDense, ref grad, ref scratch);
                    count++;
                }
            }

            // we need use double type to accumulate loss to avoid roundoff error
            // please see http://mathworld.wolfram.com/RoundoffError.html for roundoff error definition
            // finally we need to convert double type to float for function definition
            return (Float)loss;
        }

        protected VBuffer<Float> InitializeWeights(IEnumerable<Float> weights, IEnumerable<Float> biases)
        {
            Contracts.AssertValue(biases);
            Contracts.AssertValue(weights);

            // REVIEW: Support initializing the weights of a superset of features
            var initWeights = new Float[BiasCount + WeightCount];

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

            return new VBuffer<Float>(initWeights.Length, initWeights);
        }
    }
}
