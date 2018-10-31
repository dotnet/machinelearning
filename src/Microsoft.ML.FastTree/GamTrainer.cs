// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Command;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.FastTree.Internal;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Internal.CpuMath;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Training;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using Timer = Microsoft.ML.Trainers.FastTree.Internal.Timer;

[assembly: LoadableClass(typeof(GamPredictorBase.VisualizationCommand), typeof(GamPredictorBase.VisualizationCommand.Arguments), typeof(SignatureCommand),
    "GAM Vizualization Command", GamPredictorBase.VisualizationCommand.LoadName, "gamviz", DocName = "command/GamViz.md")]

[assembly: LoadableClass(typeof(void), typeof(Gam), null, typeof(SignatureEntryPointModule), "GAM")]

namespace Microsoft.ML.Trainers.FastTree
{
    using AutoResetEvent = System.Threading.AutoResetEvent;
    using SplitInfo = LeastSquaresRegressionTreeLearner.SplitInfo;

    /// <summary>
    /// Generalized Additive Model Learner.
    /// </summary>
    public abstract partial class GamTrainerBase<TArgs, TTransformer, TPredictor> : TrainerEstimatorBase<TTransformer, TPredictor>
        where TTransformer: ISingleFeaturePredictionTransformer<TPredictor>
        where TArgs : GamTrainerBase<TArgs, TTransformer, TPredictor>.ArgumentsBase, new()
        where TPredictor : IPredictorProducing<float>
    {
        public abstract class ArgumentsBase : LearnerInputBaseWithWeight
        {
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "The entropy (regularization) coefficient between 0 and 1", ShortName = "e")]
            public double EntropyCoefficient;

            /// Only consider a gain if its likelihood versus a random choice gain is above a certain value.
            /// So 0.95 would mean restricting to gains that have less than a 0.05 change of being generated randomly through choice of a random split.
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Tree fitting gain confidence requirement (should be in the range [0,1) ).", ShortName = "gainconf")]
            public int GainConfidenceLevel;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Total number of iterations over all features", ShortName = "iter", SortOrder = 1)]
            [TGUI(SuggestedSweeps = "200,1500,9500")]
            [TlcModule.SweepableDiscreteParamAttribute("NumIterations", new object[] { 200, 1500, 9500 })]
            public int NumIterations = 9500;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "The number of threads to use", ShortName = "t", NullName = "<Auto>")]
            public int? NumThreads = null;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "The learning rate", ShortName = "lr", SortOrder = 4)]
            [TGUI(SuggestedSweeps = "0.001,0.1;log")]
            [TlcModule.SweepableFloatParamAttribute("LearningRates", 0.001f, 0.1f, isLogScale: true)]
            public double LearningRates = 0.002; // Small learning rate.

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Whether to utilize the disk or the data's native transposition facilities (where applicable) when performing the transpose", ShortName = "dt")]
            public bool? DiskTranspose;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Maximum number of distinct values (bins) per feature", ShortName = "mb")]
            public int MaxBins = 255; // Save one for undefs.

            [Argument(ArgumentType.AtMostOnce, HelpText = "Upper bound on absolute value of single output", ShortName = "mo")]
            public double MaxOutput = Double.PositiveInfinity;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Sample each query 1 in k times in the GetDerivatives function", ShortName = "sr")]
            public int GetDerivativesSampleRate = 1;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "The seed of the random number generator", ShortName = "r1")]
            public int RngSeed = 123;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Minimum number of training instances required to form a partition", ShortName = "mi", SortOrder = 3)]
            [TGUI(SuggestedSweeps = "1,10,50")]
            [TlcModule.SweepableDiscreteParamAttribute("MinDocuments", new object[] { 1, 10, 50 })]
            public int MinDocuments = 10;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Whether to collectivize features during dataset preparation to speed up training", ShortName = "flocks", Hide = true)]
            public bool FeatureFlocks = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Enable post-training pruning to avoid overfitting. (a validation set is required)", ShortName = "pruning")]
            public bool EnablePruning = true;
        }

        internal const string Summary = "Trains a gradient boosted stump per feature, on all features simultaneously, " +
                                         "to fit target values using least-squares. It mantains " +
                                         "no interactions between features.";
        private const string RegisterName = "GamTraining";

        //Parameters of training
        protected readonly TArgs Args;
        private readonly double _gainConfidenceInSquaredStandardDeviations;
        private readonly double _entropyCoefficient;

        //Dataset information
        protected Dataset TrainSet;
        protected Dataset ValidSet;
        /// <summary>
        /// Whether a validation set was passed in
        /// </summary>
        protected bool HasValidSet => ValidSet != null;
        protected ScoreTracker TrainSetScore;
        protected ScoreTracker ValidSetScore;
        protected TestHistory PruningTest;
        protected int PruningLossIndex;
        protected int InputLength;
        private LeastSquaresRegressionTreeLearner.LeafSplitCandidates _leafSplitCandidates;
        private SufficientStatsBase[] _histogram;
        private ILeafSplitStatisticsCalculator _leafSplitHelper;
        private ObjectiveFunctionBase _objectiveFunction;
        private bool HasWeights => TrainSet?.SampleWeights != null;

        // Training datastructures
        private SubGraph _subGraph;

        //Results of training
        protected double MeanEffect;
        protected double[][] BinEffects;
        protected int[] FeatureMap;

        public override TrainerInfo Info { get; }
        private protected virtual bool NeedCalibration => false;

        protected IParallelTraining ParallelTraining;

        private protected GamTrainerBase(IHostEnvironment env,
            string name,
            SchemaShape.Column label,
            string featureColumn,
            string weightColumn,
            int minDocumentsInLeafs,
            double learningRate,
            Action<TArgs> advancedSettings)
            : base(Contracts.CheckRef(env, nameof(env)).Register(name), TrainerUtils.MakeR4VecFeature(featureColumn), label, TrainerUtils.MakeR4ScalarWeightColumn(weightColumn))
        {
            Args = new TArgs();

            Args.MinDocuments = minDocumentsInLeafs;
            Args.LearningRates = learningRate;

            //apply the advanced args, if the user supplied any
            advancedSettings?.Invoke(Args);

            Args.LabelColumn = label.Name;
            Args.FeatureColumn = featureColumn;

            if (weightColumn != null)
                Args.WeightColumn = weightColumn;

            Info = new TrainerInfo(normalization: false, calibration: NeedCalibration, caching: false, supportValid: true);
            _gainConfidenceInSquaredStandardDeviations = Math.Pow(ProbabilityFunctions.Probit(1 - (1 - Args.GainConfidenceLevel) * 0.5), 2);
            _entropyCoefficient = Args.EntropyCoefficient * 1e-6;

            InitializeThreads();
        }

        private protected GamTrainerBase(IHostEnvironment env, TArgs args, string name, SchemaShape.Column label)
            : base(Contracts.CheckRef(env, nameof(env)).Register(name), TrainerUtils.MakeR4VecFeature(args.FeatureColumn),
                  label, TrainerUtils.MakeR4ScalarWeightColumn(args.WeightColumn, args.WeightColumn.IsExplicit))
        {
            Contracts.CheckValue(env, nameof(env));
            Host.CheckValue(args, nameof(args));

            Host.CheckParam(args.LearningRates > 0, nameof(args.LearningRates), "Must be positive.");
            Host.CheckParam(args.NumThreads == null || args.NumThreads > 0, nameof(args.NumThreads), "Must be positive.");
            Host.CheckParam(0 <= args.EntropyCoefficient && args.EntropyCoefficient <= 1, nameof(args.EntropyCoefficient), "Must be in [0, 1].");
            Host.CheckParam(0 <= args.GainConfidenceLevel && args.GainConfidenceLevel < 1, nameof(args.GainConfidenceLevel), "Must be in [0, 1).");
            Host.CheckParam(0 < args.MaxBins, nameof(args.MaxBins), "Must be posittive.");
            Host.CheckParam(0 < args.NumIterations, nameof(args.NumIterations), "Must be positive.");
            Host.CheckParam(0 < args.MinDocuments, nameof(args.MinDocuments), "Must be positive.");

            Args = args;

            Info = new TrainerInfo(normalization: false, calibration: NeedCalibration, caching: false, supportValid: true);
            _gainConfidenceInSquaredStandardDeviations = Math.Pow(ProbabilityFunctions.Probit(1 - (1 - Args.GainConfidenceLevel) * 0.5), 2);
            _entropyCoefficient = Args.EntropyCoefficient * 1e-6;

            InitializeThreads();
        }

        protected void TrainBase(TrainContext context)
        {
            using (var ch = Host.Start("Training"))
            {
                ch.CheckValue(context, nameof(context));

                // Create the datasets
                ConvertData(context.TrainingSet, context.ValidationSet);

                // Define scoring and testing
                DefineScoreTrackers();
                if (HasValidSet)
                    DefinePruningTest();
                InputLength = context.TrainingSet.Schema.Feature.Type.ValueCount;

                TrainCore(ch);
            }
        }

        private void DefineScoreTrackers()
        {
            TrainSetScore = new ScoreTracker("train", TrainSet, null);
            if (HasValidSet)
                ValidSetScore = new ScoreTracker("valid", ValidSet, null);
        }

        protected abstract void DefinePruningTest();

        internal abstract void CheckLabel(RoleMappedData data);

        private void ConvertData(RoleMappedData trainData, RoleMappedData validationData)
        {
            trainData.CheckFeatureFloatVector();
            trainData.CheckOptFloatWeight();
            CheckLabel(trainData);

            var useTranspose = UseTranspose(Args.DiskTranspose, trainData);
            var instanceConverter = new ExamplesToFastTreeBins(Host, Args.MaxBins, useTranspose, !Args.FeatureFlocks, Args.MinDocuments, float.PositiveInfinity);

            ParallelTraining.InitEnvironment();
            TrainSet = instanceConverter.FindBinsAndReturnDataset(trainData, PredictionKind, ParallelTraining, null, false);
            FeatureMap = instanceConverter.FeatureMap;
            if (validationData != null)
                ValidSet = instanceConverter.GetCompatibleDataset(validationData, PredictionKind, null, false);
            Host.Assert(FeatureMap == null || FeatureMap.Length == TrainSet.NumFeatures);
        }

        private bool UseTranspose(bool? useTranspose, RoleMappedData data)
        {
            Host.AssertValue(data);
            Host.AssertValue(data.Schema.Feature);

            if (useTranspose.HasValue)
                return useTranspose.Value;

            ITransposeDataView td = data.Data as ITransposeDataView;
            return td != null && td.TransposeSchema.GetSlotType(data.Schema.Feature.Index) != null;
        }

        private void TrainCore(IChannel ch)
        {
            Contracts.CheckValue(ch, nameof(ch));
            // REVIEW:Get rid of this lock then we completly remove all static classes from Gam such as BlockingThreadPool.
            lock (FastTreeShared.TrainLock)
            {
                using (Timer.Time(TimerEvent.TotalInitialization))
                    Initialize(ch);
                using (Timer.Time(TimerEvent.TotalTrain))
                    TrainMainEffectsModel(ch);
            }
        }

        /// <summary>
        /// Training algorithm for the single-feature functions f(x)
        /// </summary>
        /// <param name="ch">The channel to write to</param>
        private void TrainMainEffectsModel(IChannel ch)
        {
            Contracts.AssertValue(ch);
            int iterations = Args.NumIterations;

            ch.Info("Starting to train ...");

            using (var pch = Host.StartProgressChannel("GAM training"))
            {
                _objectiveFunction = CreateObjectiveFunction();
                var sumWeights = HasWeights ? TrainSet.SampleWeights.Sum() : 0;

                int iteration = 0;
                pch.SetHeader(new ProgressHeader("iterations"), e => e.SetProgress(0, iteration, iterations));
                for (int i = iteration; iteration < iterations; iteration++)
                {
                    using (Timer.Time(TimerEvent.Iteration))
                    {
                        var gradient = _objectiveFunction.GetGradient(ch, TrainSetScore.Scores);
                        var sumTargets = gradient.Sum();

                        SumUpsAcrossFlocks(gradient, sumTargets, sumWeights);
                        TrainOnEachFeature(gradient, TrainSetScore.Scores, sumTargets, sumWeights, iteration);
                        UpdateScores(iteration);
                    }
                }
            }

            CombineGraphs(ch);
        }

        private void SumUpsAcrossFlocks(double[] gradient, double sumTargets, double sumWeights)
        {
            var sumupTask = ThreadTaskManager.MakeTask(
               (flockIndex) =>
               {
                   _histogram[flockIndex].Sumup(
                             TrainSet.FlockToFirstFeature(flockIndex),
                             null,
                             TrainSet.NumDocs,
                             sumTargets,
                             sumWeights,
                             gradient,
                             TrainSet.SampleWeights,
                             null);
               }, TrainSet.NumFlocks);

            sumupTask.RunTask();
        }

        private void TrainOnEachFeature(double[] gradient, double[] scores, double sumTargets, double sumWeights, int iteration)
        {
            var trainTask = ThreadTaskManager.MakeTask(
                (feature) =>
                {
                    TrainingIteration(feature, gradient, scores, sumTargets, sumWeights, iteration);
                }, TrainSet.NumFeatures);
            trainTask.RunTask();
        }
        private void TrainingIteration(int globalFeatureIndex, double[] gradient, double[] scores,
            double sumTargets, double sumWeights, int iteration)
        {
            int flockIndex;
            int subFeatureIndex;
            TrainSet.MapFeatureToFlockAndSubFeature(globalFeatureIndex, out flockIndex, out subFeatureIndex);

            // Compute the split for the feature
            _histogram[flockIndex].FindBestSplitForFeature(_leafSplitHelper, _leafSplitCandidates,
                _leafSplitCandidates.Targets.Length, sumTargets, sumWeights,
                globalFeatureIndex, flockIndex, subFeatureIndex, Args.MinDocuments, HasWeights,
                _gainConfidenceInSquaredStandardDeviations, _entropyCoefficient,
                TrainSet.Flocks[flockIndex].Trust(subFeatureIndex), 0);

            // Adjust the model
            if (_leafSplitCandidates.FeatureSplitInfo[globalFeatureIndex].Gain > 0)
                ConvertTreeToGraph(globalFeatureIndex, iteration);
        }

        /// <summary>
        /// Update scores for all tracked datasets
        /// </summary>
        private void UpdateScores(int iteration)
        {
            // Pass scores by reference to be updated and manually trigger the update callbacks
            UpdateScoresForSet(TrainSet, TrainSetScore.Scores, iteration);
            TrainSetScore.SendScoresUpdatedMessage();

            if (HasValidSet)
            {
                UpdateScoresForSet(ValidSet, ValidSetScore.Scores, iteration);
                ValidSetScore.SendScoresUpdatedMessage();
            }
        }

        /// <summary>
        /// Updates the scores for a dataset.
        /// </summary>
        /// <param name="dataset">The dataset to use.</param>
        /// <param name="scores">The current scores for this dataset</param>
        /// <param name="iteration">The iteration of the algorithm.
        /// Used to look up the sub-graph to use to update the score.</param>
        /// <returns></returns>
        private void UpdateScoresForSet(Dataset dataset, double[] scores, int iteration)
        {
            DefineDocumentThreadBlocks(dataset.NumDocs, BlockingThreadPool.NumThreads, out int[] threadBlocks);

            var updateTask = ThreadTaskManager.MakeTask(
                (threadIndex) =>
                {
                    int startIndexInclusive = threadBlocks[threadIndex];
                    int endIndexExclusive = threadBlocks[threadIndex + 1];
                    for (int featureIndex = 0; featureIndex < _subGraph.Splits.Length; featureIndex++)
                    {
                        var featureIndexer = dataset.GetIndexer(featureIndex);
                        for (int doc = startIndexInclusive; doc < endIndexExclusive; doc++)
                        {
                            if (featureIndexer[doc] <= _subGraph.Splits[featureIndex][iteration].SplitPoint)
                                scores[doc] += _subGraph.Splits[featureIndex][iteration].LteValue;
                            else
                                scores[doc] += _subGraph.Splits[featureIndex][iteration].GtValue;
                        }
                    }
                }, BlockingThreadPool.NumThreads);
            updateTask.RunTask();
        }

        /// <summary>
        /// Combine the single-feature single-tree graphs to a single-feature model
        /// </summary>
        private void CombineGraphs(IChannel ch)
        {
            // Prune backwards to the best iteration
            int bestIteration = Args.NumIterations;
            if (Args.EnablePruning && PruningTest != null)
            {
                ch.Info("Pruning");
                var finalResult = PruningTest.ComputeTests().ToArray()[PruningLossIndex];
                string lossFunctionName = finalResult.LossFunctionName;
                double bestLoss = finalResult.FinalValue;
                if (PruningTest != null)
                {
                    bestIteration = PruningTest.BestIteration;
                    bestLoss = PruningTest.BestResult.FinalValue;
                }
                if (bestIteration != Args.NumIterations)
                    ch.Info($"Best Iteration ({lossFunctionName}): {bestIteration} @ {bestLoss:G6} (vs {Args.NumIterations} @ {finalResult.FinalValue:G6}).");
                else
                    ch.Info("No pruning necessary. More iterations may be necessary.");
            }

            // Combine the graphs to compute the per-feature (binned) Effects
            BinEffects = new double[TrainSet.NumFeatures][];
            for (int featureIndex = 0; featureIndex < TrainSet.NumFeatures; featureIndex++)
            {
                TrainSet.MapFeatureToFlockAndSubFeature(featureIndex, out int flockIndex, out int subFeatureIndex);
                int numOfBins = TrainSet.Flocks[flockIndex].BinCount(subFeatureIndex);
                BinEffects[featureIndex] = new double[numOfBins];

                for (int iteration = 0; iteration < bestIteration; iteration++)
                {
                    var splitPoint = _subGraph.Splits[featureIndex][iteration].SplitPoint;
                    for (int bin = 0; bin <= splitPoint; bin++)
                        BinEffects[featureIndex][bin] += _subGraph.Splits[featureIndex][iteration].LteValue;
                    for (int bin = (int)splitPoint + 1; bin < numOfBins; bin++)
                        BinEffects[featureIndex][bin] += _subGraph.Splits[featureIndex][iteration].GtValue;
                }
            }

            // Center the graph around zero
            CenterGraph();
        }

        /// <summary>
        /// Distribute the documents into blocks to be computed on each thread
        /// </summary>
        /// <param name="numDocs">The number of documents in the dataset</param>
        /// <param name="blocks">An array containing the starting point for each thread;
        /// the next position is the exclusive ending point for the thread.</param>
        /// <param name="numThreads">The number of threads used.</param>
        private void DefineDocumentThreadBlocks(int numDocs, int numThreads, out int[] blocks)
        {
            int extras = numDocs % numThreads;
            int documentsPerThread = numDocs / numThreads;
            blocks = new int[numThreads + 1];
            blocks[0] = 0;
            for (int t = 0; t < extras; t++)
                blocks[t + 1] = blocks[t] + documentsPerThread + 1;
            for (int t = extras; t < numThreads; t++)
                blocks[t + 1] = blocks[t] + documentsPerThread;
        }

        /// <summary>
        /// Center the graph using the mean response per feature on the training set.
        /// </summary>
        private void CenterGraph()
        {
            // Define this once
            DefineDocumentThreadBlocks(TrainSet.NumDocs, BlockingThreadPool.NumThreads, out int[] trainThreadBlocks);

            // Compute the mean of each Effect
            var meanEffects = new double[BinEffects.Length];
            var updateTask = ThreadTaskManager.MakeTask(
                (threadIndex) =>
                {
                    int startIndexInclusive = trainThreadBlocks[threadIndex];
                    int endIndexExclusive = trainThreadBlocks[threadIndex + 1];
                    for (int featureIndex = 0; featureIndex < BinEffects.Length; featureIndex++)
                    {
                        var featureIndexer = TrainSet.GetIndexer(featureIndex);
                        for (int doc = startIndexInclusive; doc < endIndexExclusive; doc++)
                        {
                            var bin = featureIndexer[doc];
                            double totalEffect;
                            double newTotalEffect;
                            do
                            {
                                totalEffect = meanEffects[featureIndex];
                                newTotalEffect = totalEffect + BinEffects[featureIndex][bin];

                            } while (totalEffect !=
                                     Interlocked.CompareExchange(ref meanEffects[featureIndex], newTotalEffect, totalEffect));
                            // Update the shared effect, being careful of threading
                        }
                    }
                }, BlockingThreadPool.NumThreads);
            updateTask.RunTask();

            // Compute the intercept and center each graph
            MeanEffect = 0.0;
            for (int featureIndex = 0; featureIndex < BinEffects.Length; featureIndex++)
            {
                // Compute the mean effect
                meanEffects[featureIndex] /= TrainSet.NumDocs;

                // Shift the mean from the bins into the intercept
                MeanEffect += meanEffects[featureIndex];
                for (int bin=0; bin < BinEffects[featureIndex].Length; ++bin)
                    BinEffects[featureIndex][bin] -= meanEffects[featureIndex];
            }
        }

        private void ConvertTreeToGraph(int globalFeatureIndex, int iteration)
        {
            SplitInfo splitinfo = _leafSplitCandidates.FeatureSplitInfo[globalFeatureIndex];
            _subGraph.Splits[globalFeatureIndex][iteration].SplitPoint = splitinfo.Threshold;
            _subGraph.Splits[globalFeatureIndex][iteration].LteValue = Args.LearningRates * splitinfo.LteOutput;
            _subGraph.Splits[globalFeatureIndex][iteration].GtValue = Args.LearningRates * splitinfo.GTOutput;
        }

        private void InitializeGamHistograms()
        {
            _histogram = new SufficientStatsBase[TrainSet.Flocks.Length];
            for (int i = 0; i < TrainSet.Flocks.Length; i++)
                _histogram[i] = TrainSet.Flocks[i].CreateSufficientStats(HasWeights);
        }

        private void Initialize(IChannel ch)
        {
            using (Timer.Time(TimerEvent.InitializeTraining))
            {
                InitializeGamHistograms();
                _subGraph = new SubGraph(TrainSet.NumFeatures, Args.NumIterations);
                _leafSplitCandidates = new LeastSquaresRegressionTreeLearner.LeafSplitCandidates(TrainSet);
                _leafSplitHelper = new LeafSplitHelper(HasWeights);
            }
        }

        private void InitializeThreads()
        {
            ParallelTraining = new SingleTrainer();

            int numThreads = Args.NumThreads ?? Environment.ProcessorCount;
            if (Host.ConcurrencyFactor > 0 && numThreads > Host.ConcurrencyFactor)
                using (var ch = Host.Start("GamTrainer"))
                {
                    numThreads = Host.ConcurrencyFactor;
                    ch.Warning("The number of threads specified in trainer arguments is larger than the concurrency factor "
                        + "setting of the environment. Using {0} training threads instead.", numThreads);
                }

            ThreadTaskManager.Initialize(numThreads);
        }

        protected abstract ObjectiveFunctionBase CreateObjectiveFunction();

        private class LeafSplitHelper : ILeafSplitStatisticsCalculator
        {
            private bool _hasWeights;

            public LeafSplitHelper(bool hasWeights)
            {
                _hasWeights = hasWeights;
            }

            /// <summary>
            /// Returns the split gain for a particular leaf. Used on two leaves to calculate
            /// the squared error gain for a particular leaf.
            /// </summary>
            /// <param name="count">Number of documents in this leaf</param>
            /// <param name="sumTargets">Sum of the target values for this leaf</param>
            /// <param name="sumWeights">Sum of the weights for this leaf, not meaningful if
            /// <see cref="HasWeights"/> is <c>false</c></param>
            /// <returns>The gain in least squared error</returns>
            public double GetLeafSplitGain(int count, double sumTargets, double sumWeights)
            {
                if (!_hasWeights)
                    return (sumTargets * sumTargets) / count;
                return -4.0 * (Math.Abs(sumTargets) + sumWeights);
            }

            /// <summary>
            /// Calculates the output value for a leaf after splitting.
            /// </summary>
            /// <param name="count">Number of documents in this leaf</param>
            /// <param name="sumTargets">Sum of the target values for this leaf</param>
            /// <param name="sumWeights">Sum of the weights for this leaf, not meaningful if
            /// <see cref="HasWeights"/> is <c>false</c></param>
            /// <returns>The output value for a leaf</returns>
            public double CalculateSplittedLeafOutput(int count, double sumTargets, double sumWeights)
            {
                if (!_hasWeights)
                    return sumTargets / count;
                Contracts.Assert(sumWeights != 0);
                return sumTargets / sumWeights;
            }
        }

        private struct SubGraph
        {

            public Stump[][] Splits;

            public SubGraph(int numFeatures, int numIterations)
            {
                Splits = new Stump[numFeatures][];
                for (int i =0; i < numFeatures; ++i)
                {
                    Splits[i] = new Stump[numIterations];
                    for (int j = 0; j < numIterations; j++)
                        Splits[i][j] = new Stump(0, 0, 0);
                }
            }

            public struct Stump
            {
                public uint SplitPoint;
                public double LteValue;
                public double GtValue;

                public Stump(uint splitPoint, double lteValue, double gtValue)
                {
                    SplitPoint = splitPoint;
                    LteValue = lteValue;
                    GtValue = gtValue;
                }
            }
        }
    }

    public abstract class GamPredictorBase : PredictorBase<float>,
        IValueMapper, ICanSaveModel, ICanSaveInTextFormat, ICanSaveSummary
    {
        private readonly double[][] _binUpperBounds;
        private readonly double[][] _binEffects;
        private readonly double _intercept;
        private readonly int _numFeatures;
        private readonly ColumnType _inputType;
        // These would be the bins for a totally sparse input.
        private readonly int[] _binsAtAllZero;
        // The output value for all zeros
        private readonly double _valueAtAllZero;

        private readonly int[] _featureMap;
        private readonly int _inputLength;
        private readonly Dictionary<int, int> _inputFeatureToDatasetFeatureMap;

        public ColumnType InputType => _inputType;

        public ColumnType OutputType => NumberType.Float;

        private protected GamPredictorBase(IHostEnvironment env, string name,
            int inputLength, Dataset trainSet, double meanEffect, double[][] binEffects, int[] featureMap)
            : base(env, name)
        {
            Host.CheckValue(trainSet, nameof(trainSet));
            Host.CheckParam(trainSet.NumFeatures <= inputLength, nameof(inputLength), "Must be at least as large as dataset number of features");
            Host.CheckParam(featureMap == null || featureMap.Length == trainSet.NumFeatures, nameof(featureMap), "Not of right size");
            Host.CheckValue(binEffects, nameof(binEffects));
            Host.CheckParam(binEffects.Length == trainSet.NumFeatures, nameof(binEffects), "Not of right size");

            _inputLength = inputLength;

            _numFeatures = binEffects.Length;
            _inputType = new VectorType(NumberType.Float, _inputLength);
            _featureMap = featureMap;

            _intercept = meanEffect;

            //No features were filtered.
            if (_featureMap == null)
                _featureMap = Utils.GetIdentityPermutation(trainSet.NumFeatures);

            _inputFeatureToDatasetFeatureMap = new Dictionary<int, int>(_featureMap.Length);
            for (int i = 0; i < _featureMap.Length; i++)
            {
                Host.CheckParam(0 <= _featureMap[i] && _featureMap[i] < inputLength, nameof(_featureMap), "Contains out of range feature vaule");
                Host.CheckParam(!_inputFeatureToDatasetFeatureMap.ContainsValue(_featureMap[i]), nameof(_featureMap), "Contains duplicate mappings");
                _inputFeatureToDatasetFeatureMap[_featureMap[i]] = i;
            }

            //keep only bin effect and upperbounds where the effect changes.
            int flockIndex;
            int subFeatureIndex;
            _binUpperBounds = new double[_numFeatures][];
            _binEffects = new double[_numFeatures][];
            var newBinEffects = new List<double>();
            var newBinBoundaries = new List<double>();
            _binsAtAllZero = new int[_numFeatures];

            for (int i = 0; i < _numFeatures; i++)
            {
                trainSet.MapFeatureToFlockAndSubFeature(i, out flockIndex, out subFeatureIndex);
                double[] binUpperBound = trainSet.Flocks[flockIndex].BinUpperBounds(subFeatureIndex);
                double[] binEffect = binEffects[i];
                Host.CheckValue(binEffect, nameof(binEffects), "Array contained null entries");
                Host.CheckParam(binUpperBound.Length == binEffect.Length, nameof(binEffects), "Array contained wrong number of effects");
                double value = binEffect[0];
                for (int j = 0; j < binEffect.Length; j++)
                {
                    double element = binEffect[j];
                    if (element != value)
                    {
                        newBinEffects.Add(value);
                        newBinBoundaries.Add(binUpperBound[j - 1]);
                        value = element;
                    }
                }

                newBinBoundaries.Add(binUpperBound[binEffect.Length - 1]);
                newBinEffects.Add(binEffect[binEffect.Length - 1]);
                _binUpperBounds[i] = newBinBoundaries.ToArray();

                // Center the effect around 0, and move the mean into the intercept
                _binEffects[i] = newBinEffects.ToArray();
                _valueAtAllZero += _binEffects[i][0];
                newBinEffects.Clear();
                newBinBoundaries.Clear();
            }
        }

        protected GamPredictorBase(IHostEnvironment env, string name, ModelLoadContext ctx)
            : base(env, name)
        {
            Host.CheckValue(ctx, nameof(ctx));

            BinaryReader reader = ctx.Reader;

            _numFeatures = reader.ReadInt32();
            Host.CheckDecode(_numFeatures >= 0);
            _inputLength = reader.ReadInt32();
            Host.CheckDecode(_inputLength >= 0);
            _intercept = reader.ReadDouble();

            _binEffects = new double[_numFeatures][];
            _binUpperBounds = new double[_numFeatures][];
            _binsAtAllZero = new int[_numFeatures];
            for (int i = 0; i < _numFeatures; i++)
            {
                _binEffects[i] = reader.ReadDoubleArray();
                Host.CheckDecode(Utils.Size(_binEffects[i]) >= 1);
            }
            for (int i = 0; i < _numFeatures; i++)
            {
                _binUpperBounds[i] = reader.ReadDoubleArray(_binEffects[i].Length);
                // Ideally should verify that the sum of these matches _baseOutput,
                // but due to differences in JIT over time and other considerations,
                // it's possible that the sum may change even in the absence of
                // model corruption.
                _valueAtAllZero += GetBinEffect(i, 0, out _binsAtAllZero[i]);
            }
            int len = reader.ReadInt32();
            Host.CheckDecode(len >= 0);

            _inputFeatureToDatasetFeatureMap = new Dictionary<int, int>(len);
            _featureMap = Utils.CreateArray(_numFeatures, -1);
            for (int i = 0; i < len; i++)
            {
                int key = reader.ReadInt32();
                Host.CheckDecode(0 <= key && key < _inputLength);
                int val = reader.ReadInt32();
                Host.CheckDecode(0 <= val && val < _numFeatures);
                Host.CheckDecode(!_inputFeatureToDatasetFeatureMap.ContainsKey(key));
                Host.CheckDecode(_featureMap[val] == -1);
                _inputFeatureToDatasetFeatureMap[key] = val;
                _featureMap[val] = key;
            }

            _inputType = new VectorType(NumberType.Float, _inputLength);
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));

            ctx.Writer.Write(_numFeatures);
            Host.Assert(_numFeatures >= 0);
            ctx.Writer.Write(_inputLength);
            Host.Assert(_inputLength >= 0);
            ctx.Writer.Write(_intercept);
            for (int i = 0; i < _numFeatures; i++)
                ctx.Writer.WriteDoubleArray(_binEffects[i]);
            int diff = _binEffects.Sum(e => e.Take(e.Length - 1).Select((ef, i) => ef != e[i + 1] ? 1 : 0).Sum());
            int bound = _binEffects.Sum(e => e.Length - 1);

            for (int i = 0; i < _numFeatures; i++)
            {
                ctx.Writer.WriteDoublesNoCount(_binUpperBounds[i], _binUpperBounds[i].Length);
                Host.Assert(_binUpperBounds[i].Length == _binEffects[i].Length);
            }
            ctx.Writer.Write(_inputFeatureToDatasetFeatureMap.Count);
            foreach (KeyValuePair<int, int> kvp in _inputFeatureToDatasetFeatureMap)
            {
                ctx.Writer.Write(kvp.Key);
                ctx.Writer.Write(kvp.Value);
            }
        }

        public ValueMapper<TIn, TOut> GetMapper<TIn, TOut>()
        {
            Host.Check(typeof(TIn) == typeof(VBuffer<float>));
            Host.Check(typeof(TOut) == typeof(float));

            ValueMapper<VBuffer<float>, float> del = Map;
            return (ValueMapper<TIn, TOut>)(Delegate)del;
        }

        private void Map(ref VBuffer<float> features, ref float response)
        {
            Host.CheckParam(features.Length == _inputLength, nameof(features), "Bad length of input");

            double value = _intercept;
            if (features.IsDense)
            {
                for (int i = 0; i < features.Count; ++i)
                {
                    if (_inputFeatureToDatasetFeatureMap.TryGetValue(i, out int j))
                        value += GetBinEffect(j, features.Values[i]);
                }
            }
            else
            {
                // Add in the precomputed results for all features
                value += _valueAtAllZero;
                for (int i = 0; i < features.Count; ++i)
                {
                    if (_inputFeatureToDatasetFeatureMap.TryGetValue(features.Indices[i], out int j))
                        // Add the value and subtract the value at zero that was previously accounted for
                        value += GetBinEffect(j, features.Values[i]) - GetBinEffect(j, 0);
                }
            }

            response = (float)value;
        }

        /// <summary>
        /// Returns a vector of feature contributions for a given example.
        /// <paramref name="builder"/> is used as a buffer to accumulate the contributions across trees.
        /// If <paramref name="builder"/> is null, it will be created, otherwise it will be reused.
        /// </summary>
        internal void GetFeatureContributions(ref VBuffer<float> features, ref VBuffer<float> contribs, ref BufferBuilder<float> builder)
        {
            if (builder == null)
                builder = new BufferBuilder<float>(R4Adder.Instance);

            // The model is Intercept + Features
            builder.Reset(features.Length + 1, false);
            builder.AddFeature(0, (float)_intercept);

            if (features.IsDense)
            {
                for (int i = 0; i < features.Count; ++i)
                {
                    if (_inputFeatureToDatasetFeatureMap.TryGetValue(i, out int j))
                        builder.AddFeature(i+1, (float) GetBinEffect(j, features.Values[i]));
                }
            }
            else
            {
                int k = -1;
                int index = features.Indices[++k];
                for (int i = 0; i < _numFeatures; ++i)
                {
                    if (_inputFeatureToDatasetFeatureMap.TryGetValue(i, out int j))
                    {
                        double value;
                        if (i == index)
                        {
                            // Get the computed value
                            value = GetBinEffect(j, features.Values[index]);
                            // Increment index to the next feature
                            if (k < features.Indices.Length - 1)
                                index = features.Indices[++k];
                        }
                        else
                            // For features not defined, the impact is the impact at 0
                            value = GetBinEffect(i, 0);
                        builder.AddFeature(i + 1, (float)value);
                    }
                }
            }

            builder.GetResult(ref contribs);

            return;
        }

        internal double GetFeatureBinsAndScore(ref VBuffer<float> features, int[] bins)
        {
            Host.CheckParam(features.Length == _inputLength, nameof(features));
            Host.CheckParam(Utils.Size(bins) == _numFeatures, nameof(bins));

            double value = _intercept;
            if (features.IsDense)
            {
                for (int i = 0; i < features.Count; ++i)
                {
                    if (_inputFeatureToDatasetFeatureMap.TryGetValue(i, out int j))
                        value += GetBinEffect(j, features.Values[i], out bins[j]);
                }
            }
            else
            {
                // Add in the precomputed results for all features
                value += _valueAtAllZero;
                Array.Copy(_binsAtAllZero, bins, _numFeatures);

                // Update the results for features we have
                for (int i = 0; i < features.Count; ++i)
                {
                    if (_inputFeatureToDatasetFeatureMap.TryGetValue(features.Indices[i], out int j))
                        // Add the value and subtract the value at zero that was previously accounted for
                        value += GetBinEffect(j, features.Values[i], out bins[j]) - GetBinEffect(j, 0);
                }
            }
            return value;
        }

        private double GetBinEffect(int featureIndex, double featureValue)
        {
            Contracts.Assert(0 <= featureIndex && featureIndex < _numFeatures);
            int index = Algorithms.FindFirstGE(_binUpperBounds[featureIndex], featureValue);
            return _binEffects[featureIndex][index];
        }

        private double GetBinEffect(int featureIndex, double featureValue, out int binIndex)
        {
            Contracts.Assert(0 <= featureIndex && featureIndex < _numFeatures);
            binIndex = Algorithms.FindFirstGE(_binUpperBounds[featureIndex], featureValue);
            return _binEffects[featureIndex][binIndex];
        }

        public void SaveAsText(TextWriter writer, RoleMappedSchema schema)
        {
            Host.CheckValue(writer, nameof(writer));
            Host.CheckValueOrNull(schema);

            writer.WriteLine("\xfeffFeature index table"); // add BOM to tell excel this is UTF-8
            writer.WriteLine($"Number of features:\t{_numFeatures+1:D}");
            writer.WriteLine("Feature Index\tFeature Name");

            // REVIEW: We really need some unit tests around text exporting (for this, and other learners).
            // A useful test in this case would be a model trained with:
            // maml.exe train data=Samples\breast-cancer-withheader.txt loader=text{header+ col=Label:0 col=F1:1-4 col=F2:4 col=F3:5-*}
            //    xf =expr{col=F2 expr=x:0.0} xf=concat{col=Features:F1,F2,F3} tr=gam out=bubba2.zip
            // Write out the intercept
            writer.WriteLine("-1\tIntercept");

            var names = default(VBuffer<ReadOnlyMemory<char>>);
            MetadataUtils.GetSlotNames(schema, RoleMappedSchema.ColumnRole.Feature, _inputLength, ref names);

            for (int internalIndex = 0; internalIndex < _numFeatures; internalIndex++)
            {
                int featureIndex = _featureMap[internalIndex];
                var name = names.GetItemOrDefault(featureIndex);
                writer.WriteLine(!name.IsEmpty ? "{0}\t{1}" : "{0}\tFeature {0}", featureIndex, name);
            }

            writer.WriteLine();
            writer.WriteLine("Per feature binned effects:");
            writer.WriteLine("Feature Index\tFeature Value Bin Upper Bound\tOutput (effect on label)");
            writer.WriteLine($"{-1:D}\t{float.MaxValue:R}\t{_intercept:R}");
            for (int internalIndex = 0; internalIndex < _numFeatures; internalIndex++)
            {
                int featureIndex = _featureMap[internalIndex];

                double[] effects = _binEffects[internalIndex];
                double[] boundaries = _binUpperBounds[internalIndex];
                for (int i = 0; i < effects.Length; ++i)
                    writer.WriteLine($"{featureIndex:D}\t{boundaries[i]:R}\t{effects[i]:R}");
            }
        }

        public void SaveSummary(TextWriter writer, RoleMappedSchema schema)
        {
            SaveAsText(writer, schema);
        }

        /// <summary>
        /// The GAM model visualization command. Because the data access commands must access private members of
        /// <see cref="GamPredictorBase"/>, it is convenient to have the command itself nested within the base
        /// predictor class.
        /// </summary>
        public sealed class VisualizationCommand : DataCommand.ImplBase<VisualizationCommand.Arguments>
        {
            public const string Summary = "Loads a model trained with a GAM learner, and starts an interactive web session to visualize it.";
            public const string LoadName = "GamVisualization";

            public sealed class Arguments : DataCommand.ArgumentsBase
            {
                [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to open the GAM visualization page URL", ShortName = "o", SortOrder = 3)]
                public bool Open = true;

                internal Arguments SetServerIfNeeded(IHostEnvironment env)
                {
                    // We assume that if someone invoked this, they really did mean to start the web server.
                    if (env != null && Server == null)
                        Server = ServerChannel.CreateDefaultServerFactoryOrNull(env);
                    return this;
                }
            }

            private readonly string _inputModelPath;
            private readonly bool _open;

            public VisualizationCommand(IHostEnvironment env, Arguments args)
                : base(env, args.SetServerIfNeeded(env), LoadName)
            {
                Host.CheckValue(args, nameof(args));
                Host.CheckValue(args.Server, nameof(args.Server));
                Host.CheckNonWhiteSpace(args.InputModelFile, nameof(args.InputModelFile));

                _inputModelPath = args.InputModelFile;
                _open = args.Open;
            }

            public override void Run()
            {
                using (var ch = Host.Start("Run"))
                {
                    Run(ch);
                }
            }

            private sealed class Context
            {
                private readonly GamPredictorBase _pred;
                private readonly RoleMappedData _data;

                private readonly VBuffer<ReadOnlyMemory<char>> _featNames;
                // The scores.
                private readonly float[] _scores;
                // The labels.
                private readonly float[] _labels;
                // For every feature, and for every bin, there is a list of documents with that feature.
                private readonly List<int>[][] _binDocsList;
                // Whenever the predictor is "modified," we up this version. This value is returned for anything
                // that is subject to change, and can be used by client web code to detect whenever something
                // may have happened behind its back.
                private long _version;
                private long _saveVersion;

                // Non-null if this object was created with an evaluator *and* scores and labels is non-empty.
                private readonly RoleMappedData _dataForEvaluator;
                // Non-null in the same conditions that the above is non-null.
                private readonly IEvaluator _eval;

                //the map of categorical indices, as defined in MetadataUtils
                private readonly int[] _catsMap;

                /// <summary>
                /// These are the number of input features, as opposed to the number of features used within GAM
                /// which may be lower.
                /// </summary>
                public int NumFeatures => _pred.InputType.VectorSize;

                public Context(IChannel ch, GamPredictorBase pred, RoleMappedData data, IEvaluator eval)
                {
                    Contracts.AssertValue(ch);
                    ch.AssertValue(pred);
                    ch.AssertValue(data);
                    ch.AssertValueOrNull(eval);

                    _saveVersion = -1;
                    _pred = pred;
                    _data = data;
                    var schema = _data.Schema;
                    ch.Check(schema.Feature.Type.ValueCount == _pred._inputLength);

                    int len = schema.Feature.Type.ValueCount;
                    if (schema.Schema.HasSlotNames(schema.Feature.Index, len))
                        schema.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, schema.Feature.Index, ref _featNames);
                    else
                        _featNames = VBufferUtils.CreateEmpty<ReadOnlyMemory<char>>(len);

                    var numFeatures = _pred._binEffects.Length;
                    _binDocsList = new List<int>[numFeatures][];
                    for (int f = 0; f < numFeatures; f++)
                    {
                        var binDocList = new List<int>[_pred._binEffects[f].Length];
                        for (int e = 0; e < _pred._binEffects[f].Length; e++)
                            binDocList[e] = new List<int>();
                        _binDocsList[f] = binDocList;
                    }
                    var labels = new List<float>();
                    var scores = new List<float>();

                    int[] bins = new int[numFeatures];
                    using (var cursor = new FloatLabelCursor(_data, CursOpt.Label | CursOpt.Features))
                    {
                        int doc = 0;
                        while (cursor.MoveNext())
                        {
                            labels.Add(cursor.Label);
                            var score = _pred.GetFeatureBinsAndScore(ref cursor.Features, bins);
                            scores.Add((float)score);
                            for (int f = 0; f < numFeatures; f++)
                                _binDocsList[f][bins[f]].Add(doc);
                            ++doc;
                        }

                        _labels = labels.ToArray();
                        labels = null;
                        _scores = scores.ToArray();
                        scores = null;
                    }

                    ch.Assert(_scores.Length == _labels.Length);
                    if (_labels.Length > 0 && eval != null)
                    {
                        _eval = eval;
                        var builder = new ArrayDataViewBuilder(pred.Host);
                        builder.AddColumn(DefaultColumnNames.Label, NumberType.Float, _labels);
                        builder.AddColumn(DefaultColumnNames.Score, NumberType.Float, _scores);
                        _dataForEvaluator = new RoleMappedData(builder.GetDataView(), opt: false,
                            RoleMappedSchema.ColumnRole.Label.Bind(DefaultColumnNames.Label),
                            new RoleMappedSchema.ColumnRole(MetadataUtils.Const.ScoreValueKind.Score).Bind(DefaultColumnNames.Score));
                    }

                    _data.Schema.Schema.TryGetColumnIndex(DefaultColumnNames.Features, out int featureIndex);
                    MetadataUtils.TryGetCategoricalFeatureIndices(_data.Schema.Schema, featureIndex, out _catsMap);
                }

                public FeatureInfo GetInfoForIndex(int index) => FeatureInfo.GetInfoForIndex(this, index);
                public IEnumerable<FeatureInfo> GetInfos() => FeatureInfo.GetInfos(this);

                public long SetEffect(int feat, int bin, double effect)
                {
                    // Another version with multiple effects, perhaps?
                    int internalIndex;
                    if (!_pred._inputFeatureToDatasetFeatureMap.TryGetValue(feat, out internalIndex))
                        return -1;
                    var effects = _pred._binEffects[internalIndex];
                    if (bin < 0 || bin > effects.Length)
                        return -1;

                    lock (_pred)
                    {
                        var deltaEffect = effect - effects[bin];
                        effects[bin] = effect;
                        foreach (var docIndex in _binDocsList[internalIndex][bin])
                            _scores[docIndex] += (float)deltaEffect;
                        return checked(++_version);
                    }
                }

                public MetricsInfo GetMetrics()
                {
                    if (_eval == null)
                        return null;

                    lock (_pred)
                    {
                        var metricDict = _eval.Evaluate(_dataForEvaluator);
                        IDataView metricsView;
                        if (!metricDict.TryGetValue(MetricKinds.OverallMetrics, out metricsView))
                            return null;
                        Contracts.AssertValue(metricsView);
                        return new MetricsInfo(_version, EvaluateUtils.GetMetrics(metricsView).ToArray());
                    }
                }

                /// <summary>
                /// This will write out a file, if needed. In all cases if something is written it will return
                /// a version number, with an indication based on sign of whether anything was actually written
                /// in this call.
                /// </summary>
                /// <param name="host">The host from the command</param>
                /// <param name="ch">The channel from the command</param>
                /// <param name="outFile">The (optionally empty) output file</param>
                /// <returns>Returns <c>null</c> if the model file could not be saved because <paramref name="outFile"/>
                /// was <c>null</c> or whitespace. Otherwise, if the current version if newer than the last version saved,
                /// it will save, and return that version. (In this case, the number is non-negative.) Otherwise, if the current
                /// version was the last version saved, then it will return the bitwise not of that version number (in this case,
                /// the number is negative).</returns>
                public long? SaveIfNeeded(IHost host, IChannel ch, string outFile)
                {
                    Contracts.AssertValue(ch);
                    ch.AssertValue(host);
                    ch.AssertValueOrNull(outFile);

                    if (string.IsNullOrWhiteSpace(outFile))
                        return null;

                    lock (_pred)
                    {
                        ch.Assert(_saveVersion <= _version);
                        if (_saveVersion == _version)
                            return ~_version;

                        // Note that this data pipe is the data pipe that was defined for the gam visualization
                        // command, which may not be quite the same thing as the data pipe in the original model,
                        // in the event that the user specified different loader settings, defined new transforms,
                        // etc.
                        using (var file = host.CreateOutputFile(outFile))
                            TrainUtils.SaveModel(host, ch, file, _pred, _data);
                        return _saveVersion = _version;
                    }
                }

                public sealed class MetricsInfo
                {
                    public long Version { get; }
                    public KeyValuePair<string, double>[] Metrics { get; }

                    public MetricsInfo(long version, KeyValuePair<string, double>[] metrics)
                    {
                        Version = version;
                        Metrics = metrics;
                    }
                }

                public sealed class FeatureInfo
                {
                    public int Index { get; }
                    public string Name { get; }

                    /// <summary>
                    /// The upper bounds of each bin.
                    /// </summary>
                    public IEnumerable<double> UpperBounds { get; }

                    /// <summary>
                    /// The amount added to the model for a document falling in a given bin.
                    /// </summary>
                    public IEnumerable<double> BinEffects { get; }

                    /// <summary>
                    /// The number of documents in each bin.
                    /// </summary>
                    public IEnumerable<int> DocCounts { get; }

                    /// <summary>
                    /// The version of the GAM context that has these values.
                    /// </summary>
                    public long Version { get; }

                    /// <summary>
                    /// For features belonging to the same categorical, this value will be the same,
                    /// Set to -1 for non-categoricals.
                    /// </summary>
                    public int CategoricalFeatureIndex { get; }

                    private FeatureInfo(Context context, int index, int internalIndex, int[] catsMap)
                    {
                        Contracts.AssertValue(context);
                        Contracts.Assert(context._pred._inputFeatureToDatasetFeatureMap.ContainsKey(index)
                            && context._pred._inputFeatureToDatasetFeatureMap[index] == internalIndex);
                        Index = index;
                        var name = context._featNames.GetItemOrDefault(index).ToString();
                        Name = string.IsNullOrEmpty(name) ? $"f{index}" : name;
                        var up = context._pred._binUpperBounds[internalIndex];
                        UpperBounds = up.Take(up.Length - 1);
                        BinEffects = context._pred._binEffects[internalIndex];
                        DocCounts = context._binDocsList[internalIndex].Select(Utils.Size);
                        Version = context._version;
                        CategoricalFeatureIndex = -1;

                        if (catsMap != null && index < catsMap[catsMap.Length - 1])
                        {
                            for (int i = 0; i < catsMap.Length; i += 2)
                            {
                                if (index >= catsMap[i] && index <= catsMap[i + 1])
                                {
                                    CategoricalFeatureIndex = i;
                                    break;
                                }
                            }
                        }
                    }

                    public static FeatureInfo GetInfoForIndex(Context context, int index)
                    {
                        Contracts.AssertValue(context);
                        Contracts.Assert(0 <= index && index < context._pred.InputType.ValueCount);
                        lock (context._pred)
                        {
                            int internalIndex;
                            if (!context._pred._inputFeatureToDatasetFeatureMap.TryGetValue(index, out internalIndex))
                                return null;
                            return new FeatureInfo(context, index, internalIndex, context._catsMap);
                        }
                    }

                    public static FeatureInfo[] GetInfos(Context context)
                    {
                        lock (context._pred)
                        {
                            return Utils.BuildArray(context._pred._numFeatures,
                                i => new FeatureInfo(context, context._pred._featureMap[i], i, context._catsMap));
                        }
                    }
                }
            }

            /// <summary>
            /// Attempts to initialize required items, from the input model file. In the event that anything goes
            /// wrong, this method will throw.
            /// </summary>
            /// <param name="ch">The channel</param>
            /// <returns>A structure containing essential information about the GAM dataset that enables
            /// operations on top of that structure.</returns>
            private Context Init(IChannel ch)
            {
                IDataLoader loader;
                IPredictor rawPred;
                RoleMappedSchema schema;
                LoadModelObjects(ch, true, out rawPred, true, out schema, out loader);
                bool hadCalibrator = false;

                var calibrated = rawPred as CalibratedPredictorBase;
                while (calibrated != null)
                {
                    hadCalibrator = true;
                    rawPred = calibrated.SubPredictor;
                    calibrated = rawPred as CalibratedPredictorBase;
                }
                var pred = rawPred as GamPredictorBase;
                ch.CheckUserArg(pred != null, nameof(Args.InputModelFile), "Predictor was not a " + nameof(GamPredictorBase));
                var data = new RoleMappedData(loader, schema.GetColumnRoleNames(), opt: true);
                if (hadCalibrator && !string.IsNullOrWhiteSpace(Args.OutputModelFile))
                    ch.Warning("If you save the GAM model, only the GAM model, not the wrapping calibrator, will be saved.");

                return new Context(ch, pred, data, InitEvaluator(pred));
            }

            private IEvaluator InitEvaluator(GamPredictorBase pred)
            {
                switch (pred.PredictionKind)
                {
                    case PredictionKind.BinaryClassification:
                        return new BinaryClassifierEvaluator(Host, new BinaryClassifierEvaluator.Arguments());
                    case PredictionKind.Regression:
                        return new RegressionEvaluator(Host, new RegressionEvaluator.Arguments());
                    default:
                        return null;
                }
            }

            private void Run(IChannel ch)
            {
                // First we're going to initialize a structure with lots of information about the predictor, trainer, etc.
                var context = Init(ch);

                // REVIEW: What to do with the data? Not sure. Take a sample? We could have
                // a very compressed one, since we can just "bin" everything based on pred._binUpperBounds. Anyway
                // whatever we choose to do, ultimately it will be exposed as some delegate on the server channel.
                // Maybe binning actually isn't wise, *if* we want people to be able to set their own split points
                // (which seems plausible). In the current version of the viz you can only set bin effects, but
                // "splitting" a bin might be desirable in some cases, maybe. Or not.

                // Now we have a gam predictor,
                AutoResetEvent ev = new AutoResetEvent(false);
                using (var server = InitServer(ch))
                using (var sch = Host.StartServerChannel("predictor/gam"))
                {
                    // The number of features.
                    sch?.Register("numFeatures", () => context.NumFeatures);
                    // Info for a particular feature.
                    sch?.Register<int, Context.FeatureInfo>("info", context.GetInfoForIndex);
                    // Info for all features.
                    sch?.Register("infos", context.GetInfos);
                    // Modification of the model.
                    sch?.Register<int, int, double, long>("setEffect", context.SetEffect);
                    // Getting the metrics.
                    sch?.Register("metrics", context.GetMetrics);
                    sch?.Register("canSave", () => !string.IsNullOrEmpty(Args.OutputModelFile));
                    sch?.Register("save", () => context.SaveIfNeeded(Host, ch, Args.OutputModelFile));
                    sch?.Register("quit", () =>
                    {
                        var retVal = context.SaveIfNeeded(Host, ch, Args.OutputModelFile);
                        ev.Set();
                        return retVal;
                    });

                    // Targets and scores for data.
                    sch?.Publish();

                    if (sch != null)
                    {
                        ch.Info("GAM viz server is ready and waiting.");
                        Uri uri = server.BaseAddress;
                        // Believe it or not, this is actually the recommended procedure according to MSDN.
                        if (_open)
                            System.Diagnostics.Process.Start(uri.AbsoluteUri + "content/GamViz/");
                        ev.WaitOne();
                        ch.Info("Quit signal received. Quitter.");
                    }
                    else
                        ch.Info("No server, exiting immediately.");
                }
            }
        }
    }

    public static class Gam
    {
        [TlcModule.EntryPoint(Name = "Trainers.GeneralizedAdditiveModelRegressor", Desc = RegressionGamTrainer.Summary, UserName = RegressionGamTrainer.UserNameValue, ShortName = RegressionGamTrainer.ShortName)]
        public static CommonOutputs.RegressionOutput TrainRegression(IHostEnvironment env, RegressionGamTrainer.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainGAM");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<RegressionGamTrainer.Arguments, CommonOutputs.RegressionOutput>(host, input,
                () => new RegressionGamTrainer(host, input),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.WeightColumn));
        }

        [TlcModule.EntryPoint(Name = "Trainers.GeneralizedAdditiveModelBinaryClassifier", Desc = BinaryClassificationGamTrainer.Summary, UserName = BinaryClassificationGamTrainer.UserNameValue, ShortName = BinaryClassificationGamTrainer.ShortName)]
        public static CommonOutputs.BinaryClassificationOutput TrainBinary(IHostEnvironment env, BinaryClassificationGamTrainer.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainGAM");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<BinaryClassificationGamTrainer.Arguments, CommonOutputs.BinaryClassificationOutput>(host, input,
                () => new BinaryClassificationGamTrainer(host, input),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.WeightColumn));
        }
    }
}
