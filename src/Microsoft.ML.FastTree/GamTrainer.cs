// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.CpuMath;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.FastTree;

[assembly: LoadableClass(typeof(void), typeof(Gam), null, typeof(SignatureEntryPointModule), "GAM")]

namespace Microsoft.ML.Trainers.FastTree
{
    using SplitInfo = LeastSquaresRegressionTreeLearner.SplitInfo;

    /// <summary>
    /// Base class for GAM trainers.
    /// </summary>
    public abstract partial class GamTrainerBase<TOptions, TTransformer, TPredictor> : TrainerEstimatorBase<TTransformer, TPredictor>
        where TTransformer : ISingleFeaturePredictionTransformer<TPredictor>
        where TOptions : GamTrainerBase<TOptions, TTransformer, TPredictor>.OptionsBase, new()
        where TPredictor : class
    {
        /// <summary>
        /// Base class for GAM-based trainer options.
        /// </summary>
        public abstract class OptionsBase : TrainerInputBaseWithWeight
        {
            /// <summary>
            /// The entropy (regularization) coefficient between 0 and 1.
            /// </summary>
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "The entropy (regularization) coefficient between 0 and 1", ShortName = "e")]
            public double EntropyCoefficient;

            /// <summary>
            /// Tree fitting gain confidence requirement. Only consider a gain if its likelihood versus a random choice gain is above this value.
            /// </summary>
            /// <value>
            /// Value of 0.95 would mean restricting to gains that have less than a 0.05 chance of being generated randomly through choice of a random split.
            /// Valid range is [0,1).
            /// </value>
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Tree fitting gain confidence requirement (should be in the range [0,1) ).", ShortName = "gainconf")]
            public int GainConfidenceLevel;

            /// <summary>
            /// Total number of passes over the training data.
            /// </summary>
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Total number of iterations over all features", ShortName = "iter", SortOrder = 1)]
            [TGUI(SuggestedSweeps = "200,1500,9500")]
            [TlcModule.SweepableDiscreteParamAttribute("NumIterations", new object[] { 200, 1500, 9500 })]
            public int NumberOfIterations = GamDefaults.NumberOfIterations;

            /// <summary>
            /// The number of threads to use.
            /// </summary>
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "The number of threads to use", ShortName = "t", NullName = "<Auto>")]
            public int? NumberOfThreads = null;

            /// <summary>
            /// The learning rate.
            /// </summary>
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "The learning rate", ShortName = "lr", SortOrder = 4)]
            [TGUI(SuggestedSweeps = "0.001,0.1;log")]
            [TlcModule.SweepableFloatParamAttribute("LearningRates", 0.001f, 0.1f, isLogScale: true)]
            public double LearningRate = GamDefaults.LearningRate;

            /// <summary>
            /// Whether to utilize the disk or the data's native transposition facilities (where applicable) when performing the transpose.
            /// </summary>
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Whether to utilize the disk or the data's native transposition facilities (where applicable) when performing the transpose", ShortName = "dt")]
            public bool? DiskTranspose;

            /// <summary>
            /// The maximum number of distinct values (bins) per feature.
            /// </summary>
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Maximum number of distinct values (bins) per feature", ShortName = "mb")]
            public int MaximumBinCountPerFeature = GamDefaults.MaximumBinCountPerFeature;

            /// <summary>
            /// The upper bound on the absolute value of a single tree output.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Upper bound on absolute value of single output", ShortName = "mo")]
            public double MaximumTreeOutput = double.PositiveInfinity;

            /// <summary>
            /// Sample each query 1 in k times in the GetDerivatives function.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Sample each query 1 in k times in the GetDerivatives function", ShortName = "sr")]
            public int GetDerivativesSampleRate = 1;

            /// <summary>
            /// The seed of the random number generator.
            /// </summary>
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "The seed of the random number generator", ShortName = "r1")]
            public int Seed = 123;

            /// <summary>
            /// The minimal number of data points required to form a new tree leaf.
            /// </summary>
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Minimum number of training instances required to form a partition", ShortName = "mi", SortOrder = 3)]
            [TGUI(SuggestedSweeps = "1,10,50")]
            [TlcModule.SweepableDiscreteParamAttribute("MinDocuments", new object[] { 1, 10, 50 })]
            public int MinimumExampleCountPerLeaf = 10;

            /// <summary>
            /// Whether to collectivize features during dataset preparation to speed up training.
            /// </summary>
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Whether to collectivize features during dataset preparation to speed up training", ShortName = "flocks", Hide = true)]
            public bool FeatureFlocks = true;

            /// <summary>
            /// Enable post-training tree pruning to avoid overfitting. It requires a validation set.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Enable post-training pruning to avoid overfitting. (a validation set is required)", ShortName = "pruning")]
            public bool EnablePruning = true;
        }

        internal const string Summary = "Trains a gradient boosted stump per feature, on all features simultaneously, " +
                                         "to fit target values using least-squares. It mantains " +
                                         "no interactions between features.";
        private const string RegisterName = "GamTraining";

        //Parameters of training
        private protected readonly TOptions GamTrainerOptions;
        private readonly double _gainConfidenceInSquaredStandardDeviations;
        private readonly double _entropyCoefficient;

        //Dataset information
        private protected Dataset TrainSet;
        private protected Dataset ValidSet;
        /// <summary>
        /// Whether a validation set was passed in
        /// </summary>
        private protected bool HasValidSet => ValidSet != null;
        private protected ScoreTracker TrainSetScore;
        private protected ScoreTracker ValidSetScore;
        private protected TestHistory PruningTest;
        private protected int PruningLossIndex;
        private protected int InputLength;
        private LeastSquaresRegressionTreeLearner.LeafSplitCandidates _leafSplitCandidates;
        private SufficientStatsBase[] _histogram;
        private ILeafSplitStatisticsCalculator _leafSplitHelper;
        private ObjectiveFunctionBase _objectiveFunction;
        private bool HasWeights => TrainSet?.SampleWeights != null;

        // Training data structures
        private SubGraph _subGraph;

        //Results of training
        private protected double MeanEffect;
        private protected double[][] BinEffects;
        private protected double[][] BinUpperBounds;
        private protected int[] FeatureMap;

        public override TrainerInfo Info { get; }
        private protected virtual bool NeedCalibration => false;

        private protected IParallelTraining ParallelTraining;

        private protected GamTrainerBase(IHostEnvironment env,
            string name,
            SchemaShape.Column label,
            string featureColumnName,
            string weightCrowGroupColumnName,
            int numberOfIterations,
            double learningRate,
            int maximumBinCountPerFeature)
            : base(Contracts.CheckRef(env, nameof(env)).Register(name), TrainerUtils.MakeR4VecFeature(featureColumnName), label, TrainerUtils.MakeR4ScalarWeightColumn(weightCrowGroupColumnName))
        {
            GamTrainerOptions = new TOptions();
            GamTrainerOptions.NumberOfIterations = numberOfIterations;
            GamTrainerOptions.LearningRate = learningRate;
            GamTrainerOptions.MaximumBinCountPerFeature = maximumBinCountPerFeature;

            GamTrainerOptions.LabelColumnName = label.Name;
            GamTrainerOptions.FeatureColumnName = featureColumnName;

            if (weightCrowGroupColumnName != null)
                GamTrainerOptions.ExampleWeightColumnName = weightCrowGroupColumnName;

            Info = new TrainerInfo(normalization: false, calibration: NeedCalibration, caching: false, supportValid: true);
            _gainConfidenceInSquaredStandardDeviations = Math.Pow(ProbabilityFunctions.Probit(1 - (1 - GamTrainerOptions.GainConfidenceLevel) * 0.5), 2);
            _entropyCoefficient = GamTrainerOptions.EntropyCoefficient * 1e-6;

            InitializeThreads();
        }

        private protected GamTrainerBase(IHostEnvironment env, TOptions options, string name, SchemaShape.Column label)
            : base(Contracts.CheckRef(env, nameof(env)).Register(name), TrainerUtils.MakeR4VecFeature(options.FeatureColumnName),
                  label, TrainerUtils.MakeR4ScalarWeightColumn(options.ExampleWeightColumnName))
        {
            Contracts.CheckValue(env, nameof(env));
            Host.CheckValue(options, nameof(options));

            Host.CheckParam(options.LearningRate > 0, nameof(options.LearningRate), "Must be positive.");
            Host.CheckParam(options.NumberOfThreads == null || options.NumberOfThreads > 0, nameof(options.NumberOfThreads), "Must be positive.");
            Host.CheckParam(0 <= options.EntropyCoefficient && options.EntropyCoefficient <= 1, nameof(options.EntropyCoefficient), "Must be in [0, 1].");
            Host.CheckParam(0 <= options.GainConfidenceLevel && options.GainConfidenceLevel < 1, nameof(options.GainConfidenceLevel), "Must be in [0, 1).");
            Host.CheckParam(0 < options.MaximumBinCountPerFeature, nameof(options.MaximumBinCountPerFeature), "Must be posittive.");
            Host.CheckParam(0 < options.NumberOfIterations, nameof(options.NumberOfIterations), "Must be positive.");
            Host.CheckParam(0 < options.MinimumExampleCountPerLeaf, nameof(options.MinimumExampleCountPerLeaf), "Must be positive.");

            GamTrainerOptions = options;

            Info = new TrainerInfo(normalization: false, calibration: NeedCalibration, caching: false, supportValid: true);
            _gainConfidenceInSquaredStandardDeviations = Math.Pow(ProbabilityFunctions.Probit(1 - (1 - GamTrainerOptions.GainConfidenceLevel) * 0.5), 2);
            _entropyCoefficient = GamTrainerOptions.EntropyCoefficient * 1e-6;

            InitializeThreads();
        }

        private protected void TrainBase(TrainContext context)
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
                InputLength = context.TrainingSet.Schema.Feature.Value.Type.GetValueCount();

                TrainCore(ch);
            }
        }

        private void DefineScoreTrackers()
        {
            TrainSetScore = new ScoreTracker("train", TrainSet, null);
            if (HasValidSet)
                ValidSetScore = new ScoreTracker("valid", ValidSet, null);
        }

        private protected abstract void DefinePruningTest();

        private protected abstract void CheckLabel(RoleMappedData data);

        private void ConvertData(RoleMappedData trainData, RoleMappedData validationData)
        {
            trainData.CheckFeatureFloatVector();
            trainData.CheckOptFloatWeight();
            CheckLabel(trainData);

            var useTranspose = UseTranspose(GamTrainerOptions.DiskTranspose, trainData);
            var instanceConverter = new ExamplesToFastTreeBins(Host, GamTrainerOptions.MaximumBinCountPerFeature, useTranspose, !GamTrainerOptions.FeatureFlocks, GamTrainerOptions.MinimumExampleCountPerLeaf, float.PositiveInfinity);

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
            Host.Assert(data.Schema.Feature.HasValue);

            if (useTranspose.HasValue)
                return useTranspose.Value;
            return (data.Data as ITransposeDataView)?.GetSlotType(data.Schema.Feature.Value.Index) != null;
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
            int iterations = GamTrainerOptions.NumberOfIterations;

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
                globalFeatureIndex, flockIndex, subFeatureIndex, GamTrainerOptions.MinimumExampleCountPerLeaf, HasWeights,
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
            int bestIteration = GamTrainerOptions.NumberOfIterations;
            if (GamTrainerOptions.EnablePruning && PruningTest != null)
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
                if (bestIteration != GamTrainerOptions.NumberOfIterations)
                    ch.Info($"Best Iteration ({lossFunctionName}): {bestIteration} @ {bestLoss:G6} (vs {GamTrainerOptions.NumberOfIterations} @ {finalResult.FinalValue:G6}).");
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

            // Redefine the bins s.t. bins only mark changes in effects
            CreateEfficientBinning();
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
                for (int bin = 0; bin < BinEffects[featureIndex].Length; ++bin)
                    BinEffects[featureIndex][bin] -= meanEffects[featureIndex];
            }
        }

        /// <summary>
        /// Process bins such that only bin upper bounds and bin effects remain where
        /// the effect changes.
        /// </summary>
        private protected void CreateEfficientBinning()
        {
            BinUpperBounds = new double[TrainSet.NumFeatures][];
            var newBinEffects = new List<double>();
            var newBinBoundaries = new List<double>();

            for (int i = 0; i < TrainSet.NumFeatures; i++)
            {
                TrainSet.MapFeatureToFlockAndSubFeature(i, out int flockIndex, out int subFeatureIndex);
                double[] binUpperBound = TrainSet.Flocks[flockIndex].BinUpperBounds(subFeatureIndex);
                double value = BinEffects[i][0];
                for (int j = 0; j < BinEffects[i].Length; j++)
                {
                    double element = BinEffects[i][j];
                    if (element != value)
                    {
                        newBinEffects.Add(value);
                        newBinBoundaries.Add(binUpperBound[j - 1]);
                        value = element;
                    }
                }
                // Catch the last value
                newBinBoundaries.Add(binUpperBound[BinEffects[i].Length - 1]);
                newBinEffects.Add(BinEffects[i][BinEffects[i].Length - 1]);

                // Overwrite the old arrays with the efficient arrays
                BinUpperBounds[i] = newBinBoundaries.ToArray();
                BinEffects[i] = newBinEffects.ToArray();
                newBinEffects.Clear();
                newBinBoundaries.Clear();
            }
        }

        private void ConvertTreeToGraph(int globalFeatureIndex, int iteration)
        {
            SplitInfo splitinfo = _leafSplitCandidates.FeatureSplitInfo[globalFeatureIndex];
            _subGraph.Splits[globalFeatureIndex][iteration].SplitPoint = splitinfo.Threshold;
            _subGraph.Splits[globalFeatureIndex][iteration].LteValue = GamTrainerOptions.LearningRate * splitinfo.LteOutput;
            _subGraph.Splits[globalFeatureIndex][iteration].GtValue = GamTrainerOptions.LearningRate * splitinfo.GTOutput;
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
                _subGraph = new SubGraph(TrainSet.NumFeatures, GamTrainerOptions.NumberOfIterations);
                _leafSplitCandidates = new LeastSquaresRegressionTreeLearner.LeafSplitCandidates(TrainSet);
                _leafSplitHelper = new LeafSplitHelper(HasWeights);
            }
        }

        private void InitializeThreads()
        {
            ParallelTraining = new SingleTrainer();
            ThreadTaskManager.Initialize(GamTrainerOptions.NumberOfThreads ?? Environment.ProcessorCount);
        }

        private protected abstract ObjectiveFunctionBase CreateObjectiveFunction();

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
                for (int i = 0; i < numFeatures; ++i)
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

    internal static class Gam
    {
        [TlcModule.EntryPoint(Name = "Trainers.GeneralizedAdditiveModelRegressor", Desc = GamRegressionTrainer.Summary, UserName = GamRegressionTrainer.UserNameValue, ShortName = GamRegressionTrainer.ShortName)]
        public static CommonOutputs.RegressionOutput TrainRegression(IHostEnvironment env, GamRegressionTrainer.Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainGAM");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return TrainerEntryPointsUtils.Train<GamRegressionTrainer.Options, CommonOutputs.RegressionOutput>(host, input,
                () => new GamRegressionTrainer(host, input),
                () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumnName),
                () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.ExampleWeightColumnName));
        }

        [TlcModule.EntryPoint(Name = "Trainers.GeneralizedAdditiveModelBinaryClassifier", Desc = GamBinaryTrainer.Summary, UserName = GamBinaryTrainer.UserNameValue, ShortName = GamBinaryTrainer.ShortName)]
        public static CommonOutputs.BinaryClassificationOutput TrainBinary(IHostEnvironment env, GamBinaryTrainer.Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainGAM");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return TrainerEntryPointsUtils.Train<GamBinaryTrainer.Options, CommonOutputs.BinaryClassificationOutput>(host, input,
                () => new GamBinaryTrainer(host, input),
                () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumnName),
                () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.ExampleWeightColumnName));
        }
    }

    internal static class GamDefaults
    {
        internal const int NumberOfIterations = 9500;
        internal const int MaximumBinCountPerFeature = 255;
        internal const double LearningRate = 0.002; // A small value
    }
}
