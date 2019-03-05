// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.Data.DataView;
using Microsoft.ML.Calibrators;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Data.Conversion;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Model.Pfa;
using Microsoft.ML.Transforms;
using Microsoft.ML.TreePredictor;
using Newtonsoft.Json.Linq;

// All of these reviews apply in general to fast tree and random forest implementations.
//REVIEW: Decouple train method in Application.cs to have boosting and random forest logic seperate.
//REVIEW: Do we need to keep all the fast tree based testers?

namespace Microsoft.ML.Trainers.FastTree
{
    [BestFriend]
    internal delegate void SignatureTreeEnsembleTrainer();

    /// <summary>
    /// FastTreeTrainerBase is generic class and can't have shared object among classes.
    /// This class is to provide common for all classes object which we can use for lock purpose.
    /// </summary>
    internal static class FastTreeShared
    {
        public static readonly object TrainLock = new object();
    }

    public abstract class FastTreeTrainerBase<TOptions, TTransformer, TModel> :
        TrainerEstimatorBaseWithGroupId<TTransformer, TModel>
        where TTransformer : ISingleFeaturePredictionTransformer<TModel>
        where TOptions : TreeOptions, new()
        where TModel : class
    {
        private protected readonly TOptions FastTreeTrainerOptions;
        private protected readonly bool AllowGC;
        private protected int FeatureCount;
        private protected InternalTreeEnsemble TrainedEnsemble;
        private protected RoleMappedData ValidData;
        /// <summary>
        /// If not null, it's a test data set passed in from training context. It will be converted to one element in
        /// <see cref="Tests"/> by calling <see cref="ExamplesToFastTreeBins.GetCompatibleDataset"/> in <see cref="InitializeTests"/>.
        /// </summary>
        private protected RoleMappedData TestData;
        private protected IParallelTraining ParallelTraining;
        private protected OptimizationAlgorithm OptimizationAlgorithm;
        private protected Dataset TrainSet;
        private protected Dataset ValidSet;
        /// <summary>
        /// Data sets used to evaluate the prediction scores produced the trained model during the triaining process.
        /// </summary>
        private protected Dataset[] TestSets;
        private protected int[] FeatureMap;
        /// <summary>
        /// In the training process, <see cref="TrainSet"/>, <see cref="ValidSet"/>, <see cref="TestSets"/> would be
        /// converted into <see cref="Tests"/> for efficient model evaluation.
        /// </summary>
        private protected List<Test> Tests;
        private protected TestHistory PruningTest;
        private protected int[] CategoricalFeatures;

        // Test for early stopping.
        private protected Test TrainTest;
        private protected Test ValidTest;

        private protected double[] InitTrainScores;
        private protected double[] InitValidScores;
        private protected double[][] InitTestScores;
        private protected InternalTreeEnsemble Ensemble;

        private protected bool HasValidSet => ValidSet != null;

        private const string RegisterName = "FastTreeTraining";
        // random for active features selection
        private Random _featureSelectionRandom;

        private protected string InnerOptions => CmdParser.GetSettings(Host, FastTreeTrainerOptions, new TOptions());

        public override TrainerInfo Info { get; }

        private protected virtual bool NeedCalibration => false;

        /// <summary>
        /// Constructor to use when instantiating the classes deriving from here through the API.
        /// </summary>
        private protected FastTreeTrainerBase(IHostEnvironment env,
            SchemaShape.Column label,
            string featureColumn,
            string weightColumn,
            string groupIdColumn,
            int numLeaves,
            int numTrees,
            int minDatapointsInLeaves)
            : base(Contracts.CheckRef(env, nameof(env)).Register(RegisterName), TrainerUtils.MakeR4VecFeature(featureColumn), label, TrainerUtils.MakeR4ScalarWeightColumn(weightColumn), TrainerUtils.MakeU4ScalarColumn(groupIdColumn))
        {
            FastTreeTrainerOptions = new TOptions();

            // set up the directly provided values
            // override with the directly provided values.
            FastTreeTrainerOptions.NumLeaves = numLeaves;
            FastTreeTrainerOptions.NumTrees = numTrees;
            FastTreeTrainerOptions.MinDocumentsInLeafs = minDatapointsInLeaves;

            FastTreeTrainerOptions.LabelColumnName = label.Name;
            FastTreeTrainerOptions.FeatureColumnName = featureColumn;
            FastTreeTrainerOptions.ExampleWeightColumnName = weightColumn;
            FastTreeTrainerOptions.RowGroupColumnName = groupIdColumn;

            // The discretization step renders this trainer non-parametric, and therefore it does not need normalization.
            // Also since it builds its own internal discretized columnar structures, it cannot benefit from caching.
            // Finally, even the binary classifiers, being logitboost, tend to not benefit from external calibration.
            Info = new TrainerInfo(normalization: false, caching: false, calibration: NeedCalibration, supportValid: true, supportTest: true);
            // REVIEW: CLR 4.6 has a bug that is only exposed in Scope, and if we trigger GC.Collect in scope environment
            // with memory consumption more than 5GB, GC get stuck in infinite loop.
            // Before, we could check a specific type of the environment here, but now it is internal, so we will need another
            // mechanism to detect that we are running in Scope.
            AllowGC = true;

            Initialize(env);
        }

        /// <summary>
        /// Constructor that is used when invoking the classes deriving from this, through maml.
        /// </summary>
        private protected FastTreeTrainerBase(IHostEnvironment env, TOptions options, SchemaShape.Column label)
            : base(Contracts.CheckRef(env, nameof(env)).Register(RegisterName), TrainerUtils.MakeR4VecFeature(options.FeatureColumnName), label, TrainerUtils.MakeR4ScalarWeightColumn(options.ExampleWeightColumnName),
                  TrainerUtils.MakeU4ScalarColumn(options.RowGroupColumnName))
        {
            Host.CheckValue(options, nameof(options));
            FastTreeTrainerOptions = options;
            // The discretization step renders this trainer non-parametric, and therefore it does not need normalization.
            // Also since it builds its own internal discretized columnar structures, it cannot benefit from caching.
            // Finally, even the binary classifiers, being logitboost, tend to not benefit from external calibration.
            Info = new TrainerInfo(normalization: false, caching: false, calibration: NeedCalibration, supportValid: true, supportTest: true);
            // REVIEW: CLR 4.6 has a bug that is only exposed in Scope, and if we trigger GC.Collect in scope environment
            // with memory consumption more than 5GB, GC get stuck in infinite loop.
            // Before, we could check a specific type of the environment here, but now it is internal, so we will need another
            // mechanism to detect that we are running in Scope.
            AllowGC = true;

            Initialize(env);
        }

        private protected abstract void PrepareLabels(IChannel ch);

        private protected abstract void InitializeTests();

        private protected abstract Test ConstructTestForTrainingData();

        private protected abstract OptimizationAlgorithm ConstructOptimizationAlgorithm(IChannel ch);

        private protected abstract TreeLearner ConstructTreeLearner(IChannel ch);

        private protected abstract ObjectiveFunctionBase ConstructObjFunc(IChannel ch);

        private protected virtual float GetMaxLabel()
        {
            return float.PositiveInfinity;
        }

        private void Initialize(IHostEnvironment env)
        {
            int numThreads = FastTreeTrainerOptions.NumThreads ?? Environment.ProcessorCount;
            if (Host.ConcurrencyFactor > 0 && numThreads > Host.ConcurrencyFactor)
            {
                using (var ch = Host.Start("FastTreeTrainerBase"))
                {
                    numThreads = Host.ConcurrencyFactor;
                    ch.Warning("The number of threads specified in trainer arguments is larger than the concurrency factor "
                        + "setting of the environment. Using {0} training threads instead.", numThreads);
                }
            }
            ParallelTraining = FastTreeTrainerOptions.ParallelTrainer != null ? FastTreeTrainerOptions.ParallelTrainer.CreateComponent(env) : new SingleTrainer();
            ParallelTraining.InitEnvironment();

            Tests = new List<Test>();

            InitializeThreads(numThreads);
        }

        private protected void ConvertData(RoleMappedData trainData)
        {
            AnnotationUtils.TryGetCategoricalFeatureIndices(trainData.Schema.Schema, trainData.Schema.Feature.Value.Index, out CategoricalFeatures);
            var useTranspose = UseTranspose(FastTreeTrainerOptions.DiskTranspose, trainData) && (ValidData == null || UseTranspose(FastTreeTrainerOptions.DiskTranspose, ValidData));
            var instanceConverter = new ExamplesToFastTreeBins(Host, FastTreeTrainerOptions.MaxBins, useTranspose, !FastTreeTrainerOptions.FeatureFlocks, FastTreeTrainerOptions.MinDocumentsInLeafs, GetMaxLabel());

            TrainSet = instanceConverter.FindBinsAndReturnDataset(trainData, PredictionKind, ParallelTraining, CategoricalFeatures, FastTreeTrainerOptions.CategoricalSplit);
            FeatureMap = instanceConverter.FeatureMap;
            if (ValidData != null)
                ValidSet = instanceConverter.GetCompatibleDataset(ValidData, PredictionKind, CategoricalFeatures, FastTreeTrainerOptions.CategoricalSplit);
            if (TestData != null)
                TestSets = new[] { instanceConverter.GetCompatibleDataset(TestData, PredictionKind, CategoricalFeatures, FastTreeTrainerOptions.CategoricalSplit) };
        }

        private bool UseTranspose(bool? useTranspose, RoleMappedData data)
        {
            Host.AssertValue(data);
            Host.Assert(data.Schema.Feature.HasValue);

            if (useTranspose.HasValue)
                return useTranspose.Value;

            var itdv = data.Data as ITransposeDataView;
            return itdv?.GetSlotType(data.Schema.Feature.Value.Index) != null;
        }

        private protected void TrainCore(IChannel ch)
        {
            Contracts.CheckValue(ch, nameof(ch));
            // REVIEW:Get rid of this lock then we completly remove all static classes from FastTree such as BlockingThreadPool.
            lock (FastTreeShared.TrainLock)
            {
                using (Timer.Time(TimerEvent.TotalInitialization))
                {
                    CheckOptions(ch);
                    PrintPrologInfo(ch);

                    Initialize(ch);
                    PrintMemoryStats(ch);
                }
                using (Timer.Time(TimerEvent.TotalTrain))
                    Train(ch);
                if (FastTreeTrainerOptions.ExecutionTimes)
                    PrintExecutionTimes(ch);
                TrainedEnsemble = Ensemble;
                if (FeatureMap != null)
                    TrainedEnsemble.RemapFeatures(FeatureMap);
                ParallelTraining.FinalizeEnvironment();
            }
        }

        private protected virtual bool ShouldStop(IChannel ch, ref IEarlyStoppingCriterion earlyStopping, ref int bestIteration)
        {
            bestIteration = Ensemble.NumTrees;
            return false;
        }
        private protected virtual int GetBestIteration(IChannel ch) => Ensemble.NumTrees;

        private protected virtual void InitializeThreads(int numThreads)
        {
            ThreadTaskManager.Initialize(numThreads);
        }

        private protected virtual void PrintExecutionTimes(IChannel ch)
        {
            ch.Info("Execution time breakdown:\n{0}", Timer.GetString());
        }

        private protected virtual void CheckOptions(IChannel ch)
        {
            FastTreeTrainerOptions.Check(ch);

            IntArray.CompatibilityLevel = FastTreeTrainerOptions.FeatureCompressionLevel;

            // change arguments
            if (FastTreeTrainerOptions.HistogramPoolSize < 2)
                FastTreeTrainerOptions.HistogramPoolSize = FastTreeTrainerOptions.NumLeaves * 2 / 3;
            if (FastTreeTrainerOptions.HistogramPoolSize > FastTreeTrainerOptions.NumLeaves - 1)
                FastTreeTrainerOptions.HistogramPoolSize = FastTreeTrainerOptions.NumLeaves - 1;

            if (FastTreeTrainerOptions.BaggingSize > 0)
            {
                int bagCount = FastTreeTrainerOptions.NumTrees / FastTreeTrainerOptions.BaggingSize;
                if (bagCount * FastTreeTrainerOptions.BaggingSize != FastTreeTrainerOptions.NumTrees)
                    throw ch.Except("Number of trees should be a multiple of number bag size");
            }

            if (!(0 <= FastTreeTrainerOptions.GainConfidenceLevel && FastTreeTrainerOptions.GainConfidenceLevel < 1))
                throw ch.Except("Gain confidence level must be in the range [0,1)");

#if OLD_DATALOAD
#if !NO_STORE
            if (_args.offloadBinsToFileStore)
            {
                if (!string.IsNullOrEmpty(_args.offloadBinsDirectory) && !Directory.Exists(_args.offloadBinsDirectory))
                {
                    try
                    {
                        Directory.CreateDirectory(_args.offloadBinsDirectory);
                    }
                    catch (Exception e)
                    {
                        throw ch.Except(e, "Failure creating bins offload directory {0} - Exception {1}", _args.offloadBinsDirectory, e.Message);
                    }
                }
            }
#endif
#endif
        }

        /// <summary>
        /// A virtual method that is used to print header of test graph.
        /// Appliations that need printing test graph are supposed to override
        /// it to print specific test graph header.
        /// </summary>
        /// <returns> string representation of test graph header </returns>
        private protected virtual string GetTestGraphHeader() => string.Empty;

        /// <summary>
        /// A virtual method that is used to print a single line of test graph.
        /// Applications that need printing test graph are supposed to override
        /// it to print a specific line of test graph after a new iteration is finished.
        /// </summary>
        /// <returns> string representation of a line of test graph </returns>
        private protected virtual string GetTestGraphLine() => string.Empty;

        /// <summary>
        /// A virtual method that is used to compute test results after each iteration is finished.
        /// </summary>
        private protected virtual void ComputeTests()
        {
        }

        private protected void PrintTestGraph(IChannel ch)
        {
            // we call Tests computing no matter whether we require to print test graph
            ComputeTests();

            if (!FastTreeTrainerOptions.PrintTestGraph)
                return;

            if (Ensemble.NumTrees == 0)
                ch.Info(GetTestGraphHeader());
            else
                ch.Info(GetTestGraphLine());

            return;
        }

        private protected virtual void Initialize(IChannel ch)
        {
            #region Load/Initialize State

            using (Timer.Time(TimerEvent.InitializeLabels))
                PrepareLabels(ch);
            using (Timer.Time(TimerEvent.InitializeTraining))
            {
                InitializeEnsemble();
                OptimizationAlgorithm = ConstructOptimizationAlgorithm(ch);
            }
            using (Timer.Time(TimerEvent.InitializeTests))
                InitializeTests();
            if (AllowGC)
            {
                GC.Collect(2, GCCollectionMode.Forced);
                GC.Collect(2, GCCollectionMode.Forced);
            }
            #endregion
        }

#if !NO_STORE
        /// <summary>
        /// Calculates the percentage of feature bins that will fit into memory based on current available memory in the machine.
        /// </summary>
        /// <returns>A float number between 0 and 1 indicating the percentage of features to load.
        ///         The number will not be smaller than two times the feature fraction value</returns>
        private float GetFeaturePercentInMemory(IChannel ch)
        {
            const float maxFeaturePercentValue = 1.0f;

            float availableMemory = GetMachineAvailableBytes();

            ch.Info("Available memory in the machine is = {0} bytes", availableMemory.ToString("N", CultureInfo.InvariantCulture));

            float minFeaturePercentThreshold = _args.preloadFeatureBinsBeforeTraining ? (float)_args.featureFraction * 2 : (float)_args.featureFraction;

            if (minFeaturePercentThreshold >= maxFeaturePercentValue)
            {
                return maxFeaturePercentValue;
            }

            // Initial free memory allowance in bytes for single and parallel fastrank modes
            float freeMemoryAllowance = 1024 * 1024 * 512;

            if (_optimizationAlgorithm.TreeLearner != null)
            {
                // Get the size of memory in bytes needed by the tree learner internal data structures
                freeMemoryAllowance += _optimizationAlgorithm.TreeLearner.GetSizeOfReservedMemory();
            }

            availableMemory = (availableMemory > freeMemoryAllowance) ? availableMemory - freeMemoryAllowance : 0;

            long featureSize = TrainSet.FeatureSetSize;

            if (ValidSet != null)
            {
                featureSize += ValidSet.FeatureSetSize;
            }

            if (TestSets != null)
            {
                foreach (var item in TestSets)
                {
                    featureSize += item.FeatureSetSize;
                }
            }

            ch.Info("Total Feature bins size is = {0} bytes", featureSize.ToString("N", CultureInfo.InvariantCulture));

            return Math.Min(Math.Max(minFeaturePercentThreshold, availableMemory / featureSize), maxFeaturePercentValue);
        }
#endif

        private protected bool[] GetActiveFeatures()
        {
            var activeFeatures = Utils.CreateArray(TrainSet.NumFeatures, true);
            if (FastTreeTrainerOptions.FeatureFraction < 1.0)
            {
                if (_featureSelectionRandom == null)
                    _featureSelectionRandom = new Random(FastTreeTrainerOptions.FeatureSelectSeed);

                for (int i = 0; i < TrainSet.NumFeatures; ++i)
                {
                    if (activeFeatures[i])
                        activeFeatures[i] = _featureSelectionRandom.NextDouble() <= FastTreeTrainerOptions.FeatureFraction;
                }
            }

            return activeFeatures;
        }

        private string GetDatasetStatistics(Dataset set)
        {
            long datasetSize = set.SizeInBytes();
            int skeletonSize = set.Skeleton.SizeInBytes();
            return string.Format("set contains {0} query-doc pairs in {1} queries with {2} features and uses {3} MB ({4} MB for features)",
                set.NumDocs, set.NumQueries, set.NumFeatures, datasetSize / 1024 / 1024, (datasetSize - skeletonSize) / 1024 / 1024);
        }

        private protected virtual void PrintMemoryStats(IChannel ch)
        {
            Contracts.AssertValue(ch);
            ch.Trace("Training {0}", GetDatasetStatistics(TrainSet));

            if (ValidSet != null)
                ch.Trace("Validation {0}", GetDatasetStatistics(ValidSet));
            if (TestSets != null)
            {
                for (int i = 0; i < TestSets.Length; ++i)
                    ch.Trace("ComputeTests[{1}] {0}",
                        GetDatasetStatistics(TestSets[i]), i);
            }

            if (AllowGC)
                ch.Trace("GC Total Memory = {0} MB", GC.GetTotalMemory(true) / 1024 / 1024);
            Process currentProcess = Process.GetCurrentProcess();
            ch.Trace("Working Set = {0} MB", currentProcess.WorkingSet64 / 1024 / 1024);
            ch.Trace("Virtual Memory = {0} MB",
                currentProcess.VirtualMemorySize64 / 1024 / 1024);
            ch.Trace("Private Memory = {0} MB",
                currentProcess.PrivateMemorySize64 / 1024 / 1024);
            ch.Trace("Peak Working Set = {0} MB", currentProcess.PeakWorkingSet64 / 1024 / 1024);
            ch.Trace("Peak Virtual Memory = {0} MB",
                currentProcess.PeakVirtualMemorySize64 / 1024 / 1024);
        }

        private protected bool AreSamplesWeighted(IChannel ch)
        {
            return TrainSet.SampleWeights != null;
        }

        private void InitializeEnsemble()
        {
            Ensemble = new InternalTreeEnsemble();
        }

        /// <summary>
        /// Creates weights wrapping (possibly, trivial) for gradient target values.
        /// </summary>
        private protected virtual IGradientAdjuster MakeGradientWrapper(IChannel ch)
        {
            if (AreSamplesWeighted(ch))
                return new QueryWeightsGradientWrapper();
            else
                return new TrivialGradientWrapper();
        }

#if !NO_STORE
        /// <summary>
        /// Unloads feature bins being used in the current iteration.
        /// </summary>
        /// <param name="featureToUnload">Boolean array indicating the features to unload</param>
        private void UnloadFeatureBins(bool[] featureToUnload)
        {
            foreach (ScoreTracker scoreTracker in this._optimizationAlgorithm.TrackedScores)
            {
                for (int i = 0; i < scoreTracker.Dataset.Features.Length; i++)
                {
                    if (featureToUnload[i])
                    {
                        // Only return buffers to the pool that were allocated using the pool
                        // So far only this type of IntArrays below have buffer pool support.
                        // This is to avoid unexpected leaks in case a new IntArray is added but we are not allocating it from the pool.
                        if (scoreTracker.Dataset.Features[i].Bins is DenseIntArray ||
                            scoreTracker.Dataset.Features[i].Bins is DeltaSparseIntArray ||
                            scoreTracker.Dataset.Features[i].Bins is DeltaRepeatIntArray)
                        {
                            scoreTracker.Dataset.Features[i].Bins.ReturnBuffer();
                            scoreTracker.Dataset.Features[i].Bins = null;
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Worker thread delegate that loads features for the next training iteration
        /// </summary>
        /// <param name="state">thread state object</param>
        private void LazyFeatureLoad(object state)
        {
            bool[] featuresToLoad = (bool[])state;

            foreach (ScoreTracker scoreTracker in this._optimizationAlgorithm.TrackedScores)
            {
                for (int i = 0; i < scoreTracker.Dataset.Features.Length; i++)
                {
                    if (featuresToLoad[i])
                    {
                        // just using the Bins property so feature bins are loaded into memory
                        IntArray bins = scoreTracker.Dataset.Features[i].Bins;
                    }
                }
            }
        }

        /// <summary>
        /// Iterates through the feature sets needed in future tree training iterations (i.e. in ActiveFeatureSetQueue),
        /// using the same order as they were enqueued, and it returns the initial active features based on the percentage parameter.
        /// </summary>
        /// <param name="pctFeatureThreshold">A float value between 0 and 1 indicating maximum percentage of features to return</param>
        /// <returns>Array indicating calculated feature list</returns>
        private bool[] GetNextFeaturesByThreshold(float pctFeatureThreshold)
        {
            int totalUniqueFeatureCount = 0;
            bool[] nextActiveFeatures = new bool[TrainSet.NumFeatures];

            if (pctFeatureThreshold == 1.0f)
            {
                // return all features to load
                return nextActiveFeatures.Select(x => x = true).ToArray();
            }

            int maxNumberOfFeatures = (int)(pctFeatureThreshold * TrainSet.NumFeatures);

            for (int i = 0; i < _activeFeatureSetQueue.Count; i++)
            {
                bool[] tempActiveFeatures = _activeFeatureSetQueue.ElementAt(i);

                for (int j = 0; j < tempActiveFeatures.Length; j++)
                {
                    if (tempActiveFeatures[j] && !nextActiveFeatures[j])
                    {
                        nextActiveFeatures[j] = true;
                        if (totalUniqueFeatureCount++ > maxNumberOfFeatures)
                            return nextActiveFeatures;
                    }
                }
            }

            return nextActiveFeatures;
        }

        /// <summary>
        /// Adds several items in the ActiveFeature queue
        /// </summary>
        /// <param name="numberOfItems">Number of items to add</param>
        private void GenerateActiveFeatureLists(int numberOfItems)
        {
            for (int i = 0; i < numberOfItems; i++)
            {
                _activeFeatureSetQueue.Enqueue(GetActiveFeatures());
            }
        }
#endif

        private protected virtual BaggingProvider CreateBaggingProvider()
        {
            Contracts.Assert(FastTreeTrainerOptions.BaggingSize > 0);
            return new BaggingProvider(TrainSet, FastTreeTrainerOptions.NumLeaves, FastTreeTrainerOptions.RngSeed, FastTreeTrainerOptions.BaggingTrainFraction);
        }

        private protected virtual bool ShouldRandomStartOptimizer()
        {
            return false;
        }

        private protected virtual void Train(IChannel ch)
        {
            Contracts.AssertValue(ch);
            int numTotalTrees = FastTreeTrainerOptions.NumTrees;

            ch.Info(
                "Reserved memory for tree learner: {0} bytes",
                OptimizationAlgorithm.TreeLearner.GetSizeOfReservedMemory());

#if !NO_STORE
            if (_args.offloadBinsToFileStore)
            {
                // Initialize feature percent to load before loading any features
                _featurePercentToLoad = GetFeaturePercentInMemory(ch);
                ch.Info("Using featurePercentToLoad = {0} ", _featurePercentToLoad);
            }
#endif

            // random starting point
            bool revertRandomStart = false;
            if (Ensemble.NumTrees < numTotalTrees && ShouldRandomStartOptimizer())
            {
                ch.Info("Randomizing start point");
                OptimizationAlgorithm.TrainingScores.RandomizeScores(FastTreeTrainerOptions.RngSeed, false);
                revertRandomStart = true;
            }

            ch.Info("Starting to train ...");

            BaggingProvider baggingProvider = FastTreeTrainerOptions.BaggingSize > 0 ? CreateBaggingProvider() : null;

#if OLD_DATALOAD
#if !NO_STORE
            // Preload
            GenerateActiveFeatureLists(_args.numTrees);
            Thread featureLoadThread = null;

            // Initial feature load
            if (_args.offloadBinsToFileStore)
            {
                FileObjectStore<IntArrayFormatter>.GetDefaultInstance().SealObjectStore();
                if (_args.preloadFeatureBinsBeforeTraining)
                {
                    StartFeatureLoadThread(GetNextFeaturesByThreshold(_featurePercentToLoad)).Join();
                }
            }
#endif
#endif

            IEarlyStoppingCriterion earlyStoppingRule = null;
            int bestIteration = 0;
            int emptyTrees = 0;
            using (var pch = Host.StartProgressChannel("FastTree training"))
            {
                pch.SetHeader(new ProgressHeader("trees"), e => e.SetProgress(0, Ensemble.NumTrees, numTotalTrees));
                while (Ensemble.NumTrees < numTotalTrees)
                {
                    using (Timer.Time(TimerEvent.Iteration))
                    {
#if NO_STORE
                        bool[] activeFeatures = GetActiveFeatures();
#else
                        bool[] activeFeatures = _activeFeatureSetQueue.Dequeue();
#endif

                        if (FastTreeTrainerOptions.BaggingSize > 0 && Ensemble.NumTrees % FastTreeTrainerOptions.BaggingSize == 0)
                        {
                            baggingProvider.GenerateNewBag();
                            OptimizationAlgorithm.TreeLearner.Partitioning =
                                baggingProvider.GetCurrentTrainingPartition();
                        }

#if !NO_STORE
                        if (_args.offloadBinsToFileStore)
                        {
                            featureLoadThread = StartFeatureLoadThread(GetNextFeaturesByThreshold(_featurePercentToLoad));
                            if (!_args.preloadFeatureBinsBeforeTraining)
                                featureLoadThread.Join();
                        }
#endif

                        // call the weak learner
                        var tree = OptimizationAlgorithm.TrainingIteration(ch, activeFeatures);
                        if (tree == null)
                        {
                            emptyTrees++;
                            numTotalTrees--;
                        }
                        else if (FastTreeTrainerOptions.BaggingSize > 0 && Ensemble.Trees.Count() > 0)
                        {
                            ch.Assert(Ensemble.Trees.Last() == tree);
                            Ensemble.Trees.Last()
                                .AddOutputsToScores(OptimizationAlgorithm.TrainingScores.Dataset,
                                    OptimizationAlgorithm.TrainingScores.Scores,
                                    baggingProvider.GetCurrentOutOfBagPartition().Documents);
                        }

                        CustomizedTrainingIteration(tree);

                        using (Timer.Time(TimerEvent.Test))
                        {
                            PrintIterationMessage(ch, pch);
                            PrintTestResults(ch);
                        }

                        // revert randomized start
                        if (revertRandomStart)
                        {
                            revertRandomStart = false;
                            ch.Info("Reverting random score assignment");
                            OptimizationAlgorithm.TrainingScores.RandomizeScores(FastTreeTrainerOptions.RngSeed, true);
                        }

#if !NO_STORE
                        if (_args.offloadBinsToFileStore)
                        {
                            // Unload only features that are not needed for the next iteration
                            bool[] featuresToUnload = activeFeatures;

                            if (_args.preloadFeatureBinsBeforeTraining)
                            {
                                featuresToUnload =
                                    activeFeatures.Zip(GetNextFeaturesByThreshold(_featurePercentToLoad),
                                        (current, next) => current && !next).ToArray();
                            }

                            UnloadFeatureBins(featuresToUnload);

                            if (featureLoadThread != null &&
                                _args.preloadFeatureBinsBeforeTraining)
                            {
                                // wait for loading the features needed for the next iteration
                                featureLoadThread.Join();
                            }
                        }
#endif
                        if (ShouldStop(ch, ref earlyStoppingRule, ref bestIteration))
                            break;
                    }
                }

                if (emptyTrees > 0)
                {
                    ch.Warning("{0} of the boosting iterations failed to grow a tree. This is commonly because the " +
                        "minimum documents in leaf hyperparameter was set too high for this dataset.", emptyTrees);
                }
            }

            if (earlyStoppingRule != null)
            {
                Contracts.Assert(numTotalTrees == 0 || bestIteration > 0);
                // REVIEW: Need to reconcile with future progress reporting changes.
                ch.Info("The training is stopped at {0} and iteration {1} is picked",
                    Ensemble.NumTrees, bestIteration);
            }
            else
            {
                bestIteration = GetBestIteration(ch);
            }

            OptimizationAlgorithm.FinalizeLearning(bestIteration);
            Ensemble.PopulateRawThresholds(TrainSet);
            ParallelTraining.FinalizeTreeLearner();
        }

#if !NO_STORE
        /// <summary>
        /// Gets the available bytes performance counter on the local machine
        /// </summary>
        /// <returns>Available bytes number</returns>
        private float GetMachineAvailableBytes()
        {
            using (var availableBytes = new System.Diagnostics.PerformanceCounter("Memory", "Available Bytes", true))
            {
                return availableBytes.NextValue();
            }
        }
#endif

        // This method is called at the end of each training iteration, with the tree that was learnt on that iteration.
        // Note that this tree can be null if no tree was learnt this iteration.
        private protected virtual void CustomizedTrainingIteration(InternalRegressionTree tree)
        {
        }

        private protected virtual void PrintIterationMessage(IChannel ch, IProgressChannel pch)
        {
            // REVIEW: report some metrics, not just number of trees?
            int iteration = Ensemble.NumTrees;
            if (iteration % 50 == 49)
                pch.Checkpoint(iteration + 1);
        }

        private protected virtual void PrintTestResults(IChannel ch)
        {
            if (FastTreeTrainerOptions.TestFrequency != int.MaxValue && (Ensemble.NumTrees % FastTreeTrainerOptions.TestFrequency == 0 || Ensemble.NumTrees == FastTreeTrainerOptions.NumTrees))
            {
                var sb = new StringBuilder();
                using (var sw = new StringWriter(sb))
                {
                    foreach (var t in Tests)
                    {
                        var results = t.ComputeTests();
                        sw.Write(t.FormatInfoString());
                    }
                }

                if (sb.Length > 0)
                    ch.Info(sb.ToString());
            }
        }
        private protected virtual void PrintPrologInfo(IChannel ch)
        {
            Contracts.AssertValue(ch);
            ch.Trace("Host = {0}", Environment.MachineName);
            ch.Trace("CommandLine = {0}", CmdParser.GetSettings(Host, FastTreeTrainerOptions, new TOptions()));
            ch.Trace("GCSettings.IsServerGC = {0}", System.Runtime.GCSettings.IsServerGC);
            ch.Trace("{0}", FastTreeTrainerOptions);
        }

        private protected ScoreTracker ConstructScoreTracker(Dataset set)
        {
            // If not found contruct one
            ScoreTracker st = null;
            if (set == TrainSet)
                st = OptimizationAlgorithm.GetScoreTracker("train", TrainSet, InitTrainScores);
            else if (set == ValidSet)
                st = OptimizationAlgorithm.GetScoreTracker("valid", ValidSet, InitValidScores);
            else
            {
                for (int t = 0; t < TestSets.Length; ++t)
                {
                    if (set == TestSets[t])
                    {
                        double[] initTestScores = InitTestScores?[t];
                        st = OptimizationAlgorithm.GetScoreTracker(string.Format("test[{0}]", t), TestSets[t], initTestScores);
                    }
                }
            }
            Contracts.Check(st != null, "unknown dataset passed to ConstructScoreTracker");
            return st;
        }

        private double[] ComputeScoresSmart(IChannel ch, Dataset set)
        {
            if (!FastTreeTrainerOptions.CompressEnsemble)
            {
                foreach (var st in OptimizationAlgorithm.TrackedScores)
                    if (st.Dataset == set)
                    {
                        ch.Trace("Computing scores fast");
                        return st.Scores;
                    }
            }
            return ComputeScoresSlow(ch, set);
        }

        private double[] ComputeScoresSlow(IChannel ch, Dataset set)
        {
            ch.Trace("Computing scores slow");
            double[] scores = new double[set.NumDocs];
            Ensemble.GetOutputs(set, scores);
            double[] initScores = GetInitScores(set);
            if (initScores != null)
            {
                Contracts.Check(scores.Length == initScores.Length, "Length of initscores and scores mismatch");
                for (int i = 0; i < scores.Length; i++)
                    scores[i] += initScores[i];
            }
            return scores;
        }

        private double[] GetInitScores(Dataset set)
        {
            if (set == TrainSet)
                return InitTrainScores;
            if (set == ValidSet)
                return InitValidScores;
            for (int i = 0; TestSets != null && i < TestSets.Length; i++)
            {
                if (set == TestSets[i])
                    return InitTestScores?[i];
            }
            throw Contracts.Except("Queried for unknown set");
        }
    }

    internal abstract class DataConverter
    {
        private protected readonly int NumFeatures;
        public abstract int NumExamples { get; }

        private protected readonly float MaxLabel;

        private protected readonly PredictionKind PredictionKind;

        /// <summary>
        /// The per-feature bin upper bounds. Implementations may differ on when all of the items
        /// in this array are initialized to non-null values but it must happen at least no later
        /// than immediately after we return from <see cref="GetDataset"/>.
        /// </summary>
        public readonly Double[][] BinUpperBounds;

        /// <summary>
        /// In the event that any features are filtered, this will contain the feature map, where
        /// the indices are the indices of features within the dataset, and the tree as we are
        /// learning, and the values are the indices of the features within the original input
        /// data. This array is used to "rehydrate" the tree once we finish training, so that the
        /// feature indices are once again over the full set of features, as opposed to the subset
        /// of features we actually trained on. This can be null in the event that no filtering
        /// occurred.
        /// </summary>
        /// <seealso cref="InternalTreeEnsemble.RemapFeatures"/>
        public int[] FeatureMap;

        private protected readonly IHost Host;

        private protected readonly int[] CategoricalFeatureIndices;

        private protected readonly bool CategoricalSplit;

        private protected bool UsingMaxLabel
        {
            get { return MaxLabel != float.PositiveInfinity; }
        }

        private DataConverter(RoleMappedData data, IHost host, Double[][] binUpperBounds, float maxLabel,
            PredictionKind kind, int[] categoricalFeatureIndices, bool categoricalSplit)
        {
            Contracts.AssertValue(host, "host");
            Host = host;
            Host.CheckValue(data, nameof(data));
            data.CheckFeatureFloatVector(out int featLen);
            data.CheckOptFloatWeight();
            data.CheckOptGroup();

            NumFeatures = featLen;
            if (binUpperBounds != null)
            {
                Host.AssertValue(binUpperBounds);
                Host.Assert(Utils.Size(binUpperBounds) == NumFeatures);
                Host.Assert(binUpperBounds.All(b => b != null));
                BinUpperBounds = binUpperBounds;
            }
            else
                BinUpperBounds = new Double[NumFeatures][];
            MaxLabel = maxLabel;
            PredictionKind = kind;
            CategoricalSplit = categoricalSplit;
            CategoricalFeatureIndices = categoricalFeatureIndices;
        }

        public static DataConverter Create(RoleMappedData data, IHost host, int maxBins,
            float maxLabel, bool diskTranspose, bool noFlocks, int minDocsPerLeaf, PredictionKind kind,
            IParallelTraining parallelTraining, int[] categoricalFeatureIndices, bool categoricalSplit)
        {
            Contracts.AssertValue(host, "host");
            host.AssertValue(data);
            host.Assert(maxBins > 0);
            DataConverter conv;
            using (var ch = host.Start("CreateConverter"))
            {
                if (!diskTranspose)
                    conv = new MemImpl(data, host, maxBins, maxLabel, noFlocks, minDocsPerLeaf, kind,
                        parallelTraining, categoricalFeatureIndices, categoricalSplit);
                else
                    conv = new DiskImpl(data, host, maxBins, maxLabel, kind, parallelTraining, categoricalFeatureIndices, categoricalSplit);
            }
            return conv;
        }

        public static DataConverter Create(RoleMappedData data, IHost host, Double[][] binUpperBounds,
            float maxLabel, bool diskTranspose, bool noFlocks, PredictionKind kind, int[] categoricalFeatureIndices, bool categoricalSplit)
        {
            Contracts.AssertValue(host, "host");
            host.AssertValue(data);
            DataConverter conv;
            using (var ch = host.Start("CreateConverter"))
            {
                if (!diskTranspose)
                    conv = new MemImpl(data, host, binUpperBounds, maxLabel, noFlocks, kind, categoricalFeatureIndices, categoricalSplit);
                else
                    conv = new DiskImpl(data, host, binUpperBounds, maxLabel, kind, categoricalFeatureIndices, categoricalSplit);
            }
            return conv;
        }

#if !CORECLR
        private protected void GetFeatureIniContent(RoleMappedData data, ref VBuffer<ReadOnlyMemory<char>> content)
        {
            // The existing implementations will have verified this by the time this utility
            // function is called.
            Host.AssertValue(data);
            var feat = data.Schema.Feature;
            Host.AssertValue(feat);
            Host.Assert(feat.Type.ValueCount > 0);

            var sch = data.Schema.Schema;
            var type = sch.GetMetadataTypeOrNull(BingBinLoader.IniContentMetadataKind, feat.Index);
            if (type == null || type.VectorSize != feat.Type.ValueCount || !type.IsVector || !type.ItemType.IsText)
                content = new VBuffer<ReadOnlyMemory<char>>(feat.Type.ValueCount, 0, content.Values, content.Indices);
            else
                sch.GetMetadata(BingBinLoader.IniContentMetadataKind, feat.Index, ref content);
        }
#endif

        public abstract Dataset GetDataset();

        /// <summary>
        /// Bins and input vector of feature values.
        /// </summary>
        /// <param name="binFinder">The instead of the bin finder to use</param>
        /// <param name="values">The values for one particular feature value across all examples</param>
        /// <param name="maxBins">The maximum number of bins to find</param>
        /// <param name="minDocsPerLeaf"></param>
        /// <param name="upperBounds">The bin upper bounds, maximum length will be <paramref name="maxBins"/></param>
        /// <returns>Whether finding the bins was successful or not. It will be unsuccessful iff <paramref name="values"/>
        /// has any missing values. In that event, the out parameters will be left as null.</returns>
        private protected static bool CalculateBins(BinFinder binFinder, in VBuffer<double> values, int maxBins, int minDocsPerLeaf,
            out double[] upperBounds)
        {
            return binFinder.FindBins(in values, maxBins, minDocsPerLeaf, out upperBounds);
        }

        private static IEnumerable<KeyValuePair<int, int>> NonZeroBinnedValuesForSparse(ReadOnlySpan<double> values, ReadOnlySpan<int> indices, Double[] binUpperBounds)
        {
            Contracts.Assert(values.Length == indices.Length);
            Contracts.Assert(Algorithms.FindFirstGE(binUpperBounds, 0) == 0);
            var result = new List<KeyValuePair<int, int>>();
            for (int i = 0; i < values.Length; ++i)
            {
                int ge = Algorithms.FindFirstGE(binUpperBounds, values[i]);
                if (ge != 0)
                    result.Add(new KeyValuePair<int, int>(indices[i], ge));
            }
            return result;
        }

        private FeatureFlockBase CreateOneHotFlock(IChannel ch,
                List<int> features, int[] binnedValues, int[] lastOn, ValuesList[] instanceList,
                ref int[] forwardIndexerWork, ref VBuffer<double> temp, bool categorical)
        {
            Contracts.AssertValue(ch);
            ch.Assert(0 <= features.Min() && features.Max() < NumFeatures);
            ch.Assert(features.Count > 0);

            if (features.Count == 1)
            {
                // Singleton.
                int fi = features[0];
                var values = instanceList[fi];
                values.CopyTo(NumExamples, ref temp);
                return CreateSingletonFlock(ch, in temp, binnedValues, BinUpperBounds[fi]);
            }
            // Multiple, one hot.
            int[] hotFeatureStarts = new int[features.Count + 1];
            // The position 0 is reserved as the "cold" position for all features in the slot.
            // This corresponds to all features being in their first bin (for example, cold). So the
            // first feature's "hotness" starts at 1. HOWEVER, for the purpose of defining the
            // bins, we start with this array computed off by one. Once we define the bins, we
            // will correct it.
            hotFeatureStarts[0] = 0;
            // There are as many hot positions per feature as there are number of bin upper
            // bounds, minus 1. (The first bin is the "cold" position.)
            for (int i = 1; i < hotFeatureStarts.Length; ++i)
                hotFeatureStarts[i] = hotFeatureStarts[i - 1] + BinUpperBounds[features[i - 1]].Length - 1;
            IntArrayBits flockBits = IntArray.NumBitsNeeded(hotFeatureStarts[hotFeatureStarts.Length - 1] + 1);

            int min = features[0];
            int lim = features[features.Count - 1] + 1;
            var ind = new ValuesList.ForwardIndexer(instanceList, features.ToArray(), ref forwardIndexerWork);
            int[] f2sf = Utils.CreateArray(lim - min, -1);
            for (int i = 0; i < features.Count; ++i)
                f2sf[features[i] - min] = i;

            int hotCount = 0;
            for (int i = 0; i < lastOn.Length; ++i)
            {
                int fi = lastOn[i];
                if (fi < min || fi >= lim)
                {
                    // All of the features would bin to 0, so we're in the "cold" position.
                    binnedValues[i] = 0;
#if false // This would be a very nice test to have, but for some situations it's too slow, even for debug builds. Consider reactivating temporarily if actively working on flocks.
                                // Assert that all the features really would be cold for this position.
                                Contracts.Assert(Enumerable.Range(min, lim - min).All(f => ind[f, i] < BinUpperBounds[f][0]));
#endif
                    continue;
                }
                ch.Assert(min <= fi && fi < lim);
                int subfeature = f2sf[fi - min];
                ch.Assert(subfeature >= 0);
                Double val = ind[subfeature, i];
#if false // Same note, too slow even for debug builds.
                            // Assert that all the other features really would be cold for this position.
                            Contracts.Assert(Enumerable.Range(min, fi - min).Concat(Enumerable.Range(fi + 1, lim - (fi + 1))).All(f => ind[f, i] < BinUpperBounds[f][0]));
#endif
                Double[] bub = BinUpperBounds[fi];
                ch.Assert(bub.Length > 1);
                int bin = Algorithms.FindFirstGE(bub, val);
                ch.Assert(0 < bin && bin < bub.Length); // If 0, should not have been considered "on", so what the heck?
                binnedValues[i] = hotFeatureStarts[subfeature] + bin;
                hotCount++;
            }
#if DEBUG
            int limBin = (1 << (int)flockBits);
            Contracts.Assert(flockBits == IntArrayBits.Bits32 || binnedValues.All(b => b < limBin));
#endif
            // Correct the hot feature starts now that we're done binning.
            for (int f = 0; f < hotFeatureStarts.Length; ++f)
                hotFeatureStarts[f]++;
            // Construct the int array of binned values.
            const double sparsifyThreshold = 0.7;

            IntArrayType type = hotCount < (1 - sparsifyThreshold) * NumExamples
                ? IntArrayType.Sparse
                : IntArrayType.Dense;
            IntArray bins = IntArray.New(NumExamples, type, flockBits, binnedValues);

            var bups = features.Select(fi => BinUpperBounds[fi]).ToArray(features.Count);
            return new OneHotFeatureFlock(bins, hotFeatureStarts, bups, categorical);
        }

        private FeatureFlockBase CreateOneHotFlockCategorical(IChannel ch,
                List<int> features, int[] binnedValues, int[] lastOn, bool categorical)
        {
            Contracts.AssertValue(ch);
            ch.Assert(0 <= features.Min() && features.Max() < NumFeatures);
            ch.Assert(features.Count > 1);

            // Multiple, one hot.
            int[] hotFeatureStarts = new int[features.Count + 1];
            // The position 0 is reserved as the "cold" position for all features in the slot.
            // This corresponds to all features being in their first bin (for example, cold). So the
            // first feature's "hotness" starts at 1. HOWEVER, for the purpose of defining the
            // bins, we start with this array computed off by one. Once we define the bins, we
            // will correct it.
            hotFeatureStarts[0] = 0;
            // There are as many hot positions per feature as there are number of bin upper
            // bounds, minus 1. (The first bin is the "cold" position.)
            for (int i = 1; i < hotFeatureStarts.Length; ++i)
                hotFeatureStarts[i] = hotFeatureStarts[i - 1] + BinUpperBounds[features[i - 1]].Length - 1;
            IntArrayBits flockBits = IntArray.NumBitsNeeded(hotFeatureStarts[hotFeatureStarts.Length - 1] + 1);

            int min = features[0];
            int lim = features[features.Count - 1] + 1;
            int[] f2sf = Utils.CreateArray(lim - min, -1);
            for (int i = 0; i < features.Count; ++i)
                f2sf[features[i] - min] = i;

            int hotCount = 0;
            for (int i = 0; i < lastOn.Length; ++i)
            {
                int fi = lastOn[i];
                if (fi < min || fi >= lim)
                {
                    // All of the features would bin to 0, so we're in the "cold" position.
                    binnedValues[i] = 0;
#if false // This would be a very nice test to have, but for some situations it's too slow, even for debug builds. Consider reactivating temporarily if actively working on flocks.
                                // Assert that all the features really would be cold for this position.
                                Contracts.Assert(Enumerable.Range(min, lim - min).All(f => ind[f, i] < BinUpperBounds[f][0]));
#endif
                    continue;
                }
                ch.Assert(min <= fi && fi < lim);
                int subfeature = f2sf[fi - min];
                ch.Assert(subfeature >= 0);
#if false // Same note, too slow even for debug builds.
                            // Assert that all the other features really would be cold for this position.
                            Contracts.Assert(Enumerable.Range(min, fi - min).Concat(Enumerable.Range(fi + 1, lim - (fi + 1))).All(f => ind[f, i] < BinUpperBounds[f][0]));
#endif
                Double[] bub = BinUpperBounds[fi];
                ch.Assert(bub.Length == 2);
                //REVIEW: leaving out check for the value to reduced memory consuption and going with
                //leap of faith based on what the user told.
                binnedValues[i] = hotFeatureStarts[subfeature] + 1;
                hotCount++;
            }
#if DEBUG
            int limBin = (1 << (int)flockBits);
            Contracts.Assert(flockBits == IntArrayBits.Bits32 || binnedValues.All(b => b < limBin));
#endif
            // Correct the hot feature starts now that we're done binning.
            for (int f = 0; f < hotFeatureStarts.Length; ++f)
                hotFeatureStarts[f]++;
            // Construct the int array of binned values.
            const double sparsifyThreshold = 0.7;

            IntArrayType type = hotCount < (1 - sparsifyThreshold) * NumExamples
                ? IntArrayType.Sparse
                : IntArrayType.Dense;
            IntArray bins = IntArray.New(NumExamples, type, flockBits, binnedValues);

            var bups = features.Select(fi => BinUpperBounds[fi]).ToArray(features.Count);
            return new OneHotFeatureFlock(bins, hotFeatureStarts, bups, categorical);
        }

        /// <summary>
        /// Create a new feature flock with a given name, values and specified bin bounds.
        /// </summary>
        /// <param name="ch"></param>
        /// <param name="values">The values for this feature, that will be binned.</param>
        /// <param name="binnedValues">A working array of length equal to the length of the input feature vector</param>
        /// <param name="binUpperBounds">The upper bounds of the binning of this feature.</param>
        /// <returns>A derived binned derived feature vector.</returns>
        private protected static SingletonFeatureFlock CreateSingletonFlock(IChannel ch, in VBuffer<double> values, int[] binnedValues,
            Double[] binUpperBounds)
        {
            Contracts.AssertValue(ch);
            ch.Assert(Utils.Size(binUpperBounds) > 0);
            ch.AssertValue(binnedValues);
            ch.Assert(binnedValues.Length == values.Length);

            // TODO: Consider trying to speed up FindFirstGE by making a "map" like is done in the fastrank code
            // TODO: Cache binnedValues
            int zeroBin = Algorithms.FindFirstGE(binUpperBounds, 0);

            // TODO: Make this a settable parameter / use the sparsifyThreshold already in the parameters
            const double sparsifyThreshold = 0.7;

            IntArray bins = null;

            var valuesValues = values.GetValues();
            var numBitsNeeded = IntArray.NumBitsNeeded(binUpperBounds.Length);
            if (numBitsNeeded == IntArrayBits.Bits0)
                bins = new Dense0BitIntArray(values.Length);
            else if (!values.IsDense && zeroBin == 0 && valuesValues.Length < (1 - sparsifyThreshold) * values.Length)
            {
                // Special code to go straight from our own sparse format to a sparse IntArray.
                // Note: requires zeroBin to be 0 because that's what's assumed in FastTree code
                var nonZeroValues = NonZeroBinnedValuesForSparse(valuesValues, values.GetIndices(), binUpperBounds);
                bins = new DeltaSparseIntArray(values.Length, numBitsNeeded, nonZeroValues);
            }
            else
            {
                // Fill the binnedValues array and convert using normal IntArray code
                int firstBinCount = 0;
                if (!values.IsDense)
                {
                    if (zeroBin != 0)
                    {
                        for (int i = 0; i < values.Length; i++)
                            binnedValues[i] = zeroBin;
                    }
                    else
                        Array.Clear(binnedValues, 0, values.Length);
                    var valuesIndices = values.GetIndices();
                    for (int i = 0; i < valuesValues.Length; ++i)
                    {
                        if ((binnedValues[valuesIndices[i]] = Algorithms.FindFirstGE(binUpperBounds, valuesValues[i])) == 0)
                            firstBinCount++;
                    }
                    if (zeroBin == 0)
                        firstBinCount += values.Length - valuesValues.Length;
                }
                else
                {
                    for (int i = 0; i < valuesValues.Length; i++)
                    {
                        if (valuesValues[i] == 0)
                            binnedValues[i] = zeroBin;
                        else
                            binnedValues[i] = Algorithms.FindFirstGE(binUpperBounds, valuesValues[i]);
                        if (binnedValues[i] == 0)
                            firstBinCount++;
                    }
                }
                // This sparsity check came from the FastRank code.
                double firstBinFrac = (double)firstBinCount / binnedValues.Length;
                IntArrayType arrayType = firstBinFrac > sparsifyThreshold ? IntArrayType.Sparse : IntArrayType.Dense;
                bins = IntArray.New(values.Length, arrayType, IntArray.NumBitsNeeded(binUpperBounds.Length), binnedValues);
            }
            return new SingletonFeatureFlock(bins, binUpperBounds);
        }

        private sealed class DiskImpl : DataConverter
        {
            private readonly int _numExamples;
            private readonly Dataset _dataset;

            public override int NumExamples { get { return _numExamples; } }

            public DiskImpl(RoleMappedData data, IHost host, int maxBins, float maxLabel, PredictionKind kind,
                IParallelTraining parallelTraining, int[] categoricalFeatureIndices, bool categoricalSplit)
                : base(data, host, null, maxLabel, kind, categoricalFeatureIndices, categoricalSplit)
            {
                // use parallel training for training data
                Host.AssertValue(parallelTraining);
                _dataset = Construct(data, ref _numExamples, maxBins, parallelTraining);
            }

            public DiskImpl(RoleMappedData data, IHost host,
                double[][] binUpperBounds, float maxLabel, PredictionKind kind, int[] categoricalFeatureIndices, bool categoricalSplit)
                : base(data, host, binUpperBounds, maxLabel, kind, categoricalFeatureIndices, categoricalSplit)
            {
                _dataset = Construct(data, ref _numExamples, -1, null);
            }

            public override Dataset GetDataset()
            {
                return _dataset;
            }

            private static int AddColumnIfNeeded(DataViewSchema.Column? info, List<int> toTranspose)
            {
                if (!info.HasValue)
                    return -1;
                // It is entirely possible that a single column could have two roles,
                // and so be added twice, but this case is handled by the transposer.
                var idx = info.Value.Index;
                toTranspose.Add(idx);
                return idx;
            }

            private ValueMapper<VBuffer<T1>, VBuffer<T2>> GetCopier<T1, T2>(DataViewType itemType1, DataViewType itemType2)
            {
                var conv = Conversions.Instance.GetStandardConversion<T1, T2>(itemType1, itemType2, out bool identity);
                if (identity)
                {
                    ValueMapper<VBuffer<T1>, VBuffer<T1>> identityResult =
                        (in VBuffer<T1> src, ref VBuffer<T1> dst) => src.CopyTo(ref dst);
                    return (ValueMapper<VBuffer<T1>, VBuffer<T2>>)(object)identityResult;
                }
                return
                    (in VBuffer<T1> src, ref VBuffer<T2> dst) =>
                    {
                        var srcValues = src.GetValues();
                        var editor = VBufferEditor.Create(ref dst, src.Length, srcValues.Length);
                        if (srcValues.Length > 0)
                        {
                            if (!src.IsDense)
                            {
                                src.GetIndices().CopyTo(editor.Indices);
                            }
                            for (int i = 0; i < srcValues.Length; ++i)
                                conv(in srcValues[i], ref editor.Values[i]);
                        }
                        dst = editor.Commit();
                    };
            }

            private Dataset Construct(RoleMappedData examples, ref int numExamples, int maxBins, IParallelTraining parallelTraining)
            {
                Host.AssertValue(examples);
                Host.Assert(examples.Schema.Feature.HasValue);

                if (parallelTraining == null)
                    Host.AssertValue(BinUpperBounds);

                Dataset result;
                using (var ch = Host.Start("Conversion"))
                {
                    // Add a missing value filter on the features.
                    // REVIEW: Possibly filter out missing labels, but we don't do this in current FastTree conversion.
                    //var missingArgs = new MissingValueFilter.Arguments();
                    //missingArgs.column = new string[] { examples.Schema.Feature.Name };
                    //IDataView data = new MissingValueFilter(missingArgs, Host, examples.Data);
                    IDataView data = examples.Data;

                    // Convert the label column, if one exists.
                    var labelName = examples.Schema.Label?.Name;
                    if (labelName != null)
                    {
                        var convArgs = new LabelConvertTransform.Arguments();
                        var convCol = new LabelConvertTransform.Column() { Name = labelName, Source = labelName };
                        convArgs.Columns = new LabelConvertTransform.Column[] { convCol };
                        data = new LabelConvertTransform(Host, convArgs, data);
                    }
                    // Convert the group column, if one exists.
                    if (examples.Schema.Group?.Name is string groupName)
                        data = new TypeConvertingTransformer(Host, new TypeConvertingEstimator.ColumnOptions(groupName, DataKind.UInt64, groupName)).Transform(data);

                    // Since we've passed it through a few transforms, reconstitute the mapping on the
                    // newly transformed data.
                    examples = new RoleMappedData(data, examples.Schema.GetColumnRoleNames());

                    // Get the index of the columns in the transposed view, while we're at it composing
                    // the list of the columns we want to transpose.
                    var toTranspose = new List<int>();
                    int featIdx = AddColumnIfNeeded(examples.Schema.Feature, toTranspose);
                    int labelIdx = AddColumnIfNeeded(examples.Schema.Label, toTranspose);
                    int groupIdx = AddColumnIfNeeded(examples.Schema.Group, toTranspose);
                    int weightIdx = AddColumnIfNeeded(examples.Schema.Weight, toTranspose);
                    Host.Assert(1 <= toTranspose.Count && toTranspose.Count <= 4);
                    ch.Info("Changing data from row-wise to column-wise on disk");
                    // Note that if these columns are already transposed, then this will be a no-op.
                    using (Transposer trans = Transposer.Create(Host, data, false, toTranspose.ToArray()))
                    {
                        VBuffer<float> temp = default(VBuffer<float>);
                        // Construct the derived features.
                        var features = new FeatureFlockBase[NumFeatures];
                        BinFinder finder = new BinFinder();
                        FeaturesToContentMap fmap = new FeaturesToContentMap(examples.Schema);

                        var hasMissingPred = Conversions.Instance.GetHasMissingPredicate<float>(((ITransposeDataView)trans).GetSlotType(featIdx));
                        // There is no good mechanism to filter out rows with missing feature values on transposed data.
                        // So, we instead perform one featurization pass which, if successful, will remain one pass but,
                        // if we ever encounter missing values will become a "detect missing features" pass, which will
                        // in turn inform a necessary featurization pass secondary
                        SlotDropper slotDropper = null;
                        bool[] localConstructBinFeatures = Utils.CreateArray<bool>(NumFeatures, true);

                        if (parallelTraining != null)
                            localConstructBinFeatures = parallelTraining.GetLocalBinConstructionFeatures(NumFeatures);

                        using (var pch = Host.StartProgressChannel("FastTree disk-based bins initialization"))
                        {
                            for (; ; )
                            {
                                bool hasMissing = false;
                                using (var cursor = trans.GetSlotCursor(featIdx))
                                {
                                    HashSet<int> constructed = new HashSet<int>();
                                    var getter = SubsetGetter(cursor.GetGetter<float>(), slotDropper);
                                    numExamples = slotDropper?.DstLength ?? trans.RowCount;

                                    // Perhaps we should change the binning to just work over singles.
                                    VBuffer<double> doubleTemp = default(VBuffer<double>);
                                    var copier = GetCopier<float, Double>(NumberDataViewType.Single, NumberDataViewType.Double);
                                    int iFeature = 0;
                                    pch.SetHeader(new ProgressHeader("features"), e => e.SetProgress(0, iFeature, features.Length));
                                    while (cursor.MoveNext())
                                    {
                                        iFeature = cursor.SlotIndex;
                                        if (!localConstructBinFeatures[iFeature])
                                            continue;

                                        Host.Assert(iFeature < features.Length);
                                        Host.Assert(features[iFeature] == null);
                                        getter(ref temp);
                                        Host.Assert(temp.Length == numExamples);

                                        // First get the bin bounds, constructing them if they do not exist.
                                        if (BinUpperBounds[iFeature] == null)
                                        {
                                            constructed.Add(iFeature);
                                            ch.Assert(maxBins > 0);
                                            finder = finder ?? new BinFinder();
                                            // Must copy over, as bin calculation is potentially destructive.
                                            copier(in temp, ref doubleTemp);
                                            hasMissing = !CalculateBins(finder, in doubleTemp, maxBins, 0,
                                                out BinUpperBounds[iFeature]);
                                        }
                                        else
                                            hasMissing = hasMissingPred(in temp);

                                        if (hasMissing)
                                        {
                                            // Let's just be a little extra safe, since it's so easy to check and the results if there
                                            // is a bug in the upstream pipeline would be very severe.
                                            ch.Check(slotDropper == null,
                                                "Multiple passes over the data seem to be producing different data. There is a bug in the upstream pipeline.");

                                            // Destroy any constructed bin upper bounds. We'll calculate them over the next pass.
                                            foreach (var i in constructed)
                                                BinUpperBounds[i] = null;
                                            // Determine what rows have missing values.
                                            slotDropper = ConstructDropSlotRanges(cursor, getter, ref temp);
                                            ch.Assert(slotDropper.DstLength < temp.Length);
                                            ch.Warning("{0} of {1} examples will be skipped due to missing feature values",
                                                temp.Length - slotDropper.DstLength, temp.Length);

                                            break;
                                        }
                                        Host.AssertValue(BinUpperBounds[iFeature]);
                                    }
                                }
                                if (hasMissing == false)
                                    break;
                            }

                            // Sync up global boundaries.
                            if (parallelTraining != null)
                                parallelTraining.SyncGlobalBoundary(NumFeatures, maxBins, BinUpperBounds);

                            List<FeatureFlockBase> flocks = new List<FeatureFlockBase>();
                            using (var cursor = trans.GetSlotCursor(featIdx))
                            using (var catCursor = trans.GetSlotCursor(featIdx))
                            {
                                var getter = SubsetGetter(cursor.GetGetter<float>(), slotDropper);
                                var catGetter = SubsetGetter(catCursor.GetGetter<float>(), slotDropper);
                                numExamples = slotDropper?.DstLength ?? trans.RowCount;

                                // Perhaps we should change the binning to just work over singles.
                                VBuffer<double> doubleTemp = default(VBuffer<double>);

                                int[] binnedValues = new int[numExamples];
                                var copier = GetCopier<float, Double>(NumberDataViewType.Single, NumberDataViewType.Double);
                                int iFeature = 0;
                                if (CategoricalSplit && CategoricalFeatureIndices != null)
                                {
                                    int[] lastOn = new int[NumExamples];
                                    for (int i = 0; i < lastOn.Length; ++i)
                                        lastOn[i] = -1;
                                    List<int> pending = new List<int>();
                                    int catRangeIndex = 0;
                                    for (iFeature = 0; iFeature < NumFeatures;)
                                    {
                                        if (catRangeIndex < CategoricalFeatureIndices.Length &&
                                            CategoricalFeatureIndices[catRangeIndex] == iFeature)
                                        {
                                            pending.Clear();
                                            bool oneHot = true;
                                            for (int iFeatureLocal = iFeature;
                                                iFeatureLocal <= CategoricalFeatureIndices[catRangeIndex + 1];
                                                ++iFeatureLocal)
                                            {
                                                Double[] bup = BinUpperBounds[iFeatureLocal];
                                                if (bup.Length == 1)
                                                {
                                                    // This is a trivial feature. Skip it.
                                                    continue;
                                                }
                                                Contracts.Assert(Utils.Size(bup) > 0);

                                                Double firstBin = bup[0];
                                                GetFeatureValues(catCursor, iFeatureLocal, catGetter, ref temp, ref doubleTemp, copier);
                                                bool add = false;
                                                var doubleTempValues = doubleTemp.GetValues();
                                                var doubleTempIndices = doubleTemp.GetIndices();
                                                for (int index = 0; index < doubleTempValues.Length; ++index)
                                                {
                                                    if (doubleTempValues[index] <= firstBin)
                                                        continue;

                                                    int iindex = doubleTemp.IsDense ? index : doubleTempIndices[index];
                                                    int last = lastOn[iindex];

                                                    if (doubleTempValues[index] != 1 || (last != -1 && last >= iFeature))
                                                    {
                                                        catRangeIndex += 2;
                                                        pending.Clear();
                                                        oneHot = false;
                                                        break;
                                                    }

                                                    lastOn[iindex] = iFeatureLocal;
                                                    add = true;
                                                }

                                                if (!oneHot)
                                                    break;

                                                if (add)
                                                    pending.Add(iFeatureLocal);
                                            }

                                            if (!oneHot)
                                                continue;

                                            if (pending.Count > 0)
                                            {
                                                flocks.Add(CreateOneHotFlockCategorical(ch, pending, binnedValues,
                                                    lastOn, true));
                                            }
                                            iFeature = CategoricalFeatureIndices[catRangeIndex + 1] + 1;
                                            catRangeIndex += 2;
                                        }
                                        else
                                        {
                                            GetFeatureValues(cursor, iFeature, getter, ref temp, ref doubleTemp, copier);
                                            double[] upperBounds = BinUpperBounds[iFeature++];
                                            Host.AssertValue(upperBounds);
                                            if (upperBounds.Length == 1)
                                                continue; //trivial feature, skip it.

                                            flocks.Add(CreateSingletonFlock(ch, in doubleTemp, binnedValues, upperBounds));
                                        }
                                    }
                                }
                                else
                                {
                                    for (int i = 0; i < NumFeatures; i++)
                                    {
                                        GetFeatureValues(cursor, i, getter, ref temp, ref doubleTemp, copier);
                                        double[] upperBounds = BinUpperBounds[i];
                                        Host.AssertValue(upperBounds);
                                        if (upperBounds.Length == 1)
                                            continue; //trivial feature, skip it.

                                        flocks.Add(CreateSingletonFlock(ch, in doubleTemp, binnedValues, upperBounds));
                                    }
                                }

                                Contracts.Assert(FeatureMap == null);

                                FeatureMap = Enumerable.Range(0, NumFeatures).Where(f => BinUpperBounds[f].Length > 1).ToArray();
                                features = flocks.ToArray();
                            }
                        }

                        // Construct the labels.
                        short[] ratings = new short[numExamples];
                        Double[] actualLabels = new Double[numExamples];

                        if (labelIdx >= 0)
                        {
                            trans.GetSingleSlotValue<float>(labelIdx, ref temp);
                            slotDropper?.DropSlots(ref temp, ref temp);

                            var tempValues = temp.GetValues();
                            var tempIndices = temp.GetIndices();
                            for (int i = 0; i < tempValues.Length; ++i)
                            {
                                int ii = temp.IsDense ? i : tempIndices[i];
                                var label = tempValues[i];
                                if (UsingMaxLabel && !(0 <= label && label <= MaxLabel))
                                    throw Host.Except("Found invalid label {0}. Value should be between 0 and {1}, inclusive.", label, MaxLabel);
                                ratings[ii] = (short)label;
                                actualLabels[ii] = (Double)label;
                            }
                        }

                        // Construct the boundaries and query IDs.
                        int[] boundaries;
                        ulong[] qids;
                        if (PredictionKind == PredictionKind.Ranking)
                        {
                            if (groupIdx < 0)
                                throw ch.Except("You need to provide {0} column for Ranking problem", DefaultColumnNames.GroupId);
                            VBuffer<ulong> groupIds = default(VBuffer<ulong>);
                            trans.GetSingleSlotValue<ulong>(groupIdx, ref groupIds);
                            slotDropper?.DropSlots(ref groupIds, ref groupIds);

                            ConstructBoundariesAndQueryIds(in groupIds, out boundaries, out qids);
                        }
                        else
                        {
                            if (groupIdx >= 0)
                                ch.Warning("This is not ranking problem, Group Id '{0}' column will be ignored", examples.Schema.Group.Value.Name);
                            const int queryChunkSize = 100;
                            qids = new ulong[(numExamples - 1) / queryChunkSize + 1];
                            boundaries = new int[qids.Length + 1];
                            for (int i = 0; i < qids.Length; ++i)
                            {
                                qids[i] = (ulong)i;
                                boundaries[i + 1] = boundaries[i] + queryChunkSize;
                            }
                            boundaries[boundaries.Length - 1] = numExamples;
                        }
                        // Construct the doc IDs. Doesn't really matter what these are.
                        ulong[] dids = Enumerable.Range(0, numExamples).Select(d => (ulong)d).ToArray(numExamples);

                        var skeleton = new Dataset.DatasetSkeleton(ratings, boundaries, qids, dids, new double[0][], actualLabels);

                        Host.Assert(features.All(f => f != null));
                        result = new Dataset(skeleton, features);
                    }
                }
                return result;
            }

            private void GetFeatureValues(SlotCursor cursor, int iFeature, ValueGetter<VBuffer<float>> getter,
                ref VBuffer<float> temp, ref VBuffer<double> doubleTemp, ValueMapper<VBuffer<float>, VBuffer<double>> copier)
            {
                while (cursor.MoveNext())
                {

                    Contracts.Assert(iFeature >= cursor.SlotIndex);

                    if (iFeature == cursor.SlotIndex)
                        break;
                }

                Contracts.Assert(cursor.SlotIndex == iFeature);

                getter(ref temp);
                copier(in temp, ref doubleTemp);
            }

            private static ValueGetter<VBuffer<T>> SubsetGetter<T>(ValueGetter<VBuffer<T>> getter, SlotDropper slotDropper)
            {
                if (slotDropper == null)
                    return getter;

                return slotDropper.SubsetGetter(getter);
            }

            /// <summary>
            /// Returns a slot dropper object that has ranges of slots to be dropped,
            /// based on an examination of the feature values.
            /// </summary>
            private static SlotDropper ConstructDropSlotRanges(SlotCursor cursor,
                ValueGetter<VBuffer<float>> getter, ref VBuffer<float> temp)
            {
                // The iteration here is slightly differently from a usual cursor iteration. Here, temp
                // already holds the value of the cursor's current position, and we don't really want
                // to re-fetch it, and the cursor is necessarily advanced.
                Contracts.Assert(cursor.SlotIndex >= 0);
                BitArray rowHasMissing = new BitArray(temp.Length);
                for (; ; )
                {
                    foreach (var kv in temp.Items())
                    {
                        if (float.IsNaN(kv.Value))
                            rowHasMissing.Set(kv.Key, true);
                    }
                    if (!cursor.MoveNext())
                        break;
                    getter(ref temp);
                }

                List<int> minSlots = new List<int>();
                List<int> maxSlots = new List<int>();
                bool previousBit = false;
                for (int i = 0; i < rowHasMissing.Length; i++)
                {
                    bool currentBit = rowHasMissing.Get(i);
                    if (currentBit && !previousBit)
                    {
                        minSlots.Add(i);
                        maxSlots.Add(i);
                    }
                    else if (currentBit)
                        maxSlots[maxSlots.Count - 1] = i;

                    previousBit = currentBit;
                }

                Contracts.Assert(maxSlots.Count == minSlots.Count);

                return new SlotDropper(temp.Length, minSlots.ToArray(), maxSlots.ToArray());
            }

            private static void ConstructBoundariesAndQueryIds(in VBuffer<ulong> groupIds, out int[] boundariesArray, out ulong[] qidsArray)
            {
                List<ulong> qids = new List<ulong>();
                List<int> boundaries = new List<int>();

                ulong last = 0;
                if (groupIds.Length > 0)
                    groupIds.GetItemOrDefault(0, ref last);
                int count = 0;
                foreach (ulong groupId in groupIds.DenseValues())
                {
                    if (count == 0 || last != groupId)
                    {
                        qids.Add(last = groupId);
                        boundaries.Add(count);
                    }
                    count++;
                }
                boundaries.Add(count);
                qidsArray = qids.ToArray();
                boundariesArray = boundaries.ToArray();
            }
        }

        // REVIEW: Our data conversion is extremely inefficient. Fix it!
        private sealed class MemImpl : DataConverter
        {
            private readonly RoleMappedData _data;

            // instanceList[feature] is the vector of values for the given feature
            private readonly ValuesList[] _instanceList;

            private readonly List<short> _targetsList;
            private readonly List<double> _actualTargets;
            private readonly List<double> _weights;
            private readonly List<int> _boundaries;
            private readonly long _numMissingInstances;
            private readonly int _numExamples;
            private readonly bool _noFlocks;
            private readonly int _minDocsPerLeaf;

            public override int NumExamples
            {
                get { return _numExamples; }
            }

            private MemImpl(RoleMappedData data, IHost host, double[][] binUpperBounds, float maxLabel, bool dummy,
                bool noFlocks, PredictionKind kind, int[] categoricalFeatureIndices, bool categoricalSplit)
                : base(data, host, binUpperBounds, maxLabel, kind, categoricalFeatureIndices, categoricalSplit)
            {
                _data = data;
                // Array of List<double> objects for each feature, containing values for that feature over all rows
                _instanceList = new ValuesList[NumFeatures];
                for (int i = 0; i < _instanceList.Length; i++)
                    _instanceList[i] = new ValuesList();
                // Labels.
                _targetsList = new List<short>();
                _actualTargets = new List<double>();
                _weights = data.Schema.Weight != null ? new List<double>() : null;
                _boundaries = new List<int>();
                _noFlocks = noFlocks;

                MakeBoundariesAndCheckLabels(out _numMissingInstances, out long numInstances);
                if (numInstances > Utils.ArrayMaxSize)
                    throw Host.ExceptParam(nameof(data), "Input data had {0} rows, but can only accomodate {1}", numInstances, Utils.ArrayMaxSize);
                _numExamples = (int)numInstances;
            }

            public MemImpl(RoleMappedData data, IHost host, int maxBins, float maxLabel, bool noFlocks, int minDocsPerLeaf,
                PredictionKind kind, IParallelTraining parallelTraining, int[] categoricalFeatureIndices, bool categoricalSplit)
                : this(data, host, null, maxLabel, dummy: true, noFlocks: noFlocks, kind: kind,
                      categoricalFeatureIndices: categoricalFeatureIndices, categoricalSplit: categoricalSplit)
            {
                // Convert features to binned values.
                _minDocsPerLeaf = minDocsPerLeaf;
                InitializeBins(maxBins, parallelTraining);
            }

            public MemImpl(RoleMappedData data, IHost host, double[][] binUpperBounds, float maxLabel,
                bool noFlocks, PredictionKind kind, int[] categoricalFeatureIndices, bool categoricalSplit)
                : this(data, host, binUpperBounds, maxLabel, dummy: true, noFlocks: noFlocks, kind: kind,
                      categoricalFeatureIndices: categoricalFeatureIndices, categoricalSplit: categoricalSplit)
            {
                Host.AssertValue(binUpperBounds);
            }

            private void MakeBoundariesAndCheckLabels(out long missingInstances, out long totalInstances)
            {
                using (var ch = Host.Start("InitBoundariesAndLabels"))
                using (var pch = Host.StartProgressChannel("FastTree data preparation"))
                {
                    long featureValues = 0;
                    // Warn at about 2 GB usage.
                    const long featureValuesWarnThreshold = (2L << 30) / sizeof(Double);
                    bool featureValuesWarned = false;
                    const string featureValuesWarning = "We seem to be processing a lot of data. Consider using the FastTree diskTranspose+ (or dt+) option, for slower but more memory efficient transposition.";
                    const int queryChunkSize = 100;

                    // Populate the feature values array and labels.
                    ch.Info("Changing data from row-wise to column-wise");

                    long pos = 0;
                    double rowCountDbl = (double?)_data.Data.GetRowCount() ?? Double.NaN;
                    pch.SetHeader(new ProgressHeader("examples"),
                        e => e.SetProgress(0, pos, rowCountDbl));
                    // REVIEW: Should we ignore rows with bad label, weight, or group? The previous code seemed to let
                    // them through (but filtered out bad features).
                    CursOpt curOptions = CursOpt.Label | CursOpt.Features;
                    bool hasGroup = false;
                    if (PredictionKind == PredictionKind.Ranking)
                    {
                        hasGroup = _data.Schema.Group != null;

                        if (hasGroup)
                            curOptions |= CursOpt.Group;
                    }
                    else
                    {
                        if (_data.Schema.Group != null)
                            ch.Warning("This is not ranking problem, Group Id '{0}' column will be ignored", _data.Schema.Group.Value.Name);
                    }

                    if (_data.Schema.Weight.HasValue)
                        curOptions |= CursOpt.Weight;

                    using (var cursor = new FloatLabelCursor(_data, curOptions))
                    {
                        ulong groupPrev = 0;

                        while (cursor.MoveNext())
                        {
                            pos = cursor.KeptRowCount - 1;
                            int index = checked((int)pos);
                            ch.Assert(pos >= 0);

                            // If we have no group, then the group number should not change.
                            Host.Assert(hasGroup || cursor.Group == groupPrev);
                            if (hasGroup)
                            {
                                // If we are either at the start of iteration, or a new
                                // group has started, add the boundary and register the
                                // new group identifier.
                                if (pos == 0 || cursor.Group != groupPrev)
                                {
                                    _boundaries.Add(index);
                                    groupPrev = cursor.Group;
                                }
                            }
                            else if (pos % queryChunkSize == 0)
                            {
                                // If there are no groups, it is best to just put the
                                // boundaries at regular intervals.
                                _boundaries.Add(index);
                            }

                            if (UsingMaxLabel)
                            {
                                if (cursor.Label < 0 || cursor.Label > MaxLabel)
                                    throw ch.Except("Found invalid label {0}. Value should be between 0 and {1}, inclusive.", cursor.Label, MaxLabel);
                            }

                            foreach (var kvp in cursor.Features.Items())
                                _instanceList[kvp.Key].Add(index, kvp.Value);

                            _actualTargets.Add(cursor.Label);
                            if (_weights != null)
                                _weights.Add(cursor.Weight);
                            _targetsList.Add((short)cursor.Label);
                            featureValues += cursor.Features.GetValues().Length;

                            if (featureValues > featureValuesWarnThreshold && !featureValuesWarned)
                            {
                                ch.Warning(featureValuesWarning);
                                featureValuesWarned = true;
                            }
                        }

                        _boundaries.Add(checked((int)cursor.KeptRowCount));
                        totalInstances = cursor.KeptRowCount;
                        missingInstances = cursor.BadFeaturesRowCount;
                    }

                    ch.Check(totalInstances > 0, "All instances skipped due to missing features.");

                    if (missingInstances > 0)
                        ch.Warning("Skipped {0} instances with missing features during training", missingInstances);
                }
            }

            private void InitializeBins(int maxBins, IParallelTraining parallelTraining)
            {
                // Find upper bounds for each bin for each feature.
                using (var ch = Host.Start("InitBins"))
                using (var pch = Host.StartProgressChannel("FastTree in-memory bins initialization"))
                {
                    BinFinder binFinder = new BinFinder();
                    VBuffer<double> temp = default(VBuffer<double>);
                    int len = _numExamples;
                    bool[] localConstructBinFeatures = parallelTraining.GetLocalBinConstructionFeatures(NumFeatures);
                    int iFeature = 0;
                    pch.SetHeader(new ProgressHeader("features"), e => e.SetProgress(0, iFeature, NumFeatures));
                    List<int> trivialFeatures = new List<int>();
                    for (iFeature = 0; iFeature < NumFeatures; iFeature++)
                    {
                        if (!localConstructBinFeatures[iFeature])
                            continue;
                        // The following strange call will actually sparsify.
                        _instanceList[iFeature].CopyTo(len, ref temp);
                        // REVIEW: In principle we could also put the min docs per leaf information
                        // into here, and collapse bins somehow as we determine the bins, so that "trivial"
                        // bins on the head or tail of the bin distribution are never actually considered.
                        CalculateBins(binFinder, in temp, maxBins, _minDocsPerLeaf,
                            out double[] binUpperBounds);
                        BinUpperBounds[iFeature] = binUpperBounds;
                    }
                    parallelTraining.SyncGlobalBoundary(NumFeatures, maxBins, BinUpperBounds);
                }
            }

            public override Dataset GetDataset()
            {
                using (var ch = Host.Start("BinFeatures"))
                using (var pch = Host.StartProgressChannel("FastTree feature conversion"))
                {
                    FeatureFlockBase[] flocks = CreateFlocks(ch, pch).ToArray();
                    ch.Trace("{0} features stored in {1} flocks.", NumFeatures, flocks.Length);
                    return new Dataset(CreateDatasetSkeleton(), flocks);
                }
            }

            private NHotFeatureFlock CreateNHotFlock(IChannel ch, List<int> features)
            {
                Contracts.AssertValue(ch);
                ch.Assert(Utils.Size(features) > 1);

                // Copy pasta from above.
                int[] hotFeatureStarts = new int[features.Count + 1];
                for (int i = 1; i < hotFeatureStarts.Length; ++i)
                    hotFeatureStarts[i] = hotFeatureStarts[i - 1] + BinUpperBounds[features[i - 1]].Length - 1;
                IntArrayBits flockBits = IntArray.NumBitsNeeded(hotFeatureStarts[hotFeatureStarts.Length - 1] + 1);

                var kvEnums = new IEnumerator<KeyValuePair<int, int>>[features.Count];
                var delta = new List<byte>();
                var values = new List<int>();

                try
                {
                    for (int i = 0; i < features.Count; ++i)
                        kvEnums[i] = _instanceList[features[i]].Binned(BinUpperBounds[features[i]], NumExamples).GetEnumerator();
                    Heap<int> heap = new Heap<int>(
                        (i, j) =>
                        {
                            ch.AssertValue(kvEnums[i]);
                            ch.AssertValue(kvEnums[j]);
                            int irow = kvEnums[i].Current.Key;
                            int jrow = kvEnums[j].Current.Key;
                            if (irow == jrow) // If we're on the same row, prefer the "smaller" feature.
                                return j < i;
                            // Earlier rows should go first.
                            return jrow < irow;
                        });
                    // Do the initial population of the heap.
                    for (int i = 0; i < kvEnums.Length; ++i)
                    {
                        if (kvEnums[i].MoveNext())
                            heap.Add(i);
                        else
                        {
                            kvEnums[i].Dispose();
                            kvEnums[i] = null;
                        }
                    }
                    // Iteratively build the delta-sparse and int arrays.
                    // REVIEW: Could be hinted as having capacity count hot, but may do more harm than good.
                    int last = 0;
                    while (heap.Count > 0)
                    {
                        int i = heap.Pop();
                        var kvEnum = kvEnums[i];
                        ch.AssertValue(kvEnum);
                        var kvp = kvEnum.Current;
                        ch.Assert(kvp.Key >= last);
                        ch.Assert(kvp.Value > 0);
                        while (kvp.Key - last > Byte.MaxValue)
                        {
                            delta.Add(Byte.MaxValue);
                            values.Add(0);
                            last += Byte.MaxValue;
                        }
                        ch.Assert(kvp.Key - last <= Byte.MaxValue);
                        // Note that kvp.Key - last might be zero, in the case where we are representing multiple
                        // values for a single row.
                        delta.Add((byte)(kvp.Key - last));
                        values.Add(kvp.Value + hotFeatureStarts[i]);
                        ch.Assert(kvp.Key > last || values.Count == 1 || values[values.Count - 1] > values[values.Count - 2]);
                        last = kvp.Key;
                        if (kvEnum.MoveNext())
                            heap.Add(i);
                        else
                        {
                            kvEnum.Dispose();
                            kvEnums[i] = null;
                        }
                    }
                }
                finally
                {
                    // Need to dispose the enumerators.
                    foreach (var enumerator in kvEnums)
                    {
                        if (enumerator != null)
                            enumerator.Dispose();
                    }
                }

                // Correct the hot feature starts now that we're done binning.
                for (int f = 0; f < hotFeatureStarts.Length; ++f)
                    hotFeatureStarts[f]++;
                var denseBins = (DenseIntArray)IntArray.New(values.Count, IntArrayType.Dense, flockBits, values);
                var bups = features.Select(fi => BinUpperBounds[fi]).ToArray(features.Count);
                return new NHotFeatureFlock(denseBins, delta.ToArray(), NumExamples, hotFeatureStarts, bups);
            }

            private IEnumerable<FeatureFlockBase> CreateFlocks(IChannel ch, IProgressChannel pch)
            {
                int iFeature = 0;
                FeatureMap = Enumerable.Range(0, NumFeatures).Where(f => BinUpperBounds[f].Length > 1).ToArray();

                foreach (FeatureFlockBase flock in CreateFlocksCore(ch, pch))
                {
                    Contracts.Assert(flock.Count > 0);
                    Contracts.Assert(iFeature + flock.Count <= FeatureMap.Length);
                    int min = FeatureMap[iFeature];
                    int lim = iFeature + flock.Count == FeatureMap.Length
                        ? NumFeatures
                        : FeatureMap[iFeature + flock.Count];
                    for (int i = min; i < lim; ++i)
                        _instanceList[i] = null;
                    iFeature += flock.Count;
                    yield return flock;
                }
                ch.Assert(iFeature <= NumFeatures); // Some could have been filtered.
                ch.Assert(iFeature == FeatureMap.Length);
                if (iFeature == 0)
                {
                    // It is possible to filter out all features. In such a case as this we introduce a dummy
                    // "trivial" feature, so that the learning code downstream does not choke.
                    yield return new SingletonFeatureFlock(new Dense0BitIntArray(NumExamples), BinUpperBounds[0]);
                    FeatureMap = new[] { 0 };
                }
            }

            private IEnumerable<FeatureFlockBase> CreateFlocksCore(IChannel ch, IProgressChannel pch)
            {
                int iFeature = 0;
                pch.SetHeader(new ProgressHeader("features"), e => e.SetProgress(0, iFeature, NumFeatures));
                VBuffer<double> temp = default(VBuffer<double>);
                // Working array for bins.
                int[] binnedValues = new int[NumExamples];

                if (_noFlocks)
                {
                    for (iFeature = 0; iFeature < NumFeatures; ++iFeature)
                    {
                        var bup = BinUpperBounds[iFeature];
                        ch.Assert(Utils.Size(bup) > 0);
                        if (bup.Length == 1) // Trivial.
                            continue;
                        var values = _instanceList[iFeature];
                        _instanceList[iFeature] = null;
                        values.CopyTo(NumExamples, ref temp);
                        yield return CreateSingletonFlock(ch, in temp, binnedValues, bup);
                    }
                    yield break;
                }

                List<int> pending = new List<int>();
                int[] forwardIndexerWork = null;

                if (CategoricalSplit && CategoricalFeatureIndices != null)
                {
                    int[] lastOn = new int[NumExamples];
                    for (int i = 0; i < lastOn.Length; ++i)
                        lastOn[i] = -1;

                    int catRangeIndex = 0;
                    for (iFeature = 0; iFeature < NumFeatures;)
                    {
                        if (catRangeIndex < CategoricalFeatureIndices.Length)
                        {
                            if (CategoricalFeatureIndices[catRangeIndex] == iFeature)
                            {
                                bool isOneHot = true;
                                for (int iFeatureLocal = iFeature;
                                    iFeatureLocal <= CategoricalFeatureIndices[catRangeIndex + 1];
                                    ++iFeatureLocal)
                                {
                                    Double[] bup = BinUpperBounds[iFeatureLocal];
                                    if (bup.Length == 1)
                                    {
                                        // This is a trivial feature. Skip it.
                                        continue;
                                    }
                                    Contracts.Assert(Utils.Size(bup) > 0);

                                    Double firstBin = bup[0];
                                    using (IEnumerator<int> hotEnumerator = _instanceList[iFeatureLocal].AllIndicesGT(NumExamples, firstBin).GetEnumerator())
                                    {
                                        while (hotEnumerator.MoveNext())
                                        {
                                            int last = lastOn[hotEnumerator.Current];

                                            //Not a one-hot flock, bail.
                                            if (last >= iFeature)
                                            {
                                                isOneHot = false;
                                                pending.Clear();
                                                break;
                                            }

                                            lastOn[hotEnumerator.Current] = iFeatureLocal;
                                        }
                                    }

                                    pending.Add(iFeatureLocal);
                                }

                                if (pending.Count > 0)
                                {
                                    yield return CreateOneHotFlock(ch, pending, binnedValues, lastOn, _instanceList,
                                        ref forwardIndexerWork, ref temp, true);

                                    pending.Clear();
                                }

                                if (isOneHot)
                                    iFeature = CategoricalFeatureIndices[catRangeIndex + 1] + 1;

                                catRangeIndex += 2;
                            }
                            else
                            {
                                foreach (var flock in CreateFlocksCore(ch, pch, iFeature, CategoricalFeatureIndices[catRangeIndex]))
                                    yield return flock;

                                iFeature = CategoricalFeatureIndices[catRangeIndex];
                            }
                        }
                        else
                        {
                            foreach (var flock in CreateFlocksCore(ch, pch, iFeature, NumFeatures))
                                yield return flock;

                            iFeature = NumFeatures;
                        }
                    }
                }
                else
                {
                    foreach (var flock in CreateFlocksCore(ch, pch, 0, NumFeatures))
                        yield return flock;
                }
            }

            private IEnumerable<FeatureFlockBase> CreateFlocksCore(IChannel ch, IProgressChannel pch, int startFeatureIndex, int featureLim)
            {
                int iFeature = startFeatureIndex;
                VBuffer<double> temp = default(VBuffer<double>);
                // Working array for bins.
                int[] binnedValues = new int[NumExamples];
                // Holds what feature for an example was last "on", that is, will have
                // to be explicitly represented. This was the last feature for which AllIndicesGE
                // returned an index.
                int[] lastOn = new int[NumExamples];
                for (int i = 0; i < lastOn.Length; ++i)
                    lastOn[i] = -1;
                int[] forwardIndexerWork = null;
                // What creations are pending?
                List<int> pending = new List<int>();

                Func<FeatureFlockBase> createOneHotFlock =
                    () => CreateOneHotFlock(ch, pending, binnedValues, lastOn, _instanceList,
                        ref forwardIndexerWork, ref temp, false);

                Func<FeatureFlockBase> createNHotFlock =
                    () => CreateNHotFlock(ch, pending);

                // The exclusive upper bound of what features have already been incorporated
                // into a flock.
                int limMade = startFeatureIndex;
                int countBins = 1; // Count of bins we'll need to represent. Starts at 1, accumulates "hot" features.
                // Tracking for n-hot flocks.
                long countHotRows = 0; // The count of hot "rows"
                long hotNThreshold = (long)(0.1 * NumExamples);
                bool canBeOneHot = true;

                Func<FeatureFlockBase> createFlock =
                    () =>
                    {
                        ch.Assert(pending.Count > 0);
                        FeatureFlockBase flock;
                        if (canBeOneHot)
                            flock = createOneHotFlock();
                        else
                            flock = createNHotFlock();
                        canBeOneHot = true;
                        limMade = iFeature;
                        pending.Clear();
                        countHotRows = 0;
                        countBins = 1;
                        return flock;
                    };

                for (; iFeature < featureLim; ++iFeature)
                {
                    Double[] bup = BinUpperBounds[iFeature];
                    Contracts.Assert(Utils.Size(bup) > 0);
                    if (bup.Length == 1)
                    {
                        // This is a trivial feature. Skip it.
                        continue;
                    }
                    ValuesList values = _instanceList[iFeature];

                    if (countBins > Utils.ArrayMaxSize - (bup.Length - 1))
                    {
                        // It can happen that a flock could be created with more than Utils.ArrayMaxSize
                        // bins, in the case where we bin over a training dataset with many features with
                        // many bins (for example, 1 million features with 10k bins each), and then in a subsequent
                        // validation dataset we have these features suddenly become one-hot. Practically
                        // this will never happen, of course, but it is still possible. If this ever happens,
                        // we create the flock before this becomes an issue.
                        ch.Assert(0 < countBins && countBins <= Utils.ArrayMaxSize);
                        ch.Assert(limMade < iFeature);
                        ch.Assert(pending.Count > 0);
                        yield return createFlock();
                    }
                    countBins += bup.Length - 1;
                    Double firstBin = bup[0];
                    int localHotRows = 0;
                    // The number of bits we would use if we incorporated the current feature in to the
                    // existing running flock.
                    IntArrayBits newBits = IntArray.NumBitsNeeded(countBins);

                    if (canBeOneHot)
                    {
                        using (IEnumerator<int> hotEnumerator = values.AllIndicesGT(NumExamples, firstBin).GetEnumerator())
                        {
                            if (pending.Count > 0)
                            {
                                // There are prior features we haven't yet flocked. So we are still contemplating
                                // "flocking" this prior feature with this feature (and possibly features beyond).
                                // The enumeration will need to run the appropriate checks.
                                while (hotEnumerator.MoveNext())
                                {
                                    int i = hotEnumerator.Current;
                                    ++localHotRows;
                                    var last = lastOn[i];
                                    Contracts.Assert(last < iFeature);
                                    if (last >= limMade)
                                    {
                                        // We've encountered an overlapping feature. We now need to decide whether we want
                                        // to continue accumulating into a flock and so make this n-hot flock, or cut it off
                                        // now and create a one-hot flock.
                                        if (countHotRows < hotNThreshold)
                                        {
                                            // We may want to create an N-hot flock.
                                            int superLocalHot = values.CountIndicesGT(NumExamples, firstBin);
                                            if (countHotRows + superLocalHot < hotNThreshold)
                                            {
                                                // If this succeeds, we want to create an N-hot flock including this.
                                                canBeOneHot = false;
                                                localHotRows = superLocalHot;
                                                break; // Future iterations will create the n-hot.
                                            }
                                            // If the test above failed, then we want to create a one-hot of [limMade, iFeature),
                                            // and keep going on this guy.
                                        }

                                        // We've decided to create a one-hot flock. Before continuing to fill in lastOn, use
                                        // lastOn in its current state to create a flock from limMade inclusive, to f
                                        // exclusive, and make "f" the new limMade. Note that we continue to fill in lastOn
                                        // once we finish this.
                                        ch.Assert(limMade < iFeature);
                                        ch.Assert(canBeOneHot);
                                        yield return createFlock();
                                        lastOn[i] = iFeature;
                                        // Now that we've made the feature there's no need continually check against lastOn[i]'s
                                        // prior values. Fall through to the limMade == iFeature case.
                                        break;
                                    }
                                    lastOn[i] = iFeature;
                                }
                            }

                            if (canBeOneHot)
                            {
                                // In the event that hotEnumerator was exhausted in the above loop, the following is a no-op.
                                while (hotEnumerator.MoveNext())
                                {
                                    // There is no prior feature to flock, so there's no need to track anything yet.
                                    // Just populate lastOn appropriately.
                                    ++localHotRows;
                                    lastOn[hotEnumerator.Current] = iFeature;
                                }
                            }
                        }
                        ch.Assert(values.CountIndicesGT(NumExamples, firstBin) == localHotRows);
                        pending.Add(iFeature); // Have not yet flocked this feature.
                    }
                    else
                    {
                        // No need to track in lastOn, since we're no longer contemplating this being one-hot.
                        ch.Assert(limMade < iFeature);
                        ch.Assert(countHotRows < hotNThreshold);
                        ch.Assert(!canBeOneHot);
                        localHotRows = values.CountIndicesGT(NumExamples, firstBin);
                        if (countHotRows + localHotRows >= hotNThreshold)
                        {
                            // Too dense if we add iFeature to the mix. Make an n-hot of [limMade, iFeature),
                            // then decrement iFeature so that we reconsider it in light of being a candidate
                            // for one-hot or singleton. Do not add to pending, as its status will be considered
                            // in the next pass.
                            yield return createFlock();
                            --iFeature;
                        }
                        else // Have not yet flocked as feature.
                            pending.Add(iFeature);
                    }
                    countHotRows += localHotRows;
                }
                Contracts.Assert(limMade < featureLim);
                if (pending.Count > 0)
                    yield return createFlock();
            }

            /// <summary>
            /// Create an artificial metadata object to pad the Dataset
            /// </summary>
            private Dataset.DatasetSkeleton CreateDatasetSkeleton()
            {
                ulong[] docIds = new ulong[_numExamples]; // All zeros is fine
                ulong[] queryIds = new ulong[_boundaries.Count - 1]; // All zeros is fine
                var ds = UsingMaxLabel
                    ? new Dataset.DatasetSkeleton(_targetsList.ToArray(), _boundaries.ToArray(), queryIds, docIds, new double[0][])
                    : new Dataset.DatasetSkeleton(_targetsList.ToArray(), _boundaries.ToArray(), queryIds, docIds, new double[0][], _actualTargets.ToArray());
                //AP TODO change it to have weights=null when dataset is unweighted in order to avoid potential long memory scan
                if (_weights != null)
                    ds.SampleWeights = _weights.ToArray();
                return ds;
            }
        }

        // REVIEW: Change this, as well as the bin finding code and bin upper bounds, to be float instead of Double.

        /// <summary>
        /// A mutable list of index,value that may be kept sparse or dense.
        /// </summary>
        private sealed class ValuesList
        {
            private bool _isSparse;
            private List<Double> _dense;
            private int _nonZeroElements; // when dense, is the number of non-zero elements (for determining when to sparsify)
            private List<KeyValuePair<int, Double>> _sparse;

            public ValuesList()
            {
                _dense = new List<Double>();
            }

            public void Add(int index, Double value)
            {
                if (!_isSparse)
                {
                    // Check if adding this element will make the array sparse.
                    if (ShouldSparsify(_nonZeroElements + 1, index + 1))
                        Sparsify();
                    else
                    {
                        // Add zeros if needed.
                        while (_dense.Count < index)
                            _dense.Add(default(Double));
                        // Add the value.
                        _dense.Add(value);
                        if (value != 0)
                            _nonZeroElements++;
                        return;
                    }
                }
                // Note this also may happen because we just sparsified.
                Contracts.Assert(_isSparse);
                if (value != 0)
                    _sparse.Add(new KeyValuePair<int, Double>(index, value));
            }

            private bool ShouldSparsify(int nonZeroElements, int totalElements)
            {
                // TODO: We need a better solution here. Also, maybe should start sparse and become dense instead?
                return (double)nonZeroElements / totalElements < 0.25 && totalElements > 10;
            }

            private void Sparsify()
            {
                _sparse = new List<KeyValuePair<int, Double>>(_nonZeroElements);
                for (int i = 0; i < _dense.Count; i++)
                {
                    if (_dense[i] != 0)
                        _sparse.Add(new KeyValuePair<int, Double>(i, _dense[i]));
                }
                _isSparse = true;
                _dense = null;
            }

            /// <summary>
            /// Returns the count of all positions greater than an indicated value.
            /// </summary>
            /// <param name="length">The limit of indices to check</param>
            /// <param name="gtValue">The value against which the greater-than
            /// comparison is made</param>
            /// <returns>The count of all indices in the range of 0 to <paramref name="length"/>
            /// exclusive whose values are greater than <paramref name="gtValue"/></returns>
            public int CountIndicesGT(int length, Double gtValue)
            {
                Contracts.Assert(0 <= length);
                if (_isSparse)
                {
                    Contracts.Assert(_sparse.Count == 0 || _sparse[_sparse.Count - 1].Key < length);
                    return _sparse.Count(kvp => kvp.Value > gtValue) + (0 > gtValue ? length - _sparse.Count : 0);
                }
                else
                {
                    Contracts.Assert(_dense.Count <= length);
                    return _dense.Count(v => v > gtValue) + (0 > gtValue ? length - _dense.Count : 0);
                }
            }

            /// <summary>
            /// Return all indices that are greater than an indicated value.
            /// </summary>
            /// <param name="lim">The limit of indices to return</param>
            /// <param name="gtValue">The value against which the greater-than
            /// comparison is made</param>
            /// <returns>All indices in the range of 0 to <paramref name="lim"/> exclusive
            /// whose values are greater than <paramref name="gtValue"/>, in
            /// increasing order</returns>
            public IEnumerable<int> AllIndicesGT(int lim, Double gtValue)
            {
                Contracts.Assert(0 <= lim);
                if (_isSparse)
                {
                    Contracts.Assert(_sparse.Count == 0 || _sparse[_sparse.Count - 1].Key < lim);
                    if (0 > gtValue)
                    {
                        // All implicitly defined sparse values will have to be returned.
                        int prev = -1;
                        foreach (var kvp in _sparse)
                        {
                            Contracts.Assert(prev < kvp.Key);
                            while (++prev < kvp.Key)
                                yield return prev;
                            if (kvp.Value > gtValue)
                                yield return kvp.Key;
                        }
                        // Return the "leftovers."
                        while (++prev < lim)
                            yield return prev;
                    }
                    else
                    {
                        // Only explicitly defined values have to be returned.
                        foreach (var kvp in _sparse)
                        {
                            if (kvp.Value > gtValue)
                                yield return kvp.Key;
                        }
                    }
                }
                else
                {
                    Contracts.Assert(_dense.Count <= lim);
                    for (int i = 0; i < _dense.Count; ++i)
                    {
                        if (_dense[i] > gtValue)
                            yield return i;
                    }
                    if (0 > gtValue)
                    {
                        // All implicitly defined post-dense values will have to be returned,
                        // assuming there are any (this set is only non-empty when listLim < lim).
                        for (int i = _dense.Count; i < lim; ++i)
                            yield return i;
                    }
                }
            }

            public void CopyTo(int length, ref VBuffer<Double> dst)
            {
                Contracts.Assert(0 <= length);
                VBufferEditor<double> editor;
                if (!_isSparse)
                {
                    Contracts.Assert(_dense.Count <= length);
                    if (ShouldSparsify(_nonZeroElements, length))
                        Sparsify();
                    else
                    {
                        editor = VBufferEditor.Create(ref dst, length);
                        if (_dense.Count < length)
                        {
                            _dense.CopyTo(editor.Values);
                            editor.Values.Slice(_dense.Count, length - _dense.Count).Clear();
                        }
                        else
                            _dense.CopyTo(editor.Values, length);
                        dst = editor.Commit();
                        return;
                    }
                }
                int count = _sparse.Count;
                Contracts.Assert(count <= length);
                editor = VBufferEditor.Create(ref dst, length, count);
                for (int i = 0; i < _sparse.Count; ++i)
                {
                    editor.Indices[i] = _sparse[i].Key;
                    editor.Values[i] = _sparse[i].Value;
                }
                Contracts.Assert(Utils.IsIncreasing(0, editor.Indices, count, length));
                dst = editor.Commit();
            }

            /// <summary>
            /// An enumerable of the row/bin pair of every non-zero bin row according to the
            /// binning passed into this method.
            /// </summary>
            /// <param name="binUpperBounds">The binning to use for the enumeration</param>
            /// <param name="length">The number of rows in this feature</param>
            /// <returns>An enumerable that returns a pair of every row-index and binned value,
            /// where the row indices are increasing, the binned values are positive</returns>
            public IEnumerable<KeyValuePair<int, int>> Binned(double[] binUpperBounds, int length)
            {
                Contracts.Assert(Utils.Size(binUpperBounds) > 0);
                Contracts.Assert(0 <= length);

                int zeroBin = Algorithms.FindFirstGE(binUpperBounds, 0);
                IntArrayBits numBitsNeeded = IntArray.NumBitsNeeded(binUpperBounds.Length);
                if (numBitsNeeded == IntArrayBits.Bits0)
                    yield break;
                if (!_isSparse)
                {
                    Contracts.Assert(_dense.Count <= length);
                    if (ShouldSparsify(_nonZeroElements, length))
                        Sparsify();
                }

                if (_isSparse)
                {
                    Contracts.AssertValue(_sparse);
                    if (zeroBin == 0)
                    {
                        // We can skip all implicit values in sparse.
                        foreach (var kvp in _sparse)
                        {
                            Contracts.Assert(kvp.Key < length);
                            int binned = Algorithms.FindFirstGE(binUpperBounds, kvp.Value);
                            if (binned > 0)
                                yield return new KeyValuePair<int, int>(kvp.Key, binned);
                        }
                        yield break;
                    }

                    Contracts.Assert(zeroBin != 0);
                    int last = -1;
                    foreach (var kvp in _sparse)
                    {
                        Contracts.Assert(kvp.Key < length);
                        while (++last < kvp.Key)
                            yield return new KeyValuePair<int, int>(last, zeroBin);
                        int binned = Algorithms.FindFirstGE(binUpperBounds, kvp.Value);
                        if (binned > 0)
                            yield return new KeyValuePair<int, int>(kvp.Key, binned);
                    }
                    while (++last < length)
                        yield return new KeyValuePair<int, int>(last, zeroBin);

                    yield break;
                }
                Contracts.Assert(!_isSparse);
                Contracts.AssertValue(_dense);
                Contracts.Assert(_dense.Count <= length);
                for (int i = 0; i < _dense.Count; ++i)
                {
                    int binned = Algorithms.FindFirstGE(binUpperBounds, _dense[i]);
                    if (binned > 0)
                        yield return new KeyValuePair<int, int>(i, binned);
                }
                if (zeroBin > 0)
                {
                    for (int i = _dense.Count; i < length; ++i)
                        yield return new KeyValuePair<int, int>(i, zeroBin);
                }
            }

            public sealed class ForwardIndexer
            {
                // All of the _values list. We are only addressing _min through _lim.
                private readonly ValuesList[] _values;
                // Parallel to the subsequence of _values in min to lim, indicates the index where
                // we should start to look for the next value, if the corresponding value list in
                // _values is sparse. If the corresponding value list is dense the entry at this
                // position is not used.
                private readonly int[] _perFeaturePosition;
                private readonly int[] _featureIndices;
#if DEBUG
                // Holds for each feature the row index that it was previously accessed on.
                // Purely for validation purposes.
                private int[] _lastRow;
#endif

                /// <summary>
                /// Access the value of a particular feature, at a particular row.
                /// </summary>
                /// <param name="featureIndex">A feature index, which indexes not the global feature indices,
                /// but the index into the subset of features specified at the constructor time</param>
                /// <param name="rowIndex">The row index to access, which must be non-decreasing, and must
                /// indeed be actually increasing for access on the same feature (for example, if you have two features,
                /// it is OK to access <c>[1, 5]</c>, then <c>[0, 5]</c>, but once this is done you cannot
                /// access the same feature at the same position.</param>
                /// <returns></returns>
                public Double this[int featureIndex, int rowIndex]
                {
                    get
                    {
                        Contracts.Assert(0 <= featureIndex && featureIndex < _featureIndices.Length);
                        Contracts.Assert(rowIndex >= 0);
                        var values = _values[_featureIndices[featureIndex]];
#if DEBUG
                        int lastRow = _lastRow[featureIndex];
                        Contracts.Assert(rowIndex > lastRow);
                        _lastRow[featureIndex] = rowIndex;
#endif
                        if (!values._isSparse)
                            return rowIndex < values._dense.Count ? values._dense[rowIndex] : 0;
                        int last = _perFeaturePosition[featureIndex];
                        var sp = values._sparse;
#if DEBUG
                        // The next value of _sparse (assuming there is one) should have been past the last access.
                        // That is, sp[last].Key, if it exist, must be greater than lastRow.
                        Contracts.Assert(sp.Count == 0 || sp[last].Key > lastRow);
#endif
                        while (last < sp.Count)
                        {
                            var s = sp[last++];
                            if (s.Key < rowIndex)
                                continue;
                            if (s.Key > rowIndex)
                            {
                                // We'd previously put last past this element,
                                // have to put it back a bit.
                                last--;
                                break;
                            }
                            Contracts.Assert(s.Key == rowIndex);
                            _perFeaturePosition[featureIndex] = last;
                            return s.Value;
                        }
                        _perFeaturePosition[featureIndex] = last;
                        return 0;
                    }
                }

                /// <summary>
                /// Initialize a forward indexer.
                /// </summary>
                /// <param name="values">Holds the values of the features</param>
                /// <param name="features">The array of feature indices this will index</param>
                /// <param name="workArray">A possibly shared working array, once used by this forward
                /// indexer it should not be used in any previously created forward indexer</param>
                public ForwardIndexer(ValuesList[] values, int[] features, ref int[] workArray)
                {
                    Contracts.AssertValue(values);
                    Contracts.AssertValueOrNull(workArray);
                    Contracts.AssertValue(features);
                    Contracts.Assert(Utils.IsIncreasing(0, features, values.Length));
                    Contracts.Assert(features.All(i => values[i] != null));
                    _values = values;
                    _featureIndices = features;
                    Utils.EnsureSize(ref workArray, _featureIndices.Length, keepOld: false);
                    Contracts.AssertValue(workArray); // Should be initialized now.
                    _perFeaturePosition = workArray;
                    Array.Clear(_perFeaturePosition, 0, _featureIndices.Length);
#if DEBUG
                    _lastRow = new int[features.Length];
                    for (int i = 0; i < _lastRow.Length; ++i)
                        _lastRow[i] = -1;
#endif
                }
            }
        }
    }

    internal sealed class ExamplesToFastTreeBins
    {
        private readonly int _maxBins;
        private readonly float _maxLabel;
        private readonly IHost _host;
        private readonly bool _diskTranspose;
        private readonly bool _noFlocks;
        private readonly int _minDocsPerLeaf;

        /// <summary> Bin boundaries </summary>
        public double[][] BinUpperBounds
        {
            get;
            private set;
        }

        public int[] FeatureMap { get; private set; }

        public ExamplesToFastTreeBins(IHostEnvironment env, int maxBins, bool diskTranspose, bool noFlocks, int minDocsPerLeaf, float maxLabel)
        {
            Contracts.AssertValue(env);
            _host = env.Register("Converter");

            _maxBins = maxBins;
            _maxLabel = maxLabel;
            _diskTranspose = diskTranspose;
            _noFlocks = noFlocks;
            _minDocsPerLeaf = minDocsPerLeaf;
        }

        public Dataset FindBinsAndReturnDataset(RoleMappedData data, PredictionKind kind, IParallelTraining parallelTraining,
            int[] categoricalFeaturIndices, bool categoricalSplit)
        {
            using (var ch = _host.Start("InitDataset"))
            {
                ch.Info("Making per-feature arrays");
                var convData = DataConverter.Create(data, _host, _maxBins, _maxLabel, _diskTranspose, _noFlocks,
                    _minDocsPerLeaf, kind, parallelTraining, categoricalFeaturIndices, categoricalSplit);

                ch.Info("Processed {0} instances", convData.NumExamples);
                ch.Info("Binning and forming Feature objects");
                Dataset d = convData.GetDataset();
                BinUpperBounds = convData.BinUpperBounds;
                FeatureMap = convData.FeatureMap;
                return d;
            }
        }

        public Dataset GetCompatibleDataset(RoleMappedData data, PredictionKind kind, int[] categoricalFeatures, bool categoricalSplit)
        {
            _host.AssertValue(BinUpperBounds);
            var convData = DataConverter.Create(data, _host, BinUpperBounds, _maxLabel, _diskTranspose, _noFlocks, kind,
                categoricalFeatures, categoricalSplit);

            return convData.GetDataset();
        }
    }

    public abstract class TreeEnsembleModelParameters :
        ModelParametersBase<float>,
        IValueMapper,
        ICanSaveInTextFormat,
        ICanSaveInIniFormat,
        ICanSaveInSourceCode,
        ICanSaveSummary,
        ICanGetSummaryInKeyValuePairs,
        ITreeEnsemble,
        IPredictorWithFeatureWeights<float>,
        IFeatureContributionMapper,
        ICalculateFeatureContribution,
        ICanGetSummaryAsIRow,
        ISingleCanSavePfa,
        ISingleCanSaveOnnx
    {
        // The below two properties are necessary for tree Visualizer
        [BestFriend]
        internal InternalTreeEnsemble TrainedEnsemble { get; }

        int ITreeEnsemble.NumTrees => TrainedEnsemble.NumTrees;

        // Inner args is used only for documentation purposes when saving comments to INI files.
        private protected readonly string InnerOptions;

        // The total number of features used in training (takes the value of zero if the
        // written version of the loaded model is less than VerNumFeaturesSerialized)
        private protected readonly int NumFeatures;

        // Maximum index of the split features of trainedEnsemble trees
        private protected readonly int MaxSplitFeatIdx;

        private protected abstract uint VerNumFeaturesSerialized { get; }

        private protected abstract uint VerDefaultValueSerialized { get; }

        private protected abstract uint VerCategoricalSplitSerialized { get; }

        internal readonly DataViewType InputType;
        DataViewType IValueMapper.InputType => InputType;

        protected readonly DataViewType OutputType;
        DataViewType IValueMapper.OutputType => OutputType;

        bool ICanSavePfa.CanSavePfa => true;

        bool ICanSaveOnnx.CanSaveOnnx(OnnxContext ctx) => true;

        /// <summary>
        /// Used to determine the contribution of each feature to the score of an example by <see cref="FeatureContributionCalculatingTransformer"/>.
        /// The calculation of feature contribution essentially consists in determining which splits in the tree have the most impact
        /// on the final score and assigning the value of the impact to the features determining the split. More precisely, the contribution of a feature
        /// is equal to the change in score produced by exploring the opposite sub-tree every time a decision node for the given feature is encountered.
        /// Consider a simple case with a single decision tree that has a decision node for the binary feature F1. Given an example that has feature F1
        /// equal to true, we can calculate the score it would have obtained if we chose the subtree corresponding to the feature F1 being equal to false
        /// while keeping the other features constant. The contribution of feature F1 for the given example is the difference between the original score
        /// and the score obtained by taking the opposite decision at the node corresponding to feature F1. This algorithm extends naturally to models with
        /// many decision trees.
        /// </summary>
        FeatureContributionCalculator ICalculateFeatureContribution.FeatureContributionCalculator => new FeatureContributionCalculator(this);

        /// The following function is used in both FastTree and LightGBM so <see cref="BestFriendAttribute"/> is required.
        [BestFriend]
        private protected TreeEnsembleModelParameters(IHostEnvironment env, string name, InternalTreeEnsemble trainedEnsemble, int numFeatures, string innerArgs)
            : base(env, name)
        {
            Host.CheckValue(trainedEnsemble, nameof(trainedEnsemble));
            Host.CheckParam(numFeatures > 0, nameof(numFeatures), "must be positive");
            Host.CheckValueOrNull(innerArgs);

            // REVIEW: When we make the predictor wrapper, we may want to further "optimize"
            // the trained ensemble to, for instance, resize arrays so that they are of the length
            // the actual number of leaves/nodes, or remove unnecessary arrays, and so forth.
            TrainedEnsemble = trainedEnsemble;
            InnerOptions = innerArgs;
            NumFeatures = numFeatures;

            MaxSplitFeatIdx = trainedEnsemble.GetMaxFeatureIndex();
            Contracts.Assert(NumFeatures > MaxSplitFeatIdx);

            InputType = new VectorType(NumberDataViewType.Single, NumFeatures);
            OutputType = NumberDataViewType.Single;
        }

        private protected TreeEnsembleModelParameters(IHostEnvironment env, string name, ModelLoadContext ctx, VersionInfo ver)
            : base(env, name, ctx)
        {
            // *** Binary format ***
            // Ensemble
            // int: Inner args string id
            // int: Number of features (VerNumFeaturesSerialized)
            // <PredictionKind> specific stuff
            ctx.CheckVersionInfo(ver);
            bool usingDefaultValues = false;
            bool categoricalSplits = false;
            if (ctx.Header.ModelVerWritten >= VerDefaultValueSerialized)
                usingDefaultValues = true;

            if (ctx.Header.ModelVerWritten >= VerCategoricalSplitSerialized)
                categoricalSplits = true;

            TrainedEnsemble = new InternalTreeEnsemble(ctx, usingDefaultValues, categoricalSplits);
            MaxSplitFeatIdx = TrainedEnsemble.GetMaxFeatureIndex();

            InnerOptions = ctx.LoadStringOrNull();
            if (ctx.Header.ModelVerWritten >= VerNumFeaturesSerialized)
            {
                NumFeatures = ctx.Reader.ReadInt32();
                // It is possible that the number of features is 0 when an old model is loaded and then saved with the new version.
                Host.CheckDecode(NumFeatures >= 0);
            }

            // In the days of TLC <= 2.7 before we had a data pipeline, there was
            // some auxiliary structure called the "ContentMap." This structure is
            // no longer necessary or helpful since the data pipeline is in
            // TLC >= 3.0 supposed to be independent of any predictor specific
            // tricks.

            InputType = new VectorType(NumberDataViewType.Single, NumFeatures);
            OutputType = NumberDataViewType.Single;
        }

        [BestFriend]
        private protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);

            // *** Binary format ***
            // Ensemble
            // int: Inner args string id
            // int: Number of features (VerNumFeaturesSerialized)
            // <PredictionKind> specific stuff
            TrainedEnsemble.Save(ctx);
            ctx.SaveStringOrNull(InnerOptions);
            Host.Assert(NumFeatures >= 0);
            ctx.Writer.Write(NumFeatures);
        }

        ValueMapper<TIn, TOut> IValueMapper.GetMapper<TIn, TOut>()
        {
            Host.Check(typeof(TIn) == typeof(VBuffer<float>));
            Host.Check(typeof(TOut) == typeof(float));

            ValueMapper<VBuffer<float>, float> del = Map;
            return (ValueMapper<TIn, TOut>)(Delegate)del;
        }

        private protected virtual void Map(in VBuffer<float> src, ref float dst)
        {
            int inputVectorSize = InputType.GetVectorSize();
            if (inputVectorSize > 0)
                Host.Check(src.Length == inputVectorSize);
            else
                Host.Check(src.Length > MaxSplitFeatIdx);

            dst = (float)TrainedEnsemble.GetOutput(in src);
        }

        ValueMapper<TSrc, VBuffer<float>> IFeatureContributionMapper.GetFeatureContributionMapper<TSrc, TDst>(int top, int bottom, bool normalize)
        {
            Host.Check(typeof(TSrc) == typeof(VBuffer<float>));
            Host.Check(typeof(TDst) == typeof(VBuffer<float>));
            Host.Check(top >= 0, "top must be non-negative");
            Host.Check(bottom >= 0, "bottom must be non-negative");

            BufferBuilder<float> builder = null;
            ValueMapper<VBuffer<float>, VBuffer<float>> del =
                (in VBuffer<float> src, ref VBuffer<float> dst) =>
                {
                    FeatureContributionMap(in src, ref dst, ref builder);
                    Numeric.VectorUtils.SparsifyNormalize(ref dst, top, bottom, normalize);
                };
            return (ValueMapper<TSrc, VBuffer<float>>)(Delegate)del;
        }

        private void FeatureContributionMap(in VBuffer<float> src, ref VBuffer<float> dst, ref BufferBuilder<float> builder)
        {
            int inputVectorSize = InputType.GetVectorSize();
            if (inputVectorSize > 0)
                Host.Check(src.Length == inputVectorSize);
            else
                Host.Check(src.Length > MaxSplitFeatIdx);

            TrainedEnsemble.GetFeatureContributions(in src, ref dst, ref builder);
        }

        /// <summary>
        /// write out a C# representation of the ensemble
        /// </summary>
        void ICanSaveInSourceCode.SaveAsCode(TextWriter writer, RoleMappedSchema schema)
        {
            Host.CheckValueOrNull(schema);
            SaveEnsembleAsCode(writer, schema);
        }

        /// <summary>
        /// Output the INI model to a given writer
        /// </summary>
        void ICanSaveInTextFormat.SaveAsText(TextWriter writer, RoleMappedSchema schema)
        {
            Host.CheckValue(writer, nameof(writer));
            Host.CheckValueOrNull(schema);
            ((ICanSaveInIniFormat)this).SaveAsIni(writer, schema);
        }

        /// <summary>
        /// Output the INI model to a given writer
        /// </summary>
        void ICanSaveInIniFormat.SaveAsIni(TextWriter writer, RoleMappedSchema schema, ICalibrator calibrator)
        {
            Host.CheckValue(writer, nameof(writer));
            var ensembleIni = FastTreeIniFileUtils.TreeEnsembleToIni(Host, TrainedEnsemble, schema, calibrator,
                InnerOptions, appendFeatureGain: true, includeZeroGainFeatures: false);
            writer.WriteLine(ensembleIni);
        }

        JToken ISingleCanSavePfa.SaveAsPfa(BoundPfaContext ctx, JToken input)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.CheckValue(input, nameof(input));
            return TrainedEnsemble.AsPfa(ctx, input);
        }

        private enum NodeMode
        {
            [Description("BRANCH_LEQ")]
            BranchLEq,
            [Description("BRANCH_LT")]
            BranchLT,
            [Description("BRANCH_GTE")]
            BranchGte,
            [Description("BRANCH_GT")]
            BranchGT,
            [Description("BRANCH_EQ")]
            BranchEq,
            [Description("BRANCH_LT")]
            BranchNeq,
            [Description("LEAF")]
            Leaf
        };

        private enum PostTransform
        {
            [Description("NONE")]
            None,
            [Description("SOFTMAX")]
            SoftMax,
            [Description("LOGISTIC")]
            Logstic,
            [Description("SOFTMAX_ZERO")]
            SoftMaxZero
        }

        private enum AggregateFunction
        {
            [Description("AVERAGE")]
            Average,
            [Description("SUM")]
            Sum,
            [Description("MIN")]
            Min,
            [Description("MAX")]
            Max
        }

        bool ISingleCanSaveOnnx.SaveAsOnnx(OnnxContext ctx, string[] outputNames, string featureColumn)
        {
            Host.CheckValue(ctx, nameof(ctx));

            //Nodes.
            var nodesTreeids = new List<long>();
            var nodesIds = new List<long>();
            var nodesFeatureIds = new List<long>();
            var nodeModes = new List<string>();
            var nodesValues = new List<double>();
            var nodeHitrates = new List<long>();
            var missingValueTracksTrue = new List<bool>();
            var nodesTrueNodeIds = new List<long>();
            var nodesFalseNodeIds = new List<long>();
            var nodesBaseValues = new List<float>();

            //Leafs.
            var classTreeIds = new List<long>();
            var classNodeIds = new List<long>();
            var classIds = new List<long>();
            var classWeights = new List<double>();

            int treeIndex = -1;
            foreach (var tree in TrainedEnsemble.Trees)
            {
                treeIndex++;
                for (int nodeIndex = 0; nodeIndex < tree.NumNodes; nodeIndex++)
                {
                    nodesTreeids.Add(treeIndex);
                    nodeModes.Add(NodeMode.BranchLEq.GetDescription());
                    nodesIds.Add(nodeIndex);
                    nodesFeatureIds.Add(tree.SplitFeature(nodeIndex));
                    nodesValues.Add(tree.RawThresholds[nodeIndex]);
                    nodesTrueNodeIds.Add(tree.LteChild[nodeIndex] < 0 ? ~tree.LteChild[nodeIndex] + tree.NumNodes : tree.LteChild[nodeIndex]);
                    nodesFalseNodeIds.Add(tree.GtChild[nodeIndex] < 0 ? ~tree.GtChild[nodeIndex] + tree.NumNodes : tree.GtChild[nodeIndex]);
                    if (tree.DefaultValueForMissing?[nodeIndex] <= tree.RawThresholds[nodeIndex])
                        missingValueTracksTrue.Add(true);
                    else
                        missingValueTracksTrue.Add(false);

                    nodeHitrates.Add(0);
                }

                for (int leafIndex = 0; leafIndex < tree.NumLeaves; leafIndex++)
                {
                    int nodeIndex = tree.NumNodes + leafIndex;
                    nodesTreeids.Add(treeIndex);
                    nodesBaseValues.Add(0);
                    nodeModes.Add(NodeMode.Leaf.GetDescription());
                    nodesIds.Add(nodeIndex);
                    nodesFeatureIds.Add(0);
                    nodesValues.Add(0);
                    nodesTrueNodeIds.Add(0);
                    nodesFalseNodeIds.Add(0);
                    missingValueTracksTrue.Add(false);
                    nodeHitrates.Add(0);

                    classTreeIds.Add(treeIndex);
                    classNodeIds.Add(nodeIndex);
                    classIds.Add(0);
                    classWeights.Add(tree.LeafValues[leafIndex]);
                }
            }

            string opType = "TreeEnsembleRegressor";
            var node = ctx.CreateNode(opType, new[] { featureColumn }, outputNames, ctx.GetNodeName(opType));

            node.AddAttribute("post_transform", PostTransform.None.GetDescription());
            node.AddAttribute("n_targets", 1);
            node.AddAttribute("base_values", new List<float>() { 0 });
            node.AddAttribute("aggregate_function", AggregateFunction.Sum.GetDescription());
            node.AddAttribute("nodes_treeids", nodesTreeids);
            node.AddAttribute("nodes_nodeids", nodesIds);
            node.AddAttribute("nodes_featureids", nodesFeatureIds);
            node.AddAttribute("nodes_modes", nodeModes);
            node.AddAttribute("nodes_values", nodesValues);
            node.AddAttribute("nodes_truenodeids", nodesTrueNodeIds);
            node.AddAttribute("nodes_falsenodeids", nodesFalseNodeIds);
            node.AddAttribute("nodes_missing_value_tracks_true", missingValueTracksTrue);
            node.AddAttribute("target_treeids", classTreeIds);
            node.AddAttribute("target_nodeids", classNodeIds);
            node.AddAttribute("target_ids", classIds);
            node.AddAttribute("target_weights", classWeights);

            return true;
        }

        void ICanSaveSummary.SaveSummary(TextWriter writer, RoleMappedSchema schema)
        {
            writer.WriteLine();
            writer.WriteLine("Per-feature gain summary for the boosted tree ensemble:");

            foreach (var pair in ((ICanGetSummaryInKeyValuePairs)this).GetSummaryInKeyValuePairs(schema))
            {
                Host.Assert(pair.Value is Double);
                writer.WriteLine("\t{0}\t{1}", pair.Key, (Double)pair.Value);
            }
        }

        private IEnumerable<KeyValuePair<string, Double>> GetSortedFeatureGains(RoleMappedSchema schema)
        {
            var gainMap = new FeatureToGainMap(TrainedEnsemble.Trees.ToList(), normalize: true);

            var names = default(VBuffer<ReadOnlyMemory<char>>);
            AnnotationUtils.GetSlotNames(schema, RoleMappedSchema.ColumnRole.Feature, NumFeatures, ref names);
            var ordered = gainMap.OrderByDescending(pair => pair.Value);
            Double max = ordered.FirstOrDefault().Value;
            Double normFactor = max == 0 ? 1.0 : (1.0 / Math.Sqrt(max));
            foreach (var pair in ordered)
            {
                var name = names.GetItemOrDefault(pair.Key).ToString();
                if (string.IsNullOrEmpty(name))
                    name = $"f{pair.Key}";
                yield return new KeyValuePair<string, Double>(name, Math.Sqrt(pair.Value) * normFactor);
            }
        }

        ///<inheritdoc/>
        IList<KeyValuePair<string, object>> ICanGetSummaryInKeyValuePairs.GetSummaryInKeyValuePairs(RoleMappedSchema schema)
        {
            List<KeyValuePair<string, object>> results = new List<KeyValuePair<string, object>>();

            var ordered = GetSortedFeatureGains(schema);
            foreach (var pair in ordered)
                results.Add(new KeyValuePair<string, object>(pair.Key, pair.Value));
            return results;
        }

        /// <summary>
        /// returns a C# representation of the ensemble
        /// </summary>
        private void SaveEnsembleAsCode(TextWriter writer, RoleMappedSchema schema)
        {
            Host.AssertValueOrNull(schema);

            var names = default(VBuffer<ReadOnlyMemory<char>>);
            AnnotationUtils.GetSlotNames(schema, RoleMappedSchema.ColumnRole.Feature, NumFeatures, ref names);

            int i = 0;
            foreach (InternalRegressionTree tree in TrainedEnsemble.Trees)
            {
                writer.Write("double treeOutput{0}=", i);
                SaveTreeAsCode(tree, writer, in names);
                writer.Write(";\n");
                i++;
            }
            writer.Write("double output = ");
            for (int j = 0; j < i; j++)
                writer.Write((j > 0 ? "+" : "") + "treeOutput" + j);
            writer.Write(";");
        }

        /// <summary>
        /// Convert a single tree to code, called recursively
        /// </summary>
        private void SaveTreeAsCode(InternalRegressionTree tree, TextWriter writer, in VBuffer<ReadOnlyMemory<char>> names)
        {
            ToCSharp(tree, writer, 0, in names);
        }

        // converts a subtree into a C# expression
        private void ToCSharp(InternalRegressionTree tree, TextWriter writer, int node, in VBuffer<ReadOnlyMemory<char>> names)
        {
            if (node < 0)
            {
                writer.Write(FloatUtils.ToRoundTripString(tree.LeafValue(~node)));
                //_output[~node].ToString());
            }
            else
            {
                var name = names.GetItemOrDefault(tree.SplitFeature(node)).ToString();
                if (string.IsNullOrEmpty(name))
                    name = $"f{tree.SplitFeature(node)}";

                writer.Write("(({0} > {1}) ? ", name, FloatUtils.ToRoundTripString(tree.RawThreshold(node)));
                ToCSharp(tree, writer, tree.GetGtChildForNode(node), in names);
                writer.Write(" : ");
                ToCSharp(tree, writer, tree.GetLteChildForNode(node), in names);
                writer.Write(")");
            }
        }

        public void GetFeatureWeights(ref VBuffer<float> weights)
        {
            var numFeatures = Math.Max(NumFeatures, MaxSplitFeatIdx + 1);
            FeatureToGainMap gainMap = new FeatureToGainMap(TrainedEnsemble.Trees.ToList(), normalize: true);

            // If there are no trees or no splits, there are no gains.
            if (gainMap.Count == 0)
            {
                VBufferUtils.Resize(ref weights, numFeatures, 0);
                return;
            }

            Double max = gainMap.Values.Max();
            Double normFactor = max == 0 ? 1.0 : (1.0 / Math.Sqrt(max));
            var bldr = new BufferBuilder<float>(R4Adder.Instance);
            bldr.Reset(numFeatures, false);
            foreach (var pair in gainMap)
                bldr.AddFeature(pair.Key, (float)(Math.Sqrt(pair.Value) * normFactor));
            bldr.GetResult(ref weights);
        }

        ITree[] ITreeEnsemble.GetTrees()
        {
            return TrainedEnsemble.Trees.Select(k => new Tree(k)).ToArray();
        }

        public float GetLeafValue(int treeId, int leafId)
        {
            return (float)TrainedEnsemble.GetTreeAt(treeId).LeafValue(leafId);
        }

        /// <summary>
        /// Returns the leaf node in the requested tree for the given feature vector, and populates 'path' with the list of
        /// internal nodes in the path from the root to that leaf. If 'path' is null a new list is initialized. All elements
        /// in 'path' are cleared before filling in the current path nodes.
        /// </summary>
        public int GetLeaf(int treeId, in VBuffer<float> features, ref List<int> path)
        {
            return TrainedEnsemble.GetTreeAt(treeId).GetLeaf(in features, ref path);
        }

        DataViewRow ICanGetSummaryAsIRow.GetSummaryIRowOrNull(RoleMappedSchema schema)
        {
            var names = default(VBuffer<ReadOnlyMemory<char>>);
            AnnotationUtils.GetSlotNames(schema, RoleMappedSchema.ColumnRole.Feature, NumFeatures, ref names);
            var metaBuilder = new DataViewSchema.Annotations.Builder();
            metaBuilder.AddSlotNames(NumFeatures, names.CopyTo);

            var weights = default(VBuffer<Single>);
            ((IHaveFeatureWeights)this).GetFeatureWeights(ref weights);
            var builder = new DataViewSchema.Annotations.Builder();
            builder.Add<VBuffer<float>>("Gains", new VectorType(NumberDataViewType.Single, NumFeatures), weights.CopyTo, metaBuilder.ToAnnotations());

            return AnnotationUtils.AnnotationsAsRow(builder.ToAnnotations());
        }

        DataViewRow ICanGetSummaryAsIRow.GetStatsIRowOrNull(RoleMappedSchema schema)
        {
            return null;
        }

        private sealed class Tree : ITree<VBuffer<float>>
        {
            private readonly InternalRegressionTree _regTree;

            public Tree(InternalRegressionTree regTree)
            {
                _regTree = regTree;
            }

            public int[] GtChild => _regTree.GtChild;

            public int[] LteChild => _regTree.LteChild;

            public int NumNodes => _regTree.NumNodes;

            public int NumLeaves => _regTree.NumLeaves;

            public int GetLeaf(in VBuffer<float> feat)
            {
                return _regTree.GetLeaf(in feat);
            }

            public INode GetNode(int nodeId, bool isLeaf, IEnumerable<string> featuresNames = null)
            {
                var keyValues = new Dictionary<string, object>();
                if (isLeaf)
                {
                    keyValues.Add(NodeKeys.LeafValue, _regTree.LeafValue(nodeId));
                }
                else
                {
                    if (featuresNames != null)
                    {
                        if (featuresNames is FeatureNameCollection features)
                        {
                            if (_regTree.CategoricalSplit[nodeId])
                            {
                                string featureList = string.Join(" OR \n",
                                    _regTree.CategoricalSplitFeatures[nodeId].Select(feature => features[feature]));

                                keyValues.Add(NodeKeys.SplitName, featureList);
                            }
                            else
                                keyValues.Add(NodeKeys.SplitName, features[_regTree.SplitFeature(nodeId)]);
                        }
                    }
                    keyValues.Add(NodeKeys.Threshold, string.Format("<= {0}", _regTree.RawThreshold(nodeId)));
                    if (_regTree.SplitGains != null)
                        keyValues.Add(NodeKeys.SplitGain, _regTree.SplitGains[nodeId]);
                    if (_regTree.GainPValues != null)
                        keyValues.Add(NodeKeys.GainValue, _regTree.GainPValues[nodeId]);
                    if (_regTree.PreviousLeafValues != null)
                        keyValues.Add(NodeKeys.PreviousLeafValue, _regTree.PreviousLeafValues[nodeId]);
                }

                return new TreeNode(keyValues);
            }

            public double GetLeafValue(int leafId)
            {
                return _regTree.LeafValue(leafId);
            }
        }

        private sealed class TreeNode : INode
        {
            public TreeNode(Dictionary<string, object> keyValues)
            {
                KeyValues = keyValues;
            }

            public Dictionary<string, object> KeyValues { get; }
        }
    }

    /// <summary>
    /// <see cref="TreeEnsembleModelParametersBasedOnRegressionTree"/> is derived from
    /// <see cref="TreeEnsembleModelParameters"/> plus a strongly-typed public attribute,
    /// <see cref="TrainedTreeEnsemble"/>, for exposing trained model's details to users.
    /// Its function, <see cref="CreateTreeEnsembleFromInternalDataStructure"/>, is
    /// called to create <see cref="TrainedTreeEnsemble"/> inside <see cref="TreeEnsembleModelParameters"/>.
    /// Note that the major difference between <see cref="TreeEnsembleModelParametersBasedOnQuantileRegressionTree"/>
    /// and <see cref="TreeEnsembleModelParametersBasedOnRegressionTree"/> is the type of
    /// <see cref="TrainedTreeEnsemble"/>.
    /// </summary>
    public abstract class TreeEnsembleModelParametersBasedOnRegressionTree : TreeEnsembleModelParameters
    {
        /// <summary>
        /// An ensemble of trees exposed to users. It is a wrapper on the <see langword="internal"/>
        /// <see cref="InternalTreeEnsemble"/> in <see cref="ML.Trainers.FastTree.TreeEnsemble{T}"/>.
        /// </summary>
        public RegressionTreeEnsemble TrainedTreeEnsemble { get; }

        [BestFriend]
        private protected TreeEnsembleModelParametersBasedOnRegressionTree(IHostEnvironment env, string name, InternalTreeEnsemble trainedEnsemble, int numFeatures, string innerArgs)
            : base(env, name, trainedEnsemble, numFeatures, innerArgs)
        {
            TrainedTreeEnsemble = CreateTreeEnsembleFromInternalDataStructure();
        }

        [BestFriend]
        private protected TreeEnsembleModelParametersBasedOnRegressionTree(IHostEnvironment env, string name, ModelLoadContext ctx, VersionInfo ver)
            : base(env, name, ctx, ver)
        {
            TrainedTreeEnsemble = CreateTreeEnsembleFromInternalDataStructure();
        }

        private RegressionTreeEnsemble CreateTreeEnsembleFromInternalDataStructure()
        {
            var trees = TrainedEnsemble.Trees.Select(tree => new RegressionTree(tree));
            var treeWeights = TrainedEnsemble.Trees.Select(tree => tree.Weight);
            return new RegressionTreeEnsemble(trees, treeWeights, TrainedEnsemble.Bias);
        }
    }

    /// <summary>
    /// <see cref="TreeEnsembleModelParametersBasedOnQuantileRegressionTree"/> is derived from
    /// <see cref="TreeEnsembleModelParameters"/> plus a strongly-typed public attribute,
    /// <see cref="TrainedTreeEnsemble"/>, for exposing trained model's details to users.
    /// Its function, <see cref="CreateTreeEnsembleFromInternalDataStructure"/>, is
    /// called to create <see cref="TrainedTreeEnsemble"/> inside <see cref="TreeEnsembleModelParameters"/>.
    /// Note that the major difference between <see cref="TreeEnsembleModelParametersBasedOnQuantileRegressionTree"/>
    /// and <see cref="TreeEnsembleModelParametersBasedOnRegressionTree"/> is the type of
    /// <see cref="TrainedTreeEnsemble"/>.
    /// </summary>
    public abstract class TreeEnsembleModelParametersBasedOnQuantileRegressionTree : TreeEnsembleModelParameters
    {
        /// <summary>
        /// An ensemble of trees exposed to users. It is a wrapper on the <see langword="internal"/>
        /// <see cref="InternalTreeEnsemble"/> in <see cref="ML.Trainers.FastTree.TreeEnsemble{T}"/>.
        /// </summary>
        public QuantileRegressionTreeEnsemble TrainedTreeEnsemble { get; }

        [BestFriend]
        private protected TreeEnsembleModelParametersBasedOnQuantileRegressionTree(IHostEnvironment env, string name, InternalTreeEnsemble trainedEnsemble, int numFeatures, string innerArgs)
            : base(env, name, trainedEnsemble, numFeatures, innerArgs)
        {
            TrainedTreeEnsemble = CreateTreeEnsembleFromInternalDataStructure();
        }

        private protected TreeEnsembleModelParametersBasedOnQuantileRegressionTree(IHostEnvironment env, string name, ModelLoadContext ctx, VersionInfo ver)
            : base(env, name, ctx, ver)
        {
            TrainedTreeEnsemble = CreateTreeEnsembleFromInternalDataStructure();
        }

        private QuantileRegressionTreeEnsemble CreateTreeEnsembleFromInternalDataStructure()
        {
            var trees = TrainedEnsemble.Trees.Select(tree => new QuantileRegressionTree((InternalQuantileRegressionTree)tree));
            var treeWeights = TrainedEnsemble.Trees.Select(tree => tree.Weight);
            return new QuantileRegressionTreeEnsemble(trees, treeWeights, TrainedEnsemble.Bias);
        }
    }
}
