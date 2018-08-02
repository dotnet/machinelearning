// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.Runtime.Internal.CpuMath;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.FastTree.Internal
{
#if USE_SINGLE_PRECISION
    using FloatType = System.Single;
#else
    using FloatType = System.Double;
#endif

    /// <summary>
    /// Trains regression trees.
    /// </summary>
    public class LeastSquaresRegressionTreeLearner : TreeLearner, ILeafSplitStatisticsCalculator
    {
        // parameters
        protected readonly int MinDocsInLeaf;
        protected readonly int MinDocsInLeafGlobal;
        protected readonly bool AllowEmptyTrees;

        protected readonly double EntropyCoefficient;
        protected readonly double FeatureFirstUsePenalty;
        protected readonly double FeatureReusePenalty;
        protected readonly double SoftmaxTemperature;
        protected readonly double GainConfidenceInSquaredStandardDeviations;
        public readonly double MinDocsPercentageForCategoricalSplit;
        public readonly int MinDocsForCategoricalSplit;
        public readonly Bundle Bundling;
        public readonly double Bias;

        // Multithread task to find best threshold.
        private IThreadTask _calculateLeafSplitCandidates;

        protected SplitInfo[] BestSplitInfoPerLeaf;
        protected HashSet<int> CategoricalThresholds;

        // Reusable data structures that contain the temp memory used when searching for the best
        // feature/threshold for a certain leaf. This data structure is allocated once, and reused again and again,
        // to prevent repeated reallocation and garbage-collection. We keep two of these around, since we
        // typically search for the best feature/threshold for two leaves in parallel
        protected readonly LeafSplitCandidates SmallerChildSplitCandidates;
        protected readonly LeafSplitCandidates LargerChildSplitCandidates;

        // histogram arrays, used to cache histograms for future use
        protected readonly MappedObjectPool<SufficientStatsBase[]> HistogramArrayPool;
        protected SufficientStatsBase[] ParentHistogramArray;
        protected SufficientStatsBase[] SmallerChildHistogramArray;
        protected SufficientStatsBase[] LargerChildHistogramArray;

        // which features are active
        protected bool[] ActiveFeatures;

        // how many times each feature has been split on, for diversity penalty
        protected readonly int[] FeatureUseCount;

        protected readonly Random Rand;

        protected readonly double SplitFraction;
        protected readonly bool FilterZeros;
        protected readonly double BsrMaxTreeOutput;

        // size of reserved memory
        private readonly long _sizeOfReservedMemory;

        private IParallelTraining _parallelTraining;

        public int MaxCategoricalGroupsPerNode { get; }

        public int MaxCategoricalSplitPointsPerNode { get; }

        /// <summary>
        /// Creates a new LeastSquaresRegressionTreeLearner
        /// </summary>
        /// <param name="trainData">Data to train from</param>
        /// <param name="numLeaves">Maximum leaves in tree</param>
        /// <param name="minDocsInLeaf">Minimum allowable documents in leaf</param>
        /// <param name="entropyCoefficient">Add the information gain of a split to the gain
        /// times this value. Practically, this will favor more balanced splits</param>
        /// <param name="featureFirstUsePenalty">Features never used before effectively
        /// have this amount subtracted from their gain</param>
        /// <param name="featureReusePenalty">Features used before effectively have
        /// this amount subtracted from their gain</param>
        /// <param name="softmaxTemperature">Regularization parameter, where we become
        /// increasingly likely to select a non-optimal split feature the higher the
        /// temperature is</param>
        /// <param name="histogramPoolSize">Number of feature histograms to cache</param>
        /// <param name="randomSeed">The seed to use for sampling</param>
        /// <param name="splitFraction"></param>
        /// <param name="filterZeros">Whether we should ignore zero lambdas for the
        /// purpose of tree learning (generally a bad idea except for when zero indicates
        /// that the document should be ignored)</param>
        /// <param name="allowEmptyTrees">If false, failure to split the root will result in error, or if
        /// true will result in null being returned when we try to fit a tree</param>
        /// <param name="gainConfidenceLevel">Only consider a gain if its likelihood versus a random
        /// choice gain is above a certain value (so 0.95 would mean restricting to gains that have less
        /// than a 0.05 change of being generated randomly through choice of a random split).</param>
        /// <param name="maxCategoricalGroupsPerNode">Maximum categorical split points to consider when splitting on a
        /// categorical feature.</param>
        /// <param name="maxCategoricalSplitPointPerNode"></param>
        /// <param name="bsrMaxTreeOutput">-1 if best step ranking is to be disabled, otherwise it
        /// is interpreted as being similar to the maximum output for a leaf</param>
        /// <param name="parallelTraining"></param>
        /// <param name="minDocsPercentageForCategoricalSplit"></param>
        /// <param name="bundling"></param>
        /// <param name="minDocsForCategoricalSplit"></param>
        /// <param name="bias"></param>
        public LeastSquaresRegressionTreeLearner(Dataset trainData, int numLeaves, int minDocsInLeaf, double entropyCoefficient,
            double featureFirstUsePenalty, double featureReusePenalty, double softmaxTemperature, int histogramPoolSize,
            int randomSeed, double splitFraction, bool filterZeros, bool allowEmptyTrees, double gainConfidenceLevel,
            int maxCategoricalGroupsPerNode, int maxCategoricalSplitPointPerNode,
            double bsrMaxTreeOutput, IParallelTraining parallelTraining, double minDocsPercentageForCategoricalSplit,
            Bundle bundling, int minDocsForCategoricalSplit, double bias)
            : base(trainData, numLeaves)
        {
            MinDocsInLeaf = minDocsInLeaf;
            AllowEmptyTrees = allowEmptyTrees;
            EntropyCoefficient = entropyCoefficient * 1e-6;
            FeatureFirstUsePenalty = featureFirstUsePenalty;
            FeatureReusePenalty = featureReusePenalty;
            SoftmaxTemperature = softmaxTemperature;
            MaxCategoricalGroupsPerNode = maxCategoricalGroupsPerNode;
            MaxCategoricalSplitPointsPerNode = maxCategoricalSplitPointPerNode;
            BestSplitInfoPerLeaf = new SplitInfo[numLeaves];
            MinDocsPercentageForCategoricalSplit = minDocsPercentageForCategoricalSplit;
            MinDocsForCategoricalSplit = minDocsForCategoricalSplit;
            Bundling = bundling;
            Bias = bias;

            _calculateLeafSplitCandidates = ThreadTaskManager.MakeTask(
                FindBestThresholdForFlockThreadWorker, TrainData.NumFlocks);

            // allocate histogram pool and calculate reserved memory size.
            _sizeOfReservedMemory = 0L;

            SufficientStatsBase[][] histogramPool = new SufficientStatsBase[histogramPoolSize][];
            for (int i = 0; i < histogramPoolSize; i++)
            {
                histogramPool[i] = new SufficientStatsBase[TrainData.NumFlocks];
                for (int j = 0; j < TrainData.NumFlocks; j++)
                {
                    var ss = histogramPool[i][j] = TrainData.Flocks[j].CreateSufficientStats(HasWeights);
                    _sizeOfReservedMemory += ss.SizeInBytes();
                }

                // Each reference has the same size as IntPtr, so we estimate
                // histogramPool[i] uses IntPtr.Size * TrainData.NumFlocks bytes.
                _sizeOfReservedMemory += (long)IntPtr.Size * TrainData.NumFlocks;
            }

            // wrap the pool of histogram arrays in a MappedObjectPool object
            HistogramArrayPool = new MappedObjectPool<SufficientStatsBase[]>(histogramPool, numLeaves - 1);

            MakeSplitCandidateArrays(TrainData, out SmallerChildSplitCandidates, out LargerChildSplitCandidates);

            FeatureUseCount = new int[TrainData.NumFeatures];
            Rand = new Random(randomSeed);
            SplitFraction = splitFraction;
            FilterZeros = filterZeros;
            BsrMaxTreeOutput = bsrMaxTreeOutput;

            GainConfidenceInSquaredStandardDeviations = ProbabilityFunctions.Probit(1 - (1 - gainConfidenceLevel) * 0.5);
            GainConfidenceInSquaredStandardDeviations *= GainConfidenceInSquaredStandardDeviations;

            MinDocsInLeafGlobal = MinDocsInLeaf;
            _parallelTraining = parallelTraining;
            _parallelTraining.InitTreeLearner(TrainData, numLeaves, MaxCategoricalSplitPointsPerNode, ref MinDocsInLeaf);

        }

        public override long GetSizeOfReservedMemory()
        {
            return base.GetSizeOfReservedMemory() + _sizeOfReservedMemory;
        }

        protected virtual void MakeSplitCandidateArrays(Dataset data, out LeafSplitCandidates smallerCandidates, out LeafSplitCandidates largerCandidates)
        {
            smallerCandidates = new LeafSplitCandidates(data);
            largerCandidates = new LeafSplitCandidates(data);
        }

        protected virtual RegressionTree NewTree()
        {
            return new RegressionTree(NumLeaves);
        }

        protected virtual void MakeDummyRootSplit(RegressionTree tree, double rootTarget, double[] targets)
        {
            // Pick a random feature and split on it:
            SplitInfo newRootSplitInfo = new SplitInfo();
            newRootSplitInfo.Feature = 0;
            newRootSplitInfo.Threshold = 0;
            newRootSplitInfo.LteOutput = rootTarget;
            newRootSplitInfo.GTOutput = rootTarget;
            newRootSplitInfo.Gain = 0;
            newRootSplitInfo.LteCount = -1;
            newRootSplitInfo.GTCount = -1;
            BestSplitInfoPerLeaf[0] = newRootSplitInfo;
            int dummyGtChild;
            int dummyLteChild;
            PerformSplit(tree, 0, targets, out dummyLteChild, out dummyGtChild);
        }

        /// <summary>
        /// Learns a new tree for the current outputs
        /// </summary>
        /// <returns>A regression tree</returns>
        public sealed override RegressionTree FitTargets(IChannel ch, bool[] activeFeatures, double[] targets)
        {
            int maxLeaves = base.NumLeaves;
            using (Timer.Time(TimerEvent.TreeLearnerGetTree))
            {
                // create a new tree
                RegressionTree tree = NewTree();
                // Not use weak reference here to avoid the change of activeFeatures in the ParallelInterface.
                tree.ActiveFeatures = (bool[])activeFeatures.Clone();

                // clear memory
                Initialize(activeFeatures);

                // find the best split of the root node.
                FindBestSplitOfRoot(targets);

                int bestLeaf = 0;
                SplitInfo rootSplitInfo = BestSplitInfoPerLeaf[0];
                if (Double.IsNaN(rootSplitInfo.Gain) || Double.IsNegativeInfinity(rootSplitInfo.Gain))
                {
                    // REVIEW: In the event that we cannot split on the root, it might be best to have a "dummy"
                    // root split that has the average output simply. However it is somewhat awkward to do this in a way
                    // that will actually work on all datasets, since it may be impossible for any split to exist, at all.
                    // It might be best to handle this with a "bias" on the aggregator apart from the trees, or through some
                    // other means, if this scenario proves to be important. (It almost certainly won't. People don't invoke
                    // a tree learner because they're interested in getting a predictor that outputs a constant value.)
                    ch.Check(AllowEmptyTrees, "Impossible to perform a leaf split, but the learner was configured to not allow empty trees");
                    return null; // We cannot split the root. Give up!
                }

                // tally split
                ++FeatureUseCount[rootSplitInfo.Feature];
                int gtChild;
                int lteChild;
                PerformSplit(tree, 0, targets, out lteChild, out gtChild);

                // perform numLeaves-2 more splits
                for (int split = 0; split < maxLeaves - 2; ++split)
                {
                    // For each the two new leaves, find the best feature and threshold to split on.
                    FindBestSplitOfSiblings(lteChild, gtChild, Partitioning, targets);

                    // choose the leaf with the highest gain
                    bestLeaf = BestSplitInfoPerLeaf.Select(info => info.Gain).ArgMax(tree.NumLeaves);
                    SplitInfo bestLeafSplitInfo = BestSplitInfoPerLeaf[bestLeaf];

                    // check if any progress can be made
                    if (bestLeafSplitInfo.Gain <= 0)
                        break;

                    // tally split
                    //REVIEW: Should we tally for all features within a categorical ORs?
                    ++FeatureUseCount[bestLeafSplitInfo.Feature];

                    PerformSplit(tree, bestLeaf, targets, out lteChild, out gtChild);
                }
                _parallelTraining.FinalizeIteration();
                // return the tree
                return tree;
            }
        }

        protected virtual void PerformSplit(RegressionTree tree, int bestLeaf, double[] targets, out int lteChild, out int gtChild)
        {
            SplitInfo bestSplitInfo = BestSplitInfoPerLeaf[bestLeaf];

            // perform the split in the tree, get new node names
            int newInteriorNodeIndex = tree.Split(bestLeaf, bestSplitInfo.Feature, bestSplitInfo.CategoricalFeatureIndices, bestSplitInfo.CategoricalSplitRange,
                bestSplitInfo.CategoricalSplit, bestSplitInfo.Threshold, bestSplitInfo.LteOutput, bestSplitInfo.GTOutput, bestSplitInfo.Gain, bestSplitInfo.GainPValue);

            gtChild = ~tree.GetGtChildForNode(newInteriorNodeIndex);
            lteChild = bestLeaf; // lte inherits name from parent

            // update the partitioning
            if (bestSplitInfo.CategoricalSplit)
            {
                Contracts.Assert(TrainData.Flocks[bestSplitInfo.Flock] is OneHotFeatureFlock);

                if (CategoricalThresholds == null)
                    CategoricalThresholds = new HashSet<int>();

                CategoricalThresholds.Clear();
                int flockFirstFeatureIndex = TrainData.FlockToFirstFeature(bestSplitInfo.Flock);
                foreach (var index in bestSplitInfo.CategoricalFeatureIndices)
                {
                    int localIndex = index - flockFirstFeatureIndex;

                    Contracts.Assert(localIndex >= 0);

                    CategoricalThresholds.Add(localIndex);
                }

                Partitioning.Split(bestLeaf, (TrainData.Flocks[bestSplitInfo.Flock] as OneHotFeatureFlock)?.Bins, CategoricalThresholds, gtChild);
            }
            else
                Partitioning.Split(bestLeaf, TrainData.GetIndexer(bestSplitInfo.Feature), bestSplitInfo.Threshold, gtChild);

            _parallelTraining.PerformGlobalSplit(bestLeaf, lteChild, gtChild, bestSplitInfo);
            Contracts.Assert(bestSplitInfo.GTCount < 0 || Partitioning.NumDocsInLeaf(gtChild) == bestSplitInfo.GTCount
                || !(_parallelTraining is SingleTrainer));
        }

        /// <summary>
        /// Clears data structures
        /// </summary>
        private void Initialize(bool[] activeFeatures)
        {
            _parallelTraining.InitIteration(ref activeFeatures);
            ActiveFeatures = activeFeatures;
            HistogramArrayPool.Reset();
            Partitioning.Initialize();
        }

        protected bool HasWeights => TrainData?.SampleWeights != null;

        protected double[] GetTargetWeights()
        {
            return TrainData.SampleWeights;
        }

        /// <summary>
        /// finds best feature/threshold split of root node. Fills in BestSplitInfoPerLeaf[0]
        /// </summary>
        protected virtual void FindBestSplitOfRoot(double[] targets)
        {
            using (Timer.Time(TimerEvent.FindBestSplit))
            using (Timer.Time(TimerEvent.FindBestSplitOfRoot))
            {
                var smallSplitInit = Task.Factory.StartNew(() =>
                {
                    // Initialize.
                    using (Timer.Time(TimerEvent.FindBestSplitInit))
                    {
                        if (Partitioning.NumDocs == TrainData.NumDocs)
                        {
                            // This case _is_ important: if filterZeros==false (the default) we can use more optimal index-free sumups routines.
                            SmallerChildSplitCandidates.Initialize(targets, GetTargetWeights(), FilterZeros);
                        }
                        else
                            SmallerChildSplitCandidates.Initialize(0, Partitioning, targets, GetTargetWeights(), FilterZeros);
                    }
                });

                ParentHistogramArray = null;
                HistogramArrayPool.Get(0, out SmallerChildHistogramArray);
                LargerChildSplitCandidates.Initialize();
                smallSplitInit.Wait();

                // Run multithread task that fills in _smallerChildSplitCandidates (finds best threshold for each feature).
                using (Timer.Time(TimerEvent.CalculateLeafSplitCandidates))
                    _calculateLeafSplitCandidates.RunTask();

                // Find gain-maximizing feature to split on.
                FindAndSetBestFeatureForLeaf(SmallerChildSplitCandidates);
                _parallelTraining.FindGlobalBestSplit(SmallerChildSplitCandidates, null, FindBestThresholdFromRawArray, BestSplitInfoPerLeaf);
            }
        }

        /// <summary>
        /// Finds best feature/threshold split of <paramref name="lteChild"/> and <paramref name="gtChild"/>,
        /// and fills in the corresponding elements of <see cref="BestSplitInfoPerLeaf"/>.
        /// </summary>
        protected virtual void FindBestSplitOfSiblings(int lteChild, int gtChild, DocumentPartitioning partitioning, double[] targets)
        {
            using (Timer.Time(TimerEvent.FindBestSplit))
            using (Timer.Time(TimerEvent.FindBestSplitOfSiblings))
            {
                int numDocsInLteChild = partitioning.NumDocsInLeaf(lteChild);
                int numDocsInGtChild = partitioning.NumDocsInLeaf(gtChild);

                _parallelTraining.GetGlobalDataCountInLeaf(lteChild, ref numDocsInLteChild);
                _parallelTraining.GetGlobalDataCountInLeaf(gtChild, ref numDocsInGtChild);

                // shortcut
                if (numDocsInGtChild < MinDocsInLeafGlobal * 2 && numDocsInLteChild < MinDocsInLeafGlobal * 2)
                {
                    BestSplitInfoPerLeaf[lteChild].Gain = double.NegativeInfinity;
                    BestSplitInfoPerLeaf[gtChild].Gain = double.NegativeInfinity;
                    return;
                }

                ParentHistogramArray = null;

                // initialize the "leafSplitCandidates" object for the smaller and larger child
                // also, get histogram arrays from the pool
                if (numDocsInLteChild < numDocsInGtChild)
                {
                    using (Timer.Time(TimerEvent.FindBestSplitInit))
                    {
                        var smallSplitInit = Task.Factory.StartNew(() => SmallerChildSplitCandidates.Initialize(lteChild, partitioning, targets, GetTargetWeights(), FilterZeros));
                        LargerChildSplitCandidates.Initialize(gtChild, partitioning, targets, GetTargetWeights(), FilterZeros);
                        smallSplitInit.Wait();
                    }
                    if (HistogramArrayPool.Get(lteChild, out LargerChildHistogramArray))
                        ParentHistogramArray = LargerChildHistogramArray;
                    HistogramArrayPool.Steal(lteChild, gtChild);
                    HistogramArrayPool.Get(lteChild, out SmallerChildHistogramArray);
                }
                else
                {
                    using (Timer.Time(TimerEvent.FindBestSplitInit))
                    {
                        var smallSplitInit = Task.Factory.StartNew(() => SmallerChildSplitCandidates.Initialize(gtChild, partitioning, targets, GetTargetWeights(), FilterZeros));
                        LargerChildSplitCandidates.Initialize(lteChild, partitioning, targets, GetTargetWeights(), FilterZeros);
                        smallSplitInit.Wait();
                    }
                    if (HistogramArrayPool.Get(lteChild, out LargerChildHistogramArray))
                        ParentHistogramArray = LargerChildHistogramArray;
                    HistogramArrayPool.Get(gtChild, out SmallerChildHistogramArray);
                }

                // run multithread task that fills in _smallerChildSplitCandidates and _largerChildSplitCandidates (finds best threshold for each feature)
                using (Timer.Time(TimerEvent.CalculateLeafSplitCandidates))
                    _calculateLeafSplitCandidates.RunTask();

                // for each leaf, find best feature to split on
                FindAndSetBestFeatureForLeaf(SmallerChildSplitCandidates);
                FindAndSetBestFeatureForLeaf(LargerChildSplitCandidates);
                _parallelTraining.FindGlobalBestSplit(SmallerChildSplitCandidates, LargerChildSplitCandidates, FindBestThresholdFromRawArray, BestSplitInfoPerLeaf);
            }
        }

        /// <summary>
        /// After the gain for each feature has been computed, this function chooses the gain maximizing feature
        /// and sets its info in the right places
        /// This method is overriden in MPI version of the code
        /// </summary>
        /// <param name="leafSplitCandidates">the FindBestThesholdleafSplitCandidates data structure that contains the best split information</param>
        protected virtual void FindAndSetBestFeatureForLeaf(LeafSplitCandidates leafSplitCandidates)
        {
            int bestFeature;
            if (SoftmaxTemperature == 0)
            {
                if (SplitFraction < 1.0)
                    bestFeature = leafSplitCandidates.FeatureSplitInfo.Select(info => info.Gain)
                        .ArgMaxRand(Rand, SplitFraction);
                else
                {
                    var infos = leafSplitCandidates.FeatureSplitInfo;
                    bestFeature = 0;
                    double max = infos[0].Gain;

                    if (leafSplitCandidates.FlockToBestFeature != null)
                    {
                        // If we have this array, we can use it in this case to zoom directly
                        // to the best candidate among all features in the flock.
                        for (int flock = 0; flock < leafSplitCandidates.FlockToBestFeature.Length; ++flock)
                        {
                            int bestFeatInFlock = leafSplitCandidates.FlockToBestFeature[flock];
                            if (bestFeatInFlock != -1 && infos[bestFeatInFlock].Gain > max)
                                max = infos[bestFeature = bestFeatInFlock].Gain;
                        }
                    }
                    else
                    {
                        for (int f = 1; f < infos.Length; ++f)
                        {
                            if (infos[f].Gain > max)
                                max = infos[bestFeature = f].Gain;
                        }
                    }
                }
            }
            else
                bestFeature = leafSplitCandidates.FeatureSplitInfo.Select(info => info.Gain / SoftmaxTemperature).SoftArgMax(Rand);

            SetBestFeatureForLeaf(leafSplitCandidates, bestFeature);
        }

        protected virtual void SetBestFeatureForLeaf(LeafSplitCandidates leafSplitCandidates, int bestFeature)
        {
            int leaf = leafSplitCandidates.LeafIndex;
            BestSplitInfoPerLeaf[leaf] = leafSplitCandidates.FeatureSplitInfo[bestFeature];
            if (BestSplitInfoPerLeaf[leaf].CategoricalSplit)
                Array.Sort(BestSplitInfoPerLeaf[leaf].CategoricalFeatureIndices);

            BestSplitInfoPerLeaf[leaf].Feature = bestFeature;
        }

        /// <summary>
        /// The multithreading entry-point: finds the best threshold for a given flock at a given leaf.
        /// </summary>
        private void FindBestThresholdForFlockThreadWorker(int flock)
        {
            int featureMin = TrainData.FlockToFirstFeature(flock);
            int featureLim = featureMin + TrainData.Flocks[flock].Count;
            // Check if any feature is active.
            if (ActiveFeatures != null)
            {
                bool anyActive = false;
                for (int f = featureMin; f < featureLim; ++f)
                {
                    if (ActiveFeatures[f])
                    {
                        anyActive = true;
                        break;
                    }
                }
                if (!anyActive)
                    return;
            }

            // Check splittability, exiting early if it is not splittable.
            var smallStats = SmallerChildHistogramArray[flock];
            if (ParentHistogramArray == null)
            {
                for (int i = 0; i < smallStats.IsSplittable.Length; ++i)
                    smallStats.IsSplittable[i] = true;
            }
            else
            {
                Contracts.Assert(ParentHistogramArray[flock].Flock == smallStats.Flock);
                Array.Copy(ParentHistogramArray[flock].IsSplittable, smallStats.IsSplittable, smallStats.IsSplittable.Length);
                if (!ParentHistogramArray[flock].IsSplittable.Any() && _parallelTraining.IsSkipNonSplittableHistogram())
                    return;
            }

            // Compute the flock's sufficient statistics of smaller child.
            Sumup(smallStats, featureMin, SmallerChildSplitCandidates);

            FindBestThresholdFromHistogram(smallStats, SmallerChildSplitCandidates, flock);

            if (LargerChildSplitCandidates.LeafIndex < 0)
                return;

            var largeStats = LargerChildHistogramArray[flock];
            // Compute the sufficient statistics histogram of larger child. Larger child inherits the sufficient statistics
            // from the parent if present, so we can, if available, compute it using a simple subtraction.
            if (ParentHistogramArray == null)
                Sumup(largeStats, featureMin, LargerChildSplitCandidates);
            else
                largeStats.Subtract(smallStats);

            // Use the histogram to find best threshold for larger child.
            FindBestThresholdFromHistogram(largeStats, LargerChildSplitCandidates, flock);

        }

        private void Sumup(SufficientStatsBase stats, int featureMin, LeafSplitCandidates candidates)
        {
            stats.Sumup(
                featureMin,
                ActiveFeatures,
                candidates.NumDocsInLeaf,
                candidates.SumTargets,
                candidates.SumWeights,
                candidates.Targets,
                candidates.Weights,
                candidates.DocIndices);
        }

        /// <summary>
        /// Returns the set of features that are active within a particular range.
        /// </summary>
        /// <param name="min">The inclusive lower bound of the feature indices</param>
        /// <param name="lim">The exclusive upper bound of the feature indices</param>
        /// <returns>The feature indices within the range that are active</returns>
        public IEnumerable<int> GetActiveFeatures(int min, int lim)
        {
            Contracts.Assert(0 <= min && min <= lim && lim <= TrainData.NumFeatures);
            if (ActiveFeatures == null)
            {
                // If there is no _activeFeatures array, then all the features are active.
                for (int i = min; i < lim; ++i)
                    yield return i;
            }
            else
            {
                for (int i = min; i < lim; ++i)
                {
                    if (ActiveFeatures[i])
                        yield return i;
                }
            }
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
            if (!HasWeights)
            {
                if (BsrMaxTreeOutput < 0)
                    return (sumTargets * sumTargets) / count;
                // For the BSR case, fall through to below with sweight
                // receiving the "natural" weight.
                sumWeights = count;
            }
            // Function is created in a way to produce results in the same scale
            // as from LeastSquareRegressionTreeLearner + Adjust tree output.
            // Best x is starget / (2 * sweight).
            // F(x) = 4.0 * x * (starget - x * sweight) -> max
            double absSumTargets = Math.Abs(sumTargets);
            if (BsrMaxTreeOutput <= 0 || absSumTargets < 2.0 * sumWeights * BsrMaxTreeOutput)
            {
                Contracts.Assert(sumWeights != 0);
                return sumTargets * sumTargets / sumWeights;
            }
            return 4.0 * BsrMaxTreeOutput * (absSumTargets - BsrMaxTreeOutput * sumWeights);
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
            if (BsrMaxTreeOutput >= 0)
                sumTargets *= 0.5;
            if (!HasWeights)
                return sumTargets / count;
            Contracts.Assert(sumWeights != 0);
            return sumTargets / sumWeights;
        }

        /// <summary>
        /// Finds the best threshold to split on, and sets the appropriate values in the LeafSplitCandidates data structure.
        /// </summary>
        /// <param name="histogram">The sufficient stats accumulated for the flock</param>
        /// <param name="leafSplitCandidates">The LeafSplitCandidates data structure</param>
        /// <param name="flock">The index of the flock containing this feature</param>
        protected virtual void FindBestThresholdFromHistogram(SufficientStatsBase histogram,
            LeafSplitCandidates leafSplitCandidates, int flock)
        {
            // Cache histograms for the parallel interface.
            int featureMin = TrainData.FlockToFirstFeature(flock);
            int featureLim = featureMin + TrainData.Flocks[flock].Count;
            foreach (var feature in GetActiveFeatures(featureMin, featureLim))
            {
                int subfeature = feature - featureMin;
                _parallelTraining.CacheHistogram(leafSplitCandidates == SmallerChildSplitCandidates, feature, subfeature, histogram, HasWeights);
            }

            if (!_parallelTraining.IsNeedFindLocalBestSplit())
                return;

            Contracts.AssertValue(histogram);
            Contracts.AssertValue(leafSplitCandidates);
            Contracts.Assert(0 <= flock && flock < TrainData.NumFlocks);
            Contracts.Assert(histogram.Flock == TrainData.Flocks[flock]);

            if (TrainData.Flocks[flock].Categorical && leafSplitCandidates.NumDocsInLeaf > 100)
            {
                if (Bundling == Bundle.None)
                {
                    histogram.FillSplitCandidatesCategorical(this, leafSplitCandidates,
                    flock, FeatureUseCount, FeatureFirstUsePenalty, FeatureReusePenalty, MinDocsInLeaf,
                    HasWeights, GainConfidenceInSquaredStandardDeviations,
                    EntropyCoefficient);
                }
                else if (Bundling == Bundle.AggregateLowPopulation)
                {
                    histogram.FillSplitCandidatesCategoricalLowPopulation(this, leafSplitCandidates,
                    flock, FeatureUseCount, FeatureFirstUsePenalty, FeatureReusePenalty, MinDocsInLeaf,
                    HasWeights, GainConfidenceInSquaredStandardDeviations,
                    EntropyCoefficient);
                }
                else if (Bundling == Bundle.Adjacent)
                {
                    histogram.FillSplitCandidatesCategoricalNeighborBundling(this, leafSplitCandidates,
                    flock, FeatureUseCount, FeatureFirstUsePenalty, FeatureReusePenalty, MinDocsInLeaf,
                    HasWeights, GainConfidenceInSquaredStandardDeviations,
                    EntropyCoefficient);
                }
            }
            else
            {
                histogram.FillSplitCandidates(this, leafSplitCandidates,
                    flock, FeatureUseCount, FeatureFirstUsePenalty, FeatureReusePenalty, MinDocsInLeaf,
                    HasWeights, GainConfidenceInSquaredStandardDeviations,
                    EntropyCoefficient);
            }

        }

        protected void FindBestThresholdFromRawArray(LeafSplitCandidates leafSplitCandidates, int feature, int flock, int subfeature,
            int[] countByBin, FloatType[] sumTargetsByBin, FloatType[] sumWeightsByBin,
            int numDocsInLeaf, double sumTargets, double sumWeights, double varianceTargets, out SplitInfo bestSplit)
        {
            double bestSumGTTargets = double.NaN;
            double bestSumGTWeights = double.NaN;
            double bestShiftedGain = double.NegativeInfinity;
            double trust = TrainData.Flocks[flock].Trust(subfeature);

            int bestGTCount = -1;

            const double eps = 1e-10;
            double sumGTTargets = 0.0;
            double sumGTWeights = eps;
            int gtCount = 0;

            int totalCount = numDocsInLeaf;
            sumWeights += 2 * eps;
            double gainShift = GetLeafSplitGain(totalCount, sumTargets, sumWeights);

            // We get to this more explicit handling of the zero case since, under the influence of
            // numerical error, especially under single precision, the histogram computed values can
            // be wildly inaccurate even to the point where 0 unshifted gain may become a strong
            // criteria.
            double minShiftedGain = GainConfidenceInSquaredStandardDeviations <= 0 ? 0.0 :
                (GainConfidenceInSquaredStandardDeviations * varianceTargets
                * totalCount / (totalCount - 1) + gainShift);

            // re-evaluate if the histogram is splittable
            // histogram.IsSplittable[subfeature] = false;

            double minDocsForThis = MinDocsInLeafGlobal / trust;
            int numBin = TrainData.Flocks[flock].BinCount(subfeature);
            uint bestThreshold = (uint)numBin;
            bool hasWeights = HasWeights;
            int min = 0;
            // No store stats for bin 0, so the max is numBin - 2.
            int max = numBin - 2;
            for (int i = max; i >= min; --i)
            {
                sumGTTargets += sumTargetsByBin[i];
                if (hasWeights)
                    sumGTWeights += sumWeightsByBin[i];
                gtCount += countByBin[i];

                // Advance until GTCount is high enough.
                if (gtCount < minDocsForThis)
                    continue;
                int lteCount = totalCount - gtCount;

                // If LTECount is too small, we are finished.
                if (lteCount < minDocsForThis)
                    break;

                // Calculate the shifted gain, including the LTE child.
                double currentShiftedGain = GetLeafSplitGain(gtCount, sumGTTargets, sumGTWeights)
                    + GetLeafSplitGain(lteCount, sumTargets - sumGTTargets, sumWeights - sumGTWeights);

                // Test whether we are meeting the min shifted gain confidence criteria for this split.
                if (currentShiftedGain < minShiftedGain)
                    continue;

                if (EntropyCoefficient > 0)
                {
                    // Consider the entropy of the split.
                    double entropyGain = (totalCount * Math.Log(totalCount) - lteCount * Math.Log(lteCount) - gtCount * Math.Log(gtCount));
                    currentShiftedGain += EntropyCoefficient * entropyGain;
                }

                // Is t the best threshold so far?
                if (currentShiftedGain > bestShiftedGain)
                {
                    bestGTCount = gtCount;
                    bestSumGTTargets = sumGTTargets;
                    bestSumGTWeights = sumGTWeights;
                    bestThreshold = (uint)i;
                    bestShiftedGain = currentShiftedGain;
                }
            }

            // set the appropriate place in the output vectors
            leafSplitCandidates.FeatureSplitInfo[feature].Feature = feature;
            leafSplitCandidates.FeatureSplitInfo[feature].Threshold = bestThreshold;
            leafSplitCandidates.FeatureSplitInfo[feature].LteOutput = CalculateSplittedLeafOutput(totalCount - bestGTCount, sumTargets - bestSumGTTargets, sumWeights - bestSumGTWeights);
            leafSplitCandidates.FeatureSplitInfo[feature].GTOutput = CalculateSplittedLeafOutput(bestGTCount, bestSumGTTargets, bestSumGTWeights);
            leafSplitCandidates.FeatureSplitInfo[feature].LteCount = totalCount - bestGTCount;
            leafSplitCandidates.FeatureSplitInfo[feature].GTCount = bestGTCount;

            // note: may want to use a function of count that decays with iteration #
            double usePenalty = (FeatureUseCount[feature] == 0) ?
                FeatureFirstUsePenalty : FeatureReusePenalty * Math.Log(FeatureUseCount[feature] + 1);

            leafSplitCandidates.FeatureSplitInfo[feature].Gain = (bestShiftedGain - gainShift) * trust - usePenalty;
            double erfcArg = Math.Sqrt((bestShiftedGain - gainShift) * (totalCount - 1) / (2 * varianceTargets * totalCount));
            leafSplitCandidates.FeatureSplitInfo[feature].GainPValue = ProbabilityFunctions.Erfc(erfcArg);

            if (leafSplitCandidates.FlockToBestFeature != null)
            {
                if (leafSplitCandidates.FlockToBestFeature[flock] == -1 ||
                    leafSplitCandidates.FeatureSplitInfo[leafSplitCandidates.FlockToBestFeature[flock]].Gain <
                    leafSplitCandidates.FeatureSplitInfo[feature].Gain)
                {
                    leafSplitCandidates.FlockToBestFeature[flock] = feature;
                }
            }
            bestSplit = leafSplitCandidates.FeatureSplitInfo[feature];
        }

        /// <summary>
        /// Contains the memory data structures required for finding the best threshold for a given
        /// feature at a given leaf.
        /// </summary>
        public sealed class LeafSplitCandidates
        {
            private int _leafIndex;
            private int _numDocsInLeaf;
            private double _sumTargets;
            private double _sumWeights;
            private double _sumSquaredTargets;
            private int[] _docIndices;
            private int[] _docIndicesCopy;
            public readonly FloatType[] Targets;
            public readonly double[] Weights;
            public readonly SplitInfo[] FeatureSplitInfo;
            // Note that the range of this map is the feature index for the dataset, not
            // the feature index within the corresponding flock. -1 if there is no applicable
            // best feature found for this flock.
            public readonly int[] FlockToBestFeature;

            public LeafSplitCandidates(Dataset data)
            {
                FeatureSplitInfo = new SplitInfo[data.NumFeatures];
                if (data.NumFlocks < data.NumFeatures / 2)
                    FlockToBestFeature = new int[data.NumFlocks];

                Clear();
                _docIndicesCopy = _docIndices = new int[data.NumDocs];
                Targets = new FloatType[data.NumDocs];
                if (data.SampleWeights != null)
                    Weights = new double[data.NumDocs];
            }

            public int LeafIndex
            {
                get { return _leafIndex; }
            }

            public int NumDocsInLeaf
            {
                get { return _numDocsInLeaf; }
            }

            public double SumTargets
            {
                get { return _sumTargets; }
            }

            public double SumWeights
            {
                get { return _sumWeights; }
            }

            public double SumSquaredTargets
            {
                get { return _sumSquaredTargets; }
            }

            public double VarianceTargets
            {
                get
                {
                    double denom = Weights == null ? NumDocsInLeaf : SumWeights;
                    return (SumSquaredTargets - SumTargets / denom) / (denom - 1);
                }
            }

            public int[] DocIndices
            {
                get { return _docIndices; }
            }

            public int SizeInBytes(int maxCatSplitPoints)
            {
                return sizeof(int) * 2
                       + sizeof(double) * 3
                       + SplitInfo.SizeInBytes(maxCatSplitPoints) * FeatureSplitInfo.Length
                       + sizeof(int) * _docIndices.Length
                       + sizeof(FloatType) * Targets.Length
                       + sizeof(FloatType) * Utils.Size(Weights)
                       + sizeof(int) * Utils.Size(FlockToBestFeature);
            }

            /// <summary>
            /// Initializes the object for a specific leaf, with a certain subset of documents.
            /// </summary>
            /// <param name="leafIndex">The leaf index</param>
            /// <param name="partitioning">The partitioning object that knows which documents have reached that leaf</param>
            /// <param name="targets">The array of targets, which the regression tree is trying to fit</param>
            /// <param name="weights">The array of weights for the document</param>
            /// <param name="filterZeros">Whether filtering of zero gradients was turned on or not</param>
            public void Initialize(int leafIndex, DocumentPartitioning partitioning, double[] targets, double[] weights, bool filterZeros)
            {
                Clear();
                _sumTargets = 0;
                _sumWeights = 0;
                _sumSquaredTargets = 0;

                _leafIndex = leafIndex;
                _docIndices = _docIndicesCopy;
                _numDocsInLeaf = partitioning.GetLeafDocuments(leafIndex, _docIndices);

                if (filterZeros)
                {
                    int nonZeroCount = 0;
                    // REVIEW: Consider removing filter zero weights.
                    for (int i = 0; i < _numDocsInLeaf; ++i)
                    {
                        int docIndex = _docIndices[i];
                        FloatType target = targets[docIndex];
                        if (target != 0.0)
                        {
                            Targets[nonZeroCount] = target;
                            _sumTargets += target;
                            _docIndices[nonZeroCount] = docIndex;
                            if (Weights != null)
                            {
                                FloatType weight = weights[docIndex];
                                Weights[nonZeroCount] = weight;
                                _sumWeights += weight;
                                // Each target is really (target/weight) since we pre-multiply
                                // the targets, and we want the weighted sum of squared targets
                                // to be weight * (target*target), which with the
                                // pre-multiplication becomes target*target/weight.
                                if (weight != 0.0)
                                    _sumSquaredTargets += target * target / weight;
                            }
                            else
                                _sumSquaredTargets += target * target;
                            nonZeroCount++;
                        }
                    }
                    _numDocsInLeaf = nonZeroCount;
                }
                else
                {
                    if (Weights == null)
                    {
                        for (int i = 0; i < _numDocsInLeaf; ++i)
                        {
                            int docIndex = _docIndices[i];
                            FloatType target = targets[docIndex];

                            Targets[i] = target;
                            _sumTargets += target;
                            _docIndices[i] = docIndex;
                            _sumSquaredTargets += target * target;
                        }
                    }
                    else
                    {
                        for (int i = 0; i < _numDocsInLeaf; ++i)
                        {
                            int docIndex = _docIndices[i];
                            FloatType target = targets[docIndex];
                            FloatType weight = weights[docIndex];

                            Targets[i] = target;
                            _sumTargets += target;
                            _docIndices[i] = docIndex;
                            Weights[i] = weight;
                            _sumWeights += weight;
                            // Each target is really (target/weight) since we pre-multiply
                            // the targets, and we want the weighted sum of squared targets
                            // to be weight * (target*target), which with the
                            // pre-multiplication becomes target*target/weight.
                            if (weight != 0.0)
                                _sumSquaredTargets += target * target / weight;
                        }
                    }
                }
            }

            /// <summary>
            /// Initializes the object for computing the root node split
            /// </summary>
            /// <param name="targets">the array of targets, which the regression tree is trying to fit</param>
            /// <param name="weights"></param>
            /// <param name="filterZeros"></param>
            public void Initialize(double[] targets, double[] weights, bool filterZeros)
            {
                Clear();
                _sumTargets = 0;
                _sumWeights = 0;
                _sumSquaredTargets = 0;

                _leafIndex = 0;
                _numDocsInLeaf = targets.Length;

                if (filterZeros)
                {
                    _docIndices = _docIndicesCopy;
                    int nonZeroCount = 0;
                    for (int i = 0; i < _numDocsInLeaf; ++i)
                    {
                        FloatType target = targets[i];
                        if (target != 0)
                        {
                            Targets[nonZeroCount] = (FloatType)target;
                            _sumTargets += target;
                            _docIndices[nonZeroCount] = i;

                            // orignal code here: nonZeroCount++
                            // is a bug and it will cause issue in next several lines of code,
                            // so we move it down to the end of if{} block.
                            if (Weights != null)
                            {
                                FloatType weight = weights[i];
                                Weights[nonZeroCount] = weight;
                                _sumWeights += weight;
                                if (weight != 0.0)
                                    _sumSquaredTargets += target * target / weight;
                            }
                            else
                                _sumSquaredTargets += target * target;

                            nonZeroCount++;
                        }
                    }
                    _numDocsInLeaf = nonZeroCount;
                }
                else
                {
                    _docIndices = null;  //This is important, it indicates that we we can use index-free routines for sumup.
                    if (Weights == null)
                    {
                        for (int i = 0; i < _numDocsInLeaf; ++i)
                        {
                            FloatType target = targets[i];
                            Targets[i] = target;
                            _sumTargets += target;
                            _sumSquaredTargets += target * target;
                        }
                    }
                    else
                    {
                        for (int i = 0; i < _numDocsInLeaf; ++i)
                        {
                            FloatType target = targets[i];
                            FloatType weight = weights[i];

                            Targets[i] = (FloatType)target;
                            _sumTargets += target;
                            Weights[i] = weight;
                            _sumWeights += weight;
                            if (weight != 0.0)
                                _sumSquaredTargets += target * target / weight;
                        }
                    }
                }
            }

            private void Clear()
            {
                _leafIndex = -1;
                for (int f = 0; f < FeatureSplitInfo.Length; f++)
                {
                    FeatureSplitInfo[f].Feature = f;
                    FeatureSplitInfo[f].Gain = double.NegativeInfinity;
                }
                if (FlockToBestFeature != null)
                {
                    for (int f = 0; f < FlockToBestFeature.Length; ++f)
                        FlockToBestFeature[f] = -1;
                }
            }

            /// <summary>
            /// Initializes the object to do nothing
            /// </summary>
            public void Initialize()
            {
                Clear();
            }
        }

        /// <summary>
        /// A struct to store information about each leaf for splitting
        /// </summary>
        public struct SplitInfo : IComparable<SplitInfo>
        {
            public int Feature;
            public uint Threshold;
            public double LteOutput;
            public double GTOutput;
            public double Gain;
            public double GainPValue;
            public int LteCount;
            public int GTCount;
            public int Flock;
            public bool CategoricalSplit;
            public int[] CategoricalSplitRange;
            public int[] CategoricalFeatureIndices;

            public void Reset()
            {
                Feature = -1;
                Gain = double.NegativeInfinity;
                CategoricalFeatureIndices = new int[0];
                CategoricalSplitRange = new int[0];
                Flock = -1;
            }

            public static int SizeInBytes(int maxCatSplitPoints)
            {
                // The AllReduce code in TLC++ requires the size to be a constant so
                // we will assume CategoricalFeatureIndices to be of size maxCatSplitPoints.
                return sizeof(int) * 4 + sizeof(double) * 4 + sizeof(uint) + sizeof(bool)
                    + sizeof(int) * 3 // For CategoricalSplitRange
                    + sizeof(int) * (maxCatSplitPoints + 1); // For CategoricalFeatureIndices
            }

            public void ToByteArray(byte[] buffer, ref int offset, int size)
            {
                Contracts.CheckValue(buffer, nameof(buffer));
                Contracts.Check(0 <= offset && offset + size <= buffer.Length);
                var startIndex = offset;
                Feature.ToByteArray(buffer, ref offset);
                Threshold.ToByteArray(buffer, ref offset);
                LteOutput.ToByteArray(buffer, ref offset);
                GTOutput.ToByteArray(buffer, ref offset);
                Gain.ToByteArray(buffer, ref offset);
                GainPValue.ToByteArray(buffer, ref offset);
                LteCount.ToByteArray(buffer, ref offset);
                GTCount.ToByteArray(buffer, ref offset);
                Flock.ToByteArray(buffer, ref offset);
                Convert.ToByte(CategoricalSplit).ToByteArray(buffer, ref offset);
                if (!CategoricalSplit)
                {
                    offset = startIndex + size;
                    return;
                }
                Contracts.Check(CategoricalSplitRange.Length == 2);
                CategoricalSplitRange.ToByteArray(buffer, ref offset);
                CategoricalFeatureIndices.ToByteArray(buffer, ref offset);
                Contracts.Check(offset - startIndex <= size);
                offset = startIndex + size;
            }

            public void FromByteArray(byte[] buffer, ref int offset, int size)
            {
                Contracts.CheckValue(buffer, nameof(buffer));
                Contracts.Check(0 <= offset && offset + size <= buffer.Length);
                var startIndex = offset;
                Feature = buffer.ToInt(ref offset);
                Threshold = buffer.ToUInt(ref offset);
                LteOutput = buffer.ToDouble(ref offset);
                GTOutput = buffer.ToDouble(ref offset);
                Gain = buffer.ToDouble(ref offset);
                GainPValue = buffer.ToDouble(ref offset);
                LteCount = buffer.ToInt(ref offset);
                GTCount = buffer.ToInt(ref offset);
                Flock = buffer.ToInt(ref offset);
                CategoricalSplit = Convert.ToBoolean(buffer.ToByte(ref offset));
                if (!CategoricalSplit)
                {
                    offset = startIndex + size;
                    return;
                }
                CategoricalSplitRange = buffer.ToIntArray(ref offset);
                Contracts.Check(CategoricalSplitRange.Length == 2);
                CategoricalFeatureIndices = buffer.ToIntArray(ref offset);
                Contracts.Check(offset - startIndex <= size);
                offset = startIndex + size;
            }

            public int CompareTo(SplitInfo other)
            {
                double myGain = Gain;
                double otherGain = other.Gain;
                if (double.IsNaN(myGain))
                    myGain = double.NegativeInfinity;

                if (double.IsNaN(otherGain))
                    otherGain = double.NegativeInfinity;

                int myFeature = Feature;
                int otherFeature = other.Feature;

                if (myFeature == -1)
                    myFeature = int.MaxValue;

                if (otherFeature == -1)
                    otherFeature = int.MaxValue;

                if (myGain > otherGain)
                    return 1;
                else if (myGain == otherGain)
                    return otherFeature - myFeature;
                else
                    return -1;
            }
        }
    }

    public interface ILeafSplitStatisticsCalculator
    {
        double CalculateSplittedLeafOutput(int count, double sumTargets, double sumWeights);

        double GetLeafSplitGain(int count, double sumTargets, double sumWeights);
    }
}
