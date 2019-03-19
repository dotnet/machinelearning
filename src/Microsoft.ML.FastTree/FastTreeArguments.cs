// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.CommandLine;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.FastTree;

[assembly: EntryPointModule(typeof(FastTreeBinaryTrainer.Options))]
[assembly: EntryPointModule(typeof(FastTreeRegressionTrainer.Options))]
[assembly: EntryPointModule(typeof(FastTreeTweedieTrainer.Options))]
[assembly: EntryPointModule(typeof(FastTreeRankingTrainer.Options))]

namespace Microsoft.ML.Trainers.FastTree
{
    [TlcModule.ComponentKind("FastTreeTrainer")]
    internal interface IFastTreeTrainerFactory : IComponentFactory<ITrainer>
    {
    }

    /// <summary>
    /// Stopping measurements for classification and regression.
    /// </summary>
    public enum EarlyStoppingMetric
    {
        /// <summary>
        /// L1-norm of gradient.
        /// </summary>
        L1Norm = 1,
        /// <summary>
        /// L2-norm of gradient.
        /// </summary>
        L2Norm = 2
    };

    /// <summary>
    /// Stopping measurements for ranking.
    /// </summary>
    public enum EarlyStoppingRankingMetric
    {
        /// <summary>
        /// NDCG@1
        /// </summary>
        NdcgAt1 = 1,
        /// <summary>
        /// NDCG@3
        /// </summary>
        NdcgAt3 = 3
    }

    // XML docs are provided in the other part of this partial class. No need to duplicate the content here.
    public sealed partial class FastTreeBinaryTrainer
    {
        /// <summary>
        /// Options for the <see cref="FastTreeBinaryTrainer"/>.
        /// </summary>
        [TlcModule.Component(Name = LoadNameValue, FriendlyName = UserNameValue, Desc = Summary)]
        public sealed class Options : BoostedTreeOptions, IFastTreeTrainerFactory
        {

            /// <summary>
            /// Whether to use derivatives optimized for unbalanced training data.
            /// </summary>
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Option for using derivatives optimized for unbalanced sets", ShortName = "us")]
            [TGUI(Label = "Optimize for unbalanced")]
            public bool UnbalancedSets = false;

            /// <summary>
            /// internal state of <see cref="EarlyStoppingMetric"/>. It should be always synced with
            /// <see cref="BoostedTreeOptions.EarlyStoppingMetrics"/>.
            /// </summary>
            // Disable 649 because Visual Studio can't detect its assignment via property.
            #pragma warning disable 649
            private EarlyStoppingMetric _earlyStoppingMetric;
            #pragma warning restore 649

            /// <summary>
            /// Early stopping metrics.
            /// </summary>
            public EarlyStoppingMetric EarlyStoppingMetric
            {
                get { return _earlyStoppingMetric; }

                set
                {
                    // Update the state of the user-facing stopping metric.
                    _earlyStoppingMetric = value;
                    // Set up internal property according to its public value.
                    EarlyStoppingMetrics = (int)_earlyStoppingMetric;
                }
            }

            /// <summary>
            /// Create a new <see cref="Options"/> object with default values.
            /// </summary>
            public Options()
            {
                // Use L1 by default.
                EarlyStoppingMetric = EarlyStoppingMetric.L1Norm;
            }

            ITrainer IComponentFactory<ITrainer>.CreateComponent(IHostEnvironment env) => new FastTreeBinaryTrainer(env, this);
        }
    }

    // XML docs are provided in the other part of this partial class. No need to duplicate the content here.
    public sealed partial class FastTreeRegressionTrainer
    {
        /// <summary>
        /// Options for the <see cref="FastTreeRegressionTrainer"/>.
        /// </summary>
        [TlcModule.Component(Name = LoadNameValue, FriendlyName = UserNameValue, Desc = Summary)]
        public sealed class Options : BoostedTreeOptions, IFastTreeTrainerFactory
        {
            /// <summary>
            /// internal state of <see cref="EarlyStoppingMetric"/>. It should be always synced with
            /// <see cref="BoostedTreeOptions.EarlyStoppingMetrics"/>.
            /// </summary>
            private EarlyStoppingMetric _earlyStoppingMetric;

            /// <summary>
            /// Early stopping metrics.
            /// </summary>
            public EarlyStoppingMetric EarlyStoppingMetric
            {
                get { return _earlyStoppingMetric; }

                set
                {
                    // Update the state of the user-facing stopping metric.
                    _earlyStoppingMetric = value;
                    // Set up internal property according to its public value.
                    EarlyStoppingMetrics = (int)_earlyStoppingMetric;
                }
            }

            /// <summary>
            /// Create a new <see cref="Options"/> object with default values.
            /// </summary>
            public Options()
            {
                EarlyStoppingMetric = EarlyStoppingMetric.L1Norm; // Use L1 by default.
            }

            ITrainer IComponentFactory<ITrainer>.CreateComponent(IHostEnvironment env) => new FastTreeRegressionTrainer(env, this);
        }
    }

    // XML docs are provided in the other part of this partial class. No need to duplicate the content here.
    public sealed partial class FastTreeTweedieTrainer
    {
        /// <summary>
        /// Options for the <see cref="FastTreeTweedieTrainer"/>.
        /// </summary>
        [TlcModule.Component(Name = LoadNameValue, FriendlyName = UserNameValue, Desc = Summary)]
        public sealed class Options : BoostedTreeOptions, IFastTreeTrainerFactory
        {
            // REVIEW: It is possible to estimate this index parameter from the distribution of data, using
            // a combination of univariate optimization and grid search, following section 4.2 of the paper. However
            // it is probably not worth doing unless and until explicitly asked for.
            /// <summary>
            /// The index parameter for the Tweedie distribution, in the range [1, 2]. 1 is Poisson loss, 2 is gamma loss,
            /// and intermediate values are compound Poisson loss.
            /// </summary>
            [Argument(ArgumentType.LastOccurenceWins, HelpText =
                "Index parameter for the Tweedie distribution, in the range [1, 2]. 1 is Poisson loss, 2 is gamma loss, " +
                "and intermediate values are compound Poisson loss.")]
            public Double Index = 1.5;

            /// <summary>
            /// internal state of <see cref="EarlyStoppingMetric"/>. It should be always synced with
            /// <see cref="BoostedTreeOptions.EarlyStoppingMetrics"/>.
            /// </summary>
            // Disable 649 because Visual Studio can't detect its assignment via property.
            #pragma warning disable 649
            private EarlyStoppingMetric _earlyStoppingMetric;
            #pragma warning restore 649

            /// <summary>
            /// Early stopping metrics.
            /// </summary>
            public EarlyStoppingMetric EarlyStoppingMetric
            {
                get { return _earlyStoppingMetric; }

                set
                {
                    // Update the state of the user-facing stopping metric.
                    _earlyStoppingMetric = value;
                    // Set up internal property according to its public value.
                    EarlyStoppingMetrics = (int)_earlyStoppingMetric;
                }
            }

            /// <summary>
            /// Create a new <see cref="Options"/> object with default values.
            /// </summary>
            public Options()
            {
                EarlyStoppingMetric = EarlyStoppingMetric.L1Norm; // Use L1 by default.
            }

            ITrainer IComponentFactory<ITrainer>.CreateComponent(IHostEnvironment env) => new FastTreeTweedieTrainer(env, this);
        }
    }

    // XML docs are provided in the other part of this partial class. No need to duplicate the content here.
    public sealed partial class FastTreeRankingTrainer
    {
        /// <summary>
        /// Options for the <see cref="FastTreeRankingTrainer"/>.
        /// </summary>
        [TlcModule.Component(Name = LoadNameValue, FriendlyName = UserNameValue, Desc = Summary)]
        public sealed class Options : BoostedTreeOptions, IFastTreeTrainerFactory
        {
            /// <summary>
            /// Comma-separated list of gains associated with each relevance label.
            /// </summary>
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Comma-separated list of gains associated to each relevance label.", ShortName = "gains")]
            [TGUI(NoSweep = true)]
            public double[] CustomGains = new double[] { 0, 3, 7, 15, 31 };

            /// <summary>
            /// Whether to train using discounted cumulative gain (DCG) instead of normalized DCG (NDCG).
            /// </summary>
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Train DCG instead of NDCG", ShortName = "dcg")]
            public bool UseDcg;

            // REVIEW: Hiding sorting for now. Should be an enum or component factory.
            [BestFriend]
            [Argument(ArgumentType.LastOccurenceWins,
                HelpText = "The sorting algorithm to use for DCG and LambdaMart calculations [DescendingStablePessimistic/DescendingStable/DescendingReverse/DescendingDotNet]",
                ShortName = "sort",
                Hide = true)]
            [TGUI(NotGui = true)]
            internal string SortingAlgorithm = "DescendingStablePessimistic";

            /// <summary>
            /// The maximum NDCG truncation to use in the
            /// <a href="https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf">LambdaMAR algorithm</a>.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "max-NDCG truncation to use in the LambdaMART algorithm", ShortName = "n", Hide = true)]
            [TGUI(NotGui = true)]
            public int NdcgTruncationLevel = 100;

            [BestFriend]
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Use shifted NDCG", Hide = true)]
            [TGUI(NotGui = true)]
            internal bool ShiftedNdcg;

            [BestFriend]
            [Argument(ArgumentType.AtMostOnce, HelpText = "Cost function parameter (w/c)", ShortName = "cf", Hide = true)]
            [TGUI(NotGui = true)]
            internal char CostFunctionParam = 'w';

            [BestFriend]
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Distance weight 2 adjustment to cost", ShortName = "dw", Hide = true)]
            [TGUI(NotGui = true)]
            internal bool DistanceWeight2;

            [BestFriend]
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Normalize query lambdas", ShortName = "nql", Hide = true)]
            [TGUI(NotGui = true)]
            internal bool NormalizeQueryLambdas;

            /// <summary>
            /// internal state of <see cref="EarlyStoppingMetric"/>. It should be always synced with
            /// <see cref="BoostedTreeOptions.EarlyStoppingMetrics"/>.
            /// </summary>
            // Disable 649 because Visual Studio can't detect its assignment via property.
            #pragma warning disable 649
            private EarlyStoppingRankingMetric _earlyStoppingMetric;
            #pragma warning restore 649

            /// <summary>
            /// Early stopping metrics.
            /// </summary>
            public EarlyStoppingRankingMetric EarlyStoppingMetric
            {
                get { return _earlyStoppingMetric; }

                set
                {
                    // Update the state of the user-facing stopping metric.
                    _earlyStoppingMetric = value;
                    // Set up internal property according to its public value.
                    EarlyStoppingMetrics = (int)_earlyStoppingMetric;
                }
            }

            /// <summary>
            /// Create a new <see cref="Options"/> object with default values.
            /// </summary>
            public Options()
            {
                EarlyStoppingMetric = EarlyStoppingRankingMetric.NdcgAt1; // Use L1 by default.
            }

            ITrainer IComponentFactory<ITrainer>.CreateComponent(IHostEnvironment env) => new FastTreeRankingTrainer(env, this);

            internal override void Check(IExceptionContext ectx)
            {
                base.Check(ectx);

                ectx.CheckUserArg(SortingAlgorithm == "DescendingStable"
                    || SortingAlgorithm == "DescendingReverse"
                    || SortingAlgorithm == "DescendingDotNet"
                    || SortingAlgorithm == "DescendingStablePessimistic",
                    nameof(SortingAlgorithm),
                        "The specified sorting algorithm is invalid. Only 'DescendingStable', 'DescendingReverse', " +
                        "'DescendingDotNet', and 'DescendingStablePessimistic' are supported.");
#if OLD_DATALOAD
                ectx.CheckUserArg(0 <= secondaryMetricShare && secondaryMetricShare <= 1, "secondaryMetricShare", "secondaryMetricShare must be between 0 and 1.");
#endif
                ectx.CheckUserArg(0 < NdcgTruncationLevel, nameof(NdcgTruncationLevel), "must be positive.");
            }
        }
    }

    public enum Bundle : byte
    {
        None = 0,
        AggregateLowPopulation = 1,
        Adjacent = 2
    }

    [BestFriend]
    internal static class Defaults
    {
        public const int NumberOfTrees = 100;
        public const int NumberOfLeaves = 20;
        public const int MinimumExampleCountPerLeaf = 10;
        public const double LearningRate = 0.2;
    }

    /// <summary>
    /// Options for tree trainers.
    /// </summary>
    public abstract class TreeOptions : TrainerInputBaseWithGroupId
    {
        /// <summary>
        /// Allows to choose Parallel FastTree Learning Algorithm.
        /// </summary>
        [Argument(ArgumentType.Multiple, HelpText = "Allows to choose Parallel FastTree Learning Algorithm", ShortName = "parag")]
        internal ISupportParallelTraining ParallelTrainer = new SingleTrainerFactory();

        /// <summary>
        /// The number of threads to use.
        /// </summary>
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "The number of threads to use", ShortName = "t", NullName = "<Auto>")]
        public int? NumberOfThreads = null;

        // this random seed is used for:
        // 1. example sampling for feature binning
        // 2. init Randomize Score
        // 3. grad Sampling Rate in Objective Function
        // 4. tree learner
        // 5. bagging provider
        // 6. emsemble compressor
        /// <summary>
        /// The seed of the random number generator.
        /// </summary>
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "The seed of the random number generator", ShortName = "r1")]
        public int Seed = 123;

        // this random seed is only for active feature selection
        /// <summary>
        /// The seed of the active feature selection.
        /// </summary>
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "The seed of the active feature selection", ShortName = "r3", Hide = true)]
        [TGUI(NotGui = true)]
        public int FeatureSelectionSeed = 123;

        /// <summary>
        /// The entropy (regularization) coefficient between 0 and 1.
        /// </summary>
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "The entropy (regularization) coefficient between 0 and 1", ShortName = "e")]
        public Double EntropyCoefficient;

        // REVIEW: Different short name from TLC FR arguments.
        /// <summary>
        /// The number of histograms in the pool (between 2 and numLeaves).
        /// </summary>
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "The number of histograms in the pool (between 2 and numLeaves)", ShortName = "ps")]
        public int HistogramPoolSize = -1;

        /// <summary>
        /// Whether to utilize the disk or the data's native transposition facilities (where applicable) when performing the transpose.
        /// </summary>
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Whether to utilize the disk or the data's native transposition facilities (where applicable) when performing the transpose", ShortName = "dt")]
        public bool? DiskTranspose;

        /// <summary>
        /// Whether to collectivize features during dataset preparation to speed up training.
        /// </summary>
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Whether to collectivize features during dataset preparation to speed up training", ShortName = "flocks", Hide = true)]
        public bool FeatureFlocks = true;

        /// <summary>
        /// Whether to do split based on multiple categorical feature values.
        /// </summary>
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Whether to do split based on multiple categorical feature values.", ShortName = "cat")]
        public bool CategoricalSplit = false;

        /// <summary>
        /// Maximum categorical split groups to consider when splitting on a categorical feature. Split groups are a collection of split points. This is used to reduce overfitting when there many categorical features.
        /// </summary>
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Maximum categorical split groups to consider when splitting on a categorical feature. " +
                                                             "Split groups are a collection of split points. This is used to reduce overfitting when " +
                                                             "there many categorical features.", ShortName = "mcg")]
        public int MaximumCategoricalGroupCountPerNode = 64;

        /// <summary>
        /// Maximum categorical split points to consider when splitting on a categorical feature.
        /// </summary>
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Maximum categorical split points to consider when splitting on a categorical feature.", ShortName = "maxcat")]
        public int MaximumCategoricalSplitPointCount = 64;

        /// <summary>
        /// Minimum categorical example percentage in a bin to consider for a split. Default is 0.1% of all training examples.
        /// </summary>
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Minimum categorical example percentage in a bin to consider for a split.", ShortName = "mdop")]
        public double MinimumExampleFractionForCategoricalSplit = 0.001;

        /// <summary>
        /// Minimum categorical example count in a bin to consider for a split.
        /// </summary>
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Minimum categorical example count in a bin to consider for a split.", ShortName = "mdo")]
        public int MinimumExamplesForCategoricalSplit = 100;

        /// <summary>
        /// Bias for calculating gradient for each feature bin for a categorical feature.
        /// </summary>
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Bias for calculating gradient for each feature bin for a categorical feature.", ShortName = "bias")]
        public double Bias = 0;

        /// <summary>
        /// Bundle low population bins. Bundle.None(0): no bundling, Bundle.AggregateLowPopulation(1): Bundle low population, Bundle.Adjacent(2): Neighbor low population bundle.
        /// </summary>
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Bundle low population bins. " +
                                                             "Bundle.None(0): no bundling, " +
                                                             "Bundle.AggregateLowPopulation(1): Bundle low population, " +
                                                             "Bundle.Adjacent(2): Neighbor low population bundle.", ShortName = "bundle")]
        public Bundle Bundling = Bundle.None;

        // REVIEW: Different default from TLC FR. I prefer the TLC FR default of 255.
        // REVIEW: Reverting back to 255 to make the same defaults of FR.
        /// <summary>
        /// Maximum number of distinct values (bins) per feature.
        /// </summary>
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Maximum number of distinct values (bins) per feature", ShortName = "mb")]
        public int MaximumBinCountPerFeature = 255;  // save one for undefs

        /// <summary>
        /// Sparsity level needed to use sparse feature representation.
        /// </summary>
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Sparsity level needed to use sparse feature representation", ShortName = "sp")]
        public Double SparsifyThreshold = 0.7;

        /// <summary>
        /// The feature first use penalty coefficient.
        /// </summary>
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "The feature first use penalty coefficient", ShortName = "ffup")]
        public Double FeatureFirstUsePenalty;

        /// <summary>
        /// The feature re-use penalty (regularization) coefficient.
        /// </summary>
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "The feature re-use penalty (regularization) coefficient", ShortName = "frup")]
        public Double FeatureReusePenalty;

        /// <summary>
        /// Tree fitting gain confidence requirement. Only consider a gain if its likelihood versus a random choice gain is above this value.
        /// </summary>
        /// <value>
        /// Value of 0.95 would mean restricting to gains that have less than a 0.05 chance of being generated randomly through choice of a random split.
        /// Valid range is [0,1).
        /// </value>
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Tree fitting gain confidence requirement (should be in the range [0,1) ).", ShortName = "gainconf")]
        public Double GainConfidenceLevel;

        /// <summary>
        /// The temperature of the randomized softmax distribution for choosing the feature.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "The temperature of the randomized softmax distribution for choosing the feature", ShortName = "smtemp")]
        public Double SoftmaxTemperature;

        /// <summary>
        /// Print execution time breakdown to ML.NET channel.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Print execution time breakdown to stdout", ShortName = "et")]
        public bool ExecutionTime;

        // REVIEW: Different from original FastRank arguments (shortname l vs. nl). Different default from TLC FR Wrapper (20 vs. 20).
        /// <summary>
        /// The max number of leaves in each regression tree.
        /// </summary>
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "The max number of leaves in each regression tree", ShortName = "nl", SortOrder = 2)]
        [TGUI(Description = "The maximum number of leaves per tree", SuggestedSweeps = "2-128;log;inc:4")]
        [TlcModule.SweepableLongParamAttribute("NumLeaves", 2, 128, isLogScale: true, stepSize: 4)]
        public int NumberOfLeaves = Defaults.NumberOfLeaves;

        /// <summary>
        /// The minimal number of data points required to form a new tree leaf.
        /// </summary>
        // REVIEW: Arrays not supported in GUI
        // REVIEW: Different shortname than FastRank module. Same as the TLC FRWrapper.
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "The minimal number of examples allowed in a leaf of a regression tree, out of the subsampled data", ShortName = "mil", SortOrder = 3)]
        [TGUI(Description = "Minimum number of training instances required to form a leaf", SuggestedSweeps = "1,10,50")]
        [TlcModule.SweepableDiscreteParamAttribute("MinDocumentsInLeafs", new object[] { 1, 10, 50 })]
        public int MinimumExampleCountPerLeaf = Defaults.MinimumExampleCountPerLeaf;

        /// <summary>
        /// Total number of decision trees to create in the ensemble.
        /// </summary>
        // REVIEW: Different shortname than FastRank module. Same as the TLC FRWrapper.
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Total number of decision trees to create in the ensemble", ShortName = "iter", SortOrder = 1)]
        [TGUI(Description = "Total number of trees constructed", SuggestedSweeps = "20,100,500")]
        [TlcModule.SweepableDiscreteParamAttribute("NumTrees", new object[] { 20, 100, 500 })]
        public int NumberOfTrees = Defaults.NumberOfTrees;

        /// <summary>
        /// The fraction of features (chosen randomly) to use on each iteration. Use 0.9 if only 90% of features is needed.
        /// Lower numbers help reduce over-fitting.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "The fraction of features (chosen randomly) to use on each iteration", ShortName = "ff")]
        public Double FeatureFraction = 1;

        /// <summary>
        /// Number of trees in each bag (0 for disabling bagging).
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Number of trees in each bag (0 for disabling bagging)", ShortName = "bag")]
        public int BaggingSize;

        /// <summary>
        /// Percentage of training examples used in each bag. Default is 0.7 (70%).
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Percentage of training examples used in each bag", ShortName = "bagfrac")]
        // REVIEW: sweeping bagfrac doesn't make sense unless 'baggingSize' is non-zero. The 'SuggestedSweeps' here
        // are used to denote 'sensible range', but the GUI will interpret this as 'you must sweep these values'. So, I'm keeping
        // the values there for the future, when we have an appropriate way to encode this information.
        // [TGUI(SuggestedSweeps = "0.5,0.7,0.9")]
        public Double BaggingExampleFraction = 0.7;

        /// <summary>
        /// The fraction of features (chosen randomly) to use on each split. If it's value is 0.9, 90% of all features would be dropped in expectation.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "The fraction of features (chosen randomly) to use on each split", ShortName = "sf")]
        public Double FeatureFractionPerSplit = 1;

        /// <summary>
        /// Smoothing parameter for tree regularization.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Smoothing paramter for tree regularization", ShortName = "s")]
        public Double Smoothing;

        /// <summary>
        /// When a root split is impossible, allow training to proceed.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "When a root split is impossible, allow training to proceed", ShortName = "allowempty,dummies", Hide = true)]
        [TGUI(NotGui = true)]
        public bool AllowEmptyTrees = true;

        /// <summary>
        /// The level of feature compression to use.
        /// </summary>
        [BestFriend]
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "The level of feature compression to use", ShortName = "fcomp", Hide = true)]
        [TGUI(NotGui = true)]
        internal int FeatureCompressionLevel = 1;

        /// <summary>
        /// Compress the tree Ensemble.
        /// </summary>
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Compress the tree Ensemble", ShortName = "cmp", Hide = true)]
        [TGUI(NotGui = true)]
        public bool CompressEnsemble;

        /// <summary>
        /// Print metrics graph for the first test set.
        /// </summary>
        [BestFriend]
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Print metrics graph for the first test set", ShortName = "graph", Hide = true)]
        [TGUI(NotGui = true)]
        internal bool PrintTestGraph;

        /// <summary>
        /// Print Train and Validation metrics in graph.
        /// </summary>
        //It is only enabled if printTestGraph is also set
        [BestFriend]
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Print Train and Validation metrics in graph", ShortName = "graphtv", Hide = true)]
        [TGUI(NotGui = true)]
        internal bool PrintTrainValidGraph;

        /// <summary>
        /// Calculate metric values for train/valid/test every k rounds.
        /// </summary>
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Calculate metric values for train/valid/test every k rounds", ShortName = "tf")]
        public int TestFrequency = int.MaxValue;

        internal virtual void Check(IExceptionContext ectx)
        {
            Contracts.AssertValue(ectx);
            ectx.CheckUserArg(NumberOfThreads == null || NumberOfThreads > 0, nameof(NumberOfThreads), "Must be positive.");
            ectx.CheckUserArg(NumberOfLeaves >= 2, nameof(NumberOfLeaves), "Must be at least 2.");
            ectx.CheckUserArg(0 <= EntropyCoefficient && EntropyCoefficient <= 1, nameof(EntropyCoefficient), "Must be between 0 and 1.");
            ectx.CheckUserArg(0 <= GainConfidenceLevel && GainConfidenceLevel < 1, nameof(GainConfidenceLevel), "Must be in [0, 1).");
            ectx.CheckUserArg(0 <= FeatureFraction && FeatureFraction <= 1, nameof(FeatureFraction), "Must be between 0 and 1.");
            ectx.CheckUserArg(0 <= FeatureFractionPerSplit && FeatureFractionPerSplit <= 1, nameof(FeatureFractionPerSplit), "Must be between 0 and 1.");
            ectx.CheckUserArg(0 <= SoftmaxTemperature, nameof(SoftmaxTemperature), "Must be non-negative.");
            ectx.CheckUserArg(0 < MaximumBinCountPerFeature, nameof(MaximumBinCountPerFeature), "Must greater than 0.");
            ectx.CheckUserArg(0 <= SparsifyThreshold && SparsifyThreshold <= 1, nameof(SparsifyThreshold), "Must be between 0 and 1.");
            ectx.CheckUserArg(0 < NumberOfTrees, nameof(NumberOfTrees), "Must be positive.");
            ectx.CheckUserArg(0 <= Smoothing && Smoothing <= 1, nameof(Smoothing), "Must be between 0 and 1.");
            ectx.CheckUserArg(0 <= BaggingSize, nameof(BaggingSize), "Must be non-negative.");
            ectx.CheckUserArg(0 <= BaggingExampleFraction && BaggingExampleFraction <= 1, nameof(BaggingExampleFraction), "Must be between 0 and 1.");
            ectx.CheckUserArg(0 <= FeatureFirstUsePenalty, nameof(FeatureFirstUsePenalty), "Must be non-negative.");
            ectx.CheckUserArg(0 <= FeatureReusePenalty, nameof(FeatureReusePenalty), "Must be non-negative.");
            ectx.CheckUserArg(0 <= MaximumCategoricalGroupCountPerNode, nameof(MaximumCategoricalGroupCountPerNode), "Must be non-negative.");
            ectx.CheckUserArg(0 <= MaximumCategoricalSplitPointCount, nameof(MaximumCategoricalSplitPointCount), "Must be non-negative.");
            ectx.CheckUserArg(0 <= MinimumExampleFractionForCategoricalSplit, nameof(MinimumExampleFractionForCategoricalSplit), "Must be non-negative.");
            ectx.CheckUserArg(0 <= MinimumExamplesForCategoricalSplit, nameof(MinimumExamplesForCategoricalSplit), "Must be non-negative.");
            ectx.CheckUserArg(Bundle.None <= Bundling && Bundling <= Bundle.Adjacent, nameof(Bundling), "Must be between 0 and 2.");
            ectx.CheckUserArg(Bias >= 0, nameof(Bias), "Must be greater than equal to zero.");
        }
    }

    /// <summary>
    /// Options for boosting tree trainers.
    /// </summary>
    public abstract class BoostedTreeOptions : TreeOptions
    {
        // REVIEW: TLC FR likes to call it bestStepRegressionTrees which might be more appropriate.
        //Use the second derivative for split gains (not just outputs). Use MaxTreeOutput to "clip" cases where the second derivative is too close to zero.
        //Turning BSR on makes larger steps in initial stages and converges to better results with fewer trees (though in the end, it asymptotes to the same results).
        /// <summary>
        /// Option for using best regression step trees.
        /// </summary>
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Option for using best regression step trees", ShortName = "bsr")]
        public bool BestStepRankingRegressionTrees = false;

        /// <summary>
        /// Determines whether to use line search for a step size.
        /// </summary>
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Should we use line search for a step size", ShortName = "ls")]
        public bool UseLineSearch;

        /// <summary>
        /// Number of post-bracket line search steps.
        /// </summary>
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Number of post-bracket line search steps", ShortName = "lssteps")]
        public int MaximumNumberOfLineSearchSteps;

        /// <summary>
        /// Minimum line search step size.
        /// </summary>
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Minimum line search step size", ShortName = "minstep")]
        public Double MinimumStepSize;

        /// <summary>
        /// Types of optimization algorithms.
        /// </summary>
        public enum OptimizationAlgorithmType { GradientDescent, AcceleratedGradientDescent, ConjugateGradientDescent };

        /// <summary>
        /// Optimization algorithm to be used.
        /// </summary>
        /// <value>
        /// See <see cref="OptimizationAlgorithmType"/> for available optimizers.
        /// </value>
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Optimization algorithm to be used (GradientDescent, AcceleratedGradientDescent)", ShortName = "oa")]
        public OptimizationAlgorithmType OptimizationAlgorithm = OptimizationAlgorithmType.GradientDescent;

        /// <summary>
        /// Early stopping rule. (Validation set (/valid) is required).
        /// </summary>
        [BestFriend]
        [Argument(ArgumentType.Multiple, HelpText = "Early stopping rule. (Validation set (/valid) is required.)", Name = "EarlyStoppingRule", ShortName = "esr", NullName = "<Disable>")]
        [TGUI(Label = "Early Stopping Rule", Description = "Early stopping rule. (Validation set (/valid) is required.)")]
        internal IEarlyStoppingCriterionFactory EarlyStoppingRuleFactory;

        /// <summary>
        /// The underlying state of <see cref="EarlyStoppingRuleFactory"/> and <see cref="EarlyStoppingRule"/>.
        /// </summary>
        private EarlyStoppingRuleBase _earlyStoppingRuleBase;

        /// <summary>
        /// Early stopping rule used to terminate training process once meeting a specified criterion. Possible choices are
        /// <see cref="EarlyStoppingRuleBase"/>'s implementations such as <see cref="TolerantEarlyStoppingRule"/> and <see cref="GeneralityLossRule"/>.
        /// </summary>
        public EarlyStoppingRuleBase EarlyStoppingRule
        {
            get { return _earlyStoppingRuleBase;  }
            set
            {
                _earlyStoppingRuleBase = value;
                EarlyStoppingRuleFactory = _earlyStoppingRuleBase.BuildFactory();
            }
        }

        /// <summary>
        /// Early stopping metrics. (For regression, 1: L1, 2:L2; for ranking, 1:NDCG@1, 3:NDCG@3).
        /// </summary>
        [BestFriend]
        [Argument(ArgumentType.AtMostOnce, HelpText = "Early stopping metrics. (For regression, 1: L1, 2:L2; for ranking, 1:NDCG@1, 3:NDCG@3)", ShortName = "esmt")]
        [TGUI(Description = "Early stopping metrics. (For regression, 1: L1, 2:L2; for ranking, 1:NDCG@1, 3:NDCG@3)")]
        internal int EarlyStoppingMetrics;

        /// <summary>
        /// Enable post-training tree pruning to avoid overfitting. It requires a validation set.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Enable post-training pruning to avoid overfitting. (a validation set is required)", ShortName = "pruning")]
        public bool EnablePruning;

        /// <summary>
        /// Use window and tolerance for pruning.
        /// </summary>
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Use window and tolerance for pruning", ShortName = "prtol")]
        public bool UseTolerantPruning;

        /// <summary>
        /// The tolerance threshold for pruning.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "The tolerance threshold for pruning", ShortName = "prth")]
        [TGUI(Description = "Pruning threshold")]
        public double PruningThreshold = 0.004;

        /// <summary>
        /// The moving window size for pruning.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "The moving window size for pruning", ShortName = "prws")]
        [TGUI(Description = "Pruning window size")]
        public int PruningWindowSize = 5;

        /// <summary>
        /// The learning rate.
        /// </summary>
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "The learning rate", ShortName = "lr", SortOrder = 4)]
        [TGUI(Label = "Learning Rate", SuggestedSweeps = "0.025-0.4;log")]
        [TlcModule.SweepableFloatParamAttribute("LearningRates", 0.025f, 0.4f, isLogScale: true)]
        public double LearningRate = Defaults.LearningRate;

        /// <summary>
        /// Shrinkage.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Shrinkage", ShortName = "shrk")]
        [TGUI(Label = "Shrinkage", SuggestedSweeps = "0.25-4;log")]
        [TlcModule.SweepableFloatParamAttribute("Shrinkage", 0.025f, 4f, isLogScale: true)]
        public Double Shrinkage = 1;

        /// <summary>
        /// Dropout rate for tree regularization.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Dropout rate for tree regularization", ShortName = "tdrop")]
        [TGUI(SuggestedSweeps = "0,0.000000001,0.05,0.1,0.2")]
        [TlcModule.SweepableDiscreteParamAttribute("DropoutRate", new object[] { 0.0f, 1E-9f, 0.05f, 0.1f, 0.2f })]
        public Double DropoutRate = 0;

        /// <summary>
        /// Sample each query 1 in k times in the GetDerivatives function.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Sample each query 1 in k times in the GetDerivatives function", ShortName = "sr")]
        public int GetDerivativesSampleRate = 1;

        /// <summary>
        /// Write the last ensemble instead of the one determined by early stopping.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Write the last ensemble instead of the one determined by early stopping", ShortName = "hl")]
        public bool WriteLastEnsemble;

        /// <summary>
        /// Upper bound on absolute value of single tree output.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Upper bound on absolute value of single tree output", ShortName = "mo")]
        public Double MaximumTreeOutput = 100;

        /// <summary>
        /// Training starts from random ordering (determined by /r1).
        /// </summary>
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Training starts from random ordering (determined by /r1)", ShortName = "rs", Hide = true)]
        [TGUI(NotGui = true)]
        public bool RandomStart;

        /// <summary>
        /// Filter zero lambdas during training.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Filter zero lambdas during training", ShortName = "fzl", Hide = true)]
        [TGUI(NotGui = true)]
        public bool FilterZeroLambdas;

#if OLD_DATALOAD
        [Argument(ArgumentType.AtMostOnce, HelpText = "The proportion of the lambdas that should be secondary metrics", ShortName = "sfrac", Hide = true)]
        [TGUI(NotGUI = true)]
        public Double secondaryMetricShare;

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Secondary lambdas by default are calculated for all pairs; this makes them calculated only for those pairs with identical labels",
            ShortName = "secondexclusive", Hide = true)]
        [TGUI(NotGUI = true)]
        public bool secondaryIsolabelExclusive;

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Force garbage collection during feature extraction each time this many features are read", ShortName = "gcfe", Hide = true)]
        [TGUI(NotGUI = true)]
        public int forceGCFeatureExtraction = 100;
#endif

        /// <summary>
        /// Freeform defining the scores that should be used as the baseline ranker.
        /// </summary>
        [BestFriend]
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Freeform defining the scores that should be used as the baseline ranker", ShortName = "basescores", Hide = true)]
        [TGUI(NotGui = true)]
        internal string BaselineScoresFormula;

        /// <summary>
        /// Baseline alpha for tradeoffs of risk (0 is normal training).
        /// </summary>
        [BestFriend]
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Baseline alpha for tradeoffs of risk (0 is normal training)", ShortName = "basealpha", Hide = true)]
        [TGUI(NotGui = true)]
        internal string BaselineAlphaRisk;

        /// <summary>
        /// The discount freeform which specifies the per position discounts of examples in a query (uses a single variable P for position where P=0 is first position).
        /// </summary>
        [BestFriend]
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "The discount freeform which specifies the per position discounts of examples in a query (uses a single variable P for position where P=0 is first position)",
            ShortName = "pdff", Hide = true)]
        [TGUI(NotGui = true)]
        internal string PositionDiscountFreeform;

#if !NO_STORE
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Offload feature bins to a file store", ShortName = "fbsopt", Hide = true)]
        [TGUI(NotGUI = true)]
        public bool offloadBinsToFileStore;

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Directory used to offload feature bins", ShortName = "fbsoptdir", Hide = true)]
        [TGUI(NotGUI = true)]
        public string offloadBinsDirectory = string.Empty;

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Preloads feature bins needed for the next iteration when bins file store is used", ShortName = "fbsoptpreload", Hide = true)]
        [TGUI(NotGUI = true)]
        public bool preloadFeatureBinsBeforeTraining;
#endif

        internal override void Check(IExceptionContext ectx)
        {
            base.Check(ectx);

            ectx.CheckUserArg(0 <= MaximumTreeOutput, nameof(MaximumTreeOutput), "Must be non-negative.");
            ectx.CheckUserArg(0 <= PruningThreshold, nameof(PruningThreshold), "Must be non-negative.");
            ectx.CheckUserArg(0 < PruningWindowSize, nameof(PruningWindowSize), "Must be positive.");
            ectx.CheckUserArg(0 < Shrinkage, nameof(Shrinkage), "Must be positive.");
            ectx.CheckUserArg(0 <= DropoutRate && DropoutRate <= 1, nameof(DropoutRate), "Must be between 0 and 1.");
            ectx.CheckUserArg(0 < GetDerivativesSampleRate, nameof(GetDerivativesSampleRate), "Must be positive.");
            ectx.CheckUserArg(0 <= MaximumNumberOfLineSearchSteps, nameof(MaximumNumberOfLineSearchSteps), "Must be non-negative.");
            ectx.CheckUserArg(0 <= MinimumStepSize, nameof(MinimumStepSize), "Must be non-negative.");
        }
    }
}
