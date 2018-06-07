// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.FastTree;
using Microsoft.ML.Runtime.Internal.Internallearn;

[assembly: EntryPointModule(typeof(FastTreeBinaryClassificationTrainer.Arguments))]
[assembly: EntryPointModule(typeof(FastTreeRegressionTrainer.Arguments))]
[assembly: EntryPointModule(typeof(FastTreeTweedieTrainer.Arguments))]
[assembly: EntryPointModule(typeof(FastTreeRankingTrainer.Arguments))]

namespace Microsoft.ML.Runtime.FastTree
{
    [TlcModule.ComponentKind("FastTreeTrainer")]
    public interface IFastTreeTrainerFactory : IComponentFactory<ITrainer>
    {
    }

    public sealed partial class FastTreeBinaryClassificationTrainer
    {
        [TlcModule.Component(Name = LoadNameValue, FriendlyName = UserNameValue, Desc = Summary)]
        public sealed class Arguments : BoostedTreeArgs, IFastTreeTrainerFactory
        {
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Should we use derivatives optimized for unbalanced sets", ShortName = "us")]
            [TGUI(Label = "Optimize for unbalanced")]
            public bool UnbalancedSets = false;

            public ITrainer CreateComponent(IHostEnvironment env) => new FastTreeBinaryClassificationTrainer(env, this);
        }
    }

    public sealed partial class FastTreeRegressionTrainer
    {
        [TlcModule.Component(Name = LoadNameValue, FriendlyName = UserNameValue, Desc = Summary)]
        public sealed class Arguments : BoostedTreeArgs, IFastTreeTrainerFactory
        {
            public Arguments()
            {
                EarlyStoppingMetrics = 1; // Use L1 by default.
            }

            public ITrainer CreateComponent(IHostEnvironment env) => new FastTreeRegressionTrainer(env, this);
        }
    }

    public sealed partial class FastTreeTweedieTrainer
    {
        [TlcModule.Component(Name = LoadNameValue, FriendlyName = UserNameValue, Desc = Summary)]
        public sealed class Arguments : BoostedTreeArgs, IFastTreeTrainerFactory
        {
            // REVIEW: It is possible to estimate this index parameter from the distribution of data, using
            // a combination of univariate optimization and grid search, following section 4.2 of the paper. However
            // it is probably not worth doing unless and until explicitly asked for.
            [Argument(ArgumentType.LastOccurenceWins, HelpText =
                "Index parameter for the Tweedie distribution, in the range [1, 2]. 1 is Poisson loss, 2 is gamma loss, " +
                "and intermediate values are compound Poisson loss.")]
            public Double Index = 1.5;

            public ITrainer CreateComponent(IHostEnvironment env) => new FastTreeTweedieTrainer(env, this);
        }
    }

    public sealed partial class FastTreeRankingTrainer
    {
        [TlcModule.Component(Name = LoadNameValue, FriendlyName = UserNameValue, Desc = Summary)]
        public sealed class Arguments : BoostedTreeArgs, IFastTreeTrainerFactory
        {
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Comma seperated list of gains associated to each relevance label.", ShortName = "gains")]
            [TGUI(NoSweep = true)]
            public string CustomGains = "0,3,7,15,31";

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Train DCG instead of NDCG", ShortName = "dcg")]
            public bool TrainDcg;

            // REVIEW: Hiding sorting for now. Should be an enum or SubComponent.
            [Argument(ArgumentType.LastOccurenceWins,
                HelpText = "The sorting algorithm to use for DCG and LambdaMart calculations [DescendingStablePessimistic/DescendingStable/DescendingReverse/DescendingDotNet]",
                ShortName = "sort",
                Hide = true)]
            [TGUI(NotGui = true)]
            public string SortingAlgorithm = "DescendingStablePessimistic";

            [Argument(ArgumentType.AtMostOnce, HelpText = "max-NDCG truncation to use in the Lambda Mart algorithm", ShortName = "n", Hide = true)]
            [TGUI(NotGui = true)]
            public int LambdaMartMaxTruncation = 100;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Use shifted NDCG", Hide = true)]
            [TGUI(NotGui = true)]
            public bool ShiftedNdcg;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Cost function parameter (w/c)", ShortName = "cf", Hide = true)]
            [TGUI(NotGui = true)]
            public char CostFunctionParam = 'w';

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Distance weight 2 adjustment to cost", ShortName = "dw", Hide = true)]
            [TGUI(NotGui = true)]
            public bool DistanceWeight2;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Normalize query lambdas", ShortName = "nql", Hide = true)]
            [TGUI(NotGui = true)]
            public bool NormalizeQueryLambdas;

            public Arguments()
            {
                EarlyStoppingMetrics = 1;
            }

            public ITrainer CreateComponent(IHostEnvironment env) => new FastTreeRankingTrainer(env, this);

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
                ectx.CheckUserArg(0 < LambdaMartMaxTruncation, nameof(LambdaMartMaxTruncation), "lambdaMartMaxTruncation must be positive.");
            }
        }
    }

    public enum Bundle : Byte
    {
        None = 0,
        AggregateLowPopulation = 1,
        Adjacent = 2
    }

    public abstract class TreeArgs : LearnerInputBaseWithGroupId
    {
        [Argument(ArgumentType.Multiple, HelpText = "Allows to choose Parallel FastTree Learning Algorithm", ShortName = "parag")]
        public ISupportParallelTraining ParallelTrainer = new SingleTrainerFactory();

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "The number of threads to use", ShortName = "t", NullName = "<Auto>")]
        public int? NumThreads = null;

        // this random seed is used for:
        // 1. doc sampling for feature binning
        // 2. init Randomize Score
        // 3. grad Sampling Rate in Objective Function
        // 4. tree learner
        // 5. bagging provider
        // 6. emsemble compressor
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "The seed of the random number generator", ShortName = "r1")]
        public int RngSeed = 123;

        // this random seed is only for active feature selection
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "The seed of the active feature selection", ShortName = "r3", Hide = true)]
        [TGUI(NotGui = true)]
        public int FeatureSelectSeed = 123;

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "The entropy (regularization) coefficient between 0 and 1", ShortName = "e")]
        public Double EntropyCoefficient;

        // REVIEW: Different short name from TLC FR arguments.
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "The number of histograms in the pool (between 2 and numLeaves)", ShortName = "ps")]
        public int HistogramPoolSize = -1;

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Whether to utilize the disk or the data's native transposition facilities (where applicable) when performing the transpose", ShortName = "dt")]
        public bool? DiskTranspose;

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Whether to collectivize features during dataset preparation to speed up training", ShortName = "flocks", Hide = true)]
        public bool FeatureFlocks = true;

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Whether to do split based on multiple categorical feature values.", ShortName = "cat")]
        public bool CategoricalSplit = false;

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Maximum categorical split groups to consider when splitting on a categorical feature. " +
                                                             "Split groups are a collection of split points. This is used to reduce overfitting when " +
                                                             "there many categorical features.", ShortName = "mcg")]
        public int MaxCategoricalGroupsPerNode = 64;

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Maximum categorical split points to consider when splitting on a categorical feature.", ShortName = "maxcat")]
        public int MaxCategoricalSplitPoints = 64;

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Minimum categorical docs percentage in a bin to consider for a split.", ShortName = "mdop")]
        public double MinDocsPercentageForCategoricalSplit = 0.001;

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Minimum categorical doc count in a bin to consider for a split.", ShortName = "mdo")]
        public int MinDocsForCategoricalSplit = 100;

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Bias for calculating gradient for each feature bin for a categorical feature.", ShortName = "bias")]
        public double Bias = 0;

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Bundle low population bins. " +
                                                             "Bundle.None(0): no bundling, " +
                                                             "Bundle.AggregateLowPopulation(1): Bundle low population, " +
                                                             "Bundle.Adjacent(2): Neighbor low population bundle.", ShortName = "bundle")]
        public Bundle Bundling = Bundle.None;

        // REVIEW: Different default from TLC FR. I prefer the TLC FR default of 255.
        // REVIEW: Reverting back to 255 to make the same defaults of FR.
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Maximum number of distinct values (bins) per feature", ShortName = "mb")]
        public int MaxBins = 255;  // save one for undefs

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Sparsity level needed to use sparse feature representation", ShortName = "sp")]
        public Double SparsifyThreshold = 0.7;

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "The feature first use penalty coefficient", ShortName = "ffup")]
        public Double FeatureFirstUsePenalty;

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "The feature re-use penalty (regularization) coefficient", ShortName = "frup")]
        public Double FeatureReusePenalty;

        /// Only consider a gain if its likelihood versus a random choice gain is above a certain value.
        /// So 0.95 would mean restricting to gains that have less than a 0.05 change of being generated randomly through choice of a random split.
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Tree fitting gain confidence requirement (should be in the range [0,1) ).", ShortName = "gainconf")]
        public Double GainConfidenceLevel;

        [Argument(ArgumentType.AtMostOnce, HelpText = "The temperature of the randomized softmax distribution for choosing the feature", ShortName = "smtemp")]
        public Double SoftmaxTemperature;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Print execution time breakdown to stdout", ShortName = "et")]
        public bool ExecutionTimes;

        // REVIEW: Different from original FastRank arguments (shortname l vs. nl). Different default from TLC FR Wrapper (20 vs. 20).
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "The max number of leaves in each regression tree", ShortName = "nl", SortOrder = 2)]
        [TGUI(Description = "The maximum number of leaves per tree", SuggestedSweeps = "2-128;log;inc:4")]
        [TlcModule.SweepableLongParamAttribute("NumLeaves", 2, 128, isLogScale:true, stepSize:4)]
        public int NumLeaves = 20;

        // REVIEW: Arrays not supported in GUI
        // REVIEW: Different shortname than FastRank module. Same as the TLC FRWrapper.
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "The minimal number of documents allowed in a leaf of a regression tree, out of the subsampled data", ShortName = "mil", SortOrder = 3)]
        [TGUI(Description = "Minimum number of training instances required to form a leaf", SuggestedSweeps = "1,10,50")]
        [TlcModule.SweepableDiscreteParamAttribute("MinDocumentsInLeafs", new object[] {1, 10, 50})]
        public int MinDocumentsInLeafs = 10;

        // REVIEW: Different shortname than FastRank module. Same as the TLC FRWrapper.
		[Argument(ArgumentType.LastOccurenceWins, HelpText = "Total number of boosted trees to create in the ensemble", ShortName = "iter", SortOrder = 1)]
        [TGUI(Description = "Total number of trees constructed", SuggestedSweeps = "20,100,500")]
        [TlcModule.SweepableDiscreteParamAttribute("NumTrees", new object[] { 20, 100, 500 })]
        public int NumTrees = 100;

        [Argument(ArgumentType.AtMostOnce, HelpText = "The fraction of features (chosen randomly) to use on each iteration", ShortName = "ff")]
        public Double FeatureFraction = 1;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Number of trees in each bag (0 for disabling bagging)", ShortName = "bag")]
        public int BaggingSize;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Percentage of training examples used in each bag", ShortName = "bagfrac")]
        // REVIEW: sweeping bagfrac doesn't make sense unless 'baggingSize' is non-zero. The 'SuggestedSweeps' here
        // are used to denote 'sensible range', but the GUI will interpret this as 'you must sweep these values'. So, I'm keeping
        // the values there for the future, when we have an appropriate way to encode this information.
        // [TGUI(SuggestedSweeps = "0.5,0.7,0.9")]
        public Double BaggingTrainFraction = 0.7;

        [Argument(ArgumentType.AtMostOnce, HelpText = "The fraction of features (chosen randomly) to use on each split", ShortName = "sf")]
        public Double SplitFraction = 1;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Smoothing paramter for tree regularization", ShortName = "s")]
        public Double Smoothing;

        [Argument(ArgumentType.AtMostOnce, HelpText = "When a root split is impossible, allow training to proceed", ShortName = "allowempty,dummies", Hide = true)]
        [TGUI(NotGui = true)]
        public bool AllowEmptyTrees = true;

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "The level of feature compression to use", ShortName = "fcomp", Hide = true)]
        [TGUI(NotGui = true)]
        public int FeatureCompressionLevel = 1;

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Compress the tree Ensemble", ShortName = "cmp", Hide = true)]
        [TGUI(NotGui = true)]
        public bool CompressEnsemble;

        // REVIEW: Not used.
        [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum Number of trees after compression", ShortName = "cmpmax", Hide = true)]
        [TGUI(NotGui = true)]
        public int MaxTreesAfterCompression = -1;

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Print metrics graph for the first test set", ShortName = "graph", Hide = true)]
        [TGUI(NotGui = true)]
        public bool PrintTestGraph;

        //It is only enabled if printTestGraph is also set
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Print Train and Validation metrics in graph", ShortName = "graphtv", Hide = true)]
        [TGUI(NotGui = true)]
        public bool PrintTrainValidGraph;

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Calculate metric values for train/valid/test every k rounds", ShortName = "tf")]
        public int TestFrequency = int.MaxValue;

        internal virtual void Check(IExceptionContext ectx)
        {
            Contracts.AssertValue(ectx);
            ectx.CheckUserArg(NumThreads == null || NumThreads > 0, nameof(NumThreads), "numThreads must be positive.");
            ectx.CheckUserArg(NumLeaves >= 2, nameof(NumLeaves), "numLeaves must be at least 2.");
            ectx.CheckUserArg(0 <= EntropyCoefficient && EntropyCoefficient <= 1, nameof(EntropyCoefficient), "entropyCoefficient must be between 0 and 1.");
            ectx.CheckUserArg(0 <= GainConfidenceLevel && GainConfidenceLevel < 1, nameof(GainConfidenceLevel), "gainConfidenceLevel must be in [0, 1).");
            ectx.CheckUserArg(0 <= FeatureFraction && FeatureFraction <= 1, nameof(FeatureFraction), "featureFraction must be between 0 and 1.");
            ectx.CheckUserArg(0 <= SplitFraction && SplitFraction <= 1, nameof(SplitFraction), "splitFraction must be between 0 and 1.");
            ectx.CheckUserArg(0 <= SoftmaxTemperature, nameof(SoftmaxTemperature), "softmaxTemperature must be non-negative.");
            ectx.CheckUserArg(0 < MaxBins, nameof(MaxBins), "maxBins must greater than 0.");
            ectx.CheckUserArg(0 <= SparsifyThreshold && SparsifyThreshold <= 1, nameof(SparsifyThreshold), "specifyThreshold must be between 0 and 1.");
            ectx.CheckUserArg(0 < NumTrees, nameof(NumTrees), "Number of trees must be positive.");
            ectx.CheckUserArg(0 <= Smoothing && Smoothing <= 1, nameof(Smoothing), "smoothing must be between 0 and 1.");
            ectx.CheckUserArg(0 <= BaggingSize, nameof(BaggingSize), "baggingSize must be non-negative.");
            ectx.CheckUserArg(0 <= BaggingTrainFraction && BaggingTrainFraction <= 1, nameof(BaggingTrainFraction), "baggingTrainFraction must be between 0 and 1.");
            ectx.CheckUserArg(0 <= FeatureFirstUsePenalty, nameof(FeatureFirstUsePenalty), "featureFirstUsePenalty must be non-negative.");
            ectx.CheckUserArg(0 <= FeatureReusePenalty, nameof(FeatureReusePenalty), "featureReusePenalty must be non-negative.");
            ectx.CheckUserArg(0 <= MaxCategoricalGroupsPerNode, nameof(MaxCategoricalGroupsPerNode), "maxCategoricalGroupsPerNode must be non-negative.");
            ectx.CheckUserArg(0 <= MaxCategoricalSplitPoints, nameof(MaxCategoricalSplitPoints), "maxCategoricalSplitPoints must be non-negative.");
            ectx.CheckUserArg(0 <= MinDocsPercentageForCategoricalSplit, nameof(MinDocsPercentageForCategoricalSplit), "minDocsPercentageForCategoricalSplit must be non-negative.");
            ectx.CheckUserArg(0 <= MinDocsForCategoricalSplit, nameof(MinDocsForCategoricalSplit), "minDocsForCategoricalSplit must be non-negative.");
            ectx.CheckUserArg(Bundle.None <= Bundling && Bundling <= Bundle.Adjacent, nameof(Bundling), "bundling must be between 0 and 2.");
            ectx.CheckUserArg(Bias >= 0, nameof(Bias), "Bias must be greater than equal to zero.");
        }
    }

    public abstract class BoostedTreeArgs : TreeArgs
    {
        // REVIEW: TLC FR likes to call it bestStepRegressionTrees which might be more appropriate.
        //Use the second derivative for split gains (not just outputs). Use MaxTreeOutput to "clip" cases where the second derivative is too close to zero.
        //Turning BSR on makes larger steps in initial stages and converges to better results with fewer trees (though in the end, it asymptotes to the same results).
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Use best regression step trees?", ShortName = "bsr")]
        public bool BestStepRankingRegressionTrees = false;

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Should we use line search for a step size", ShortName = "ls")]
        public bool UseLineSearch;

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Number of post-bracket line search steps", ShortName = "lssteps")]
        public int NumPostBracketSteps;

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Minimum line search step size", ShortName = "minstep")]
        public Double MinStepSize;

        public enum OptimizationAlgorithmType { GradientDescent, AcceleratedGradientDescent, ConjugateGradientDescent };
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Optimization algorithm to be used (GradientDescent, AcceleratedGradientDescent)", ShortName = "oa")]
        public OptimizationAlgorithmType OptimizationAlgorithm = OptimizationAlgorithmType.GradientDescent;

        [Argument(ArgumentType.Multiple, HelpText = "Early stopping rule. (Validation set (/valid) is required.)", ShortName = "esr", NullName = "<Disable>")]
        [TGUI(Label = "Early Stopping Rule", Description = "Early stopping rule. (Validation set (/valid) is required.)")]
        public IEarlyStoppingCriterionFactory EarlyStoppingRule;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Early stopping metrics. (For regression, 1: L1, 2:L2; for ranking, 1:NDCG@1, 3:NDCG@3)", ShortName = "esmt")]
        [TGUI(Description = "Early stopping metrics. (For regression, 1: L1, 2:L2; for ranking, 1:NDCG@1, 3:NDCG@3)")]
        public int EarlyStoppingMetrics;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Enable post-training pruning to avoid overfitting. (a validation set is required)", ShortName = "pruning")]
        public bool EnablePruning;

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Use window and tolerance for pruning", ShortName = "prtol")]
        public bool UseTolerantPruning;

        [Argument(ArgumentType.AtMostOnce, HelpText = "The tolerance threshold for pruning", ShortName = "prth")]
        [TGUI(Description = "Pruning threshold")]
        public Double PruningThreshold = 0.004;

        [Argument(ArgumentType.AtMostOnce, HelpText = "The moving window size for pruning", ShortName = "prws")]
        [TGUI(Description = "Pruning window size")]
        public int PruningWindowSize = 5;

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "The learning rate", ShortName = "lr", SortOrder = 4)]
        [TGUI(Label = "Learning Rate", SuggestedSweeps = "0.025-0.4;log")]
        [TlcModule.SweepableFloatParamAttribute("LearningRates", 0.025f, 0.4f, isLogScale:true)]
        public Double LearningRates = 0.2;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Shrinkage", ShortName = "shrk")]
        [TGUI(Label = "Shrinkage", SuggestedSweeps = "0.25-4;log")]
        [TlcModule.SweepableFloatParamAttribute("Shrinkage", 0.025f, 4f, isLogScale:true)]
        public Double Shrinkage = 1;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Dropout rate for tree regularization", ShortName = "tdrop")]
        [TGUI(SuggestedSweeps = "0,0.000000001,0.05,0.1,0.2")]
        [TlcModule.SweepableDiscreteParamAttribute("DropoutRate", new object[] { 0.0f, 1E-9f, 0.05f, 0.1f, 0.2f})]
        public Double DropoutRate = 0;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Sample each query 1 in k times in the GetDerivatives function", ShortName = "sr")]
        public int GetDerivativesSampleRate = 1;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Write the last ensemble instead of the one determined by early stopping", ShortName = "hl")]
        public bool WriteLastEnsemble;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Upper bound on absolute value of single tree output", ShortName = "mo")]
        public Double MaxTreeOutput = 100;

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Training starts from random ordering (determined by /r1)", ShortName = "rs", Hide = true)]
        [TGUI(NotGui = true)]
        public bool RandomStart;

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

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Freeform defining the scores that should be used as the baseline ranker", ShortName = "basescores", Hide = true)]
        [TGUI(NotGui = true)]
        public string BaselineScoresFormula;

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Baseline alpha for tradeoffs of risk (0 is normal training)", ShortName = "basealpha", Hide = true)]
        [TGUI(NotGui = true)]
        public string BaselineAlphaRisk;

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "The discount freeform which specifies the per position discounts of documents in a query (uses a single variable P for position where P=0 is first position)",
            ShortName = "pdff", Hide = true)]
        [TGUI(NotGui = true)]
        public string PositionDiscountFreeform;

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

            ectx.CheckUserArg(0 <= MaxTreeOutput, nameof(MaxTreeOutput), "maxTreeOutput must be non-negative.");
            ectx.CheckUserArg(0 <= PruningThreshold, nameof(PruningThreshold), "pruningThreshold must be non-negative.");
            ectx.CheckUserArg(0 < PruningWindowSize, nameof(PruningWindowSize), "pruningWindowSize must be positive.");
            ectx.CheckUserArg(0 < Shrinkage, nameof(Shrinkage), "shrinkage must be positive.");
            ectx.CheckUserArg(0 <= DropoutRate && DropoutRate <= 1, nameof(DropoutRate), "dropoutRate must be between 0 and 1.");
            ectx.CheckUserArg(0 < GetDerivativesSampleRate, nameof(GetDerivativesSampleRate), "getDerivativesSampleRate must be positive.");
            ectx.CheckUserArg(0 <= NumPostBracketSteps, nameof(NumPostBracketSteps), "numPostBracketSteps must be non-negative.");
            ectx.CheckUserArg(0 <= MinStepSize, nameof(MinStepSize), "minStepSize must be non-negative.");
        }
    }
}
