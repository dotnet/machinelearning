// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Reflection;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.LightGbm;

[assembly: LoadableClass(typeof(Options.TreeBooster), typeof(Options.TreeBooster.Options),
    typeof(SignatureLightGBMBooster), Options.TreeBooster.FriendlyName, Options.TreeBooster.Name)]
[assembly: LoadableClass(typeof(Options.DartBooster), typeof(Options.DartBooster.Options),
    typeof(SignatureLightGBMBooster), Options.DartBooster.FriendlyName, Options.DartBooster.Name)]
[assembly: LoadableClass(typeof(Options.GossBooster), typeof(Options.GossBooster.Options),
    typeof(SignatureLightGBMBooster), Options.GossBooster.FriendlyName, Options.GossBooster.Name)]

[assembly: EntryPointModule(typeof(Options.TreeBooster.Options))]
[assembly: EntryPointModule(typeof(Options.DartBooster.Options))]
[assembly: EntryPointModule(typeof(Options.GossBooster.Options))]

namespace Microsoft.ML.Trainers.LightGbm
{
    internal delegate void SignatureLightGBMBooster();

    [TlcModule.ComponentKind("BoosterParameterFunction")]
    public interface ISupportBoosterParameterFactory : IComponentFactory<IBoosterParameter>
    {
    }

    public interface IBoosterParameter
    {
        void UpdateParameters(Dictionary<string, object> res);
    }

    /// <summary>
    /// Options for LightGBM trainer.
    /// </summary>
    /// <remarks>
    /// LightGBM is an external library that's integrated with ML.NET. For detailed information about the parameters
    /// please see https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst.
    /// </remarks>
    public sealed class Options : TrainerInputBaseWithGroupId
    {
        public abstract class BoosterParameter<TOptions> : IBoosterParameter
            where TOptions : class, new()
        {
            private protected TOptions BoosterParameterOptions { get; }

            private protected BoosterParameter(TOptions options)
            {
                BoosterParameterOptions = options;
            }

            /// <summary>
            /// Update the parameters by specific Booster, will update parameters into "res" directly.
            /// </summary>
            internal virtual void UpdateParameters(Dictionary<string, object> res)
            {
                FieldInfo[] fields = BoosterParameterOptions.GetType().GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                foreach (var field in fields)
                {
                    var attribute = field.GetCustomAttribute<ArgumentAttribute>(false);

                    if (attribute == null)
                        continue;

                    res[GetOptionName(field.Name)] = field.GetValue(BoosterParameterOptions);
                }
            }

            void IBoosterParameter.UpdateParameters(Dictionary<string, object> res) => UpdateParameters(res);
        }

        private static string GetOptionName(string name)
        {
            if (_nameMapping.ContainsKey(name))
                return _nameMapping[name];

            // Otherwise convert the name to the light gbm argument
            StringBuilder strBuf = new StringBuilder();
            bool first = true;
            foreach (char c in name)
            {
                if (char.IsUpper(c))
                {
                    if (first)
                        first = false;
                    else
                        strBuf.Append('_');
                    strBuf.Append(char.ToLower(c));
                }
                else
                    strBuf.Append(c);
            }
            return strBuf.ToString();
        }

        // Static override name map that maps friendly names to lightGBM arguments.
        // If an argument is not here, then its name is identical to a lightGBM argument
        // and does not require a mapping, for example, Subsample.
        private static Dictionary<string, string> _nameMapping = new Dictionary<string, string>()
        {
           {nameof(TreeBooster.Options.MinimumSplitGain),               "min_split_gain" },
           {nameof(TreeBooster.Options.MaximumTreeDepth),               "max_depth"},
           {nameof(TreeBooster.Options.MinimumChildWeight),             "min_child_weight"},
           {nameof(TreeBooster.Options.SubsampleFraction),              "subsample"},
           {nameof(TreeBooster.Options.SubsampleFrequency),             "subsample_freq"},
           {nameof(TreeBooster.Options.L1Regularization),               "reg_alpha"},
           {nameof(TreeBooster.Options.L2Regularization),               "reg_lambda"},
           {nameof(TreeBooster.Options.WeightOfPositiveExamples),       "scale_pos_weight"},
           {nameof(DartBooster.Options.TreeDropFraction),               "drop_rate" },
           {nameof(DartBooster.Options.MaximumNumberOfDroppedTreesPerRound),"max_drop" },
           {nameof(DartBooster.Options.SkipDropFraction),               "skip_drop" },
           {nameof(MinimumExampleCountPerLeaf),                         "min_data_per_leaf"},
           {nameof(NumberOfLeaves),                                     "num_leaves"},
           {nameof(MaximumBinCountPerFeature),                          "max_bin" },
           {nameof(CustomGains),                                        "label_gain" },
           {nameof(MinimumExampleCountPerGroup),                        "min_data_per_group" },
           {nameof(MaximumCategoricalSplitPointCount),                  "max_cat_threshold" },
           {nameof(CategoricalSmoothing),                               "cat_smooth" },
           {nameof(L2CategoricalRegularization),                        "cat_l2" }
        };

        [BestFriend]
        internal static class Defaults
        {
            public const int NumberOfIterations = 100;
        }

        /// <summary>
        /// Gradient boosting decision tree.
        /// </summary>
        /// <remarks>
        /// For details, please see <a href="https://en.wikipedia.org/wiki/Gradient_boosting#Gradient_tree_boosting">gradient tree boosting</a>.
        /// </remarks>
        public sealed class TreeBooster : BoosterParameter<TreeBooster.Options>
        {
            internal const string Name = "gbdt";
            internal const string FriendlyName = "Tree Booster";

            /// <summary>
            /// The options for <see cref="TreeBooster"/>, used for setting <see cref="Booster"/>.
            /// </summary>
            [TlcModule.Component(Name = Name, FriendlyName = FriendlyName, Desc = "Traditional Gradient Boosting Decision Tree.")]
            public class Options : ISupportBoosterParameterFactory
            {
                /// <summary>
                /// Whether training data is unbalanced. Used by <see cref="LightGbmBinaryTrainer"/>.
                /// </summary>
                [Argument(ArgumentType.AtMostOnce, HelpText = "Use for binary classification when training data is not balanced.", ShortName = "us")]
                public bool UnbalancedSets = false;

                /// <summary>
                /// The minimum loss reduction required to make a further partition on a leaf node of the tree.
                /// </summary>
                /// <value>
                /// Larger values make the algorithm more conservative.
                /// </value>
                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "Minimum loss reduction required to make a further partition on a leaf node of the tree. the larger, " +
                        "the more conservative the algorithm will be.")]
                [TlcModule.Range(Min = 0.0)]
                public double MinimumSplitGain = 0;

                /// <summary>
                /// The maximum depth of a tree.
                /// </summary>
                /// <value>
                /// 0 means no limit.
                /// </value>
                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "Maximum depth of a tree. 0 means no limit. However, tree still grows by best-first.")]
                [TlcModule.Range(Min = 0, Max = int.MaxValue)]
                public int MaximumTreeDepth = 0;

                /// <summary>
                /// The minimum sum of instance weight needed to form a new node.
                /// </summary>
                /// <value>
                /// If the tree partition step results in a leaf node with the sum of instance weight less than <see cref="MinimumChildWeight"/>,
                /// the building process will give up further partitioning. In linear regression mode, this simply corresponds to minimum number
                /// of instances needed to be in each node. The larger, the more conservative the algorithm will be.
                /// </value>
                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "Minimum sum of instance weight(hessian) needed in a child. If the tree partition step results in a leaf " +
                        "node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression mode, " +
                        "this simply corresponds to minimum number of instances needed to be in each node. The larger, the more conservative the algorithm will be.")]
                [TlcModule.Range(Min = 0.0)]
                public double MinimumChildWeight = 0.1;

                /// <summary>
                /// The frequency of performing subsampling (bagging).
                /// </summary>
                /// <value>
                /// 0 means disable bagging. N means perform bagging at every N iterations.
                /// To enable bagging, <see cref="SubsampleFraction"/> should also be set to a value less than 1.0.
                /// </value>
                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "Subsample frequency for bagging. 0 means no subsample. "
                    + "Specifies the frequency at which the bagging occurs, where if this is set to N, the subsampling will happen at every N iterations." +
                    "This must be set with Subsample as this specifies the amount to subsample.")]
                [TlcModule.Range(Min = 0, Max = int.MaxValue)]
                public int SubsampleFrequency = 0;

                /// <summary>
                /// The fraction of training data used for creating trees.
                /// </summary>
                /// <value>
                /// Setting it to 0.5 means that LightGBM randomly picks half of the data points to grow trees.
                /// This can be used to speed up training and to reduce over-fitting. Valid range is (0,1].
                /// </value>
                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "Subsample ratio of the training instance. Setting it to 0.5 means that LightGBM randomly collected " +
                        "half of the data instances to grow trees and this will prevent overfitting. Range: (0,1].")]
                [TlcModule.Range(Inf = 0.0, Max = 1.0)]
                public double SubsampleFraction = 1;

                /// <summary>
                /// The fraction of features used when creating trees.
                /// </summary>
                /// <value>
                /// If <see cref="FeatureFraction"/> is smaller than 1.0, LightGBM will randomly select fraction of features to train each tree.
                /// For example, if you set it to 0.8, LightGBM will select 80% of features before training each tree.
                /// This can be used to speed up training and to reduce over-fitting. Valid range is (0,1].
                /// </value>
                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "Subsample ratio of columns when constructing each tree. Range: (0,1].",
                    ShortName = "ff")]
                [TlcModule.Range(Inf = 0.0, Max = 1.0)]
                public double FeatureFraction = 1;

                /// <summary>
                /// The L2 regularization term on weights.
                /// </summary>
                /// <value>
                /// Increasing this value could help reduce over-fitting.
                /// </value>
                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "L2 regularization term on weights, increasing this value will make model more conservative.",
                    ShortName = "l2")]
                [TlcModule.Range(Min = 0.0)]
                [TGUI(Label = "Lambda(L2)", SuggestedSweeps = "0,0.5,1")]
                [TlcModule.SweepableDiscreteParam("RegLambda", new object[] { 0f, 0.5f, 1f })]
                public double L2Regularization = 0.01;

                /// <summary>
                /// The L1 regularization term on weights.
                /// </summary>
                /// <value>
                /// Increasing this value could help reduce over-fitting.
                /// </value>
                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "L1 regularization term on weights, increase this value will make model more conservative.",
                    ShortName = "l1")]
                [TlcModule.Range(Min = 0.0)]
                [TGUI(Label = "Alpha(L1)", SuggestedSweeps = "0,0.5,1")]
                [TlcModule.SweepableDiscreteParam("RegAlpha", new object[] { 0f, 0.5f, 1f })]
                public double L1Regularization = 0;

                /// <summary>
                /// Controls the balance of positive and negative weights in <see cref="LightGbmBinaryTrainer"/>.
                /// </summary>
                /// <value>
                /// This is useful for training on unbalanced data. A typical value to consider is sum(negative cases) / sum(positive cases).
                /// </value>
                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "Control the balance of positive and negative weights, useful for unbalanced classes." +
                    " A typical value to consider: sum(negative cases) / sum(positive cases).",
                    ShortName = "ScalePosWeight")]
                public double WeightOfPositiveExamples = 1;

                internal virtual IBoosterParameter CreateComponent(IHostEnvironment env) => new TreeBooster(this);

                IBoosterParameter IComponentFactory<IBoosterParameter>.CreateComponent(IHostEnvironment env) => CreateComponent(env);
            }

            internal TreeBooster(Options options)
                : base(options)
            {
                Contracts.CheckUserArg(BoosterParameterOptions.MinimumSplitGain >= 0, nameof(BoosterParameterOptions.MinimumSplitGain), "must be >= 0.");
                Contracts.CheckUserArg(BoosterParameterOptions.MinimumChildWeight >= 0, nameof(BoosterParameterOptions.MinimumChildWeight), "must be >= 0.");
                Contracts.CheckUserArg(BoosterParameterOptions.SubsampleFraction > 0 && BoosterParameterOptions.SubsampleFraction <= 1, nameof(BoosterParameterOptions.SubsampleFraction), "must be in (0,1].");
                Contracts.CheckUserArg(BoosterParameterOptions.FeatureFraction > 0 && BoosterParameterOptions.FeatureFraction <= 1, nameof(BoosterParameterOptions.FeatureFraction), "must be in (0,1].");
                Contracts.CheckUserArg(BoosterParameterOptions.L2Regularization >= 0, nameof(BoosterParameterOptions.L2Regularization), "must be >= 0.");
                Contracts.CheckUserArg(BoosterParameterOptions.L1Regularization >= 0, nameof(BoosterParameterOptions.L1Regularization), "must be >= 0.");
                Contracts.CheckUserArg(BoosterParameterOptions.WeightOfPositiveExamples > 0, nameof(BoosterParameterOptions.WeightOfPositiveExamples), "must be >= 0.");
            }

            internal override void UpdateParameters(Dictionary<string, object> res)
            {
                base.UpdateParameters(res);
                res["boosting_type"] = Name;
            }
        }

        /// <summary>
        /// DART booster (Dropouts meet Multiple Additive Regression Trees)
        /// </summary>
        /// <remarks>
        /// For details, please see <a href="https://arxiv.org/abs/1505.01866">here</a>.
        /// </remarks>
        public sealed class DartBooster : BoosterParameter<DartBooster.Options>
        {
            internal const string Name = "dart";
            internal const string FriendlyName = "Tree Dropout Tree Booster";

            /// <summary>
            /// The options for <see cref="DartBooster"/>, used for setting <see cref="Booster"/>.
            /// </summary>
            [TlcModule.Component(Name = Name, FriendlyName = FriendlyName, Desc = "Dropouts meet Multiple Additive Regression Trees. See https://arxiv.org/abs/1505.01866")]
            public sealed class Options : TreeBooster.Options
            {
                /// <summary>
                /// The dropout rate, i.e. the fraction of previous trees to drop during the dropout.
                /// </summary>
                /// <value>
                /// Valid range is [0,1].
                /// </value>
                [Argument(ArgumentType.AtMostOnce, HelpText = "The drop ratio for trees. Range:[0,1].")]
                [TlcModule.Range(Inf = 0.0, Max = 1.0)]
                public double TreeDropFraction = 0.1;

                /// <summary>
                /// The maximum number of dropped trees in a boosting round.
                /// </summary>
                [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum number of dropped trees in a boosting round.")]
                [TlcModule.Range(Inf = 0, Max = int.MaxValue)]
                public int MaximumNumberOfDroppedTreesPerRound = 1;

                /// <summary>
                /// The probability of skipping the dropout procedure during a boosting iteration.
                /// </summary>
                [Argument(ArgumentType.AtMostOnce, HelpText = "Probability for not dropping in a boosting round.")]
                [TlcModule.Range(Inf = 0.0, Max = 1.0)]
                public double SkipDropFraction = 0.5;

                /// <summary>
                /// Whether to enable xgboost dart mode.
                /// </summary>
                [Argument(ArgumentType.AtMostOnce, HelpText = "True will enable xgboost dart mode.")]
                public bool XgboostDartMode = false;

                /// <summary>
                /// Whether to enable uniform drop.
                /// </summary>
                [Argument(ArgumentType.AtMostOnce, HelpText = "True will enable uniform drop.")]
                public bool UniformDrop = false;

                internal override IBoosterParameter CreateComponent(IHostEnvironment env) => new DartBooster(this);
            }

            internal DartBooster(Options options)
                : base(options)
            {
                Contracts.CheckUserArg(BoosterParameterOptions.TreeDropFraction > 0 && BoosterParameterOptions.TreeDropFraction < 1, nameof(BoosterParameterOptions.TreeDropFraction), "must be in (0,1).");
                Contracts.CheckUserArg(BoosterParameterOptions.SkipDropFraction >= 0 && BoosterParameterOptions.SkipDropFraction < 1, nameof(BoosterParameterOptions.SkipDropFraction), "must be in [0,1).");
            }

            internal override void UpdateParameters(Dictionary<string, object> res)
            {
                base.UpdateParameters(res);
                res["boosting_type"] = Name;
            }
        }

        /// <summary>
        /// Gradient-based One-Side Sampling booster.
        /// </summary>
        /// <remarks>
        /// For details, please see <a href="https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf">here</a>.
        /// </remarks>
        public sealed class GossBooster : BoosterParameter<GossBooster.Options>
        {
            internal const string Name = "goss";
            internal const string FriendlyName = "Gradient-based One-Size Sampling";

            [TlcModule.Component(Name = Name, FriendlyName = FriendlyName, Desc = "Gradient-based One-Side Sampling.")]
            public sealed class Options : TreeBooster.Options
            {
                /// <summary>
                /// The retain ratio of large gradient data.
                /// </summary>
                [Argument(ArgumentType.AtMostOnce, HelpText = "Retain ratio for large gradient instances.")]
                [TlcModule.Range(Inf = 0.0, Max = 1.0)]
                public double TopRate = 0.2;

                /// <summary>
                /// The retain ratio of small gradient data.
                /// </summary>
                [Argument(ArgumentType.AtMostOnce, HelpText = "Retain ratio for small gradient instances.")]
                [TlcModule.Range(Inf = 0.0, Max = 1.0)]
                public double OtherRate = 0.1;

                internal override IBoosterParameter CreateComponent(IHostEnvironment env) => new GossBooster(this);
            }

            internal GossBooster(Options options)
                : base(options)
            {
                Contracts.CheckUserArg(BoosterParameterOptions.TopRate > 0 && BoosterParameterOptions.TopRate < 1, nameof(BoosterParameterOptions.TopRate), "must be in (0,1).");
                Contracts.CheckUserArg(BoosterParameterOptions.OtherRate >= 0 && BoosterParameterOptions.OtherRate < 1, nameof(BoosterParameterOptions.TopRate), "must be in [0,1).");
                Contracts.Check(BoosterParameterOptions.TopRate + BoosterParameterOptions.OtherRate <= 1, "Sum of topRate and otherRate cannot be larger than 1.");
            }

            internal override void UpdateParameters(Dictionary<string, object> res)
            {
                base.UpdateParameters(res);
                res["boosting_type"] = Name;
            }
        }

        /// <summary>
        /// The evaluation metrics that are available for <see cref="EvaluationMetric"/>.
        /// </summary>
        public enum EvalMetricType
        {
            DefaultMetric,
            Rmse,
            Mae,
            Logloss,
            Error,
            Merror,
            Mlogloss,
            Auc,
            Ndcg,
            Map
        };

        /// <summary>
        /// The number of boosting iterations. A new tree is created in each iteration, so this is equivalent to the number of trees.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Number of iterations.", SortOrder = 1, ShortName = "iter")]
        [TGUI(Label = "Number of boosting iterations", SuggestedSweeps = "10,20,50,100,150,200")]
        [TlcModule.SweepableDiscreteParam("NumBoostRound", new object[] { 10, 20, 50, 100, 150, 200 })]
        public int NumberOfIterations = Defaults.NumberOfIterations;

        /// <summary>
        /// The shrinkage rate for trees, used to prevent over-fitting.
        /// </summary>
        /// <value>
        /// Valid range is (0,1].
        /// </value>
        [Argument(ArgumentType.AtMostOnce,
            HelpText = "Shrinkage rate for trees, used to prevent over-fitting. Range: (0,1].",
            SortOrder = 2, ShortName = "lr", NullName = "<Auto>")]
        [TGUI(Label = "Learning Rate", SuggestedSweeps = "0.025-0.4;log")]
        [TlcModule.SweepableFloatParamAttribute("LearningRate", 0.025f, 0.4f, isLogScale: true)]
        public double? LearningRate;

        /// <summary>
        /// The maximum number of leaves in one tree.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum leaves for trees.",
            SortOrder = 2, ShortName = "nl", NullName = "<Auto>")]
        [TGUI(Description = "The maximum number of leaves per tree", SuggestedSweeps = "2-128;log;inc:4")]
        [TlcModule.SweepableLongParamAttribute("NumLeaves", 2, 128, isLogScale: true, stepSize: 4)]
        public int? NumberOfLeaves;

        /// <summary>
        /// The minimal number of data points required to form a new tree leaf.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Minimum number of instances needed in a child.",
            SortOrder = 2, ShortName = "mil", NullName = "<Auto>")]
        [TGUI(Label = "Min Documents In Leaves", SuggestedSweeps = "1,10,20,50 ")]
        [TlcModule.SweepableDiscreteParamAttribute("MinDataPerLeaf", new object[] { 1, 10, 20, 50 })]
        public int? MinimumExampleCountPerLeaf;

        /// <summary>
        /// The maximum number of bins that feature values will be bucketed in.
        /// </summary>
        /// <remarks>
        /// The small number of bins may reduce training accuracy but may increase general power (deal with over-fitting).
        /// </remarks>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum number of bucket bin for features.", ShortName = "mb")]
        public int MaximumBinCountPerFeature = 255;

        /// <summary>
        /// Determines which booster to use.
        /// </summary>
        /// <value>
        /// Available boosters are <see cref="DartBooster"/>, <see cref="GossBooster"/>, and <see cref="TreeBooster"/>.
        /// </value>
        [Argument(ArgumentType.Multiple, HelpText = "Which booster to use, can be gbtree, gblinear or dart. gbtree and dart use tree based model while gblinear uses linear function.", SortOrder = 3)]
        public ISupportBoosterParameterFactory Booster = new TreeBooster.Options();

        /// <summary>
        /// Determines whether to output progress status during training and evaluation.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Verbose", ShortName = "v")]
        public bool Verbose = false;

        /// <summary>
        /// Controls the logging level in LighGBM.
        /// </summary>
        /// <value>
        /// <see langword="true"/> means only output Fatal errors. <see langword="false"/> means output Fatal, Warning, and Info level messages.
        /// </value>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Printing running messages.")]
        public bool Silent = true;

        /// <summary>
        /// Determines the number of threads used to run LightGBM.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Number of parallel threads used to run LightGBM.", ShortName = "nt")]
        public int? NumberOfThreads;

        /// <summary>
        /// Determines what evaluation metric to use.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce,
            HelpText = "Evaluation metrics.",
            ShortName = "em")]
        public EvalMetricType EvaluationMetric = EvalMetricType.DefaultMetric;

        /// <summary>
        /// Whether to use softmax loss. Used only by <see cref="LightGbmMulticlassTrainer"/>.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Use softmax loss for the multi classification.")]
        [TlcModule.SweepableDiscreteParam("UseSoftmax", new object[] { true, false })]
        public bool? UseSoftmax;

        /// <summary>
        /// Determines the number of rounds, after which training will stop if validation metric doesn't improve.
        /// </summary>
        /// <value>
        /// 0 means disable early stopping.
        /// </value>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Rounds of early stopping, 0 will disable it.",
            ShortName = "es")]
        public int EarlyStoppingRound = 0;

        /// <summary>
        /// Comma-separated list of gains associated with each relevance label. Used only by <see cref="LightGbmRankingTrainer"/>.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Comma separated list of gains associated to each relevance label.", ShortName = "gains")]
        [TGUI(Label = "Ranking Label Gain")]
        public string CustomGains = "0,3,7,15,31,63,127,255,511,1023,2047,4095";

        /// <summary>
        /// Parameter for the sigmoid function. Used only by <see cref="LightGbmBinaryTrainer"/>, <see cref="LightGbmMulticlassTrainer"/>, and <see cref="LightGbmRankingTrainer"/>.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Parameter for the sigmoid function. Used only in " + nameof(LightGbmBinaryTrainer) + ", " + nameof(LightGbmMulticlassTrainer) +
            " and in " + nameof(LightGbmRankingTrainer) + ".", ShortName = "sigmoid")]
        [TGUI(Label = "Sigmoid", SuggestedSweeps = "0.5,1")]
        public double Sigmoid = 0.5;

        /// <summary>
        /// Number of data points per batch, when loading data.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Number of entries in a batch when loading data.", Hide = true)]
        public int BatchSize = 1 << 20;

        /// <summary>
        /// Whether to enable categorical split or not.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Enable categorical split or not.", ShortName = "cat")]
        [TlcModule.SweepableDiscreteParam("UseCat", new object[] { true, false })]
        public bool? UseCategoricalSplit;

        /// <summary>
        /// Whether to enable special handling of missing value or not.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Enable special handling of missing value or not.")]
        [TlcModule.SweepableDiscreteParam("UseMissing", new object[] { true, false })]
        public bool HandleMissingValue = false;

        /// <summary>
        /// The minimum number of data points per categorical group.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Minimum number of instances per categorical group.", ShortName = "mdpg")]
        [TlcModule.Range(Inf = 0, Max = int.MaxValue)]
        [TlcModule.SweepableDiscreteParam("MinDataPerGroup", new object[] { 10, 50, 100, 200 })]
        public int MinimumExampleCountPerGroup = 100;

        /// <summary>
        /// When the number of categories of one feature is smaller than or equal to <see cref="MaximumCategoricalSplitPointCount"/>,
        /// one-vs-other split algorithm will be used.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Max number of categorical thresholds.", ShortName = "maxcat")]
        [TlcModule.Range(Inf = 0, Max = int.MaxValue)]
        [TlcModule.SweepableDiscreteParam("MaxCatThreshold", new object[] { 8, 16, 32, 64 })]
        public int MaximumCategoricalSplitPointCount = 32;

        /// <summary>
        /// Laplace smooth term in categorical feature split.
        /// This can reduce the effect of noises in categorical features, especially for categories with few data.
        /// </summary>
        /// <value>
        /// Constraints: <see cref="CategoricalSmoothing"/> >= 0.0
        /// </value>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Laplace smooth term in categorical feature split. Avoid the bias of small categories.")]
        [TlcModule.Range(Min = 0.0)]
        [TlcModule.SweepableDiscreteParam("CatSmooth", new object[] { 1, 10, 20 })]
        public double CategoricalSmoothing = 10;

        /// <summary>
        /// L2 regularization for categorical split.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "L2 Regularization for categorical split.")]
        [TlcModule.Range(Min = 0.0)]
        [TlcModule.SweepableDiscreteParam("CatL2", new object[] { 0.1, 0.5, 1, 5, 10 })]
        public double L2CategoricalRegularization = 10;

        /// <summary>
        /// The random seed for LightGBM to use.
        /// </summary>
        /// <value>
        /// If not specified, <see cref="MLContext"/> will generate a random seed to be used.
        /// </value>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Sets the random seed for LightGBM to use.")]
        public int? Seed;

        [Argument(ArgumentType.Multiple, HelpText = "Parallel LightGBM Learning Algorithm", ShortName = "parag")]
        internal ISupportParallel ParallelTrainer = new SingleTrainerFactory();

        internal Dictionary<string, object> ToDictionary(IHost host)
        {
            Contracts.CheckValue(host, nameof(host));
            Contracts.CheckUserArg(MaximumBinCountPerFeature > 0, nameof(MaximumBinCountPerFeature), "must be > 0.");
            Contracts.CheckUserArg(Sigmoid > 0, nameof(Sigmoid), "must be > 0.");
            Dictionary<string, object> res = new Dictionary<string, object>();

            var boosterParams = Booster.CreateComponent(host);
            boosterParams.UpdateParameters(res);

            res[GetOptionName(nameof(MaximumBinCountPerFeature))] = MaximumBinCountPerFeature;

            res["verbose"] = Silent ? "-1" : "1";
            if (NumberOfThreads.HasValue)
                res["nthread"] = NumberOfThreads.Value;

            res["seed"] = (Seed.HasValue) ? Seed : host.Rand.Next();

            string metric = null;
            switch (EvaluationMetric)
            {
                case EvalMetricType.DefaultMetric:
                    break;
                case EvalMetricType.Mae:
                    metric = "l1";
                    break;
                case EvalMetricType.Logloss:
                    metric = "binary_logloss";
                    break;
                case EvalMetricType.Error:
                    metric = "binary_error";
                    break;
                case EvalMetricType.Merror:
                    metric = "multi_error";
                    break;
                case EvalMetricType.Mlogloss:
                    metric = "multi_logloss";
                    break;
                case EvalMetricType.Rmse:
                case EvalMetricType.Auc:
                case EvalMetricType.Ndcg:
                case EvalMetricType.Map:
                    metric = EvaluationMetric.ToString().ToLower();
                    break;
            }
            if (!string.IsNullOrEmpty(metric))
                res[GetOptionName(nameof(metric))] = metric;
            res[GetOptionName(nameof(Sigmoid))] = Sigmoid;
            res[GetOptionName(nameof(CustomGains))] = CustomGains;
            res[GetOptionName(nameof(HandleMissingValue))] = HandleMissingValue;
            res[GetOptionName(nameof(MinimumExampleCountPerGroup))] = MinimumExampleCountPerGroup;
            res[GetOptionName(nameof(MaximumCategoricalSplitPointCount))] = MaximumCategoricalSplitPointCount;
            res[GetOptionName(nameof(CategoricalSmoothing))] = CategoricalSmoothing;
            res[GetOptionName(nameof(L2CategoricalRegularization))] = L2CategoricalRegularization;
            return res;
        }
    }
}
