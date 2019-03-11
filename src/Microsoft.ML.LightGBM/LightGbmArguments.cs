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
using Microsoft.ML.LightGBM;
using Microsoft.ML.Trainers;

[assembly: LoadableClass(typeof(Options.TreeBooster), typeof(Options.TreeBooster.Options),
    typeof(SignatureLightGBMBooster), Options.TreeBooster.FriendlyName, Options.TreeBooster.Name)]
[assembly: LoadableClass(typeof(Options.DartBooster), typeof(Options.DartBooster.Options),
    typeof(SignatureLightGBMBooster), Options.DartBooster.FriendlyName, Options.DartBooster.Name)]
[assembly: LoadableClass(typeof(Options.GossBooster), typeof(Options.GossBooster.Options),
    typeof(SignatureLightGBMBooster), Options.GossBooster.FriendlyName, Options.GossBooster.Name)]

[assembly: EntryPointModule(typeof(Options.TreeBooster.Options))]
[assembly: EntryPointModule(typeof(Options.DartBooster.Options))]
[assembly: EntryPointModule(typeof(Options.GossBooster.Options))]

namespace Microsoft.ML.LightGBM
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



    public class BinaryClassifierOptions : OptionBase
    {
        public enum EvaluateMetricType
        {
            DefaultMetric,
            Logloss,
            Error,
            Auc,
        };
        [Argument(ArgumentType.AtMostOnce, HelpText = "Parameter for the sigmoid function." + nameof(LightGbmBinaryTrainer) + ", " + nameof(LightGbmMulticlassTrainer) +
            " and in " + nameof(LightGbmRankingTrainer) + ".", ShortName = "sigmoid")]
        [TGUI(Label = "Sigmoid", SuggestedSweeps = "0.5,1")]
        public double Sigmoid = 0.5;

        [Argument(ArgumentType.AtMostOnce,
            HelpText = "Evaluation metrics.",
            ShortName = "em")]
        public EvaluateMetricType EvaluationMetric = EvaluateMetricType.DefaultMetric;
    }

    public class MulticlassClassifierOptions : OptionBase
    {

    }

    public class RegressionClassifierOptions : OptionBase
    {


    }

    public class RankingOptions : OptionBase
    {
        public enum EvaluateMetricType
        {
            Default,
            Ndcg,
            Map
        }

        [Argument(ArgumentType.AtMostOnce,
            HelpText = "Evaluation metrics.",
            ShortName = "em")]
        public EvaluateMetricType EvaluationMetric = EvaluateMetricType.Default;
    }

    /// <summary>
    /// Parameters names comes from LightGBM library.
    /// See https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst.
    /// </summary>
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
        // If an argument is not here, then its name is identicaltto a lightGBM argument
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


        public sealed class TreeBooster : BoosterParameter<TreeBooster.Options>
        {
            internal const string Name = "gbdt";
            internal const string FriendlyName = "Tree Booster";

            [TlcModule.Component(Name = Name, FriendlyName = FriendlyName, Desc = "Traditional Gradient Boosting Decision Tree.")]
            public class Options : ISupportBoosterParameterFactory
            {
                [Argument(ArgumentType.AtMostOnce, HelpText = "Use for binary classification when training data is not balanced.", ShortName = "us")]
                public bool UnbalancedSets = false;

                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "Minimum loss reduction required to make a further partition on a leaf node of the tree. the larger, " +
                        "the more conservative the algorithm will be.")]
                [TlcModule.Range(Min = 0.0)]
                public double MinimumSplitGain = 0;

                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "Maximum depth of a tree. 0 means no limit. However, tree still grows by best-first.")]
                [TlcModule.Range(Min = 0, Max = int.MaxValue)]
                public int MaximumTreeDepth = 0;

                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "Minimum sum of instance weight(hessian) needed in a child. If the tree partition step results in a leaf " +
                        "node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression mode, " +
                        "this simply corresponds to minimum number of instances needed to be in each node. The larger, the more conservative the algorithm will be.")]
                [TlcModule.Range(Min = 0.0)]
                public double MinimumChildWeight = 0.1;

                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "Subsample frequency for bagging. 0 means no subsample. "
                    + "Specifies the frequency at which the bagging occurs, where if this is set to N, the subsampling will happen at every N iterations." +
                    "This must be set with Subsample as this specifies the amount to subsample.")]
                [TlcModule.Range(Min = 0, Max = int.MaxValue)]
                public int SubsampleFrequency = 0;

                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "Subsample ratio of the training instance. Setting it to 0.5 means that LightGBM randomly collected " +
                        "half of the data instances to grow trees and this will prevent overfitting. Range: (0,1].")]
                [TlcModule.Range(Inf = 0.0, Max = 1.0)]
                public double SubsampleFraction = 1;

                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "Subsample ratio of columns when constructing each tree. Range: (0,1].",
                    ShortName = "ff")]
                [TlcModule.Range(Inf = 0.0, Max = 1.0)]
                public double FeatureFraction = 1;

                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "L2 regularization term on weights, increasing this value will make model more conservative.",
                    ShortName = "l2")]
                [TlcModule.Range(Min = 0.0)]
                [TGUI(Label = "Lambda(L2)", SuggestedSweeps = "0,0.5,1")]
                [TlcModule.SweepableDiscreteParam("RegLambda", new object[] { 0f, 0.5f, 1f })]
                public double L2Regularization = 0.01;

                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "L1 regularization term on weights, increase this value will make model more conservative.",
                    ShortName = "l1")]
                [TlcModule.Range(Min = 0.0)]
                [TGUI(Label = "Alpha(L1)", SuggestedSweeps = "0,0.5,1")]
                [TlcModule.SweepableDiscreteParam("RegAlpha", new object[] { 0f, 0.5f, 1f })]
                public double L1Regularization = 0;

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

        public sealed class DartBooster : BoosterParameter<DartBooster.Options>
        {
            internal const string Name = "dart";
            internal const string FriendlyName = "Tree Dropout Tree Booster";

            [TlcModule.Component(Name = Name, FriendlyName = FriendlyName, Desc = "Dropouts meet Multiple Additive Regresion Trees. See https://arxiv.org/abs/1505.01866")]
            public sealed class Options : TreeBooster.Options
            {
                [Argument(ArgumentType.AtMostOnce, HelpText = "The drop ratio for trees. Range:(0,1).")]
                [TlcModule.Range(Inf = 0.0, Max = 1.0)]
                public double TreeDropFraction = 0.1;

                [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum number of dropped trees in a boosting round.")]
                [TlcModule.Range(Inf = 0, Max = int.MaxValue)]
                public int MaximumNumberOfDroppedTreesPerRound = 1;

                [Argument(ArgumentType.AtMostOnce, HelpText = "Probability for not dropping in a boosting round.")]
                [TlcModule.Range(Inf = 0.0, Max = 1.0)]
                public double SkipDropFraction = 0.5;

                [Argument(ArgumentType.AtMostOnce, HelpText = "True will enable xgboost dart mode.")]
                public bool XgboostDartMode = false;

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

        public sealed class GossBooster : BoosterParameter<GossBooster.Options>
        {
            internal const string Name = "goss";
            internal const string FriendlyName = "Gradient-based One-Size Sampling";

            [TlcModule.Component(Name = Name, FriendlyName = FriendlyName, Desc = "Gradient-based One-Side Sampling.")]
            public sealed class Options : TreeBooster.Options
            {
                [Argument(ArgumentType.AtMostOnce, HelpText = "Retain ratio for large gradient instances.")]
                [TlcModule.Range(Inf = 0.0, Max = 1.0)]
                public double TopRate = 0.2;

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
