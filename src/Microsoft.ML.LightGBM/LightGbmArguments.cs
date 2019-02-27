﻿// Licensed to the .NET Foundation under one or more agreements.
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

    /// <summary>
    /// Parameters names comes from LightGBM library.
    /// See https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst.
    /// </summary>
    public sealed class Options : LearnerInputBaseWithGroupId
    {
        public abstract class BoosterParameter<TOptions> : IBoosterParameter
            where TOptions : class, new()
        {
            protected TOptions BoosterParameterOptions { get; }

            protected BoosterParameter(TOptions options)
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

                    res[GetArgName(field.Name)] = field.GetValue(BoosterParameterOptions);
                }
            }

            void IBoosterParameter.UpdateParameters(Dictionary<string, object> res) => UpdateParameters(res);
        }

        private static string GetArgName(string name)
        {
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

        [BestFriend]
        internal static class Defaults
        {
            public const int NumBoostRound = 100;
        }

        public sealed class TreeBooster : BoosterParameter<TreeBooster.Options>
        {
            internal const string Name = "gbdt";
            internal const string FriendlyName = "Tree Booster";

            [TlcModule.Component(Name = Name, FriendlyName = FriendlyName, Desc = "Traditional Gradient Boosting Decision Tree.")]
            public class Options : ISupportBoosterParameterFactory
            {
                [Argument(ArgumentType.AtMostOnce, HelpText = "Use for binary classification when classes are not balanced.", ShortName = "us")]
                public bool UnbalancedSets = false;

                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "Minimum loss reduction required to make a further partition on a leaf node of the tree. the larger, " +
                        "the more conservative the algorithm will be.")]
                [TlcModule.Range(Min = 0.0)]
                public double MinSplitGain = 0;

                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "Maximum depth of a tree. 0 means no limit. However, tree still grows by best-first.")]
                [TlcModule.Range(Min = 0, Max = int.MaxValue)]
                public int MaxDepth = 0;

                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "Minimum sum of instance weight(hessian) needed in a child. If the tree partition step results in a leaf " +
                        "node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression mode, " +
                        "this simply corresponds to minimum number of instances needed to be in each node. The larger, the more conservative the algorithm will be.")]
                [TlcModule.Range(Min = 0.0)]
                public double MinChildWeight = 0.1;

                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "Subsample frequency. 0 means no subsample. "
                    + "If subsampleFreq > 0, it will use a subset(ratio=subsample) to train. And the subset will be updated on every Subsample iteratinos.")]
                [TlcModule.Range(Min = 0, Max = int.MaxValue)]
                public int SubsampleFreq = 0;

                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "Subsample ratio of the training instance. Setting it to 0.5 means that LightGBM randomly collected " +
                        "half of the data instances to grow trees and this will prevent overfitting. Range: (0,1].")]
                [TlcModule.Range(Inf = 0.0, Max = 1.0)]
                public double Subsample = 1;

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
                public double RegLambda = 0.01;

                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "L1 regularization term on weights, increase this value will make model more conservative.",
                    ShortName = "l1")]
                [TlcModule.Range(Min = 0.0)]
                [TGUI(Label = "Alpha(L1)", SuggestedSweeps = "0,0.5,1")]
                [TlcModule.SweepableDiscreteParam("RegAlpha", new object[] { 0f, 0.5f, 1f })]
                public double RegAlpha = 0;

                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "Control the balance of positive and negative weights, useful for unbalanced classes." +
                    " A typical value to consider: sum(negative cases) / sum(positive cases).")]
                public double ScalePosWeight = 1;

                internal virtual IBoosterParameter CreateComponent(IHostEnvironment env) => new TreeBooster(this);

                IBoosterParameter IComponentFactory<IBoosterParameter>.CreateComponent(IHostEnvironment env) => CreateComponent(env);
            }

            internal TreeBooster(Options options)
                : base(options)
            {
                Contracts.CheckUserArg(BoosterParameterOptions.MinSplitGain >= 0, nameof(BoosterParameterOptions.MinSplitGain), "must be >= 0.");
                Contracts.CheckUserArg(BoosterParameterOptions.MinChildWeight >= 0, nameof(BoosterParameterOptions.MinChildWeight), "must be >= 0.");
                Contracts.CheckUserArg(BoosterParameterOptions.Subsample > 0 && BoosterParameterOptions.Subsample <= 1, nameof(BoosterParameterOptions.Subsample), "must be in (0,1].");
                Contracts.CheckUserArg(BoosterParameterOptions.FeatureFraction > 0 && BoosterParameterOptions.FeatureFraction <= 1, nameof(BoosterParameterOptions.FeatureFraction), "must be in (0,1].");
                Contracts.CheckUserArg(BoosterParameterOptions.ScalePosWeight > 0 && BoosterParameterOptions.ScalePosWeight <= 1, nameof(BoosterParameterOptions.ScalePosWeight), "must be in (0,1].");
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
                [Argument(ArgumentType.AtMostOnce, HelpText = "Drop ratio for trees. Range:(0,1).")]
                [TlcModule.Range(Inf = 0.0, Max = 1.0)]
                public double DropRate = 0.1;

                [Argument(ArgumentType.AtMostOnce, HelpText = "Max number of dropped tree in a boosting round.")]
                [TlcModule.Range(Inf = 0, Max = int.MaxValue)]
                public int MaxDrop = 1;

                [Argument(ArgumentType.AtMostOnce, HelpText = "Probability for not perform dropping in a boosting round.")]
                [TlcModule.Range(Inf = 0.0, Max = 1.0)]
                public double SkipDrop = 0.5;

                [Argument(ArgumentType.AtMostOnce, HelpText = "True will enable xgboost dart mode.")]
                public bool XgboostDartMode = false;

                [Argument(ArgumentType.AtMostOnce, HelpText = "True will enable uniform drop.")]
                public bool UniformDrop = false;

                internal override IBoosterParameter CreateComponent(IHostEnvironment env) => new DartBooster(this);
            }

            internal DartBooster(Options options)
                : base(options)
            {
                Contracts.CheckUserArg(BoosterParameterOptions.DropRate > 0 && BoosterParameterOptions.DropRate < 1, nameof(BoosterParameterOptions.DropRate), "must be in (0,1).");
                Contracts.CheckUserArg(BoosterParameterOptions.MaxDrop > 0, nameof(BoosterParameterOptions.MaxDrop), "must be > 0.");
                Contracts.CheckUserArg(BoosterParameterOptions.SkipDrop >= 0 && BoosterParameterOptions.SkipDrop < 1, nameof(BoosterParameterOptions.SkipDrop), "must be in [0,1).");
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
                [Argument(ArgumentType.AtMostOnce,
                    HelpText = "Retain ratio for large gradient instances.")]
                [TlcModule.Range(Inf = 0.0, Max = 1.0)]
                public double TopRate = 0.2;

                [Argument(ArgumentType.AtMostOnce,
                    HelpText =
                        "Retain ratio for small gradient instances.")]
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

        [Argument(ArgumentType.AtMostOnce, HelpText = "Number of iterations.", SortOrder = 1, ShortName = "iter")]
        [TGUI(Label = "Number of boosting iterations", SuggestedSweeps = "10,20,50,100,150,200")]
        [TlcModule.SweepableDiscreteParam("NumBoostRound", new object[] { 10, 20, 50, 100, 150, 200 })]
        public int NumBoostRound = Defaults.NumBoostRound;

        [Argument(ArgumentType.AtMostOnce,
            HelpText = "Shrinkage rate for trees, used to prevent over-fitting. Range: (0,1].",
            SortOrder = 2, ShortName = "lr", NullName = "<Auto>")]
        [TGUI(Label = "Learning Rate", SuggestedSweeps = "0.025-0.4;log")]
        [TlcModule.SweepableFloatParamAttribute("LearningRate", 0.025f, 0.4f, isLogScale: true)]
        public double? LearningRate;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum leaves for trees.",
            SortOrder = 2, ShortName = "nl", NullName = "<Auto>")]
        [TGUI(Description = "The maximum number of leaves per tree", SuggestedSweeps = "2-128;log;inc:4")]
        [TlcModule.SweepableLongParamAttribute("NumLeaves", 2, 128, isLogScale: true, stepSize: 4)]
        public int? NumLeaves;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Minimum number of instances needed in a child.",
            SortOrder = 2, ShortName = "mil", NullName = "<Auto>")]
        [TGUI(Label = "Min Documents In Leaves", SuggestedSweeps = "1,10,20,50 ")]
        [TlcModule.SweepableDiscreteParamAttribute("MinDataPerLeaf", new object[] { 1, 10, 20, 50 })]
        public int? MinDataPerLeaf;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Max number of bucket bin for features.", ShortName = "mb")]
        public int MaxBin = 255;

        [Argument(ArgumentType.Multiple, HelpText = "Which booster to use, can be gbtree, gblinear or dart. gbtree and dart use tree based model while gblinear uses linear function.", SortOrder = 3)]
        public ISupportBoosterParameterFactory Booster = new TreeBooster.Options();

        [Argument(ArgumentType.AtMostOnce, HelpText = "Verbose", ShortName = "v")]
        public bool VerboseEval = false;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Printing running messages.")]
        public bool Silent = true;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Number of parallel threads used to run LightGBM.", ShortName = "nt")]
        public int? NThread;

        [Argument(ArgumentType.AtMostOnce,
            HelpText = "Evaluation metrics.",
            ShortName = "em")]
        public EvalMetricType EvalMetric = EvalMetricType.DefaultMetric;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Use softmax loss for the multi classification.")]
        [TlcModule.SweepableDiscreteParam("UseSoftmax", new object[] { true, false })]
        public bool? UseSoftmax;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Rounds of early stopping, 0 will disable it.",
            ShortName = "es")]
        public int EarlyStoppingRound = 0;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Comma seperated list of gains associated to each relevance label.", ShortName = "gains")]
        [TGUI(Label = "Ranking Label Gain")]
        public string CustomGains = "0,3,7,15,31,63,127,255,511,1023,2047,4095";

        [Argument(ArgumentType.AtMostOnce, HelpText = "Parameter for the sigmoid function. Used only in " + nameof(LightGbmBinaryTrainer) + ", " + nameof(LightGbmMulticlassTrainer) +
            " and in " + nameof(LightGbmRankingTrainer) + ".", ShortName = "sigmoid")]
        [TGUI(Label = "Sigmoid", SuggestedSweeps = "0.5,1")]
        public double Sigmoid = 0.5;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Number of entries in a batch when loading data.", Hide = true)]
        public int BatchSize = 1 << 20;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Enable categorical split or not.", ShortName = "cat")]
        [TlcModule.SweepableDiscreteParam("UseCat", new object[] { true, false })]
        public bool? UseCat;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Enable missing value auto infer or not.")]
        [TlcModule.SweepableDiscreteParam("UseMissing", new object[] { true, false })]
        public bool UseMissing = false;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Min number of instances per categorical group.", ShortName = "mdpg")]
        [TlcModule.Range(Inf = 0, Max = int.MaxValue)]
        [TlcModule.SweepableDiscreteParam("MinDataPerGroup", new object[] { 10, 50, 100, 200 })]
        public int MinDataPerGroup = 100;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Max number of categorical thresholds.", ShortName = "maxcat")]
        [TlcModule.Range(Inf = 0, Max = int.MaxValue)]
        [TlcModule.SweepableDiscreteParam("MaxCatThreshold", new object[] { 8, 16, 32, 64 })]
        public int MaxCatThreshold = 32;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Lapalace smooth term in categorical feature spilt. Avoid the bias of small categories.")]
        [TlcModule.Range(Min = 0.0)]
        [TlcModule.SweepableDiscreteParam("CatSmooth", new object[] { 1, 10, 20 })]
        public double CatSmooth = 10;

        [Argument(ArgumentType.AtMostOnce, HelpText = "L2 Regularization for categorical split.")]
        [TlcModule.Range(Min = 0.0)]
        [TlcModule.SweepableDiscreteParam("CatL2", new object[] { 0.1, 0.5, 1, 5, 10 })]
        public double CatL2 = 10;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Sets the random seed for LightGBM to use.")]
        public int? Seed;

        [Argument(ArgumentType.Multiple, HelpText = "Parallel LightGBM Learning Algorithm", ShortName = "parag")]
        internal ISupportParallel ParallelTrainer = new SingleTrainerFactory();

        internal Dictionary<string, object> ToDictionary(IHost host)
        {
            Contracts.CheckValue(host, nameof(host));
            Contracts.CheckUserArg(MaxBin > 0, nameof(MaxBin), "must be > 0.");
            Contracts.CheckUserArg(Sigmoid > 0, nameof(Sigmoid), "must be > 0.");
            Dictionary<string, object> res = new Dictionary<string, object>();

            var boosterParams = Booster.CreateComponent(host);
            boosterParams.UpdateParameters(res);

            res[GetArgName(nameof(MaxBin))] = MaxBin;

            res["verbose"] = Silent ? "-1" : "1";
            if (NThread.HasValue)
                res["nthread"] = NThread.Value;

            res["seed"] = (Seed.HasValue) ? Seed : host.Rand.Next();

            string metric = null;
            switch (EvalMetric)
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
                    metric = EvalMetric.ToString().ToLower();
                    break;
            }
            if (!string.IsNullOrEmpty(metric))
                res["metric"] = metric;
            res["sigmoid"] = Sigmoid;
            res["label_gain"] = CustomGains;
            res[GetArgName(nameof(UseMissing))] = UseMissing;
            res[GetArgName(nameof(MinDataPerGroup))] = MinDataPerGroup;
            res[GetArgName(nameof(MaxCatThreshold))] = MaxCatThreshold;
            res[GetArgName(nameof(CatSmooth))] = CatSmooth;
            res[GetArgName(nameof(CatL2))] = CatL2;
            return res;
        }
    }
}
