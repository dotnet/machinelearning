// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;

namespace Microsoft.ML.Trainers.XGBoost
{
    [BestFriend]
    internal static class Defaults
    {
        public const int NumberOfIterations = 100;
    }

    public abstract class XGBoostTrainerBase<TOptions, TOutput, TTransformer, TModel> : TrainerEstimatorBaseWithGroupId<TTransformer, TModel>
        where TTransformer : ISingleFeaturePredictionTransformer<TModel>
        where TModel : class // IPredictorProducing<float>
        where TOptions : XGBoostTrainerBase<TOptions, TOutput, TTransformer, TModel>.OptionsBase, new()
    {
        internal const string LoadNameValue = "XGBoostPredictor";
        internal const string UserNameValue = "XGBoost Predictor";
        internal const string Summary = "The base logic for all XGBoost-based trainers.";

        private protected int FeatureCount;
        private protected InternalTreeEnsemble TrainedEnsemble;

#if false
        /// <summary>
        /// The shrinkage rate for trees, used to prevent over-fitting.
	/// Also aliased to "eta"
        /// </summary>
        /// <value>
        /// Valid range is (0,1].
        /// </value>
        [Argument(ArgumentType.AtMostOnce,
            HelpText = "Shrinkage rate for trees, used to prevent over-fitting. Range: (0,1].",
            SortOrder = 2, ShortName = "lr", NullName = "<Auto>")]
        public double? LearningRate;

        /// <summary>
        /// The maximum number of leaves in one tree.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum leaves for trees.",
            SortOrder = 2, ShortName = "nl", NullName = "<Auto>")]
        public int? NumberOfLeaves;

        /// <summary>
        /// Minimum loss reduction required to make a further partition on a leaf node of
	/// the tree. The larger gamma is, the more conservative the algorithm will be.
	/// aka: gamma
	/// range: [0,\infnty]
        /// </summary>
	public int? MinSplitLoss;
#endif

        /// <summary>
        /// Maximum depth of a tree. Increasing this value will make the model more complex and
        /// more likely to overfit. 0 indicates no limit on depth. Beware that XGBoost aggressively
        /// consumes memory when training a deep tree. exact tree method requires non-zero value.
        /// range: [0,\infnty], default=6
        /// </summary>
        public int? MaxDepth;

        /// <summary>
        /// Minimum sum of instance weight (hessian) needed in a child. If the tree partition step
        /// results in a leaf node with the sum of instance weight less than min_child_weight, then
        /// the building process will give up further partitioning. In linear regression task, this
        /// simply corresponds to minimum number of instances needed to be in each node. The larger
        /// <cref>MinChildWeight</cref> is, the more conservative the algorithm will be.
        /// range: [0,\infnty]
        /// </summary>
        public float? MinChildWeight;

        private protected XGBoostTrainerBase(IHost host,
            SchemaShape.Column feature,
            SchemaShape.Column label, SchemaShape.Column weight = default, SchemaShape.Column groupId = default) : base(host, feature, label, weight, groupId)
        {
        }

#if false
        /// <summary>
        /// L2 regularization term on weights. Increasing this value will make model more conservative
        /// </summary>
        public float? L2Regularization;

        /// <summary>
	/// L1 regularization term on weights. Increasing this value will make model more conservative.
	/// </summary>
        public float? L1Regularization;
#endif

        public class OptionsBase : TrainerInputBaseWithGroupId
        {

            // Static override name map that maps friendly names to XGBMArguments arguments.
            // If an argument is not here, then its name is identical to a lightGBM argument
            // and does not require a mapping, for example, Subsample.
            // For a complete list, see https://xgboost.readthedocs.io/en/latest/parameter.html
            private protected static Dictionary<string, string> NameMapping = new Dictionary<string, string>()
            {
#if false
               {nameof(MinSplitLoss),                         "min_split_loss"},
               {nameof(NumberOfLeaves),                       "num_leaves"},
#endif
	           {nameof(MaxDepth),                             "max_depth" },
               {nameof(MinChildWeight),                   "min_child_weight" },
#if false
    	       {nameof(L2Regularization),          	      "lambda" },
       	       {nameof(L1Regularization),          	      "alpha" }
#endif
            };

            private BoosterParameterBase.OptionsBase _boosterParameter;

#if true
            /// <summary>
            /// Determines which booster to use.
            /// </summary>
            /// <value>
            /// Available boosters are <see cref="DartBooster"/>, and <see cref="GradientBooster"/>.
            /// </value>
            [Argument(ArgumentType.Multiple,
                        HelpText = "Which booster to use, can be gbtree, gblinear or dart. gbtree and dart use tree based model while gblinear uses linear function.",
                        Name = "Booster",
                        SortOrder = 3)]
            internal IBoosterParameterFactory BoosterFactory = new GradientBooster.Options();
#endif

            /// <summary>
            /// Booster parameter to use
            /// </summary>
            public BoosterParameterBase.OptionsBase Booster
            {
                get => _boosterParameter;

                set
                {
                    _boosterParameter = value;
                    BoosterFactory = _boosterParameter;
                }

            }

            private protected string GetOptionName(string name)
            {
                if (NameMapping.ContainsKey(name))
                    return NameMapping[name];
                return XGBoostInterfaceUtils.GetOptionName(name);
            }
        }

        private protected override TModel TrainModelCore(TrainContext context)
        {
#if true
            return null;
#else
            InitializeBeforeTraining();

            Host.CheckValue(context, nameof(context));

            Dataset dtrain = null;
            Dataset dvalid = null;
            CategoricalMetaData catMetaData;
            try
            {
                using (var ch = Host.Start("Loading data for XGBoost"))
                {
                    using (var pch = Host.StartProgressChannel("Loading data for XGBoost"))
                    {
                        dtrain = LoadTrainingData(ch, context.TrainingSet, out catMetaData);
                        if (context.ValidationSet != null)
                            dvalid = LoadValidationData(ch, dtrain, context.ValidationSet, catMetaData);
                    }
                }
                using (var ch = Host.Start("Training with XGBoost"))
                {
                    using (var pch = Host.StartProgressChannel("Training with XGBoost"))
                        TrainCore(ch, pch, dtrain, catMetaData, dvalid);
                }
            }
            finally
            {
                dtrain?.Dispose();
                dvalid?.Dispose();
                DisposeParallelTraining();
            }
            return CreatePredictor();
#endif
        }

        private protected XGBoostTrainerBase(IHostEnvironment env, string name, TOptions options, SchemaShape.Column label)
           : base(Contracts.CheckRef(env, nameof(env)).Register(name), TrainerUtils.MakeR4VecFeature(options.FeatureColumnName), label,
         TrainerUtils.MakeR4ScalarWeightColumn(options.ExampleWeightColumnName), TrainerUtils.MakeU4ScalarColumn(options.RowGroupColumnName))
        {
            Host.CheckValue(options, nameof(options));
#if false
            Contracts.CheckUserArg(options.NumberOfIterations >= 0, nameof(options.NumberOfIterations), "must be >= 0.");
            Contracts.CheckUserArg(options.MaximumBinCountPerFeature > 0, nameof(options.MaximumBinCountPerFeature), "must be > 0.");
            Contracts.CheckUserArg(options.MinimumExampleCountPerGroup > 0, nameof(options.MinimumExampleCountPerGroup), "must be > 0.");
            Contracts.CheckUserArg(options.MaximumCategoricalSplitPointCount > 0, nameof(options.MaximumCategoricalSplitPointCount), "must be > 0.");
            Contracts.CheckUserArg(options.CategoricalSmoothing >= 0, nameof(options.CategoricalSmoothing), "must be >= 0.");
            Contracts.CheckUserArg(options.L2CategoricalRegularization >= 0.0, nameof(options.L2CategoricalRegularization), "must be >= 0.");
#endif

#if false
            XGBoostTrainerOptions = options;
            GbmOptions = XGBoostTrainerOption.ToDictionary(Host);
#endif
        }

        private protected abstract TModel CreatePredictor();
    }
}
