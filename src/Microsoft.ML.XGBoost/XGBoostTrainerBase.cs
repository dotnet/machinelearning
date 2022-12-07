// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.FastTree;

namespace Microsoft.ML.Trainers.XGBoost
{
    [BestFriend]
    internal static class Defaults
    {
        public const int NumberOfIterations = 100;
    }

    /// <summary>
    /// Lock for XGBoost trainer.
    /// </summary>
    internal static class XGBoostShared
    {
        // Lock for the operations that are multi-threading inside in XGBoost.
        public static readonly object LockForMultiThreadingInside = new object();
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

        private protected XGBoostTrainerBase(IHost host,
            SchemaShape.Column feature,
            SchemaShape.Column label, SchemaShape.Column weight = default, SchemaShape.Column groupId = default)
            : base(host, feature, label, weight, groupId)
        {
            System.Console.WriteLine("**** In base trainer ctor 1");
        }

        public class OptionsBase : TrainerInputBaseWithGroupId
        {

            // Static override name map that maps friendly names to XGBMArguments arguments.
            // If an argument is not here, then its name is identical to a lightGBM argument
            // and does not require a mapping, for example, Subsample.
            // For a complete list, see https://xgboost.readthedocs.io/en/latest/parameter.html
            private protected static Dictionary<string, string> NameMapping = new Dictionary<string, string>()
            {
// -------------------- xgboost ----------------------
               {nameof(MinSplitLoss),                         "min_split_loss"},
               {nameof(NumberOfLeaves),                       "max_leaves"},
               {nameof(MaxDepth),                             "max_depth" },
               
 #if false
               {nameof(MinChildWeight),                   "min_child_weight" },
    	       {nameof(L2Regularization),          	      "lambda" },
       	       {nameof(L1Regularization),          	      "alpha" },
// -------------------- lightgbm ----------------------
               {nameof(MinimumExampleCountPerLeaf),           "min_data_per_leaf"},
               {nameof(NumberOfLeaves),                       "num_leaves"},
	       #endif
               {nameof(MaximumBinCountPerFeature),            "max_bin" },
               {nameof(MinimumExampleCountPerGroup),          "min_data_per_group" },
               {nameof(MaximumCategoricalSplitPointCount),    "max_cat_threshold" },
	       	       	       #if false
               {nameof(CategoricalSmoothing),                 "cat_smooth" },
               {nameof(L2CategoricalRegularization),          "cat_l2" },
	       #endif
               {nameof(HandleMissingValue),                   "use_missing" },
               {nameof(UseZeroAsMissingValue),                "zero_as_missing" }
            };

            /// <summary>
            /// The maximum number of bins that feature values will be bucketed in.
            /// </summary>
            /// <remarks>
            /// The small number of bins may reduce training accuracy but may increase general power (deal with over-fitting).
            /// </remarks>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum number of bucket bin for features.", ShortName = "mb")]
            public int MaximumBinCountPerFeature = 255;

            /// <summary>
            /// The random seed for XGBoost to use.
            /// </summary>
            /// <value>
            /// If not specified, <see cref="MLContext"/> will generate a random seed to be used.
            /// </value>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Sets the random seed for XGBoost to use.")]
            public int? Seed;

            /// <summary>
            /// Whether to enable special handling of missing value or not.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Enable special handling of missing value or not.", ShortName = "hmv")]
            public bool HandleMissingValue = true;

            /// <summary>
            /// Whether to enable the usage of zero (0) as missing value.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Enable usage of zero (0) as missing value.", ShortName = "uzam")]
            public bool UseZeroAsMissingValue = false;

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
            /// The minimal number of data points required to form a new tree leaf.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Minimum number of instances needed in a child.",
                SortOrder = 2, ShortName = "mil", NullName = "<Auto>")]
            public int? MinimumExampleCountPerLeaf;

            /// <summary>
            /// Maximum categorical split points to consider when splitting on a categorical feature.
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
            [Argument(ArgumentType.AtMostOnce, HelpText = "Lapalace smooth term in categorical feature spilt. Avoid the bias of small categories.")]
            public double CategoricalSmoothing = 10;

            /// <summary>
            /// L2 regularization for categorical split.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "L2 Regularization for categorical split.")]
            public double L2CategoricalRegularization = 10;

            /// <summary>
            /// Minimum loss reduction required to make a further partition on a leaf node of
            /// the tree. The larger gamma is, the more conservative the algorithm will be.
            /// aka: gamma
            /// range: [0,\infnty]
            /// </summary>
            public int? MinSplitLoss;

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

            /// <summary>
            /// The minimum number of data points per categorical group.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Minimum number of instances per categorical group.", ShortName = "mdpg")]
            public int MinimumExampleCountPerGroup = 100;

            private BoosterParameterBase.OptionsBase _boosterParameter;

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

            private protected string GetOptionName(string name)
            {
                if (NameMapping.ContainsKey(name))
                    return NameMapping[name];
                return XGBoostInterfaceUtils.GetOptionName(name);
            }


            internal virtual Dictionary<string, object> ToDictionary(IHost host)
            {
                Contracts.CheckValue(host, nameof(host));
                Dictionary<string, object> res = new Dictionary<string, object>();

                var boosterParams = BoosterFactory.CreateComponent(host);
                boosterParams.UpdateParameters(res);
                res["booster"] = boosterParams.BoosterName;

                res["verbose"] = Silent ? "-1" : "1";
#if false
if (NumberOfThreads.HasValue)
                    res["nthread"] = NumberOfThreads.Value;
#endif

                res["seed"] = (Seed.HasValue) ? Seed : host.Rand.Next();
                res[GetOptionName(nameof(MaximumBinCountPerFeature))] = MaximumBinCountPerFeature;
                res[GetOptionName(nameof(HandleMissingValue))] = HandleMissingValue;
                res[GetOptionName(nameof(UseZeroAsMissingValue))] = UseZeroAsMissingValue;
                res[GetOptionName(nameof(MinimumExampleCountPerGroup))] = MinimumExampleCountPerGroup;
                res[GetOptionName(nameof(MaximumCategoricalSplitPointCount))] = MaximumCategoricalSplitPointCount;
                res[GetOptionName(nameof(CategoricalSmoothing))] = CategoricalSmoothing;
                res[GetOptionName(nameof(L2CategoricalRegularization))] = L2CategoricalRegularization;
                return res;
            }

            /// <summary>
            /// The number of boosting iterations. A new tree is created in each iteration, so this is equivalent to the number of trees.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of iterations.", SortOrder = 1, ShortName = "iter")]
            public int NumberOfIterations = Defaults.NumberOfIterations;
        }

        // Contains the passed in options when the API is called
        private protected readonly TOptions XGBoostTrainerOptions;

        /// <summary>
        /// Stores arguments as objects to convert them to invariant string type in the end so that
        /// the code is culture agnostic. When retrieving key value from this dictionary as string
        /// please convert to string invariant by string.Format(CultureInfo.InvariantCulture, "{0}", Option[key]).
        /// </summary>
        private protected readonly Dictionary<string, object> GbmOptions;

        private protected override TModel TrainModelCore(TrainContext context)
        {
            InitializeBeforeTraining();
            Host.CheckValue(context, nameof(context));

#if false
            Dataset dtrain = null;
            Dataset dvalid = null;
            CategoricalMetaData catMetaData;
#else
#pragma warning disable 0219
            DMatrix dtrain = null;
#pragma warning restore 0219
#endif
            try
            {
                using (var ch = Host.Start("Loading data for XGBoost"))
                {
                    using (var pch = Host.StartProgressChannel("Loading data for XGBoost"))
                    {
                        dtrain = LoadTrainingData(ch, context.TrainingSet);
#if false
                        if (context.ValidationSet != null)
                            dvalid = LoadValidationData(ch, dtrain, context.ValidationSet, catMetaData);
#endif
                    }

                }

                using (var ch = Host.Start("Training with XGBoost"))
                {
                    using (var pch = Host.StartProgressChannel("Training with XGBoost"))
                        TrainCore(ch, pch, dtrain);
                }
            }
            finally
            {
#if false
                dtrain?.Dispose();
                dvalid?.Dispose();
                DisposeParallelTraining();
#endif
            }
            return CreatePredictor();
        }

        private protected virtual void InitializeBeforeTraining() { }

        private DMatrix LoadTrainingData(IChannel ch, RoleMappedData trainData)
        {
            Host.AssertValue(ch);
            ch.CheckValue(trainData, nameof(trainData));

            CheckDataValid(ch, trainData);

            var loadFlags = CursOpt.AllLabels | CursOpt.AllFeatures;
            var factory = new FloatLabelCursor.Factory(trainData, loadFlags);

            int featureDimensionality = 0;
            var typ = trainData.Schema.Feature;
            if (typ.HasValue)
            {
                if (typ.Value.Type is VectorDataViewType vt)
                {
                    featureDimensionality = vt.Size;
                }
            }
            ch.Assert(featureDimensionality > 0);
#if true
            FeatureCount = featureDimensionality;
#else
            var colType = trainData.Schema.Feature.Value.Type;
            int rawNumCol = colType.GetVectorSize();
            FeatureCount = rawNumCol;
#endif

            GetDefaultParameters(ch);
            CheckAndUpdateParametersBeforeTraining(ch, trainData);

            foreach (var k in GbmOptions.Keys)
            {
                System.Console.WriteLine($"Got key {k}: [{GbmOptions[k]}]");
            }

#if false
            string param = LightGbmInterfaceUtils.JoinParameters(GbmOptions);
#else
            DMatrix dtrain = LoadDMatrix(ch, factory, featureDimensionality);
            Console.WriteLine($"DMatrix has {dtrain.GetNumRows()} rows and {dtrain.GetNumCols()} columns.");
            return dtrain;
#endif
        }

        /// <summary>
        /// Load dataset. Use row batch way to reduce peak memory cost.
        /// </summary>
        private DMatrix LoadDMatrix(IChannel ch, FloatLabelCursor.Factory factory, int featureDimensionality)
        {

            Host.AssertValue(ch);
            ch.AssertValue(factory);

            List<float[]> acc = new List<float[]>();
            List<float> accLabels = new List<float>();
            ulong numRows = 0;
            using (var cursor = factory.Create())
            {
                while (cursor.MoveNext())
                {
                    acc.Add(cursor.Features.GetValues().ToArray());
                    accLabels.Add(cursor.Label);
                    numRows++;
#if false
// FIXME: dump arrays to verify
                    string strVec = string.Join(",", (cursor.Features.GetValues().ToArray()).Select(x => x.ToString()).ToArray());
                    Console.WriteLine($"features: [{strVec}], label: {cursor.Label}");
#endif
                }
            }
            var flatArray = (acc.ToArray()).SelectMany(x => x).ToArray();

            DMatrix dmat = new DMatrix(flatArray, (uint)numRows, (uint)featureDimensionality, accLabels.ToArray());
            return dmat;
        }

        private void TrainCore(IChannel ch, IProgressChannel pch, DMatrix dtrain)
        {

            Host.AssertValue(ch);
            Host.AssertValue(pch);
#if false
            Host.AssertValue(dtrain);
            Host.AssertValueOrNull(dvalid);
            Host.CheckAlive();
#endif

#if true
            Console.WriteLine("**** Trying to get labels");
            var labfrommat = dtrain.GetLabels();
            Console.WriteLine($"Got labels of length {labfrommat.Length}.");
#endif

            // Only enable one trainer to run at one time.
            lock (XGBoostShared.LockForMultiThreadingInside)
            {
                ch.Info("XGBoost objective={0}", GbmOptions["objective"]);
                Console.WriteLine("XGBoost objective={0}", GbmOptions["objective"]);
                using (Booster bst = WrappedXGBoostTraining.Train(Host, ch, pch, GbmOptions, dtrain
#if false
               ,dvalid: dvalid, numIteration: LightGbmTrainerOptions.NumberOfIterations,
                verboseEval: LightGbmTrainerOptions.Verbose, earlyStoppingRound: LightGbmTrainerOptions.EarlyStoppingRound)
#endif
                 ))
                {
                    TrainedEnsemble = bst.DumpModel();
                }
            }

        }

        private protected XGBoostTrainerBase(IHostEnvironment env,
            string name,
            SchemaShape.Column labelColumn,
            string featureColumnName,
            string exampleWeightColumnName,
            string rowGroupColumnName, /*
            int? numberOfLeaves,
            int? minimumExampleCountPerLeaf,
            double? learningRate, */
            int numberOfIterations)
        : this(env, name, new TOptions()
        {
#if false
        NumberOfLeaves = numberOfLeaves,
        MinimumExampleCountPerLeaf = minimumExampleCountPerLeaf,
        LearningRate = learningRate,
        NumberOfIterations = numberOfIterations,
#endif
            LabelColumnName = labelColumn.Name,
            FeatureColumnName = featureColumnName,
            ExampleWeightColumnName = exampleWeightColumnName,
            RowGroupColumnName = rowGroupColumnName
        },
          labelColumn)
        {
            System.Console.WriteLine("***** In base trainer ctor 2");
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
            System.Console.WriteLine("***** In base trainer ctor 3");

            XGBoostTrainerOptions = options;
            GbmOptions = XGBoostTrainerOptions.ToDictionary(Host);
        }

        private protected virtual void CheckDataValid(IChannel ch, RoleMappedData data)
        {
            data.CheckFeatureFloatVector();
            ch.CheckParam(data.Schema.Label.HasValue, nameof(data), "Need a label column");
        }

        private static double DefaultLearningRate(
#if false
	int numRow, bool useCat, int totalCats
#endif
    )
        {
#if false
            if (useCat)
            {
                if (totalCats < 1e6)
                    return 0.1;
                else
                    return 0.15;
            }
            else if (numRow <= 100000)
                return 0.2;
            else
                return 0.25;
#else
            return 0.25;
#endif
        }

        private static int DefaultNumLeaves(
#if false
	int numRow, bool useCat, int totalCats
#endif
    )
        {
#if false
            if (useCat && totalCats > 100)
            {
                if (totalCats < 1e6)
                    return 20;
                else
                    return 30;
            }
            else if (numRow <= 100000)
                return 20;
            else
                return 30;
#else
            return 30;
#endif
        }

        private protected static int DefaultMinDataPerLeaf(
#if false
	int numRow, int numberOfLeaves, int numClass
#endif
    )
        {
#if false
            if (numClass > 1)
            {
                int ret = numRow / numberOfLeaves / numClass / 10;
                ret = Math.Max(ret, 5);
                ret = Math.Min(ret, 50);
                return ret;
            }
            else
            {
                return 20;
            }
#else
            return 20;
#endif
        }

        private protected virtual void GetDefaultParameters(IChannel ch
#if false
, int numRow, bool hasCategorical, int totalCats
#endif
    , bool hiddenMsg = false
    )
        {
            double learningRate = XGBoostTrainerOptions.LearningRate ?? DefaultLearningRate(
#if false
	    numRow, hasCategorical, totalCats
#endif
        );
            int numberOfLeaves = XGBoostTrainerOptions.NumberOfLeaves ?? DefaultNumLeaves(
#if false
	    numRow, hasCategorical, totalCats
#endif
        );

            int minimumExampleCountPerLeaf = XGBoostTrainerOptions.MinimumExampleCountPerLeaf ?? DefaultMinDataPerLeaf(
#if false
	    numRow, numberOfLeaves, 1
#endif
        );

            GbmOptions["learning_rate"] = learningRate;
            GbmOptions["num_leaves"] = numberOfLeaves;
            GbmOptions["min_data_per_leaf"] = minimumExampleCountPerLeaf;
        }

        private protected abstract TModel CreatePredictor();

        /// <summary>
        /// This function will be called before training. It will check the label/group and add parameters for specific applications.
        /// </summary>
        private protected abstract void CheckAndUpdateParametersBeforeTraining(IChannel ch,
            RoleMappedData data
#if false
	    , float[] labels, int[] groups
#endif
        );
    }
}
