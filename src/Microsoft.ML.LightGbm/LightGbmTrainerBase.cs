// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;

namespace Microsoft.ML.Trainers.LightGbm
{
    [BestFriend]
    internal static class Defaults
    {
        public const int NumberOfIterations = 100;
    }

    /// <summary>
    /// Lock for LightGBM trainer.
    /// </summary>
    internal static class LightGbmShared
    {
        // Lock for the operations that are multi-threading inside in LightGBM DLL.
        public static readonly object LockForMultiThreadingInside = new object();
        // Lock for the sampling stage, this can reduce the peak memory usage.
        public static readonly object SampleLock = new object();
    }

    /// <summary>
    /// Base class for all training with LightGBM.
    /// </summary>
    public abstract class LightGbmTrainerBase<TOptions, TOutput, TTransformer, TModel> : TrainerEstimatorBaseWithGroupId<TTransformer, TModel>
        where TTransformer : ISingleFeaturePredictionTransformer<TModel>
        where TModel : class
        where TOptions : LightGbmTrainerBase<TOptions, TOutput, TTransformer, TModel>.OptionsBase, new()
    {
        public class OptionsBase : TrainerInputBaseWithGroupId
        {
            // Static override name map that maps friendly names to lightGBM arguments.
            // If an argument is not here, then its name is identicaltto a lightGBM argument
            // and does not require a mapping, for example, Subsample.
            private protected static Dictionary<string, string> NameMapping = new Dictionary<string, string>()
            {
               {nameof(MinimumExampleCountPerLeaf),           "min_data_per_leaf"},
               {nameof(NumberOfLeaves),                       "num_leaves"},
               {nameof(MaximumBinCountPerFeature),            "max_bin" },
               {nameof(MinimumExampleCountPerGroup),          "min_data_per_group" },
               {nameof(MaximumCategoricalSplitPointCount),    "max_cat_threshold" },
               {nameof(CategoricalSmoothing),                 "cat_smooth" },
               {nameof(L2CategoricalRegularization),          "cat_l2" },
               {nameof(HandleMissingValue),                   "use_missing" }
            };

            private protected string GetOptionName(string name)
            {
                if (NameMapping.ContainsKey(name))
                    return NameMapping[name];
                return LightGbmInterfaceUtils.GetOptionName(name);
            }

            private protected OptionsBase() { }

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
            /// Available boosters are <see cref="DartBooster"/>, <see cref="GossBooster"/>, and <see cref="GradientBooster"/>.
            /// </value>
            [Argument(ArgumentType.Multiple,
                        HelpText = "Which booster to use, can be gbtree, gblinear or dart. gbtree and dart use tree based model while gblinear uses linear function.",
                        Name="Booster",
                        SortOrder = 3)]
            internal IBoosterParameterFactory BoosterFactory = new GradientBooster.Options();

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
            /// Determines the number of rounds, after which training will stop if validation metric doesn't improve.
            /// </summary>
            /// <value>
            /// 0 means disable early stopping.
            /// </value>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Rounds of early stopping, 0 will disable it.",
                ShortName = "es")]
            public int EarlyStoppingRound = 0;

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
            public bool HandleMissingValue = true;

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
            [Argument(ArgumentType.AtMostOnce, HelpText = "Lapalace smooth term in categorical feature spilt. Avoid the bias of small categories.")]
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

            private BoosterParameterBase.OptionsBase _boosterParameter;

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

            internal virtual Dictionary<string, object> ToDictionary(IHost host)
            {
                Contracts.CheckValue(host, nameof(host));
                Dictionary<string, object> res = new Dictionary<string, object>();

                var boosterParams = BoosterFactory.CreateComponent(host);
                boosterParams.UpdateParameters(res);
                res["boosting_type"] = boosterParams.BoosterName;

                res["verbose"] = Silent ? "-1" : "1";
                if (NumberOfThreads.HasValue)
                    res["nthread"] = NumberOfThreads.Value;

                res["seed"] = (Seed.HasValue) ? Seed : host.Rand.Next();

                res[GetOptionName(nameof(MaximumBinCountPerFeature))] = MaximumBinCountPerFeature;
                res[GetOptionName(nameof(HandleMissingValue))] = HandleMissingValue;
                res[GetOptionName(nameof(MinimumExampleCountPerGroup))] = MinimumExampleCountPerGroup;
                res[GetOptionName(nameof(MaximumCategoricalSplitPointCount))] = MaximumCategoricalSplitPointCount;
                res[GetOptionName(nameof(CategoricalSmoothing))] = CategoricalSmoothing;
                res[GetOptionName(nameof(L2CategoricalRegularization))] = L2CategoricalRegularization;

                return res;
            }
        }

        private sealed class CategoricalMetaData
        {
            public int NumCol;
            public int TotalCats;
            public int[] CategoricalBoudaries;
            public int[] OnehotIndices;
            public int[] OnehotBias;
            public bool[] IsCategoricalFeature;
        }

        // Contains the passed in options when the API is called
        private protected readonly TOptions LightGbmTrainerOptions;

        /// <summary>
        /// Stores argumments as objects to convert them to invariant string type in the end so that
        /// the code is culture agnostic. When retrieving key value from this dictionary as string
        /// please convert to string invariant by string.Format(CultureInfo.InvariantCulture, "{0}", Option[key]).
        /// </summary>
        private protected Dictionary<string, object> GbmOptions;

        private protected IParallel ParallelTraining;

        // Store _featureCount and _trainedEnsemble to construct predictor.
        private protected int FeatureCount;
        private protected InternalTreeEnsemble TrainedEnsemble;

        private static readonly TrainerInfo _info = new TrainerInfo(normalization: false, caching: false, supportValid: true);
        public override TrainerInfo Info => _info;

        private protected LightGbmTrainerBase(IHostEnvironment env,
            string name,
            SchemaShape.Column labelColumn,
            string featureColumnName,
            string exampleWeightColumnName,
            string rowGroupColumnName,
            int? numberOfLeaves,
            int? minimumExampleCountPerLeaf,
            double? learningRate,
            int numberOfIterations)
            : this(env, name, new TOptions()
                  {
                    NumberOfLeaves = numberOfLeaves,
                    MinimumExampleCountPerLeaf = minimumExampleCountPerLeaf,
                    LearningRate = learningRate,
                    NumberOfIterations = numberOfIterations,
                    LabelColumnName = labelColumn.Name,
                    FeatureColumnName = featureColumnName,
                    ExampleWeightColumnName = exampleWeightColumnName,
                    RowGroupColumnName = rowGroupColumnName
                  },
                  labelColumn)
        {
        }

        private protected LightGbmTrainerBase(IHostEnvironment env, string name, TOptions options, SchemaShape.Column label)
           : base(Contracts.CheckRef(env, nameof(env)).Register(name), TrainerUtils.MakeR4VecFeature(options.FeatureColumnName), label,
                 TrainerUtils.MakeR4ScalarWeightColumn(options.ExampleWeightColumnName), TrainerUtils.MakeU4ScalarColumn(options.RowGroupColumnName))
        {
            Host.CheckValue(options, nameof(options));
            Contracts.CheckUserArg(options.NumberOfIterations >= 0, nameof(options.NumberOfIterations), "must be >= 0.");
            Contracts.CheckUserArg(options.MaximumBinCountPerFeature > 0, nameof(options.MaximumBinCountPerFeature), "must be > 0.");
            Contracts.CheckUserArg(options.MinimumExampleCountPerGroup > 0, nameof(options.MinimumExampleCountPerGroup), "must be > 0.");
            Contracts.CheckUserArg(options.MaximumCategoricalSplitPointCount > 0, nameof(options.MaximumCategoricalSplitPointCount), "must be > 0.");
            Contracts.CheckUserArg(options.CategoricalSmoothing >= 0, nameof(options.CategoricalSmoothing), "must be >= 0.");
            Contracts.CheckUserArg(options.L2CategoricalRegularization >= 0.0, nameof(options.L2CategoricalRegularization), "must be >= 0.");

            LightGbmTrainerOptions = options;
            InitParallelTraining();
        }

        private protected override TModel TrainModelCore(TrainContext context)
        {
            Host.CheckValue(context, nameof(context));

            Dataset dtrain = null;
            Dataset dvalid = null;
            CategoricalMetaData catMetaData;
            try
            {
                using (var ch = Host.Start("Loading data for LightGBM"))
                {
                    using (var pch = Host.StartProgressChannel("Loading data for LightGBM"))
                    {
                        dtrain = LoadTrainingData(ch, context.TrainingSet, out catMetaData);
                        if (context.ValidationSet != null)
                            dvalid = LoadValidationData(ch, dtrain, context.ValidationSet, catMetaData);
                    }
                }
                using (var ch = Host.Start("Training with LightGBM"))
                {
                    using (var pch = Host.StartProgressChannel("Training with LightGBM"))
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
        }

        private void InitParallelTraining()
        {
            GbmOptions = LightGbmTrainerOptions.ToDictionary(Host);
            ParallelTraining = LightGbmTrainerOptions.ParallelTrainer != null ? LightGbmTrainerOptions.ParallelTrainer.CreateComponent(Host) : new SingleTrainer();

            if (ParallelTraining.ParallelType() != "serial" && ParallelTraining.NumMachines() > 1)
            {
                GbmOptions["tree_learner"] = ParallelTraining.ParallelType();
                var otherParams = ParallelTraining.AdditionalParams();
                if (otherParams != null)
                {
                    foreach (var pair in otherParams)
                        GbmOptions[pair.Key] = pair.Value;
                }

                Contracts.CheckValue(ParallelTraining.GetReduceScatterFunction(), nameof(ParallelTraining.GetReduceScatterFunction));
                Contracts.CheckValue(ParallelTraining.GetAllgatherFunction(), nameof(ParallelTraining.GetAllgatherFunction));
                LightGbmInterfaceUtils.Check(WrappedLightGbmInterface.NetworkInitWithFunctions(
                        ParallelTraining.NumMachines(),
                        ParallelTraining.Rank(),
                        ParallelTraining.GetReduceScatterFunction(),
                        ParallelTraining.GetAllgatherFunction()
                    ));
            }
        }

        private void DisposeParallelTraining()
        {
            if (ParallelTraining.NumMachines() > 1)
                LightGbmInterfaceUtils.Check(WrappedLightGbmInterface.NetworkFree());
        }

        private protected virtual void CheckDataValid(IChannel ch, RoleMappedData data)
        {
            data.CheckFeatureFloatVector();
            ch.CheckParam(data.Schema.Label.HasValue, nameof(data), "Need a label column");
        }

        private protected virtual void GetDefaultParameters(IChannel ch, int numRow, bool hasCategorical, int totalCats, bool hiddenMsg = false)
        {
            double learningRate = LightGbmTrainerOptions.LearningRate ?? DefaultLearningRate(numRow, hasCategorical, totalCats);
            int numberOfLeaves = LightGbmTrainerOptions.NumberOfLeaves ?? DefaultNumLeaves(numRow, hasCategorical, totalCats);
            int minimumExampleCountPerLeaf = LightGbmTrainerOptions.MinimumExampleCountPerLeaf ?? DefaultMinDataPerLeaf(numRow, numberOfLeaves, 1);
            GbmOptions["learning_rate"] = learningRate;
            GbmOptions["num_leaves"] = numberOfLeaves;
            GbmOptions["min_data_per_leaf"] = minimumExampleCountPerLeaf;
            if (!hiddenMsg)
            {
                if (!LightGbmTrainerOptions.LearningRate.HasValue)
                    ch.Info("Auto-tuning parameters: " + nameof(LightGbmTrainerOptions.LearningRate) + " = " + learningRate);
                if (!LightGbmTrainerOptions.NumberOfLeaves.HasValue)
                    ch.Info("Auto-tuning parameters: " + nameof(LightGbmTrainerOptions.NumberOfLeaves) + " = " + numberOfLeaves);
                if (!LightGbmTrainerOptions.MinimumExampleCountPerLeaf.HasValue)
                    ch.Info("Auto-tuning parameters: " + nameof(LightGbmTrainerOptions.MinimumExampleCountPerLeaf) + " = " + minimumExampleCountPerLeaf);
            }
        }

        [BestFriend]
        internal Dictionary<string, object> GetGbmParameters() => GbmOptions;

        private FloatLabelCursor.Factory CreateCursorFactory(RoleMappedData data)
        {
            var loadFlags = CursOpt.AllLabels | CursOpt.Features;
            if (PredictionKind == PredictionKind.Ranking)
                loadFlags |= CursOpt.Group;

            if (data.Schema.Weight.HasValue)
                loadFlags |= CursOpt.AllWeights;

            var factory = new FloatLabelCursor.Factory(data, loadFlags);
            return factory;
        }

        private static List<int> GetCategoricalBoundires(int[] categoricalFeatures, int rawNumCol)
        {
            List<int> catBoundaries = new List<int> { 0 };
            int curFidx = 0;
            int j = 0;
            while (curFidx < rawNumCol)
            {
                if (j < categoricalFeatures.Length && curFidx == categoricalFeatures[j])
                {
                    if (curFidx > catBoundaries[catBoundaries.Count - 1])
                        catBoundaries.Add(curFidx);
                    if (categoricalFeatures[j + 1] - categoricalFeatures[j] >= 0)
                    {
                        curFidx = categoricalFeatures[j + 1] + 1;
                        catBoundaries.Add(curFidx);
                    }
                    else
                    {
                        for (int i = curFidx + 1; i <= categoricalFeatures[j + 1] + 1; ++i)
                            catBoundaries.Add(i);
                        curFidx = categoricalFeatures[j + 1] + 1;
                    }
                    j += 2;
                }
                else
                {
                    catBoundaries.Add(curFidx + 1);
                    ++curFidx;
                }
            }
            return catBoundaries;
        }

        private static List<string> ConstructCategoricalFeatureMetaData(int[] categoricalFeatures, int rawNumCol, ref CategoricalMetaData catMetaData)
        {
            List<int> catBoundaries = GetCategoricalBoundires(categoricalFeatures, rawNumCol);
            catMetaData.NumCol = catBoundaries.Count - 1;
            catMetaData.CategoricalBoudaries = catBoundaries.ToArray();
            catMetaData.IsCategoricalFeature = new bool[catMetaData.NumCol];
            catMetaData.OnehotIndices = new int[rawNumCol];
            catMetaData.OnehotBias = new int[rawNumCol];
            List<string> catIndices = new List<string>();
            int j = 0;
            for (int i = 0; i < catMetaData.NumCol; ++i)
            {
                var numCat = catMetaData.CategoricalBoudaries[i + 1] - catMetaData.CategoricalBoudaries[i];
                if (numCat > 1)
                {
                    catMetaData.TotalCats += numCat;
                    catMetaData.IsCategoricalFeature[i] = true;
                    catIndices.Add(i.ToString());
                    for (int k = catMetaData.CategoricalBoudaries[i]; k < catMetaData.CategoricalBoudaries[i + 1]; ++k)
                    {
                        catMetaData.OnehotIndices[j] = i;
                        catMetaData.OnehotBias[j] = k - catMetaData.CategoricalBoudaries[i];
                        ++j;
                    }
                }
                else
                {
                    catMetaData.IsCategoricalFeature[i] = false;
                    catMetaData.OnehotIndices[j] = i;
                    catMetaData.OnehotBias[j] = 0;
                    ++j;
                }
            }
            return catIndices;
        }

        private CategoricalMetaData GetCategoricalMetaData(IChannel ch, RoleMappedData trainData, int numRow)
        {
            CategoricalMetaData catMetaData = new CategoricalMetaData();
            int[] categoricalFeatures = null;
            const int useCatThreshold = 50000;
            // Disable cat when data is too small, reduce the overfitting.
            bool useCat = LightGbmTrainerOptions.UseCategoricalSplit ?? numRow > useCatThreshold;
            if (!LightGbmTrainerOptions.UseCategoricalSplit.HasValue)
                ch.Info("Auto-tuning parameters: " + nameof(LightGbmTrainerOptions.UseCategoricalSplit) + " = " + useCat);
            if (useCat)
            {
                var featureCol = trainData.Schema.Schema[DefaultColumnNames.Features];
                AnnotationUtils.TryGetCategoricalFeatureIndices(trainData.Schema.Schema, featureCol.Index, out categoricalFeatures);
            }
            var colType = trainData.Schema.Feature.Value.Type;
            int rawNumCol = colType.GetVectorSize();
            FeatureCount = rawNumCol;
            catMetaData.TotalCats = 0;
            if (categoricalFeatures == null)
            {
                catMetaData.CategoricalBoudaries = null;
                catMetaData.NumCol = rawNumCol;
            }
            else
            {
                var catIndices = ConstructCategoricalFeatureMetaData(categoricalFeatures, rawNumCol, ref catMetaData);
                // Set categorical features
                GbmOptions["categorical_feature"] = string.Join(",", catIndices);
            }
            return catMetaData;
        }

        private Dataset LoadTrainingData(IChannel ch, RoleMappedData trainData, out CategoricalMetaData catMetaData)
        {
            // Verifications.
            Host.AssertValue(ch);
            ch.CheckValue(trainData, nameof(trainData));

            CheckDataValid(ch, trainData);

            // Load metadata first.
            var factory = CreateCursorFactory(trainData);
            GetMetainfo(ch, factory, out int numRow, out float[] labels, out float[] weights, out int[] groups);
            catMetaData = GetCategoricalMetaData(ch, trainData, numRow);
            GetDefaultParameters(ch, numRow, catMetaData.CategoricalBoudaries != null, catMetaData.TotalCats);

            CheckAndUpdateParametersBeforeTraining(ch, trainData, labels, groups);
            string param = LightGbmInterfaceUtils.JoinParameters(GbmOptions);

            Dataset dtrain;
            // To reduce peak memory usage, only enable one sampling task at any given time.
            lock (LightGbmShared.SampleLock)
            {
                CreateDatasetFromSamplingData(ch, factory, numRow,
                    param, labels, weights, groups, catMetaData, out dtrain);
            }

            // Push rows into dataset.
            LoadDataset(ch, factory, dtrain, numRow, LightGbmTrainerOptions.BatchSize, catMetaData);

            return dtrain;
        }

        private Dataset LoadValidationData(IChannel ch, Dataset dtrain, RoleMappedData validData, CategoricalMetaData catMetaData)
        {
            // Verifications.
            Host.AssertValue(ch);

            ch.CheckValue(validData, nameof(validData));

            CheckDataValid(ch, validData);

            // Load meta info first.
            var factory = CreateCursorFactory(validData);
            GetMetainfo(ch, factory, out int numRow, out float[] labels, out float[] weights, out int[] groups);

            // Construct validation dataset.
            Dataset dvalid = new Dataset(dtrain, numRow, labels, weights, groups);

            // Push rows into dataset.
            LoadDataset(ch, factory, dvalid, numRow, LightGbmTrainerOptions.BatchSize, catMetaData);

            return dvalid;
        }

        private void TrainCore(IChannel ch, IProgressChannel pch, Dataset dtrain, CategoricalMetaData catMetaData, Dataset dvalid = null)
        {
            Host.AssertValue(ch);
            Host.AssertValue(pch);
            Host.AssertValue(dtrain);
            Host.AssertValueOrNull(dvalid);

            // For multi class, the number of labels is required.
            ch.Assert(((ITrainer)this).PredictionKind != PredictionKind.MulticlassClassification || GbmOptions.ContainsKey("num_class"),
                "LightGBM requires the number of classes to be specified in the parameters.");

            // Only enable one trainer to run at one time.
            lock (LightGbmShared.LockForMultiThreadingInside)
            {
                ch.Info("LightGBM objective={0}", GbmOptions["objective"]);
                using (Booster bst = WrappedLightGbmTraining.Train(ch, pch, GbmOptions, dtrain,
                dvalid: dvalid, numIteration: LightGbmTrainerOptions.NumberOfIterations,
                verboseEval: LightGbmTrainerOptions.Verbose, earlyStoppingRound: LightGbmTrainerOptions.EarlyStoppingRound))
                {
                    TrainedEnsemble = bst.GetModel(catMetaData.CategoricalBoudaries);
                }
            }
        }

        /// <summary>
        /// Calculate the density of data. Only use top 1000 rows to calculate.
        /// </summary>
        private static double DetectDensity(FloatLabelCursor.Factory factory, int numRows = 1000)
        {
            int nonZeroCount = 0;
            int totalCount = 0;
            using (var cursor = factory.Create())
            {
                while (cursor.MoveNext() && numRows > 0)
                {
                    nonZeroCount += cursor.Features.GetValues().Length;
                    totalCount += cursor.Features.Length;
                    --numRows;
                }
            }
            return (double)nonZeroCount / totalCount;
        }

        /// <summary>
        /// Compute row count, list of labels, weights and group counts of the dataset.
        /// </summary>
        private void GetMetainfo(IChannel ch, FloatLabelCursor.Factory factory,
            out int numRow, out float[] labels, out float[] weights, out int[] groups)
        {
            ch.Check(factory.Data.Schema.Label != null, "The data should have label.");
            List<float> labelList = new List<float>();
            bool hasWeights = factory.Data.Schema.Weight != null;
            bool hasGroup = false;
            if (PredictionKind == PredictionKind.Ranking)
            {
                ch.Check(factory.Data.Schema.Group != null, "The data for ranking task should have group field.");
                hasGroup = true;
            }
            List<float> weightList = hasWeights ? new List<float>() : null;
            List<ulong> cursorGroups = hasGroup ? new List<ulong>() : null;

            using (var cursor = factory.Create())
            {
                while (cursor.MoveNext())
                {
                    if (labelList.Count == Utils.ArrayMaxSize)
                        throw ch.Except($"Dataset row count exceeded the maximum count of {Utils.ArrayMaxSize}");
                    labelList.Add(cursor.Label);
                    if (hasWeights)
                    {
                        // Default weight = 1.
                        if (float.IsNaN(cursor.Weight))
                            weightList.Add(1);
                        else
                            weightList.Add(cursor.Weight);
                    }
                    if (hasGroup)
                        cursorGroups.Add(cursor.Group);
                }
            }
            labels = labelList.ToArray();
            ConvertNaNLabels(ch, factory.Data, labels);
            numRow = labels.Length;
            ch.Check(numRow > 0, "Cannot use empty dataset.");
            weights = hasWeights ? weightList.ToArray() : null;
            groups = null;
            if (hasGroup)
            {
                List<int> groupList = new List<int>();
                int lastGroup = -1;
                for (int i = 0; i < numRow; ++i)
                {
                    if (i == 0 || cursorGroups[i] != cursorGroups[i - 1])
                    {
                        groupList.Add(1);
                        ++lastGroup;
                    }
                    else
                        ++groupList[lastGroup];
                }
                groups = groupList.ToArray();
            }
        }

        /// <summary>
        /// Convert Nan labels. Default way is converting them to zero.
        /// </summary>
        private protected virtual void ConvertNaNLabels(IChannel ch, RoleMappedData data, float[] labels)
        {
            for (int i = 0; i < labels.Length; ++i)
            {
                if (float.IsNaN(labels[i]))
                    labels[i] = 0;
            }
        }

        private static bool MoveMany(FloatLabelCursor cursor, long count)
        {
            for (long i = 0; i < count; ++i)
            {
                if (!cursor.MoveNext())
                    return false;
            }
            return true;
        }

        private void GetFeatureValueDense(IChannel ch, FloatLabelCursor cursor, CategoricalMetaData catMetaData, Random rand, out ReadOnlySpan<float> featureValues)
        {
            var cursorFeaturesValues = cursor.Features.GetValues();
            if (catMetaData.CategoricalBoudaries != null)
            {
                float[] featureValuesTemp = new float[catMetaData.NumCol];
                for (int i = 0; i < catMetaData.NumCol; ++i)
                {
                    float fv = cursorFeaturesValues[catMetaData.CategoricalBoudaries[i]];
                    if (catMetaData.IsCategoricalFeature[i])
                    {
                        int hotIdx = catMetaData.CategoricalBoudaries[i] - 1;
                        int nhot = 0;
                        for (int j = catMetaData.CategoricalBoudaries[i]; j < catMetaData.CategoricalBoudaries[i + 1]; ++j)
                        {
                            if (cursorFeaturesValues[j] > 0)
                            {
                                // Reservoir Sampling.
                                nhot++;
                                var prob = rand.NextSingle();
                                if (prob < 1.0f / nhot)
                                    hotIdx = j;
                            }
                        }
                        // All-Zero is category 0.
                        fv = hotIdx - catMetaData.CategoricalBoudaries[i] + 1;
                    }
                    featureValuesTemp[i] = fv;
                }
                featureValues = featureValuesTemp;
            }
            else
            {
                featureValues = cursorFeaturesValues;
            }
        }

        private void GetFeatureValueSparse(IChannel ch, FloatLabelCursor cursor,
            CategoricalMetaData catMetaData, Random rand, out ReadOnlySpan<int> indices,
            out ReadOnlySpan<float> featureValues, out int cnt)
        {
            var cursorFeaturesValues = cursor.Features.GetValues();
            var cursorFeaturesIndices = cursor.Features.GetIndices();
            if (catMetaData.CategoricalBoudaries != null)
            {
                List<int> featureIndices = new List<int>();
                List<float> values = new List<float>();
                int lastIdx = -1;
                int nhot = 0;
                for (int i = 0; i < cursorFeaturesValues.Length; ++i)
                {
                    float fv = cursorFeaturesValues[i];
                    int colIdx = cursorFeaturesIndices[i];
                    int newColIdx = catMetaData.OnehotIndices[colIdx];
                    if (catMetaData.IsCategoricalFeature[newColIdx])
                        fv = catMetaData.OnehotBias[colIdx] + 1;
                    if (newColIdx != lastIdx)
                    {
                        featureIndices.Add(newColIdx);
                        values.Add(fv);
                        nhot = 1;
                    }
                    else
                    {
                        // Multi-hot.
                        ++nhot;
                        var prob = rand.NextSingle();
                        if (prob < 1.0f / nhot)
                            values[values.Count - 1] = fv;
                    }
                    lastIdx = newColIdx;
                }
                indices = featureIndices.ToArray();
                featureValues = values.ToArray();
                cnt = featureIndices.Count;
            }
            else
            {
                indices = cursorFeaturesIndices;
                featureValues = cursorFeaturesValues;
                cnt = cursorFeaturesValues.Length;
            }
        }

        /// <summary>
        /// Create a dataset from the sampling data.
        /// </summary>
        private void CreateDatasetFromSamplingData(IChannel ch, FloatLabelCursor.Factory factory,
            int numRow, string param, float[] labels, float[] weights, int[] groups, CategoricalMetaData catMetaData,
            out Dataset dataset)
        {
            Host.AssertValue(ch);

            int numSampleRow = GetNumSampleRow(numRow, FeatureCount);

            var rand = Host.Rand;
            double averageStep = (double)numRow / numSampleRow;
            int totalIdx = 0;
            int sampleIdx = 0;
            double density = DetectDensity(factory);

            double[][] sampleValuePerColumn = new double[catMetaData.NumCol][];
            int[][] sampleIndicesPerColumn = new int[catMetaData.NumCol][];
            int[] nonZeroCntPerColumn = new int[catMetaData.NumCol];
            int estimateNonZeroCnt = (int)(numSampleRow * density);
            estimateNonZeroCnt = Math.Max(1, estimateNonZeroCnt);
            for (int i = 0; i < catMetaData.NumCol; i++)
            {
                nonZeroCntPerColumn[i] = 0;
                sampleValuePerColumn[i] = new double[estimateNonZeroCnt];
                sampleIndicesPerColumn[i] = new int[estimateNonZeroCnt];
            };
            using (var cursor = factory.Create())
            {
                int step = 1;
                if (averageStep > 1)
                    step = rand.Next((int)(2 * averageStep - 1)) + 1;
                while (MoveMany(cursor, step))
                {
                    if (cursor.Features.IsDense)
                    {
                        GetFeatureValueDense(ch, cursor, catMetaData, rand, out ReadOnlySpan<float> featureValues);
                        for (int i = 0; i < catMetaData.NumCol; ++i)
                        {
                            float fv = featureValues[i];
                            if (fv == 0)
                                continue;
                            int curNonZeroCnt = nonZeroCntPerColumn[i];
                            Utils.EnsureSize(ref sampleValuePerColumn[i], curNonZeroCnt + 1);
                            Utils.EnsureSize(ref sampleIndicesPerColumn[i], curNonZeroCnt + 1);
                            // sampleValuePerColumn[i] is a vector whose j-th element is added when j-th non-zero value
                            // at the i-th feature is found as scanning the training data.
                            // In other words, sampleValuePerColumn[i][j] is the j-th non-zero i-th feature in the data set.
                            // when we scan the data matrix example-by-example.
                            sampleValuePerColumn[i][curNonZeroCnt] = fv;
                            // If the data set is dense, sampleValuePerColumn[i][j] would be the i-th feature at the j-th example.
                            // If the data set is not dense, sampleValuePerColumn[i][j] would be the i-th feature at the
                            // sampleIndicesPerColumn[i][j]-th example.
                            sampleIndicesPerColumn[i][curNonZeroCnt] = sampleIdx;
                            // The number of non-zero values at the i-th feature is nonZeroCntPerColumn[i].
                            nonZeroCntPerColumn[i] = curNonZeroCnt + 1;
                        }
                    }
                    else
                    {
                        GetFeatureValueSparse(ch, cursor, catMetaData, rand, out ReadOnlySpan<int> featureIndices, out ReadOnlySpan<float> featureValues, out int cnt);
                        for (int i = 0; i < cnt; ++i)
                        {
                            int colIdx = featureIndices[i];
                            float fv = featureValues[i];
                            if (fv == 0)
                                continue;
                            int curNonZeroCnt = nonZeroCntPerColumn[colIdx];
                            Utils.EnsureSize(ref sampleValuePerColumn[colIdx], curNonZeroCnt + 1);
                            Utils.EnsureSize(ref sampleIndicesPerColumn[colIdx], curNonZeroCnt + 1);
                            sampleValuePerColumn[colIdx][curNonZeroCnt] = fv;
                            sampleIndicesPerColumn[colIdx][curNonZeroCnt] = sampleIdx;
                            nonZeroCntPerColumn[colIdx] = curNonZeroCnt + 1;
                        }
                    }
                    // Actual row indexed sampled from the original data set
                    totalIdx += step;
                    // Row index in the sub-sampled data created in this loop.
                    ++sampleIdx;
                    if (numSampleRow == sampleIdx || numRow == totalIdx)
                        break;
                    averageStep = (double)(numRow - totalIdx) / (numSampleRow - sampleIdx);
                    step = 1;
                    if (averageStep > 1)
                        step = rand.Next((int)(2 * averageStep - 1)) + 1;
                }
            }
            dataset = new Dataset(sampleValuePerColumn, sampleIndicesPerColumn, catMetaData.NumCol, nonZeroCntPerColumn, sampleIdx, numRow, param, labels, weights, groups);
        }

        /// <summary>
        /// Load dataset. Use row batch way to reduce peak memory cost.
        /// </summary>
        private void LoadDataset(IChannel ch, FloatLabelCursor.Factory factory, Dataset dataset, int numRow, int batchSize, CategoricalMetaData catMetaData)
        {
            Host.AssertValue(ch);
            ch.AssertValue(factory);
            ch.AssertValue(dataset);
            ch.Assert(dataset.GetNumRows() == numRow);
            ch.Assert(dataset.GetNumCols() == catMetaData.NumCol);
            var rand = Host.Rand;
            // To avoid array resize, batch size should bigger than size of one row.
            batchSize = Math.Max(batchSize, catMetaData.NumCol);
            double density = DetectDensity(factory);
            int numElem = 0;
            int totalRowCount = 0;
            int curRowCount = 0;

            if (density >= 0.5)
            {
                int batchRow = batchSize / catMetaData.NumCol;
                batchRow = Math.Max(1, batchRow);
                if (batchRow > numRow)
                    batchRow = numRow;

                // This can only happen if the size of ONE example(row) exceeds the max array size. This looks like a very unlikely case.
                if ((long)catMetaData.NumCol * batchRow > Utils.ArrayMaxSize)
                    throw ch.Except("Size of array exceeded the " + nameof(Utils.ArrayMaxSize));

                float[] features = new float[catMetaData.NumCol * batchRow];

                using (var cursor = factory.Create())
                {
                    while (cursor.MoveNext())
                    {
                        ch.Assert(totalRowCount < numRow);
                        CopyToArray(ch, cursor, features, catMetaData, rand, ref numElem);
                        ++totalRowCount;
                        ++curRowCount;
                        if (batchRow == curRowCount)
                        {
                            ch.Assert(numElem == curRowCount * catMetaData.NumCol);
                            // PushRows is run by multi-threading inside, so lock here.
                            lock (LightGbmShared.LockForMultiThreadingInside)
                                dataset.PushRows(features, curRowCount, catMetaData.NumCol, totalRowCount - curRowCount);
                            curRowCount = 0;
                            numElem = 0;
                        }
                    }
                    ch.Assert(totalRowCount == numRow);
                    if (curRowCount > 0)
                    {
                        ch.Assert(numElem == curRowCount * catMetaData.NumCol);
                        // PushRows is run by multi-threading inside, so lock here.
                        lock (LightGbmShared.LockForMultiThreadingInside)
                            dataset.PushRows(features, curRowCount, catMetaData.NumCol, totalRowCount - curRowCount);
                    }
                }
            }
            else
            {
                int esimateBatchRow = (int)(batchSize / (catMetaData.NumCol * density));
                esimateBatchRow = Math.Max(1, esimateBatchRow);
                float[] features = new float[batchSize];
                int[] indices = new int[batchSize];
                int[] indptr = new int[esimateBatchRow + 1];

                using (var cursor = factory.Create())
                {
                    while (cursor.MoveNext())
                    {
                        ch.Assert(totalRowCount < numRow);
                        // Need push rows to LightGBM.
                        if (numElem + cursor.Features.GetValues().Length > features.Length)
                        {
                            // Mini batch size is greater than size of one row.
                            // So, at least we have the data of one row.
                            ch.Assert(curRowCount > 0);
                            Utils.EnsureSize(ref indptr, curRowCount + 1);
                            indptr[curRowCount] = numElem;
                            // PushRows is run by multi-threading inside, so lock here.
                            lock (LightGbmShared.LockForMultiThreadingInside)
                            {
                                dataset.PushRows(indptr, indices, features,
                                    curRowCount + 1, numElem, catMetaData.NumCol, totalRowCount - curRowCount);
                            }
                            curRowCount = 0;
                            numElem = 0;
                        }
                        Utils.EnsureSize(ref indptr, curRowCount + 1);
                        indptr[curRowCount] = numElem;
                        CopyToCsr(ch, cursor, indices, features, catMetaData, rand, ref numElem);
                        ++totalRowCount;
                        ++curRowCount;
                    }
                    ch.Assert(totalRowCount == numRow);
                    if (curRowCount > 0)
                    {
                        Utils.EnsureSize(ref indptr, curRowCount + 1);
                        indptr[curRowCount] = numElem;
                        // PushRows is run by multi-threading inside, so lock here.
                        lock (LightGbmShared.LockForMultiThreadingInside)
                        {
                            dataset.PushRows(indptr, indices, features, curRowCount + 1,
                                numElem, catMetaData.NumCol, totalRowCount - curRowCount);
                        }
                    }
                }
            }
        }

        private void CopyToArray(IChannel ch, FloatLabelCursor cursor, float[] features, CategoricalMetaData catMetaData, Random rand, ref int numElem)
        {
            ch.Assert(features.Length >= numElem + catMetaData.NumCol);
            if (catMetaData.CategoricalBoudaries != null)
            {
                if (cursor.Features.IsDense)
                {
                    GetFeatureValueDense(ch, cursor, catMetaData, rand, out ReadOnlySpan<float> featureValues);
                    for (int i = 0; i < catMetaData.NumCol; ++i)
                        features[numElem + i] = featureValues[i];
                    numElem += catMetaData.NumCol;
                }
                else
                {
                    GetFeatureValueSparse(ch, cursor, catMetaData, rand, out ReadOnlySpan<int> indices, out ReadOnlySpan<float> featureValues, out int cnt);
                    int lastIdx = 0;
                    for (int i = 0; i < cnt; i++)
                    {
                        int slot = indices[i];
                        float fv = featureValues[i];
                        Contracts.Assert(slot >= lastIdx);
                        while (lastIdx < slot)
                            features[numElem + lastIdx++] = 0.0f;
                        Contracts.Assert(lastIdx == slot);
                        features[numElem + lastIdx++] = fv;
                    }
                    while (lastIdx < catMetaData.NumCol)
                        features[numElem + lastIdx++] = 0.0f;
                    numElem += catMetaData.NumCol;
                }
            }
            else
            {
                cursor.Features.CopyTo(features, numElem, 0.0f);
                numElem += catMetaData.NumCol;
            }
        }

        private void CopyToCsr(IChannel ch, FloatLabelCursor cursor,
            int[] indices, float[] features, CategoricalMetaData catMetaData, Random rand, ref int numElem)
        {
            int numValue = cursor.Features.GetValues().Length;
            if (numValue > 0)
            {
                ch.Assert(indices.Length >= numElem + numValue);
                ch.Assert(features.Length >= numElem + numValue);

                if (cursor.Features.IsDense)
                {
                    GetFeatureValueDense(ch, cursor, catMetaData, rand, out ReadOnlySpan<float> featureValues);
                    for (int i = 0; i < catMetaData.NumCol; ++i)
                    {
                        float fv = featureValues[i];
                        if (fv == 0)
                            continue;
                        features[numElem] = fv;
                        indices[numElem] = i;
                        ++numElem;
                    }
                }
                else
                {
                    GetFeatureValueSparse(ch, cursor, catMetaData, rand, out ReadOnlySpan<int> featureIndices, out ReadOnlySpan<float> featureValues, out int cnt);
                    for (int i = 0; i < cnt; ++i)
                    {
                        int colIdx = featureIndices[i];
                        float fv = featureValues[i];
                        if (fv == 0)
                            continue;
                        features[numElem] = fv;
                        indices[numElem] = colIdx;
                        ++numElem;
                    }
                }
            }
        }

        private static double DefaultLearningRate(int numRow, bool useCat, int totalCats)
        {
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
        }

        private static int DefaultNumLeaves(int numRow, bool useCat, int totalCats)
        {
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
        }

        private protected static int DefaultMinDataPerLeaf(int numRow, int numberOfLeaves, int numClass)
        {
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
        }

        private static int GetNumSampleRow(int numRow, int numCol)
        {
            // Default is 65536.
            int ret = 1 << 16;
            // If have many features, use more sampling data.
            if (numCol >= 100000)
                ret *= 4;
            ret = Math.Min(ret, numRow);
            return ret;
        }

        private protected abstract TModel CreatePredictor();

        /// <summary>
        /// This function will be called before training. It will check the label/group and add parameters for specific applications.
        /// </summary>
        private protected abstract void CheckAndUpdateParametersBeforeTraining(IChannel ch,
            RoleMappedData data, float[] labels, int[] groups);
    }
}
