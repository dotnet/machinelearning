// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Training;
using System;
using System.Collections.Generic;

namespace Microsoft.ML.Runtime.LightGBM
{
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
    public abstract class LightGbmTrainerBase<TOutput, TTransformer, TModel> : TrainerEstimatorBase<TTransformer, TModel>
        where TTransformer : ISingleFeaturePredictionTransformer<TModel>
        where TModel : IPredictorProducing<TOutput>
    {
        private sealed class CategoricalMetaData
        {
            public int NumCol;
            public int TotalCats;
            public int[] CategoricalBoudaries;
            public int[] OnehotIndices;
            public int[] OnehotBias;
            public bool[] IsCategoricalFeature;
        }

        private protected readonly LightGbmArguments Args;

        /// <summary>
        /// Stores argumments as objects to convert them to invariant string type in the end so that
        /// the code is culture agnostic. When retrieving key value from this dictionary as string
        /// please convert to string invariant by string.Format(CultureInfo.InvariantCulture, "{0}", Option[key]).
        /// </summary>
        private protected Dictionary<string, object> Options;
        private protected IParallel ParallelTraining;

        // Store _featureCount and _trainedEnsemble to construct predictor.
        private protected int FeatureCount;
        private protected FastTree.Internal.Ensemble TrainedEnsemble;

        private static readonly TrainerInfo _info = new TrainerInfo(normalization: false, caching: false, supportValid: true);
        public override TrainerInfo Info => _info;

        private protected LightGbmTrainerBase(IHostEnvironment env, string name, SchemaShape.Column label, string featureColumn,
            string weightColumn = null, string groupIdColumn = null, Action<LightGbmArguments> advancedSettings = null)
            : base(Contracts.CheckRef(env, nameof(env)).Register(name), TrainerUtils.MakeR4VecFeature(featureColumn), label, TrainerUtils.MakeR4ScalarWeightColumn(weightColumn))
        {
            Args = new LightGbmArguments();

            //apply the advanced args, if the user supplied any
            advancedSettings?.Invoke(Args);
            Args.LabelColumn = label.Name;
            Args.FeatureColumn = featureColumn;

            if (weightColumn != null)
                Args.WeightColumn = weightColumn;

            if (groupIdColumn != null)
                Args.GroupIdColumn = groupIdColumn;

            InitParallelTraining();
        }

        private protected LightGbmTrainerBase(IHostEnvironment env, string name, LightGbmArguments args, SchemaShape.Column label)
           : base(Contracts.CheckRef(env, nameof(env)).Register(name), TrainerUtils.MakeR4VecFeature(args.FeatureColumn), label, TrainerUtils.MakeR4ScalarWeightColumn(args.WeightColumn))
        {
            Host.CheckValue(args, nameof(args));

            Args = args;
            InitParallelTraining();
        }

        protected override TModel TrainModelCore(TrainContext context)
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
                    ch.Done();
                }
                using (var ch = Host.Start("Training with LightGBM"))
                {
                    using (var pch = Host.StartProgressChannel("Training with LightGBM"))
                        TrainCore(ch, pch, dtrain, catMetaData, dvalid);
                    ch.Done();
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
            Options = Args.ToDictionary(Host);
            ParallelTraining = Args.ParallelTrainer != null ? Args.ParallelTrainer.CreateComponent(Host) : new SingleTrainer();

            if (ParallelTraining.ParallelType() != "serial" && ParallelTraining.NumMachines() > 1)
            {
                Options["tree_learner"] = ParallelTraining.ParallelType();
                var otherParams = ParallelTraining.AdditionalParams();
                if (otherParams != null)
                {
                    foreach (var pair in otherParams)
                        Options[pair.Key] = pair.Value;
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

        protected virtual void CheckDataValid(IChannel ch, RoleMappedData data)
        {
            data.CheckFeatureFloatVector();
            ch.CheckParam(data.Schema.Label != null, nameof(data), "Need a label column");
        }

        protected virtual void GetDefaultParameters(IChannel ch, int numRow, bool hasCategarical, int totalCats, bool hiddenMsg=false)
        {
            double learningRate = Args.LearningRate ?? DefaultLearningRate(numRow, hasCategarical, totalCats);
            int numLeaves = Args.NumLeaves ?? DefaultNumLeaves(numRow, hasCategarical, totalCats);
            int minDataPerLeaf = Args.MinDataPerLeaf ?? DefaultMinDataPerLeaf(numRow, numLeaves, 1);
            Options["learning_rate"] = learningRate;
            Options["num_leaves"] = numLeaves;
            Options["min_data_per_leaf"] = minDataPerLeaf;
            if (!hiddenMsg)
            {
                if (!Args.LearningRate.HasValue)
                    ch.Info("Auto-tuning parameters: " + nameof(Args.LearningRate) + " = " + learningRate);
                if (!Args.NumLeaves.HasValue)
                    ch.Info("Auto-tuning parameters: " + nameof(Args.NumLeaves) + " = " + numLeaves);
                if (!Args.MinDataPerLeaf.HasValue)
                    ch.Info("Auto-tuning parameters: " + nameof(Args.MinDataPerLeaf) + " = " + minDataPerLeaf);
            }
        }

        private FloatLabelCursor.Factory CreateCursorFactory(RoleMappedData data)
        {
            var loadFlags = CursOpt.AllLabels | CursOpt.AllWeights | CursOpt.Features;
            if (PredictionKind == PredictionKind.Ranking)
                loadFlags |= CursOpt.Group;

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
            bool useCat = Args.UseCat ?? numRow > useCatThreshold;
            if (!Args.UseCat.HasValue)
                ch.Info("Auto-tuning parameters: " + nameof(Args.UseCat) + " = " + useCat);
            if (useCat)
            {
                trainData.Schema.Schema.TryGetColumnIndex(DefaultColumnNames.Features, out int featureIndex);
                MetadataUtils.TryGetCategoricalFeatureIndices(trainData.Schema.Schema, featureIndex, out categoricalFeatures);
            }
            var colType = trainData.Schema.Feature.Type;
            int rawNumCol = colType.VectorSize;
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
                Options["categorical_feature"] = string.Join(",", catIndices);
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

            Dataset dtrain;
            string param = LightGbmInterfaceUtils.JoinParameters(Options);

            // To reduce peak memory usage, only enable one sampling task at any given time.
            lock (LightGbmShared.SampleLock)
            {
                CreateDatasetFromSamplingData(ch, factory, numRow,
                    param, labels, weights, groups, catMetaData, out dtrain);
            }

            // Push rows into dataset.
            LoadDataset(ch, factory, dtrain, numRow, Args.BatchSize, catMetaData);

            // Some checks.
            CheckAndUpdateParametersBeforeTraining(ch, trainData, labels, groups);
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
            LoadDataset(ch, factory, dvalid, numRow, Args.BatchSize, catMetaData);

            return dvalid;
        }

        private void TrainCore(IChannel ch, IProgressChannel pch, Dataset dtrain, CategoricalMetaData catMetaData, Dataset dvalid = null)
        {
            Host.AssertValue(ch);
            Host.AssertValue(pch);
            Host.AssertValue(dtrain);
            Host.AssertValueOrNull(dvalid);
            // For multi class, the number of labels is required.
            ch.Assert(PredictionKind != PredictionKind.MultiClassClassification || Options.ContainsKey("num_class"),
                "LightGBM requires the number of classes to be specified in the parameters.");

            // Only enable one trainer to run at one time.
            lock (LightGbmShared.LockForMultiThreadingInside)
            {
                ch.Info("LightGBM objective={0}", Options["objective"]);
                using (Booster bst = WrappedLightGbmTraining.Train(ch, pch, Options, dtrain,
                dvalid: dvalid, numIteration: Args.NumBoostRound,
                verboseEval: Args.VerboseEval, earlyStoppingRound: Args.EarlyStoppingRound))
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
                    nonZeroCount += cursor.Features.Count;
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
                ch.Check(factory.Data.Schema != null, "The data for ranking task should have group field.");
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
        protected virtual void ConvertNaNLabels(IChannel ch, RoleMappedData data, float[] labels)
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

        private void GetFeatureValueDense(IChannel ch, FloatLabelCursor cursor, CategoricalMetaData catMetaData, IRandom rand, out float[] featureValues)
        {
            if (catMetaData.CategoricalBoudaries != null)
            {
                featureValues = new float[catMetaData.NumCol];
                for (int i = 0; i < catMetaData.NumCol; ++i)
                {
                    float fv = cursor.Features.Values[catMetaData.CategoricalBoudaries[i]];
                    if (catMetaData.IsCategoricalFeature[i])
                    {
                        int hotIdx = catMetaData.CategoricalBoudaries[i] - 1;
                        int nhot = 0;
                        for (int j = catMetaData.CategoricalBoudaries[i]; j < catMetaData.CategoricalBoudaries[i + 1]; ++j)
                        {
                            if (cursor.Features.Values[j] > 0)
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
                    featureValues[i] = fv;
                }
            }
            else
            {
                featureValues = cursor.Features.Values;
            }
        }

        private void GetFeatureValueSparse(IChannel ch, FloatLabelCursor cursor,
            CategoricalMetaData catMetaData, IRandom rand, out int[] indices,
            out float[] featureValues, out int cnt)
        {
            if (catMetaData.CategoricalBoudaries != null)
            {
                List<int> featureIndices = new List<int>();
                List<float> values = new List<float>();
                int lastIdx = -1;
                int nhot = 0;
                for (int i = 0; i < cursor.Features.Count; ++i)
                {
                    float fv = cursor.Features.Values[i];
                    int colIdx = cursor.Features.Indices[i];
                    int newColIdx = catMetaData.OnehotIndices[colIdx];
                    if (catMetaData.IsCategoricalFeature[newColIdx])
                        fv = catMetaData.OnehotBias[colIdx] + 1;
                    if (newColIdx != lastIdx)
                    {
                        featureIndices.Push(newColIdx);
                        values.Push(fv);
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
                indices = cursor.Features.Indices;
                featureValues = cursor.Features.Values;
                cnt = cursor.Features.Count;
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
            for(int i = 0; i < catMetaData.NumCol; i++)
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
                        GetFeatureValueDense(ch, cursor, catMetaData, rand, out float[] featureValues);
                        for (int i = 0; i < catMetaData.NumCol; ++i)
                        {
                            float fv = featureValues[i];
                            if (fv == 0)
                                continue;
                            int curNonZeroCnt = nonZeroCntPerColumn[i];
                            Utils.EnsureSize(ref sampleValuePerColumn[i], curNonZeroCnt + 1);
                            Utils.EnsureSize(ref sampleIndicesPerColumn[i], curNonZeroCnt + 1);
                            sampleValuePerColumn[i][curNonZeroCnt] = fv;
                            sampleIndicesPerColumn[i][curNonZeroCnt] = sampleIdx;
                            nonZeroCntPerColumn[i] = curNonZeroCnt + 1;
                        }
                    }
                    else
                    {
                        GetFeatureValueSparse(ch, cursor, catMetaData, rand, out int[] featureIndices, out float[] featureValues, out int cnt);
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
                    totalIdx += step;
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
                        if (numElem + cursor.Features.Count > features.Length)
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

        private void CopyToArray(IChannel ch, FloatLabelCursor cursor, float[] features, CategoricalMetaData catMetaData, IRandom rand, ref int numElem)
        {
            ch.Assert(features.Length >= numElem + catMetaData.NumCol);
            if (catMetaData.CategoricalBoudaries != null)
            {
                if (cursor.Features.IsDense)
                {
                    GetFeatureValueDense(ch, cursor, catMetaData, rand, out float[] featureValues);
                    for (int i = 0; i < catMetaData.NumCol; ++i)
                        features[numElem + i] = featureValues[i];
                    numElem += catMetaData.NumCol;
                }
                else
                {
                    GetFeatureValueSparse(ch, cursor, catMetaData, rand, out int[] indices, out float[] featureValues, out int cnt);
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
            int[] indices, float[] features, CategoricalMetaData catMetaData, IRandom rand, ref int numElem)
        {
            int numValue = cursor.Features.Count;
            if (numValue > 0)
            {
                ch.Assert(indices.Length >= numElem + numValue);
                ch.Assert(features.Length >= numElem + numValue);

                if (cursor.Features.IsDense)
                {
                    GetFeatureValueDense(ch, cursor, catMetaData, rand, out float[] featureValues);
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
                    GetFeatureValueSparse(ch, cursor, catMetaData, rand, out int[] featureIndices, out float[] featureValues, out int cnt);
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

        protected static int DefaultMinDataPerLeaf(int numRow, int numLeaves, int numClass)
        {
            if (numClass > 1)
            {
                int ret = numRow / numLeaves / numClass / 10;
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
        protected abstract void CheckAndUpdateParametersBeforeTraining(IChannel ch,
            RoleMappedData data, float[] labels, int[] groups);
    }
}
