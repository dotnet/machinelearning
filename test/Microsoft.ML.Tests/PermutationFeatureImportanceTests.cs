// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.RunTests;
using System;
using System.Collections.Immutable;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public class PermutationFeatureImportanceTests : BaseTestPredictors
    {
        public PermutationFeatureImportanceTests(ITestOutputHelper output) : base(output)
        {
        }

        #region Regression Tests

        /// <summary>
        /// Test PFI Regression for Dense Features
        /// </summary>
        [Fact]
        public void TestPfiRegressionOnDenseFeatures()
        {
            var data = GetDenseDataset();
            var model = ML.Regression.Trainers.OnlineGradientDescent().Fit(data);
            var pfi = ML.Regression.PermutationFeatureImportance(model, data);

            // Pfi Indices:
            // X1: 0
            // X2Important: 1
            // X3: 2
            // X4Rand: 3

            // For the following metrics lower is better, so maximum delta means more important feature, and vice versa
            Assert.True(MinDeltaIndex(pfi, m => m.L1) == 3);
            Assert.True(MaxDeltaIndex(pfi, m => m.L1) == 1);

            Assert.True(MinDeltaIndex(pfi, m => m.L2) == 3);
            Assert.True(MaxDeltaIndex(pfi, m => m.L2) == 1);

            Assert.True(MinDeltaIndex(pfi, m => m.Rms) == 3);
            Assert.True(MaxDeltaIndex(pfi, m => m.Rms) == 1);

            // For the following metrics higher is better, so minimum delta means more important feature, and vice versa
            Assert.True(MaxDeltaIndex(pfi, m => m.RSquared) == 3);
            Assert.True(MinDeltaIndex(pfi, m => m.RSquared) == 1);

            Done();
        }

        /// <summary>
        /// Test PFI Regression for Sparse Features
        /// </summary>
        [Fact]
        public void TestPfiRegressionOnSparseFeatures()
        {
            var data = GetSparseDataset();
            var model = ML.Regression.Trainers.OnlineGradientDescent().Fit(data);
            var results = ML.Regression.PermutationFeatureImportance(model, data);

            // Pfi Indices:
            // X1: 0
            // X2VBuffer-Slot-0: 1
            // X2VBuffer-Slot-1: 2
            // X2VBuffer-Slot-2: 3
            // X2VBuffer-Slot-3: 4
            // X3Important: 5

            // Permuted X2VBuffer-Slot-1 lot (f2) should have min impact on SGD metrics, X3Important -- max impact.
            // For the following metrics lower is better, so maximum delta means more important feature, and vice versa
            Assert.True(MinDeltaIndex(results, m => m.L1) == 2);
            Assert.True(MaxDeltaIndex(results, m => m.L1) == 5);

            Assert.True(MinDeltaIndex(results, m => m.L2) == 2);
            Assert.True(MaxDeltaIndex(results, m => m.L2) == 5);

            Assert.True(MinDeltaIndex(results, m => m.Rms) == 2);
            Assert.True(MaxDeltaIndex(results, m => m.Rms) == 5);

            // For the following metrics higher is better, so minimum delta means more important feature, and vice versa
            Assert.True(MaxDeltaIndex(results, m => m.RSquared) == 2);
            Assert.True(MinDeltaIndex(results, m => m.RSquared) == 5);
        }

        #endregion

        #region Binary Classification Tests
        /// <summary>
        /// Test PFI Binary Classification for Dense Features
        /// </summary>
        [Fact]
        public void TestPfiBinaryClassificationOnDenseFeatures()
        {
            var data = GetDenseDataset(TaskType.BinaryClassification);
            var model = ML.BinaryClassification.Trainers.LogisticRegression().Fit(data);
            var pfi = ML.BinaryClassification.PermutationFeatureImportance(model, data);

            // Pfi Indices:
            // X1: 0
            // X2Important: 1
            // X3: 2
            // X4Rand: 3

            // For the following metrics higher is better, so minimum delta means more important feature, and vice versa
            Assert.True(MaxDeltaIndex(pfi, m => m.Auc) == 3);
            Assert.True(MinDeltaIndex(pfi, m => m.Auc) == 1);
            Assert.True(MaxDeltaIndex(pfi, m => m.Accuracy) == 3);
            Assert.True(MinDeltaIndex(pfi, m => m.Accuracy) == 1);
            Assert.True(MaxDeltaIndex(pfi, m => m.PositivePrecision) == 3);
            Assert.True(MinDeltaIndex(pfi, m => m.PositivePrecision) == 1);
            Assert.True(MaxDeltaIndex(pfi, m => m.PositiveRecall) == 3);
            Assert.True(MinDeltaIndex(pfi, m => m.PositiveRecall) == 1);
            Assert.True(MaxDeltaIndex(pfi, m => m.NegativePrecision) == 3);
            Assert.True(MinDeltaIndex(pfi, m => m.NegativePrecision) == 1);
            Assert.True(MaxDeltaIndex(pfi, m => m.NegativeRecall) == 3);
            Assert.True(MinDeltaIndex(pfi, m => m.NegativeRecall) == 1);
            Assert.True(MaxDeltaIndex(pfi, m => m.F1Score) == 3);
            Assert.True(MinDeltaIndex(pfi, m => m.F1Score) == 1);
            Assert.True(MaxDeltaIndex(pfi, m => m.Auprc) == 3);
            Assert.True(MinDeltaIndex(pfi, m => m.Auprc) == 1);

            Done();
        }

        /// <summary>
        /// Test PFI Binary Classification for Sparse Features
        /// </summary>
        [Fact]
        public void TestPfiBinaryClassificationOnSparseFeatures()
        {
            var data = GetSparseDataset(TaskType.BinaryClassification);
            var model = ML.BinaryClassification.Trainers.LogisticRegression().Fit(data);
            var pfi = ML.BinaryClassification.PermutationFeatureImportance(model, data);

            // Pfi Indices:
            // X1: 0
            // X2VBuffer-Slot-0: 1
            // X2VBuffer-Slot-1: 2
            // X2VBuffer-Slot-2: 3
            // X2VBuffer-Slot-3: 4
            // X3Important: 5

            // For the following metrics higher is better, so minimum delta means more important feature, and vice versa
            Assert.True(MaxDeltaIndex(pfi, m => m.Auc) == 2);
            Assert.True(MinDeltaIndex(pfi, m => m.Auc) == 5);
            Assert.True(MaxDeltaIndex(pfi, m => m.Accuracy) == 2);
            Assert.True(MinDeltaIndex(pfi, m => m.Accuracy) == 5);
            Assert.True(MaxDeltaIndex(pfi, m => m.PositivePrecision) == 2);
            Assert.True(MinDeltaIndex(pfi, m => m.PositivePrecision) == 5);
            Assert.True(MaxDeltaIndex(pfi, m => m.PositiveRecall) == 2);
            Assert.True(MinDeltaIndex(pfi, m => m.PositiveRecall) == 5);
            Assert.True(MaxDeltaIndex(pfi, m => m.NegativePrecision) == 2);
            Assert.True(MinDeltaIndex(pfi, m => m.NegativePrecision) == 5);
            Assert.True(MaxDeltaIndex(pfi, m => m.NegativeRecall) == 2);
            Assert.True(MinDeltaIndex(pfi, m => m.NegativeRecall) == 5);
            Assert.True(MaxDeltaIndex(pfi, m => m.F1Score) == 2);
            Assert.True(MinDeltaIndex(pfi, m => m.F1Score) == 5);
            Assert.True(MaxDeltaIndex(pfi, m => m.Auprc) == 2);
            Assert.True(MinDeltaIndex(pfi, m => m.Auprc) == 5);

            Done();
        }
        #endregion

        #region Multiclass Classification Tests
        /// <summary>
        /// Test PFI Multiclass Classification for Dense Features
        /// </summary>
        [Fact]
        public void TestPfiMulticlassClassificationOnDenseFeatures()
        {
            var data = GetDenseDataset(TaskType.MulticlassClassification);
            var model = ML.MulticlassClassification.Trainers.LogisticRegression().Fit(data);
            var pfi = ML.MulticlassClassification.PermutationFeatureImportance(model, data);

            // Pfi Indices:
            // X1: 0
            // X2Important: 1
            // X3: 2
            // X4Rand: 3

            // For the following metrics higher is better, so minimum delta means more important feature, and vice versa
            Assert.True(MaxDeltaIndex(pfi, m => m.AccuracyMicro) == 3);
            Assert.True(MinDeltaIndex(pfi, m => m.AccuracyMicro) == 1);
            Assert.True(MaxDeltaIndex(pfi, m => m.AccuracyMacro) == 3);
            Assert.True(MinDeltaIndex(pfi, m => m.AccuracyMacro) == 1);
            Assert.True(MaxDeltaIndex(pfi, m => m.LogLossReduction) == 3);
            Assert.True(MinDeltaIndex(pfi, m => m.LogLossReduction) == 1);

            // For the following metrics-delta lower is better, so maximum delta means more important feature, and vice versa
            //  Because they are _negative_, the difference will be positive for worse classifiers.
            Assert.True(MaxDeltaIndex(pfi, m => m.LogLoss) == 1);
            Assert.True(MinDeltaIndex(pfi, m => m.LogLoss) == 3);
            for (int i = 0; i < pfi[0].PerClassLogLoss.Length; i++)
            {
                Assert.True(MaxDeltaIndex(pfi, m => m.PerClassLogLoss[i]) == 1);
                Assert.True(MinDeltaIndex(pfi, m => m.PerClassLogLoss[i]) == 3);
            }

            Done();
        }

        /// <summary>
        /// Test PFI Multiclass Classification for Sparse Features
        /// </summary>
        [Fact]
        public void TestPfiMulticlassClassificationOnSparseFeatures()
        {
            var data = GetSparseDataset(TaskType.MulticlassClassification);
            var model = ML.MulticlassClassification.Trainers.LogisticRegression(advancedSettings: args => { args.MaxIterations = 1000; }).Fit(data);
            var pfi = ML.MulticlassClassification.PermutationFeatureImportance(model, data);

            // Pfi Indices:
            // X1: 0
            // X2VBuffer-Slot-0: 1
            // X2VBuffer-Slot-1: 2 // Least important
            // X2VBuffer-Slot-2: 3
            // X2VBuffer-Slot-3: 4
            // X3Important: 5 // Most important

            // For the following metrics higher is better, so minimum delta means more important feature, and vice versa
            Assert.True(MaxDeltaIndex(pfi, m => m.AccuracyMicro) == 2);
            Assert.True(MinDeltaIndex(pfi, m => m.AccuracyMicro) == 5);
            Assert.True(MaxDeltaIndex(pfi, m => m.AccuracyMacro) == 2);
            Assert.True(MinDeltaIndex(pfi, m => m.AccuracyMacro) == 5);
            Assert.True(MaxDeltaIndex(pfi, m => m.LogLossReduction) == 2);
            Assert.True(MinDeltaIndex(pfi, m => m.LogLossReduction) == 5);

            // For the following metrics-delta lower is better, so maximum delta means more important feature, and vice versa
            //  Because they are negative metrics, the _difference_ will be positive for worse classifiers.
            Assert.True(MaxDeltaIndex(pfi, m => m.LogLoss) == 5);
            Assert.True(MinDeltaIndex(pfi, m => m.LogLoss) == 2);
            for (int i = 0; i < pfi[0].PerClassLogLoss.Length; i++)
            {
                Assert.True(MaxDeltaIndex(pfi, m => m.PerClassLogLoss[i]) == 5);
                Assert.True(MinDeltaIndex(pfi, m => m.PerClassLogLoss[i]) == 2);
            }

            Done();
        }
        #endregion

        #region Ranking Tests
        /// <summary>
        /// Test PFI Multiclass Classification for Dense Features
        /// </summary>
        [Fact]
        public void TestPfiRankingOnDenseFeatures()
        {
            var data = GetDenseDataset(TaskType.Ranking);
            var model = ML.Ranking.Trainers.FastTree().Fit(data);
            var pfi = ML.Ranking.PermutationFeatureImportance(model, data);

            // Pfi Indices:
            // X1: 0 // For Ranking, this column won't result in misorderings
            // X2Important: 1
            // X3: 2
            // X4Rand: 3

            // For the following metrics higher is better, so minimum delta means more important feature, and vice versa
            for (int i = 0; i < pfi[0].Dcg.Length; i++)
            {
                Assert.True(MaxDeltaIndex(pfi, m => m.Dcg[i]) == 0);
                Assert.True(MinDeltaIndex(pfi, m => m.Dcg[i]) == 1);
            }
            for (int i = 0; i < pfi[0].Ndcg.Length; i++)
            {
                Assert.True(MaxDeltaIndex(pfi, m => m.Ndcg[i]) == 0);
                Assert.True(MinDeltaIndex(pfi, m => m.Ndcg[i]) == 1);
            }

            Done();
        }

        /// <summary>
        /// Test PFI Multiclass Classification for Sparse Features
        /// </summary>
        [Fact]
        public void TestPfiRankingOnSparseFeatures()
        {
            var data = GetSparseDataset(TaskType.Ranking);
            var model = ML.Ranking.Trainers.FastTree().Fit(data);
            var pfi = ML.Ranking.PermutationFeatureImportance(model, data);

            // Pfi Indices:
            // X1: 0
            // X2VBuffer-Slot-0: 1
            // X2VBuffer-Slot-1: 2 // Least important
            // X2VBuffer-Slot-2: 3
            // X2VBuffer-Slot-3: 4
            // X3Important: 5 // Most important

            // For the following metrics higher is better, so minimum delta means more important feature, and vice versa
            for (int i = 0; i < pfi[0].Dcg.Length; i++)
            {
                Assert.True(MaxDeltaIndex(pfi, m => m.Dcg[i]) == 2);
                Assert.True(MinDeltaIndex(pfi, m => m.Dcg[i]) == 5);
            }
            for (int i = 0; i < pfi[0].Ndcg.Length; i++)
            {
                Assert.True(MaxDeltaIndex(pfi, m => m.Ndcg[i]) == 2);
                Assert.True(MinDeltaIndex(pfi, m => m.Ndcg[i]) == 5);
            }

            Done();
        }
        #endregion

        #region Clustering Tests
        /// <summary>
        /// Test PFI Clustering for Dense Features
        /// </summary>
        [Fact]
        public void TestPfiClusteringOnDenseFeatures()
        {
            var data = GetDenseClusteringDataset();

            var preview = data.Preview();

            var model = ML.Clustering.Trainers.KMeans("Features", clustersCount: 5, 
                advancedSettings: args =>{ args.NormalizeFeatures = NormalizeOption.No;})
                .Fit(data);
            var pfi = ML.Clustering.PermutationFeatureImportance(model, data);

            // Pfi Indices:
            // X1: 0 -- middling importance for clustering (middling range)
            // X2Important: 1 -- most important for clustering (largest range)
            // X3: 2 -- Least important for clustering (smallest range)

            // For the following metrics lower is better, so maximum delta means more important feature, and vice versa
            Assert.True(MinDeltaIndex(pfi, m => m.AvgMinScore) == 0);
            Assert.True(MaxDeltaIndex(pfi, m => m.AvgMinScore) == 2);

            // For the following metrics higher is better, so minimum delta means more important feature, and vice versa
            Assert.True(MinDeltaIndex(pfi, m => m.Nmi) == 2);
            Assert.True(MaxDeltaIndex(pfi, m => m.Nmi) == 0);

            Done();
        }
        #endregion

        #region Helpers
        /// <summary>
        /// Features: x1, x2, x3, xRand; y = 10*x1 + 20x2 + 5.5x3 + e, xRand- random and Label y is to dependant on xRand.
        /// xRand has the least importance: Evaluation metrics do not change a lot when xRand is permuted.
        /// x2 has the biggest importance.
        /// </summary>
        private IDataView GetDenseDataset(TaskType task = TaskType.Regression)
        {
            Contracts.Assert(task != TaskType.Clustering, $"TaskType {nameof(TaskType.Clustering)} not supported.");

            // Setup synthetic dataset.
            const int numberOfInstances = 1000;
            var rand = new Random(10);
            float[] yArray = new float[numberOfInstances],
                x1Array = new float[numberOfInstances],
                x2Array = new float[numberOfInstances],
                x3Array = new float[numberOfInstances],
                x4RandArray = new float[numberOfInstances];

            for (var i = 0; i < numberOfInstances; i++)
            {
                var x1 = rand.Next(1000);
                x1Array[i] = x1;
                var x2Important = rand.Next(10000);
                x2Array[i] = x2Important;
                var x3 = rand.Next(5000);
                x3Array[i] = x3;
                var x4Rand = rand.Next(1000);
                x4RandArray[i] = x4Rand;

                var noise = rand.Next(50);

                yArray[i] = (float)(10 * x1 + 20 * x2Important + 5.5 * x3 + noise);
            }

            // If binary classification, modify the labels
            if (task == TaskType.BinaryClassification || 
                task == TaskType.MulticlassClassification)
                GetBinaryClassificationLabels(yArray);
            else if (task == TaskType.Ranking)
                GetRankingLabels(yArray);

            // Create data view.
            var bldr = new ArrayDataViewBuilder(Env);
            bldr.AddColumn("X1", NumberType.Float, x1Array);
            bldr.AddColumn("X2Important", NumberType.Float, x2Array);
            bldr.AddColumn("X3", NumberType.Float, x3Array);
            bldr.AddColumn("X4Rand", NumberType.Float, x4RandArray);
            bldr.AddColumn("Label", NumberType.Float, yArray);
            if (task == TaskType.Ranking)
                bldr.AddColumn("GroupId", NumberType.U4, CreateGroupIds(yArray.Length));
            var srcDV = bldr.GetDataView();

            var pipeline = ML.Transforms.Concatenate("Features", "X1", "X2Important", "X3", "X4Rand")
                .Append(ML.Transforms.Normalize("Features"));

            // Create a keytype for Ranking
            if (task == TaskType.Ranking)
                return pipeline.Append(ML.Transforms.Conversion.MapValueToKey("GroupId"))
                    .Fit(srcDV).Transform(srcDV);

            return pipeline.Fit(srcDV).Transform(srcDV);
        }

        /// <summary>
        /// Features: x1, x2vBuff(sparce vector), x3. 
        /// y = 10x1 + 10x2vBuff + 30x3 + e.
        /// Within xBuff feature  2nd slot will be sparse most of the time.
        /// 2nd slot of xBuff has the least importance: Evaluation metrics do not change a lot when this slot is permuted.
        /// x2 has the biggest importance.
        /// </summary>
        private IDataView GetSparseDataset(TaskType task = TaskType.Regression)
        {
            // Setup synthetic dataset.
            const int numberOfInstances = 10000;
            var rand = new Random(10);
            float[] yArray = new float[numberOfInstances],
                x1Array = new float[numberOfInstances],
                x3Array = new float[numberOfInstances];

            VBuffer<float>[] vbArray = new VBuffer<float>[numberOfInstances];

            for (var i = 0; i < numberOfInstances; i++)
            {
                var x1 = rand.Next(1000);
                x1Array[i] = x1;
                var x3Important = rand.Next(10000);
                x3Array[i] = x3Important;

                VBuffer<float> vb;

                if (i % 10 != 0)
                {
                    vb = new VBuffer<float>(4, 3, new float[] { rand.Next(1000), rand.Next(1000), rand.Next(1000) }, new int[] { 0, 2, 3 });
                }
                else
                {
                    vb = new VBuffer<float>(4, 4, new float[] { rand.Next(1000), rand.Next(1000), rand.Next(1000), rand.Next(1000) }, new int[] { 0, 1, 2, 3 });
                }

                vbArray[i] = vb;

                float vbSum = 0;
                foreach (var vbValue in vb.DenseValues())
                {
                    vbSum += vbValue * 10;
                }

                var noise = rand.Next(50);
                yArray[i] = 10 * x1 + vbSum + 20 * x3Important + noise;
            }

            // If binary classification, modify the labels
            if (task == TaskType.BinaryClassification ||
                task == TaskType.MulticlassClassification)
                GetBinaryClassificationLabels(yArray);
            else if (task == TaskType.Ranking)
                GetRankingLabels(yArray);

            // Create data view.
            var bldr = new ArrayDataViewBuilder(Env);
            bldr.AddColumn("X1", NumberType.Float, x1Array);
            bldr.AddColumn("X2VBuffer", NumberType.Float, vbArray);
            bldr.AddColumn("X3Important", NumberType.Float, x3Array);
            bldr.AddColumn("Label", NumberType.Float, yArray);
            if (task == TaskType.Ranking)
                bldr.AddColumn("GroupId", NumberType.U4, CreateGroupIds(yArray.Length));
            var srcDV = bldr.GetDataView();

            var pipeline = ML.Transforms.Concatenate("Features", "X1", "X2VBuffer", "X3Important")
                .Append(ML.Transforms.Normalize("Features"));

            // Create a keytype for Ranking
            if (task == TaskType.Ranking)
                return pipeline.Append(ML.Transforms.Conversion.MapValueToKey("GroupId"))
                    .Fit(srcDV).Transform(srcDV);

            return pipeline.Fit(srcDV).Transform(srcDV);
        }

        /// <summary>
        /// Features: x1, x2, x3, xRand; y = 10*x1 + 20x2 + 5.5x3 + e, xRand- random and Label y is to dependant on xRand.
        /// xRand has the least importance: Evaluation metrics do not change a lot when xRand is permuted.
        /// x2 has the biggest importance.
        /// </summary>
        private IDataView GetDenseClusteringDataset()
        {
            // Define the cluster centers
            const int clusterCount = 5;
            float[][] clusterCenters = new float[clusterCount][];
            for (int i = 0; i < clusterCount; i++)
            {
                clusterCenters[i] = new float[3] { i, i, i };
            }

            // Create rows of data sampled from these clusters
            const int numberOfInstances = 1000;
            var rand = new Random(10);

            // The cluster spacing is 1
            float x1Scale = 0.01f;
            float x2Scale = 0.1f;
            float x3Scale = 1f;

            float[] yArray = new float[numberOfInstances];
            float[] x1Array = new float[numberOfInstances];
            float[] x2Array = new float[numberOfInstances];
            float[] x3Array = new float[numberOfInstances];

            for (var i = 0; i < numberOfInstances; i++)
            {
                // Assign a cluster
                var clusterLabel = rand.Next(clusterCount);
                yArray[i] = clusterLabel;

                x1Array[i] = clusterCenters[clusterLabel][0] + x1Scale * (float)(rand.NextDouble() - 0.5);
                x2Array[i] = clusterCenters[clusterLabel][1] + x2Scale * (float)(rand.NextDouble() - 0.5);
                x3Array[i] = clusterCenters[clusterLabel][2] + x3Scale * (float)(rand.NextDouble() - 0.5);
            }

            // Create data view.
            var bldr = new ArrayDataViewBuilder(Env);
            bldr.AddColumn("Label", NumberType.Float, yArray);
            bldr.AddColumn("X1", NumberType.Float, x1Array);
            bldr.AddColumn("X2", NumberType.Float, x2Array);
            bldr.AddColumn("X3", NumberType.Float, x3Array);
            var srcDV = bldr.GetDataView();

            var pipeline = ML.Transforms.Concatenate("Features", "X1", "X2", "X3");

            return pipeline.Fit(srcDV).Transform(srcDV);
        }

        private int MinDeltaIndex<T>(
            ImmutableArray<T> metricsDelta,
            Func<T, double> metricSelector)
        {
            var min = metricsDelta.OrderBy(m => metricSelector(m)).First();
            return metricsDelta.IndexOf(min);
        }

        private int MaxDeltaIndex<T>(
            ImmutableArray<T> metricsDelta,
            Func<T, double> metricSelector)
        {
            var max = metricsDelta.OrderByDescending(m => metricSelector(m)).First();
            return metricsDelta.IndexOf(max);
        }

        private void GetBinaryClassificationLabels(float[] rawScores)
        {
            float averageScore = GetArrayAverage(rawScores);

            // Center the response and then take the sigmoid to generate the classes
            for (int i = 0; i < rawScores.Length; i++)
                rawScores[i] = MathUtils.Sigmoid(rawScores[i] - averageScore) > 0.5 ? 1 : 0;
        }

        private void GetRankingLabels(float[] rawScores)
        {
            var min = MathUtils.Min(rawScores);
            var max = MathUtils.Max(rawScores);

            for (int i = 0; i < rawScores.Length; i++)
            {
                // Bin from [zero,one), then expand out to [0,5) and truncate
                rawScores[i] = (int)(5 * (rawScores[i] - min) / (max - min));
                if (rawScores[i] == 5)
                    rawScores[i] = 4;
            }
        }

        private float GetArrayAverage(float[] scores)
        {
            // Compute the average so we can center the response
            float averageScore = 0.0f;
            for (int i = 0; i < scores.Length; i++)
                averageScore += scores[i];
            averageScore /= scores.Length;

            return averageScore;
        }

        private uint[] CreateGroupIds(int numRows, int rowsPerGroup = 5)
        {
            var groupIds = new uint[numRows];
            // Construct groups of rowsPerGroup using a modulo counter
            uint group = 0;
            for (int i = 0; i < groupIds.Length; i++)
            {
                if (i % rowsPerGroup == 0)
                    group++;
                groupIds[i] = group;
            }

            return groupIds;
        }

        private enum TaskType
        {
            Regression,
            BinaryClassification,
            MulticlassClassification,
            Ranking,
            Clustering
        }
        #endregion
    }
}
