// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

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

        /// <summary>
        /// Features: x1, x2, x3, xRand; y = 10*x1 + 20x2 + 5.5x3 + e, xRand- random and Label y is to dependant on xRand.
        /// Test verifies that xRand has the least importance: L1, L2, RMS and Loss-Fn do not change a lot when xRand is permuted.
        /// Also test checks that x2 has the biggest importance.
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
        /// Features: x1, x2vBuff(sparce vector), x3. 
        /// y = 10x1 + 10x2vBuff + 30x3 + e.
        /// Within xBuff feature  2nd slot will be sparse most of the time.
        /// Test verifies that 2nd slot of xBuff has the least importance: L1, L2, RMS and Loss-Fn do not change a lot when this slot is permuted.
        /// Also test checks that x2 has the biggest importance.
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
        /// Features: x1, x2vBuff(sparce vector), x3. 
        /// y = 10x1 + 10x2vBuff + 30x3 + e.
        /// Within xBuff feature  2nd slot will be sparse most of the time.
        /// Test verifies that 2nd slot of xBuff has the least importance: L1, L2, RMS and Loss-Fn do not change a lot when this slot is permuted.
        /// Also test checks that x2 has the biggest importance.
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

        private IDataView GetDenseDataset(TaskType task = TaskType.Regression)
        {
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
            if (task == TaskType.BinaryClassification)
                GetBinaryClassificationScores(yArray);

            // Create data view.
            var bldr = new ArrayDataViewBuilder(Env);
            bldr.AddColumn("X1", NumberType.Float, x1Array);
            bldr.AddColumn("X2Important", NumberType.Float, x2Array);
            bldr.AddColumn("X3", NumberType.Float, x3Array);
            bldr.AddColumn("X4Rand", NumberType.Float, x4RandArray);
            bldr.AddColumn("Label", NumberType.Float, yArray);
            var srcDV = bldr.GetDataView();

            var pipeline = ML.Transforms.Concatenate("Features", "X1", "X2Important", "X3", "X4Rand")
                .Append(ML.Transforms.Normalize("Features"));
            var data = pipeline.Fit(srcDV).Transform(srcDV);

            return data;
        }

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
            if (task == TaskType.BinaryClassification)
                GetBinaryClassificationScores(yArray);

            // Create data view.
            var bldr = new ArrayDataViewBuilder(Env);
            bldr.AddColumn("X1", NumberType.Float, x1Array);
            bldr.AddColumn("X2VBuffer", NumberType.Float, vbArray);
            bldr.AddColumn("X3Important", NumberType.Float, x3Array);
            bldr.AddColumn("Label", NumberType.Float, yArray);
            var srcDV = bldr.GetDataView();

            var pipeline = ML.Transforms.Concatenate("Features", "X1", "X2VBuffer", "X3Important")
                .Append(ML.Transforms.Normalize("Features"));
            var data = pipeline.Fit(srcDV).Transform(srcDV);

            return data;
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

        private void GetBinaryClassificationScores(float[] rawScores)
        {
            // Compute the average so we can center the response
            float averageScore = 0.0f;
            for (int i = 0; i < rawScores.Length; i++)
                averageScore += rawScores[i];
            averageScore /= rawScores.Length;

            // Center the response and then take the sigmoid to generate the classes
            for (int i = 0; i < rawScores.Length; i++)
                rawScores[i] = MathUtils.Sigmoid(rawScores[i] - averageScore) > 0.5 ? 1 : 0;
        }

        private enum TaskType
        {
            Regression,
            BinaryClassification
        }
    }
}
