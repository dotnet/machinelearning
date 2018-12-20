// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.RunTests;
using System;
using Xunit;
using Xunit.Abstractions;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML.Transforms;
using System.IO;

namespace Microsoft.ML.Tests
{
    public sealed class FeatureContributionTests : TestDataPipeBase
    {
        public FeatureContributionTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void FeatureContributionEstimatorWorkout()
        {
            var data = GetSparseDataset();
            var model = ML.Regression.Trainers.OrdinaryLeastSquares().Fit(data);

            var estPipe = new FeatureContributionCalculatingEstimator(ML, model.Model, model.FeatureColumn)
                .Append(new FeatureContributionCalculatingEstimator(ML, model.Model, model.FeatureColumn, normalize: false))
                .Append(new FeatureContributionCalculatingEstimator(ML, model.Model, model.FeatureColumn, top: 0))
                .Append(new FeatureContributionCalculatingEstimator(ML, model.Model, model.FeatureColumn, bottom: 0))
                .Append(new FeatureContributionCalculatingEstimator(ML, model.Model, model.FeatureColumn, top: 0, bottom: 0));

            TestEstimatorCore(estPipe, data);
            Done();
        }

        // Tests for regression trainers that implement IFeatureContributionMapper interface.
        [Fact]
        public void TestOrdinaryLeastSquaresRegression()
        {
            TestFeatureContribution(ML.Regression.Trainers.OrdinaryLeastSquares(), GetSparseDataset(numberOfInstances: 100), "LeastSquaresRegression");
        }

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // LightGBM is 64-bit only
        public void TestLightGbmRegression()
        {
            TestFeatureContribution(ML.Regression.Trainers.LightGbm(), GetSparseDataset(numberOfInstances: 100), "LightGbmRegression");
        }

        [Fact]
        public void TestFastTreeRegression()
        {
            TestFeatureContribution(ML.Regression.Trainers.FastTree(), GetSparseDataset(numberOfInstances: 100), "FastTreeRegression");
        }

        [Fact]
        public void TestFastForestRegression()
        {
            TestFeatureContribution(ML.Regression.Trainers.FastForest(), GetSparseDataset(numberOfInstances: 100), "FastForestRegression");
        }

        [Fact]
        public void TestFastTreeTweedieRegression()
        {
            TestFeatureContribution(ML.Regression.Trainers.FastTreeTweedie(), GetSparseDataset(numberOfInstances: 100), "FastTreeTweedieRegression");
        }

        [Fact]
        public void TestSDCARegression()
        {
            TestFeatureContribution(ML.Regression.Trainers.StochasticDualCoordinateAscent(advancedSettings: args => { args.NumThreads = 1; }), GetSparseDataset(numberOfInstances: 100), "SDCARegression");
        }

        [Fact]
        public void TestOnlineGradientDescentRegression()
        {
            TestFeatureContribution(ML.Regression.Trainers.OnlineGradientDescent(), GetSparseDataset(numberOfInstances: 100), "OnlineGradientDescentRegression");
        }

        [Fact]
        public void TestPoissonRegression()
        {
            TestFeatureContribution(ML.Regression.Trainers.PoissonRegression(advancedSettings: args => { args.NumThreads = 1; }), GetSparseDataset(numberOfInstances: 100), "PoissonRegression");
        }

        [Fact]
        public void TestGAMRegression()
        {
            TestFeatureContribution(ML.Regression.Trainers.GeneralizedAdditiveModels(), GetSparseDataset(numberOfInstances: 100), "GAMRegression");
        }

        // Tests for ranking trainers that implement IFeatureContributionMapper interface.
        [Fact]
        public void TestFastTreeRanking()
        {
            TestFeatureContribution(ML.Ranking.Trainers.FastTree(), GetSparseDataset(TaskType.Ranking, 100), "FastTreeRanking");
        }

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // LightGBM is 64-bit only
        public void TestLightGbmRanking()
        {
            TestFeatureContribution(ML.Ranking.Trainers.LightGbm(), GetSparseDataset(TaskType.Ranking, 100), "LightGbmRanking");
        }
        
        // Tests for binary classification trainers that implement IFeatureContributionMapper interface.
        [Fact]
        public void TestAveragePerceptronBinary()
        {
            TestFeatureContribution(ML.BinaryClassification.Trainers.AveragedPerceptron(), GetSparseDataset(TaskType.BinaryClassification, 100), "AveragePerceptronBinary");
        }

        [Fact]
        public void TestSVMBinary()
        {
            TestFeatureContribution(ML.BinaryClassification.Trainers.LinearSupportVectorMachines(), GetSparseDataset(TaskType.BinaryClassification, 100), "SVMBinary");
        }

        [Fact]
        public void TestLogisticRegressionBinary()
        {
            TestFeatureContribution(ML.BinaryClassification.Trainers.LogisticRegression(), GetSparseDataset(TaskType.BinaryClassification, 100), "LogisticRegressionBinary");
        }

        [Fact]
        public void TestFastForestBinary()
        {
            TestFeatureContribution(ML.BinaryClassification.Trainers.FastForest(), GetSparseDataset(TaskType.BinaryClassification, 100), "FastForestBinary");
        }

        [Fact]
        public void TestFastTreeBinary()
        {
            TestFeatureContribution(ML.BinaryClassification.Trainers.FastTree(), GetSparseDataset(TaskType.BinaryClassification, 100), "FastTreeBinary");
        }

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // LightGBM is 64-bit only
        public void TestLightGbmBinary()
        {
            TestFeatureContribution(ML.BinaryClassification.Trainers.LightGbm(), GetSparseDataset(TaskType.BinaryClassification, 100), "LightGbmBinary");
        }

        [Fact]
        public void TestSDCABinary()
        {
            TestFeatureContribution(ML.BinaryClassification.Trainers.StochasticDualCoordinateAscent(advancedSettings: args => { args.NumThreads = 1; }), GetSparseDataset(TaskType.BinaryClassification, 100), "SDCABinary");
        }

        [Fact]
        public void TestSGDBinary()
        {
            TestFeatureContribution(ML.BinaryClassification.Trainers.StochasticGradientDescent(advancedSettings: args => { args.NumThreads = 1; }), GetSparseDataset(TaskType.BinaryClassification, 100), "SGDBinary");
        }

        [Fact]
        public void TestSSGDBinary()
        {
            TestFeatureContribution(ML.BinaryClassification.Trainers.SymbolicStochasticGradientDescent(), GetSparseDataset(TaskType.BinaryClassification, 100), "SSGDBinary", 4);
        }

        [Fact]
        public void TestGAMBinary()
        {
            TestFeatureContribution(ML.BinaryClassification.Trainers.GeneralizedAdditiveModels(), GetSparseDataset(TaskType.BinaryClassification, 100), "GAMBinary");
        }

        private void TestFeatureContribution(
            ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictor>, IPredictor> trainer,
            IDataView data,
            string testFile,
            int precision = 6)
        {
            // Train the model.
                var model = trainer.Fit(data);

            // Extract the predictor, check that it supports feature contribution.
            var predictor = model.Model as ICalculateFeatureContribution;
            Assert.NotNull(predictor);

            // Calculate feature contributions.
            var est = new FeatureContributionCalculatingEstimator(ML, predictor, "Features", top: 3, bottom: 0)
                .Append(new FeatureContributionCalculatingEstimator(ML, predictor, "Features", top: 0, bottom: 3))
                .Append(new FeatureContributionCalculatingEstimator(ML, predictor, "Features", top: 1, bottom: 1))
                .Append(new FeatureContributionCalculatingEstimator(ML, predictor, "Features", top: 1, bottom: 1, normalize: false));

            TestEstimatorCore(est, data);
            // Verify output.
            var outputPath = GetOutputPath("FeatureContribution", testFile + ".tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(ML, new TextSaver.Arguments { Silent = true, OutputHeader = false });
                IDataView savedData = TakeFilter.Create(ML, est.Fit(data).Transform(data), 4);
                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, savedData, fs, keepHidden: true);
            }
            CheckEquality("FeatureContribution", testFile + ".tsv", digitsOfPrecision: precision);
            Done();
        }

        /// <summary>
        /// Features: x1, x2vBuff(sparce vector), x3. 
        /// y = 10x1 + 10x2vBuff + 30x3 + e.
        /// Within xBuff feature  2nd slot will be sparse most of the time.
        /// 2nd slot of xBuff has the least importance: Evaluation metrics do not change a lot when this slot is permuted.
        /// x3 has the biggest importance.
        /// </summary>
        private IDataView GetSparseDataset(TaskType task = TaskType.Regression, int numberOfInstances = 1000)
        {
            // Setup synthetic dataset.
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
    }
}
