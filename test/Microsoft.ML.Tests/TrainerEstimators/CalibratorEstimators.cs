// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Xunit;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators
    {
        /// <summary>
        /// OVA and calibrators
        /// </summary>
        [Fact]
        public void PlattCalibratorEstimator()
        {
            var calibratorTestData = GetCalibratorTestData();

            // plattCalibrator
            var plattCalibratorEstimator = new PlattCalibratorEstimator(Env);
            var plattCalibratorTransformer = plattCalibratorEstimator.Fit(calibratorTestData.ScoredData);

            //testData
            CheckValidCalibratedData(calibratorTestData.ScoredData, plattCalibratorTransformer);

            //test estimator
            TestEstimatorCore(plattCalibratorEstimator, calibratorTestData.ScoredData);

            Done();
        }

        /// <summary>
        /// OVA and calibrators
        /// </summary>
        [Fact]
        public void FixedPlattCalibratorEstimator()
        {
            var calibratorTestData = GetCalibratorTestData();

            // fixedPlattCalibrator
            var fixedPlattCalibratorEstimator = new FixedPlattCalibratorEstimator(Env);
            var fixedPlattCalibratorTransformer = fixedPlattCalibratorEstimator.Fit(calibratorTestData.ScoredData);

            CheckValidCalibratedData(calibratorTestData.ScoredData, fixedPlattCalibratorTransformer);

            //test estimator
            TestEstimatorCore(fixedPlattCalibratorEstimator, calibratorTestData.ScoredData);

            Done();
        }

        /// <summary>
        /// OVA and calibrators
        /// </summary>
        [Fact]
        public void NaiveCalibratorEstimator()
        {
            var calibratorTestData = GetCalibratorTestData();

            // naive calibrator
            var naiveCalibratorEstimator = new NaiveCalibratorEstimator(Env);
            var naiveCalibratorTransformer = naiveCalibratorEstimator.Fit(calibratorTestData.ScoredData);

            // check data
            CheckValidCalibratedData(calibratorTestData.ScoredData, naiveCalibratorTransformer);

            //test estimator
            TestEstimatorCore(naiveCalibratorEstimator, calibratorTestData.ScoredData);

            Done();
        }
        /// <summary>
        /// OVA and calibrators
        /// </summary>
        [Fact]
        public void PavCalibratorEstimator()
        {
            var calibratorTestData = GetCalibratorTestData();

            // pav calibrator
            var pavCalibratorEstimator = new IsotonicCalibratorEstimator(Env);
            var pavCalibratorTransformer = pavCalibratorEstimator.Fit(calibratorTestData.ScoredData);

            //check data
            CheckValidCalibratedData(calibratorTestData.ScoredData, pavCalibratorTransformer);

            //test estimator
            TestEstimatorCore(pavCalibratorEstimator, calibratorTestData.ScoredData);

            Done();
        }

        CalibratorTestData GetCalibratorTestData()
        {
            var (pipeline, data) = GetBinaryClassificationPipeline();
            var binaryTrainer = ML.BinaryClassification.Trainers.AveragedPerceptron();

            pipeline = pipeline.Append(binaryTrainer);

            var transformer = pipeline.Fit(data);
            var scoredData = transformer.Transform(data);
            var scoredDataPreview = scoredData.Preview();
            Assert.True(scoredDataPreview.ColumnView.Length == 6);

            return new CalibratorTestData
            {
                Data = data,
                ScoredData = scoredData,
                Pipeline = pipeline,
                Transformer = ((TransformerChain<BinaryPredictionTransformer<LinearBinaryModelParameters>>)transformer).LastTransformer as BinaryPredictionTransformer<LinearBinaryModelParameters>,
            };
        }

        private sealed class CalibratorTestData
        {
            public IDataView Data { get; set; }
            public IDataView ScoredData { get; set; }
            public IEstimator<ITransformer> Pipeline { get; set; }

            public BinaryPredictionTransformer<LinearBinaryModelParameters> Transformer { get; set; }
        }


        private void CheckValidCalibratedData(IDataView scoredData, ITransformer transformer)
        {

            var calibratedData = transformer.Transform(scoredData).Preview();

            Assert.True(calibratedData.ColumnView.Length == 7);

            for (int i = 0; i < 10; i++)
            {
                var probability = calibratedData.RowView[i].Values[6];
                Assert.InRange((float)probability.Value, 0, 1);
            }
        }

        /// <summary>
        /// Test to confirm calibrator estimators work with classes
        /// where order of label and score columns are reversed, and
        /// where name of score column is different than the default.
        /// </summary>
        [Fact]
        public void TestNonStandardCalibratorEstimatorClasses()
        {
            var mlContext = new MLContext(0);
            // Store different possible variations of calibrator data classes.
            IDataView[] dataArray = new IDataView[]
            {
                mlContext.Data.LoadFromEnumerable<CalibratorTestInputReversedOrder>(
                    new CalibratorTestInputReversedOrder[]
                    {
                        new CalibratorTestInputReversedOrder { Score = 10, Label = true },
                        new CalibratorTestInputReversedOrder { Score = 15, Label = false }
                    }),
                mlContext.Data.LoadFromEnumerable<CalibratorTestInputUniqueScoreColumnName>(
                    new CalibratorTestInputUniqueScoreColumnName[]
                    {
                        new CalibratorTestInputUniqueScoreColumnName { Label = true, ScoreX = 10 },
                        new CalibratorTestInputUniqueScoreColumnName { Label = false, ScoreX = 15 }
                    }),
                mlContext.Data.LoadFromEnumerable<CalibratorTestInputReversedOrderAndUniqueScoreColumnName>(
                    new CalibratorTestInputReversedOrderAndUniqueScoreColumnName[]
                    {
                        new CalibratorTestInputReversedOrderAndUniqueScoreColumnName { ScoreX = 10, Label = true },
                        new CalibratorTestInputReversedOrderAndUniqueScoreColumnName { ScoreX = 15, Label = false }
                    })
            };

            // When label and/or score columns are different from their default names ("Label" and "Score", respectively), they
            // need to be manually defined as done below.
            // Successful training of estimators and transforming with transformers indicate correct label and score columns
            // have been found.
            for (int i = 0; i < dataArray.Length; i++)
            {
                // Test PlattCalibratorEstimator
                var calibratorPlattEstimator = new PlattCalibratorEstimator(Env,
                    scoreColumnName: i > 0 ? "ScoreX" : DefaultColumnNames.Score);
                var calibratorPlattTransformer = calibratorPlattEstimator.Fit(dataArray[i]);
                calibratorPlattTransformer.Transform(dataArray[i]);

                // Test FixedPlattCalibratorEstimator
                var calibratorFixedPlattEstimator = new FixedPlattCalibratorEstimator(Env,
                    scoreColumn: i > 0 ? "ScoreX" : DefaultColumnNames.Score);
                var calibratorFixedPlattTransformer = calibratorFixedPlattEstimator.Fit(dataArray[i]);
                calibratorFixedPlattTransformer.Transform(dataArray[i]);

                // Test NaiveCalibratorEstimator
                var calibratorNaiveEstimator = new NaiveCalibratorEstimator(Env,
                    scoreColumn: i > 0 ? "ScoreX" : DefaultColumnNames.Score);
                var calibratorNaiveTransformer = calibratorNaiveEstimator.Fit(dataArray[i]);
                calibratorNaiveTransformer.Transform(dataArray[i]);

                // Test IsotonicCalibratorEstimator
                var calibratorIsotonicEstimator = new IsotonicCalibratorEstimator(Env,
                    scoreColumn: i > 0 ? "ScoreX" : DefaultColumnNames.Score);
                var calibratorIsotonicTransformer = calibratorIsotonicEstimator.Fit(dataArray[i]);
                calibratorIsotonicTransformer.Transform(dataArray[i]);
            }
        }

        /// <summary>
        /// Test class where the column order of the label and score
        /// columns are reversed (by default, label column is before
        /// that of score column).
        /// </summary>
        private sealed class CalibratorTestInputReversedOrder
        {
            public float Score { get; set; }
            public bool Label { get; set; }
        }

        /// <summary>
        /// Test class where name of score column is different than
        /// the default column name of "Score".
        /// </summary>
        private sealed class CalibratorTestInputUniqueScoreColumnName
        {
            public bool Label { get; set; }
            public float ScoreX { get; set; }
        }

        /// <summary>
        /// Test class where the column order of the label and score
        /// columns are reversed (by default, label column is before
        /// that of score column), and where name of score column is
        /// different than the default column name of "Score".
        /// </summary>
        private sealed class CalibratorTestInputReversedOrderAndUniqueScoreColumnName
        {
            public float ScoreX { get; set; }
            public bool Label { get; set; }
        }

        /// <summary>
        /// Test to check backwards compatibility of calibrator estimators
        /// trained before the current version of VerWritten: 0x00010001.
        /// </summary>
        [Fact]
        public void TestCalibratorEstimatorBackwardsCompatibility()
        {
            // The legacy model being loaded below was trained and saved with
            // version as such:
            /* 
             * var mlContext = new MLContext(seed: 1);
             * var calibratorTestData = GetCalibratorTestData();
             * var plattCalibratorEstimator = new PlattCalibratorEstimator(Env);
             * var plattCalibratorTransformer = plattCalibratorEstimator.Fit(calibratorTestData.ScoredData);
             * mlContext.Model.Save(plattCalibratorTransformer, calibratorTestData.ScoredData.Schema, "calibrator-model_VerWritten_0x00010001xyz.zip");
             */

            var modelPath = GetDataPath("backcompat", "Calibrator_Model_VerWritten_0x00010001.zip");
            ITransformer oldPlattCalibratorTransformer;
            using (var fs = File.OpenRead(modelPath))
                oldPlattCalibratorTransformer = ML.Model.Load(fs, out var schema);

            var calibratorTestData = GetCalibratorTestData();
            var newPlattCalibratorEstimator = new PlattCalibratorEstimator(Env);
            var newPlattCalibratorTransformer = newPlattCalibratorEstimator.Fit(calibratorTestData.ScoredData);

            // Check that both models produce the same output
            var oldCalibratedData = oldPlattCalibratorTransformer.Transform(calibratorTestData.ScoredData).Preview();
            var newCalibratedData = newPlattCalibratorTransformer.Transform(calibratorTestData.ScoredData).Preview();

            // Check first that the produced schemas and outputs are of the same size
            Assert.True(oldCalibratedData.RowView.Length == newCalibratedData.RowView.Length);
            Assert.True(oldCalibratedData.ColumnView.Length == newCalibratedData.ColumnView.Length);

            // Then check the produced probabilities (5th value corresponds to probabilities) for
            // equality, within rounding error.
            for (int i = 0; i < 10; i++)
                Assert.True((float)oldCalibratedData.RowView[i].Values[5].Value == (float)newCalibratedData.RowView[i].Values[5].Value);

            Done();
        }
    }
}
