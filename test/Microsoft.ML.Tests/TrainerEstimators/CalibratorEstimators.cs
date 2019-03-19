// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

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
            TestEstimatorCore(calibratorTestData.Pipeline, calibratorTestData.Data);

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
            TestEstimatorCore(calibratorTestData.Pipeline, calibratorTestData.Data);

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
            TestEstimatorCore(calibratorTestData.Pipeline, calibratorTestData.Data);

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
            Assert.True(scoredDataPreview.ColumnView.Length == 5);

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


        private void CheckValidCalibratedData(IDataView scoredData, ITransformer transformer){

            var calibratedData = transformer.Transform(scoredData).Preview();

            Assert.True(calibratedData.ColumnView.Length == 6);

            for (int i = 0; i < 10; i++)
            {
                var probability = calibratedData.RowView[i].Values[5];
                Assert.InRange((float)probability.Value, 0, 1);
            }
        }

    }
}
