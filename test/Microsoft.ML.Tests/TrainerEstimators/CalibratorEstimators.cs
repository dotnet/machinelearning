// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Data.DataView;
using Microsoft.ML.Calibrator;
using Microsoft.ML.Core.Data;
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

            // platCalibrator
            var platCalibratorEstimator = new PlattCalibratorEstimator(Env, calibratorTestData.transformer.Model, "Label", "Features");
            var platCalibratorTransformer = platCalibratorEstimator.Fit(calibratorTestData.scoredData);

            //testData
            checkValidCalibratedData(calibratorTestData.scoredData, platCalibratorTransformer);

            //test estimator
            TestEstimatorCore(platCalibratorEstimator, calibratorTestData.scoredData);

            Done();
        }

        /// <summary>
        /// OVA and calibrators
        /// </summary>
        [Fact]
        public void FixedPlatCalibratorEstimator()
        {
            var calibratorTestData = GetCalibratorTestData();

            // fixedPlatCalibrator
            var fixedPlatCalibratorEstimator = new FixedPlattCalibratorEstimator(Env, calibratorTestData.transformer.Model, labelColumn: "Label", featureColumn: "Features");
            var fixedPlatCalibratorTransformer = fixedPlatCalibratorEstimator.Fit(calibratorTestData.scoredData);

            checkValidCalibratedData(calibratorTestData.scoredData, fixedPlatCalibratorTransformer);

            //test estimator
            TestEstimatorCore(calibratorTestData.pipeline, calibratorTestData.data);

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
            var naiveCalibratorEstimator = new NaiveCalibratorEstimator(Env, calibratorTestData.transformer.Model, "Label", "Features");
            var naiveCalibratorTransformer = naiveCalibratorEstimator.Fit(calibratorTestData.scoredData);

            // check data
            checkValidCalibratedData(calibratorTestData.scoredData, naiveCalibratorTransformer);

            //test estimator
            TestEstimatorCore(calibratorTestData.pipeline, calibratorTestData.data);

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
            var pavCalibratorEstimator = new PavCalibratorEstimator(Env, calibratorTestData.transformer.Model, "Label", "Features");
            var pavCalibratorTransformer = pavCalibratorEstimator.Fit(calibratorTestData.scoredData);

            //check data
            checkValidCalibratedData(calibratorTestData.scoredData, pavCalibratorTransformer);

            //test estimator
            TestEstimatorCore(calibratorTestData.pipeline, calibratorTestData.data);

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
                data = data,
                scoredData = scoredData,
                pipeline = pipeline,
                transformer = ((TransformerChain<BinaryPredictionTransformer<LinearBinaryModelParameters>>)transformer).LastTransformer as BinaryPredictionTransformer<LinearBinaryModelParameters>,
            };
        }

        private class CalibratorTestData
        {
            internal IDataView data { get; set; }
            internal IDataView scoredData { get; set; }
            internal IEstimator<ITransformer> pipeline { get; set; }

            internal BinaryPredictionTransformer<LinearBinaryModelParameters> transformer { get; set; }
        }


        void checkValidCalibratedData (IDataView scoredData, ITransformer transformer){

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
