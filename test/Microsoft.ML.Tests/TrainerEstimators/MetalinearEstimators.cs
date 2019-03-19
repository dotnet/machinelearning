// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Xunit;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators
    {
        /// <summary>
        /// OVA with all constructor args.
        /// </summary>
        [Fact]
        public void OVAWithAllConstructorArgs()
        {
            var (pipeline, data) = GetMulticlassPipeline();
            var calibrator = new PlattCalibratorEstimator(Env);
            var averagePerceptron = ML.BinaryClassification.Trainers.AveragedPerceptron(
                new AveragedPerceptronTrainer.Options { Shuffle = true });

            var ova = ML.MulticlassClassification.Trainers.OneVersusAll(averagePerceptron, imputeMissingLabelsAsNegative: true,
                calibrator: calibrator, maxCalibrationExamples: 10000, useProbabilities: true);

            pipeline = pipeline.Append(ova)
                    .Append(new KeyToValueMappingEstimator(Env, "PredictedLabel"));

            TestEstimatorCore(pipeline, data);
            Done();
        }

        /// <summary>
        /// OVA un-calibrated
        /// </summary>
        [Fact]
        public void OVAUncalibrated()
        {
            var (pipeline, data) = GetMulticlassPipeline();
            var sdcaTrainer = ML.BinaryClassification.Trainers.SdcaNonCalibrated(
                new SdcaNonCalibratedBinaryTrainer.Options { MaximumNumberOfIterations = 100, Shuffle = true, NumberOfThreads = 1 });

            pipeline = pipeline.Append(ML.MulticlassClassification.Trainers.OneVersusAll(sdcaTrainer, useProbabilities: false))
                    .Append(new KeyToValueMappingEstimator(Env, "PredictedLabel"));

            TestEstimatorCore(pipeline, data);
            Done();
        }

        /// <summary>
        /// Pairwise Coupling trainer
        /// </summary>
        [Fact]
        public void PairwiseCouplingTrainer()
        {
            var (pipeline, data) = GetMulticlassPipeline();

            var sdcaTrainer = ML.BinaryClassification.Trainers.SdcaNonCalibrated(
                new SdcaNonCalibratedBinaryTrainer.Options { MaximumNumberOfIterations = 100, Shuffle = true, NumberOfThreads = 1 });

            pipeline = pipeline.Append(ML.MulticlassClassification.Trainers.PairwiseCoupling(sdcaTrainer))
                    .Append(ML.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            TestEstimatorCore(pipeline, data);
            Done();
        }

        [Fact]
        public void MetacomponentsFeaturesRenamed()
        {
            // Create text loader.
            var options = new TextLoader.Options()
            {
                Columns = TestDatasets.irisData.GetLoaderColumns(),
                Separators = new[] { ',' },
            };
            var loader = new TextLoader(Env, options: options);

            var data = loader.Load(GetDataPath(TestDatasets.irisData.trainFilename));

            var sdcaTrainer = ML.BinaryClassification.Trainers.SdcaNonCalibrated(
                new SdcaNonCalibratedBinaryTrainer.Options {
                    LabelColumnName = "Label",
                    FeatureColumnName = "Vars",
                    MaximumNumberOfIterations = 100,
                    Shuffle = true,
                    NumberOfThreads = 1, });

            var pipeline = new ColumnConcatenatingEstimator(Env, "Vars", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                .Append(new ValueToKeyMappingEstimator(Env, "Label"), TransformerScope.TrainTest)
                .Append(ML.MulticlassClassification.Trainers.OneVersusAll(sdcaTrainer))
                .Append(new KeyToValueMappingEstimator(Env, "PredictedLabel"));

            var model = pipeline.Fit(data);

            TestEstimatorCore(pipeline, data);
            Done();
        }
    }
}
