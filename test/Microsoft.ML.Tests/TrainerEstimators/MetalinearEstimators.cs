// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;
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
                calibrator: calibrator, maximumCalibrationExampleCount: 10000, useProbabilities: true);

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
        /// Test what OVA preserves key values for label.
        /// </summary>
        [Fact]
        public void OvaKeyNames()
        {
            var textLoaderOptions = new TextLoader.Options()
            {
                Columns = new[]
                {   new TextLoader.Column("Label", DataKind.Single, 0),
                    new TextLoader.Column("Row", DataKind.Single, 1),
                    new TextLoader.Column("Column", DataKind.Single, 2),
                },
                HasHeader = true,
                Separators = new[] { '\t' }
            };
            var textLoader = ML.Data.CreateTextLoader(textLoaderOptions);
            var data = textLoader.Load(GetDataPath(TestDatasets.trivialMatrixFactorization.trainFilename));

            var ap = ML.BinaryClassification.Trainers.AveragedPerceptron();
            var ova = ML.MulticlassClassification.Trainers.OneVersusAll(ap);

            var pipeline = ML.Transforms.Conversion.MapValueToKey("Label")
                .Append(ML.Transforms.Concatenate("Features", "Row", "Column"))
                .Append(ova)
                .Append(ML.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            var model = pipeline.Fit(data);
            var result = model.Transform(data);
            Assert.NotNull(result.Schema["Score"].Annotations.Schema.GetColumnOrNull("TrainingLabelValues"));
            Assert.Equal(new[] { 1.0f, 3.0f, 2.0f }, result.GetColumn<float>("PredictedLabel").Distinct());
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
                new SdcaNonCalibratedBinaryTrainer.Options
                {
                    LabelColumnName = "Label",
                    FeatureColumnName = "Vars",
                    MaximumNumberOfIterations = 100,
                    Shuffle = true,
                    NumberOfThreads = 1,
                });

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
