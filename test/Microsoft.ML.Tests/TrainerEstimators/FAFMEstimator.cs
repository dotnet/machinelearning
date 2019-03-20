// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.SamplesUtils;
using Microsoft.ML.Trainers;
using Xunit;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators : TestDataPipeBase
    {
        [Fact]
        public void FfmBinaryClassificationWithoutArguments()
        {
            var mlContext = new MLContext(seed: 0);
            var data = DatasetUtils.GenerateFfmSamples(500);
            var dataView = mlContext.Data.LoadFromEnumerable(data);

            var pipeline = mlContext.Transforms.CopyColumns(DefaultColumnNames.Features, nameof(DatasetUtils.FfmExample.Field0))
                .Append(mlContext.BinaryClassification.Trainers.FieldAwareFactorizationMachine());

            var model = pipeline.Fit(dataView);
            var prediction = model.Transform(dataView);

            var metrics = mlContext.BinaryClassification.Evaluate(prediction);

            // Run a sanity check against a few of the metrics.
            Assert.InRange(metrics.Accuracy, 0.6, 1);
            Assert.InRange(metrics.AreaUnderRocCurve, 0.7, 1);
            Assert.InRange(metrics.AreaUnderPrecisionRecallCurve, 0.65, 1);
        }

        [Fact]
        public void FfmBinaryClassificationWithAdvancedArguments()
        {
            var mlContext = new MLContext(seed: 0);
            var data = DatasetUtils.GenerateFfmSamples(500);
            var dataView = mlContext.Data.LoadFromEnumerable(data);

            var ffmArgs = new FieldAwareFactorizationMachineTrainer.Options();

            // Customized the field names.
            ffmArgs.FeatureColumnName = nameof(DatasetUtils.FfmExample.Field0); // First field.
            ffmArgs.ExtraFeatureColumns = new[]{ nameof(DatasetUtils.FfmExample.Field1), nameof(DatasetUtils.FfmExample.Field2) };

            var pipeline = mlContext.BinaryClassification.Trainers.FieldAwareFactorizationMachine(ffmArgs);

            var model = pipeline.Fit(dataView);
            var prediction = model.Transform(dataView);

            var metrics = mlContext.BinaryClassification.Evaluate(prediction);

            // Run a sanity check against a few of the metrics.
            Assert.InRange(metrics.Accuracy, 0.9, 1);
            Assert.InRange(metrics.AreaUnderRocCurve, 0.9, 1);
            Assert.InRange(metrics.AreaUnderPrecisionRecallCurve, 0.9, 1);
        }

        [Fact]
        public void FieldAwareFactorizationMachine_Estimator()
        {
            var data = new TextLoader(Env, GetFafmBCLoaderArgs())
                    .Load(GetDataPath(TestDatasets.breastCancer.trainFilename));

            var ffmArgs = new FieldAwareFactorizationMachineTrainer.Options {
                FeatureColumnName = "Feature1", // Features from the 1st field.
                ExtraFeatureColumns = new[] { "Feature2", "Feature3",  "Feature4" }, // 2nd field's feature column, 3rd field's feature column, 4th field's feature column.
                Shuffle = false,
                NumberOfIterations = 3,
                LatentDimension = 7,
            };

            var est = ML.BinaryClassification.Trainers.FieldAwareFactorizationMachine(ffmArgs);

            TestEstimatorCore(est, data);
            var model = est.Fit(data);
            var anotherModel = est.Fit(data, data, model.Model);

            Done();
        }

        private TextLoader.Options GetFafmBCLoaderArgs()
        {
            return new TextLoader.Options()
            {
                Separator = "\t",
                HasHeader = false,
                Columns = new[]
                {
                    new TextLoader.Column("Feature1", DataKind.Single, new [] { new TextLoader.Range(1, 2) }),
                    new TextLoader.Column("Feature2", DataKind.Single, new [] { new TextLoader.Range(3, 4) }),
                    new TextLoader.Column("Feature3", DataKind.Single, new [] { new TextLoader.Range(5, 6) }),
                    new TextLoader.Column("Feature4", DataKind.Single, new [] { new TextLoader.Range(7, 9) }),
                    new TextLoader.Column("Label", DataKind.Boolean, 0)
                }
            };
        }
    }
}
