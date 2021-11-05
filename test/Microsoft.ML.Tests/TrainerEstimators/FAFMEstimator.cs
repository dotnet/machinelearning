// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.TestFrameworkCommon;
using Microsoft.ML.Trainers;
using Xunit;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators : TestDataPipeBase
    {
        [FieldAwareFactorizationMachineFact]
        public void FfmBinaryClassificationWithoutArguments()
        {
            var mlContext = new MLContext(seed: 0);
            var data = GenerateFfmSamples(500);
            var dataView = mlContext.Data.LoadFromEnumerable(data);

            var pipeline = mlContext.Transforms.CopyColumns(DefaultColumnNames.Features, nameof(FfmExample.Field0))
                .Append(mlContext.BinaryClassification.Trainers.FieldAwareFactorizationMachine());

            var model = pipeline.Fit(dataView);
            var prediction = model.Transform(dataView);

            var metrics = mlContext.BinaryClassification.Evaluate(prediction);

            // Run a sanity check against a few of the metrics.
            Assert.InRange(metrics.Accuracy, 0.6, 1);
            Assert.InRange(metrics.AreaUnderRocCurve, 0.7, 1);
            Assert.InRange(metrics.AreaUnderPrecisionRecallCurve, 0.65, 1);
        }

        [FieldAwareFactorizationMachineFact]
        public void FfmBinaryClassificationWithAdvancedArguments()
        {
            var mlContext = new MLContext(seed: 0);
            var data = GenerateFfmSamples(500);
            var dataView = mlContext.Data.LoadFromEnumerable(data);

            var ffmArgs = new FieldAwareFactorizationMachineTrainer.Options();

            // Customized the field names.
            ffmArgs.FeatureColumnName = nameof(FfmExample.Field0); // First field.
            ffmArgs.ExtraFeatureColumns = new[] { nameof(FfmExample.Field1), nameof(FfmExample.Field2) };

            var pipeline = mlContext.BinaryClassification.Trainers.FieldAwareFactorizationMachine(ffmArgs);

            var model = pipeline.Fit(dataView);
            var prediction = model.Transform(dataView);

            var metrics = mlContext.BinaryClassification.Evaluate(prediction);

            // Run a sanity check against a few of the metrics.
            Assert.InRange(metrics.Accuracy, 0.9, 1);
            Assert.InRange(metrics.AreaUnderRocCurve, 0.9, 1);
            Assert.InRange(metrics.AreaUnderPrecisionRecallCurve, 0.9, 1);
        }

        [FieldAwareFactorizationMachineFact]
        public void FieldAwareFactorizationMachine_Estimator()
        {
            var data = new TextLoader(Env, GetFafmBCLoaderArgs())
                    .Load(GetDataPath(TestDatasets.breastCancer.trainFilename));

            var ffmArgs = new FieldAwareFactorizationMachineTrainer.Options
            {
                FeatureColumnName = "Feature1", // Features from the 1st field.
                ExtraFeatureColumns = new[] { "Feature2", "Feature3", "Feature4" }, // 2nd field's feature column, 3rd field's feature column, 4th field's feature column.
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

        private const int _simpleBinaryClassSampleFeatureLength = 10;

        private class FfmExample
        {
            public bool Label;

            [VectorType(_simpleBinaryClassSampleFeatureLength)]
            public float[] Field0;

            [VectorType(_simpleBinaryClassSampleFeatureLength)]
            public float[] Field1;

            [VectorType(_simpleBinaryClassSampleFeatureLength)]
            public float[] Field2;
        }

        private static IEnumerable<FfmExample> GenerateFfmSamples(int exampleCount)
        {
            var rnd = new Random(0);
            var data = new List<FfmExample>();
            for (int i = 0; i < exampleCount; ++i)
            {
                // Initialize an example with a random label and an empty feature vector.
                var sample = new FfmExample()
                {
                    Label = rnd.Next() % 2 == 0,
                    Field0 = new float[_simpleBinaryClassSampleFeatureLength],
                    Field1 = new float[_simpleBinaryClassSampleFeatureLength],
                    Field2 = new float[_simpleBinaryClassSampleFeatureLength]
                };
                // Fill feature vector according the assigned label.
                for (int j = 0; j < 10; ++j)
                {
                    var value0 = (float)rnd.NextDouble();
                    // Positive class gets larger feature value.
                    if (sample.Label)
                        value0 += 0.2f;
                    sample.Field0[j] = value0;

                    var value1 = (float)rnd.NextDouble();
                    // Positive class gets smaller feature value.
                    if (sample.Label)
                        value1 -= 0.2f;
                    sample.Field1[j] = value1;

                    var value2 = (float)rnd.NextDouble();
                    // Positive class gets larger feature value.
                    if (sample.Label)
                        value2 += 0.8f;
                    sample.Field2[j] = value2;
                }

                data.Add(sample);
            }
            return data;
        }
    }
}
