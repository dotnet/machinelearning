﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Legacy.Data;
using Microsoft.ML.Legacy.Models;
using Microsoft.ML.Legacy.Trainers;
using Microsoft.ML.Legacy.Transforms;
using Microsoft.ML.Runtime.Api;
using Xunit;

namespace Microsoft.ML.Scenarios
{
    public partial class ScenariosTests
    {
        [Fact]
        public void TrainAndPredictIrisModelWithStringLabelTest()
        {
            string dataPath = GetDataPath("iris.data");

            var pipeline = new Legacy.LearningPipeline();

            pipeline.Add(new TextLoader(dataPath).CreateFrom<IrisDataWithStringLabel>(useHeader: false, separator: ','));

            pipeline.Add(new Dictionarizer("Label"));  // "IrisPlantType" is used as "Label" because of column attribute name on the field.

            pipeline.Add(new ColumnConcatenator(outputColumn: "Features",
                 "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));

            pipeline.Add(new StochasticDualCoordinateAscentClassifier());

            var model = pipeline.Train<IrisDataWithStringLabel, IrisPrediction>();
            string[] scoreLabels;
            model.TryGetScoreLabelNames(out scoreLabels);

            Assert.NotNull(scoreLabels);
            Assert.Equal(3, scoreLabels.Length);
            Assert.Equal("Iris-setosa", scoreLabels[0]);
            Assert.Equal("Iris-versicolor", scoreLabels[1]);
            Assert.Equal("Iris-virginica", scoreLabels[2]);

            IrisPrediction prediction = model.Predict(new IrisDataWithStringLabel()
            {
                SepalLength = 5.1f,
                SepalWidth = 3.3f,
                PetalLength = 1.6f,
                PetalWidth = 0.2f,
            });

            Assert.Equal(1, prediction.PredictedLabels[0], 2);
            Assert.Equal(0, prediction.PredictedLabels[1], 2);
            Assert.Equal(0, prediction.PredictedLabels[2], 2);

            prediction = model.Predict(new IrisDataWithStringLabel()
            {
                SepalLength = 6.4f,
                SepalWidth = 3.1f,
                PetalLength = 5.5f,
                PetalWidth = 2.2f,
            });

            Assert.Equal(0, prediction.PredictedLabels[0], 2);
            Assert.Equal(0, prediction.PredictedLabels[1], 2);
            Assert.Equal(1, prediction.PredictedLabels[2], 2);

            prediction = model.Predict(new IrisDataWithStringLabel()
            {
                SepalLength = 4.4f,
                SepalWidth = 3.1f,
                PetalLength = 2.5f,
                PetalWidth = 1.2f,
            });

            Assert.Equal(.2, prediction.PredictedLabels[0], 1);
            Assert.Equal(.8, prediction.PredictedLabels[1], 1);
            Assert.Equal(0, prediction.PredictedLabels[2], 2);

            // Note: Testing against the same data set as a simple way to test evaluation.
            // This isn't appropriate in real-world scenarios.
            string testDataPath = GetDataPath("iris.data");
            var testData = new TextLoader(testDataPath).CreateFrom<IrisDataWithStringLabel>(useHeader: false, separator: ',');

            var evaluator = new ClassificationEvaluator();
            evaluator.OutputTopKAcc = 3;
            ClassificationMetrics metrics = evaluator.Evaluate(model, testData);

            Assert.Equal(.98, metrics.AccuracyMacro);
            Assert.Equal(.98, metrics.AccuracyMicro, 2);
            Assert.Equal(.06, metrics.LogLoss, 2);
            Assert.InRange(metrics.LogLossReduction, 94, 96);
            Assert.Equal(1, metrics.TopKAccuracy);

            Assert.Equal(3, metrics.PerClassLogLoss.Length);
            Assert.Equal(0, metrics.PerClassLogLoss[0], 1);
            Assert.Equal(.1, metrics.PerClassLogLoss[1], 1);
            Assert.Equal(.1, metrics.PerClassLogLoss[2], 1);

            ConfusionMatrix matrix = metrics.ConfusionMatrix;
            Assert.Equal(3, matrix.Order);
            Assert.Equal(3, matrix.ClassNames.Count);
            Assert.Equal("Iris-setosa", matrix.ClassNames[0]);
            Assert.Equal("Iris-versicolor", matrix.ClassNames[1]);
            Assert.Equal("Iris-virginica", matrix.ClassNames[2]);

            Assert.Equal(50, matrix[0, 0]);
            Assert.Equal(50, matrix["Iris-setosa", "Iris-setosa"]);
            Assert.Equal(0, matrix[0, 1]);
            Assert.Equal(0, matrix["Iris-setosa", "Iris-versicolor"]);
            Assert.Equal(0, matrix[0, 2]);
            Assert.Equal(0, matrix["Iris-setosa", "Iris-virginica"]);

            Assert.Equal(0, matrix[1, 0]);
            Assert.Equal(0, matrix["Iris-versicolor", "Iris-setosa"]);
            Assert.Equal(48, matrix[1, 1]);
            Assert.Equal(48, matrix["Iris-versicolor", "Iris-versicolor"]);
            Assert.Equal(2, matrix[1, 2]);
            Assert.Equal(2, matrix["Iris-versicolor", "Iris-virginica"]);

            Assert.Equal(0, matrix[2, 0]);
            Assert.Equal(0, matrix["Iris-virginica", "Iris-setosa"]);
            Assert.Equal(1, matrix[2, 1]);
            Assert.Equal(1, matrix["Iris-virginica", "Iris-versicolor"]);
            Assert.Equal(49, matrix[2, 2]);
            Assert.Equal(49, matrix["Iris-virginica", "Iris-virginica"]);
        }

        public class IrisDataWithStringLabel
        {
            [Column("0")]
            public float SepalLength;

            [Column("1")]
            public float SepalWidth;

            [Column("2")]
            public float PetalLength;

            [Column("3")]
            public float PetalWidth;

            [Column("4", name: "Label")]
            public string IrisPlantType;
        }
    }
}
