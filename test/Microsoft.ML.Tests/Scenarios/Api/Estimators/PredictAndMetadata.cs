// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFrameworkCommon;
using Microsoft.ML.Trainers;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public partial class ApiScenariosTests
    {
        /// <summary>
        /// Multiclass predictions produce single PredictedLabel column and array of scores.
        /// This examples shows how to map score value to original label.
        /// In case if you don't apply KeyToValue estimator on top of predictor label we won't convert
        /// key value to original label value. This example also shows how to convert key value to original label.
        /// </summary>
        [Fact]
        public void PredictAndMetadata()
        {
            var dataPath = GetDataPath(TestDatasets.irisData.trainFilename);
            var ml = new MLContext(1);

            var data = ml.Data.LoadFromTextFile<IrisData>(dataPath, separatorChar: ',');

            var pipeline = ml.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                .Append(ml.Transforms.Conversion.MapValueToKey("Label"), TransformerScope.TrainTest)
                .Append(ml.MulticlassClassification.Trainers.SdcaMaximumEntropy(
                    new SdcaMaximumEntropyMulticlassTrainer.Options { MaximumNumberOfIterations = 100, Shuffle = true, NumberOfThreads = 1, }));

            var model = pipeline.Fit(data).GetModelFor(TransformerScope.Scoring);
            var engine = ml.Model.CreatePredictionEngine<IrisDataNoLabel, IrisPredictionNotCasted>(model);

            var testLoader = ml.Data.LoadFromTextFile(dataPath, TestDatasets.irisData.GetLoaderColumns(), separatorChar: ',', hasHeader: true);
            var testData = ml.Data.CreateEnumerable<IrisData>(testLoader, false);

            // During prediction we will get Score column with 3 float values.
            // We need to find way to map each score to original label.
            // In order to do what we need to get TrainingLabelValues from Score column.
            // TrainingLabelValues on top of Score column represent original labels for i-th value in Score array.
            VBuffer<ReadOnlyMemory<char>> originalLabels = default;
            engine.OutputSchema[nameof(IrisPrediction.Score)].Annotations.GetValue(AnnotationUtils.Kinds.TrainingLabelValues, ref originalLabels);
            // Since we apply MapValueToKey estimator with default parameters, key values
            // depends on order of occurrence in data file. Which is "Iris-setosa", "Iris-versicolor", "Iris-virginica"
            // So if we have Score column equal to [0.2, 0.3, 0.5] that's mean what score for
            // Iris-setosa is 0.2
            // Iris-versicolor is 0.3
            // Iris-virginica is 0.5.
            Assert.Equal("Iris-setosa", originalLabels.GetItemOrDefault(0).ToString());
            Assert.Equal("Iris-versicolor", originalLabels.GetItemOrDefault(1).ToString());
            Assert.Equal("Iris-virginica", originalLabels.GetItemOrDefault(2).ToString());

            // Let's look how we can convert key value for PredictedLabel to original labels.
            // We need to read KeyValues for "PredictedLabel" column.
            VBuffer<ReadOnlyMemory<char>> keys = default;
            engine.OutputSchema[nameof(IrisPrediction.PredictedLabel)].GetKeyValues(ref keys);
            foreach (var input in testData.Take(20))
            {
                var prediction = engine.Predict(input);
                // Predicted label is key type which internal representation starts from 1.
                // (0 reserved for NaN value) so in order to cast key to index in key metadata we need to distract 1 from it.
                var deciphieredLabel = keys.GetItemOrDefault((int)prediction.PredictedLabel - 1).ToString();
                Assert.True(deciphieredLabel == input.Label);
            }
        }

        [Fact]
        public void MulticlassConfusionMatrixSlotNames()
        {
            var mlContext = new MLContext(seed: 1);

            var dataPath = GetDataPath(TestDatasets.irisData.trainFilename);
            var data = mlContext.Data.LoadFromTextFile<IrisData>(dataPath, separatorChar: ',');

            var pipeline = mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Label"))
                .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(
                    new SdcaMaximumEntropyMulticlassTrainer.Options { MaximumNumberOfIterations = 100, Shuffle = true, NumberOfThreads = 1, }));

            var model = pipeline.Fit(data);

            // Evaluate the model.
            var scoredData = model.Transform(data);
            var metrics = mlContext.MulticlassClassification.Evaluate(scoredData);

            // Check that the SlotNames column is there. 
            Assert.NotNull(scoredData.Schema["Score"].Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.SlotNames));

            //Assert that the confusion matrix has the class names, in the Annotations of the Count column
            Assert.Equal("Iris-setosa", metrics.ConfusionMatrix.PredictedClassesIndicators[0].ToString());
            Assert.Equal("Iris-versicolor", metrics.ConfusionMatrix.PredictedClassesIndicators[1].ToString());
            Assert.Equal("Iris-virginica", metrics.ConfusionMatrix.PredictedClassesIndicators[2].ToString());

            var dataReader = mlContext.Data.CreateTextLoader(
                columns: new[]
                    {
                        new TextLoader.Column("Label", DataKind.Single, 0), //notice the label being loaded as a float
                        new TextLoader.Column("Features", DataKind.Single, new[]{ new TextLoader.Range(1,4) })
                    },
                hasHeader: false,
                separatorChar: '\t'
            );

            var dataPath2 = GetDataPath(TestDatasets.iris.trainFilename);
            var data2 = dataReader.Load(dataPath2);

            var singleTrainer = mlContext.BinaryClassification.Trainers.FastTree();

            // Create a training pipeline.
            var pipelineUnamed = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(singleTrainer));

            // Train the model.
            var model2 = pipelineUnamed.Fit(data2);

            // Evaluate the model.
            var scoredData2 = model2.Transform(data2);
            var metrics2 = mlContext.MulticlassClassification.Evaluate(scoredData2);

            // Check that the SlotNames column is not there. 
            Assert.Null(scoredData2.Schema["Score"].Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.SlotNames));

            //Assert that the confusion matrix has just ints, as class indicators, in the Annotations of the Count column
            Assert.Equal("0", metrics2.ConfusionMatrix.PredictedClassesIndicators[0].ToString());
            Assert.Equal("1", metrics2.ConfusionMatrix.PredictedClassesIndicators[1].ToString());
            Assert.Equal("2", metrics2.ConfusionMatrix.PredictedClassesIndicators[2].ToString());

        }
    }
}
