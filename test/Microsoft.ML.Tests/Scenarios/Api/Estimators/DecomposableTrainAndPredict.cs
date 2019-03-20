// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public partial class ApiScenariosTests
    {
        /// <summary>
        /// Decomposable train and predict: Train on Iris multiclass problem, which will require
        /// a transform on labels. Be able to reconstitute the pipeline for a prediction only task,
        /// which will essentially "drop" the transform over labels, while retaining the property
        /// that the predicted label for this has a key-type, the probability outputs for the classes
        /// have the class labels as slot names, etc. This should be do-able without ugly compromises like,
        /// say, injecting a dummy label.
        /// </summary>
        [Fact]
        void DecomposableTrainAndPredict()
        {
            var dataPath = GetDataPath(TestDatasets.irisData.trainFilename);
            var ml = new MLContext();

            var data = ml.Data.LoadFromTextFile<IrisData>(dataPath, separatorChar: ',');

            var pipeline = new ColumnConcatenatingEstimator (ml, "Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                .Append(new ValueToKeyMappingEstimator(ml, "Label"), TransformerScope.TrainTest)
                .Append(ml.MulticlassClassification.Trainers.SdcaCalibrated(
                    new SdcaCalibratedMulticlassTrainer.Options { MaximumNumberOfIterations = 100, Shuffle = true, NumberOfThreads = 1, }))
                .Append(new KeyToValueMappingEstimator(ml, "PredictedLabel"));

            var model = pipeline.Fit(data).GetModelFor(TransformerScope.Scoring);
            var engine = ml.Model.CreatePredictionEngine<IrisDataNoLabel, IrisPrediction>(model);

            var testLoader = ml.Data.LoadFromTextFile(dataPath, TestDatasets.irisData.GetLoaderColumns(), separatorChar: ',', hasHeader: true);
            var testData = ml.Data.CreateEnumerable<IrisData>(testLoader, false);
            foreach (var input in testData.Take(20))
            {
                var prediction = engine.Predict(input);
                Assert.True(prediction.PredictedLabel == input.Label);
            }
        }
    }
}
