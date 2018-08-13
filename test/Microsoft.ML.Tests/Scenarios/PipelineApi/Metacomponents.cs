// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.PipelineApi
{
    public partial class PipelineApiScenarioTests
    {
        /// <summary>
        /// Meta-components: Meta-components (e.g., components that themselves instantiate components) should not be booby-trapped.
        /// When specifying what trainer OVA should use, a user will be able to specify any binary classifier.
        /// If they specify a regression or multi-class classifier ideally that should be a compile error.
        /// </summary>
        [Fact]
        void Metacomponents()
        {
            var dataPath = GetDataPath(IrisDataPath);
            var pipeline = new LearningPipeline(seed: 1, conc: 1);
            pipeline.Add(new TextLoader(dataPath).CreateFrom<IrisData>(useHeader: false));
            pipeline.Add(new Dictionarizer(new[] { "Label" }));
            pipeline.Add(new ColumnConcatenator(outputColumn: "Features",
                "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));

            pipeline.Add(OneVersusAll.With(new StochasticDualCoordinateAscentBinaryClassifier()));

            var model = pipeline.Train<IrisData, IrisPrediction>();

            var testData = new TextLoader(dataPath).CreateFrom<IrisData>(useHeader: false);
            var evaluator = new ClassificationEvaluator();
            ClassificationMetrics metrics = evaluator.Evaluate(model, testData);
            // I can't find way to convert predictedLabel from Key to string.
            //var prediction = model.Predict(new IrisData { PetalLength = 1, PetalWidth = 2, SepalLength = 1.4f, SepalWidth = 1.6f });
        }
    }
}
