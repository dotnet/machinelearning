using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public class AspirationalExamples
    {
        public class IrisPrediction
        {
            public string PredictedLabel;
        }

        public class IrisData
        {
            public float[] Features;
        }

        public void FirstExperienceWithML()
        {
            // This is the 'getting started with ML' example, how we see it in our new API.
            // It currently doesn't compile, let alone work, but we still can discuss and improve the syntax.

            // Load the data into the system.
            string dataPath = "iris-data.txt";
            var data = TextReader.ReadFile(dataPath, c => (Label: c.LoadString(0), Features: c.LoadFloat(1, 4)));

            // Assign numeric values to text in the "Label" column, because only
            // numbers can be processed during model training.
            var transformer = data.MakeTransformer(row => (Label: row.Label.Dictionarize(), row.Features));
            var trainingData = transformer.Transform(data);

            // Train a multiclass linear classifier.
            var learner = new StochasticDualCoordinateAscentClassifier();
            var classifier = learner.Train(trainingData);

            // Obtain some predictions.
            var predictionEngine = new PredictionEngine<float[], string>(classifier, inputColumn: "Features", outputColumn: "PredictedLabel");
            string prediction = predictionEngine.Predict(new[] { 3.3f, 1.6f, 0.2f, 5.1f });
        }
    }
}
