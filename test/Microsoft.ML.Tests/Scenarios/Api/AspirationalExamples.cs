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

        public class IrisExample
        {
            public float[] Features;
        }

        public void FirstExperienceWithML()
        {
            // This is the 'getting started with ML' example, how we see it in our new API.
            // It currently doesn't compile, let alone work, but we still can discuss and improve the syntax.

            // Initialize the environment.
            using (var env = new TlcEnvironment())
            {
                // Load the data into the system.
                string dataPath = "iris-data.txt";
                var data = TextReader.FitAndRead(env, dataPath, c => (Label: c.LoadString(0), Features: c.LoadFloat(1, 4)));

                // Convert string label to integer for training.
                var preprocess = data.MakeEstimator(row => (Label: row.Label.Dictionarize(), row.Features));

                // Create a learner and train it.
                var learner = preprocess.MakeEstimator(row => row.Label.SdcaPredict(row.Features, l1Coefficient: 0.1));
                var classifier = learner.Fit(preprocess.FitAndTransform(data));

                // Add another transformer that converts the integer prediction into string.
                var finalTransformer = classifier.AppendTransformer(row => row.PredictedLabel.KeyToValue());

                // Make a prediction engine and predict.
                engine = bundle.MakePredictionEngine<IrisExample, IrisPrediction>();
                IrisPrediction prediction = engine.Predict(new IrisExample { Features = new[] { 3.3f, 1.6f, 0.2f, 5.1f } });
            }
        }
    }
}
