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
            public float SepalWidth { get; set; }
            public float SepalLength { get; set; }
            public float PetalWidth { get; set; }
            public float PetalLength { get; set; }
        }

        public void FirstExperienceWithML()
        {
            // This is the 'getting started with ML' example, how we see it in our new API.
            // It currently doesn't compile, let alone work, but we still can discuss and improve the syntax.

            // Load the data into the system.
            string dataPath = "iris-data.txt";
            var data = TextReader.FitAndRead(env, dataPath, row => (
                Label: row.ReadString(0),
                SepalWidth: row.ReadFloat(1),
                SepalLength: row.ReadFloat(2),
                PetalWidth: row.ReadFloat(3),
                PetalLength: row.ReadFloat(4)));


            var preprocess = data.Schema.MakeEstimator(row => (
                // Convert string label to key.
                Label: row.Label.DictionarizeLabel(),
                // Concatenate all features into a vector.
                Features: row.SepalWidth.ConcatWith(row.SepalLength, row.PetalWidth, row.PetalLength)));

            // Create a learner and train it.
            var learner = preprocess.AppendEstimator(row => row.Label.SdcaPredict(row.Features, l1Coefficient: 0.1));
            var classifier = learner.Fit(data);

            // Add another transformer that converts the integer prediction into string.
            var finalTransformer = classifier.AppendTransformer(row => row.PredictedLabel.KeyToValue());

            // Make a prediction engine and predict.
            engine = finalTransformer.MakePredictionEngine<IrisExample, IrisPrediction>();
            IrisPrediction prediction = engine.Predict(new IrisExample
                {
                    SepalWidth = 3.3f,
                    SepalLength = 1.6f,
                    PetalWidth = 0.2f,
                    PetalLength = 5.1f
                });
        }
    }
}
