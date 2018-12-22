using Microsoft.ML.Data;
using Microsoft.ML.LightGBM.StaticPipe;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.StaticPipe;
using System;
using System.Collections.Generic;

namespace Microsoft.ML.Samples.Static
{
    class LightGBMMulticlassWithInMemoryData
    {
        /// <summary>
        /// Number of features per example used in <see cref="MultiClassLightGbmStaticPipelineWithInMemoryData"/>.
        /// </summary>
        private const int _featureVectorLength = 10;

        /// <summary>
        /// Data point used in <see cref="MultiClassLightGbmStaticPipelineWithInMemoryData"/>. A data set there 
        /// is a collection of <see cref="NativeExample"/>.
        /// </summary>
        private class NativeExample
        {
            [VectorType(_featureVectorLength)]
            public float[] Features;
            [ColumnName("Label")]
            // One of "AA", "BB", "CC", and "DD".
            public string Label;
            public uint LabelIndex;
            // One of "AA", "BB", "CC", and "DD".
            public string PredictedLabel;
            [VectorType(4)]
            // The probabilities of being "AA", "BB", "CC", and "DD".
            public float[] Scores;

            public NativeExample()
            {
                Features = new float[_featureVectorLength];
            }
        }

        /// <summary>
        /// Helper function used to generate <see cref="NativeExample"/>s.
        /// </summary>
        private static List<NativeExample> GenerateRandomExamples(int count)
        {
            var examples = new List<NativeExample>();
            var rnd = new Random(0);
            for (int i = 0; i < count; ++i)
            {
                var example = new NativeExample();
                var res = i % 4;
                // Generate random float feature values.
                for (int j = 0; j < _featureVectorLength; ++j)
                {
                    var value = (float)rnd.NextDouble() + res * 0.2f;
                    example.Features[j] = value;
                }

                // Generate label based on feature sum.
                if (res == 0)
                    example.Label = "AA";
                else if (res == 1)
                    example.Label = "BB";
                else if (res == 2)
                    example.Label = "CC";
                else
                    example.Label = "DD";

                // The following three attributes are just placeholder for storing prediction results.
                example.LabelIndex = default;
                example.PredictedLabel = null;
                example.Scores = new float[4];

                examples.Add(example);
            }
            return examples;
        }

        public static void MultiClassLightGbmStaticPipelineWithInMemoryData()
        {
            // Create a general context for ML.NET operations. It can be used for exception tracking and logging,
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext(seed: 1, conc: 1);

            // Context for calling static classifiers. It contains constructors of classifiers and evaluation utilities.
            var ctx = new MulticlassClassificationContext(mlContext);

            // Create in-memory examples as C# native class.
            var examples = GenerateRandomExamples(1000);

            // Convert native C# class to IDataView, a consumble format to ML.NET functions.
            var dataView = ComponentCreation.CreateDataView(mlContext, examples);

            // IDataView is the data format used in dynamic-typed pipeline. To use static-typed pipeline, we need to convert
            // IDataView to DataView by calling AssertStatic(...). The basic idea is to specify the static type for each column
            // in IDataView in a lambda function.
            var staticDataView = dataView.AssertStatic(mlContext, c => (
                         Features: c.R4.Vector,
                         Label: c.Text.Scalar));

            // Create static pipeline. First, we make an estimator out of static DataView as the starting of a pipeline.
            // Then, we append necessary transforms and a classifier to the starting estimator.
            var pipe = staticDataView.MakeNewEstimator()
                    .Append(mapper: r => (
                        r.Label,
                        // Train multi-class LightGBM. The trained model maps Features to Label and probability of each class.
                        // The call of ToKey() is needed to convert string labels to integer indexes.
                        Predictions: ctx.Trainers.LightGbm(r.Label.ToKey(), r.Features)
                    ))
                    .Append(r => (
                        // Actual label.
                        r.Label,
                        // Labels are converted to keys when training LightGBM so we convert it here again for calling evaluation function.
                        LabelIndex: r.Label.ToKey(),
                        // Instance of ClassificationVectorData returned
                        r.Predictions,
                        // ToValue() is used to get the original out from the class indexes computed by ToKey().
                        // For example, if label "AA" is maped to index 0 via ToKey(), then ToValue() produces "AA" from 0.
                        PredictedLabel: r.Predictions.predictedLabel.ToValue(),
                        // Assign a new name to class probabilities.
                        Scores: r.Predictions.score
                    ));

            // Split the static-typed data into training and test sets. Only training set is used in fitting
            // the created pipeline. Metrics are computed on the test.
            var (trainingData, testingData) = ctx.TrainTestSplit(staticDataView, testFraction: 0.5);

            // Train the model.
            var model = pipe.Fit(trainingData);

            // Do prediction on the test set.
            var prediction = model.Transform(testingData);

            // Evaluate the trained model is the test set.
            var metrics = ctx.Evaluate(prediction, r => r.LabelIndex, r => r.Predictions);

            // Check if metrics are resonable.
            Console.WriteLine(metrics.AccuracyMacro); // expected value: 0.863482146891263
            Console.WriteLine(metrics.AccuracyMicro); // expected value: 0.86309523809523814

            // Convert prediction in ML.NET format to native C# class.
            var nativePredictions = new List<NativeExample>(prediction.AsDynamic.AsEnumerable<NativeExample>(mlContext, false));

            // Check predicted label and class probabilities of second-first example.
            // If you see a label with LabelIndex 1, its means its probability is the 1st element in the Scores field.
            // For example, if "AA" is indexed by 1, "BB" indexed by 2, "CC" indexed by 3, and "DD" indexed by 4, Scores is
            // ["AA" probability, "BB" probability, "CC" probability, "DD" probability].
            var nativePrediction = nativePredictions[2];
            var probAA = nativePrediction.Scores[0];
            var probBB = nativePrediction.Scores[1];
            var probCC = nativePrediction.Scores[2];
            var probDD = nativePrediction.Scores[3];

            Console.WriteLine(probAA); // expected value: 0.922597349
            Console.WriteLine(probBB); // expected value: 0.07508608
            Console.WriteLine(probCC); // expected value: 0.00221699756
            Console.WriteLine(probDD); // expected value: 9.95488E-05
        }
    }
}
