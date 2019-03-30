using System;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm.StaticPipe;
using Microsoft.ML.SamplesUtils;
using Microsoft.ML.StaticPipe;

namespace Microsoft.ML.Samples.Static
{
    class LightGBMMulticlassWithInMemoryData
    {
        public void MulticlassLightGbmStaticPipelineWithInMemoryData()
        {
            // Create a general context for ML.NET operations. It can be used for exception tracking and logging,
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Create in-memory examples as C# native class.
            var examples = DatasetUtils.GenerateRandomMulticlassClassificationExamples(1000);

            // Convert native C# class to IDataView, a consumble format to ML.NET functions.
            var dataView = mlContext.Data.LoadFromEnumerable(examples);

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
                        Predictions: mlContext.MulticlassClassification.Trainers.LightGbm(r.Label.ToKey(), r.Features)
                    ))
                    .Append(r => (
                        // Actual label.
                        r.Label,
                        // Labels are converted to keys when training LightGBM so we convert it here again for calling evaluation function.
                        LabelIndex: r.Label.ToKey(),
                        // Used to compute metrics such as accuracy.
                        r.Predictions,
                        // Assign a new name to predicted class index.
                        PredictedLabelIndex: r.Predictions.predictedLabel,
                        // Assign a new name to class probabilities.
                        Scores: r.Predictions.score
                    ));

            // Split the static-typed data into training and test sets. Only training set is used in fitting
            // the created pipeline. Metrics are computed on the test.
            var (trainingData, testingData) = mlContext.Data.TrainTestSplit(staticDataView, testFraction: 0.5);

            // Train the model.
            var model = pipe.Fit(trainingData);

            // Do prediction on the test set.
            var prediction = model.Transform(testingData);

            // Evaluate the trained model is the test set.
            var metrics = mlContext.MulticlassClassification.Evaluate(prediction, r => r.LabelIndex, r => r.Predictions);

            // Check if metrics are resonable.
            Console.WriteLine ("Macro accuracy: {0}, Micro accuracy: {1}.", 0.863482146891263, 0.86309523809523814);

            // Convert prediction in ML.NET format to native C# class.
            var nativePredictions = mlContext.Data.CreateEnumerable<DatasetUtils.MulticlassClassificationExample>(prediction.AsDynamic, false).ToList();

            // Get schema object out of the prediction. It contains annotations such as the mapping from predicted label index
            // (e.g., 1) to its actual label (e.g., "AA"). The call to "AsDynamic" converts our statically-typed pipeline into
            // a dynamically-typed one only for extracting annotations. In the future, annotations in statically-typed pipeline should
            // be accessible without dynamically-typed things.
            var schema = prediction.AsDynamic.Schema;

            // Retrieve the mapping from labels to label indexes.
            var labelBuffer = new VBuffer<ReadOnlyMemory<char>>(); 
            schema[nameof(DatasetUtils.MulticlassClassificationExample.PredictedLabelIndex)].Annotations.GetValue("KeyValues", ref labelBuffer);
            // nativeLabels is { "AA" , "BB", "CC", "DD" }
            var nativeLabels = labelBuffer.DenseValues().ToArray(); // nativeLabels[nativePrediction.PredictedLabelIndex - 1] is the original label indexed by nativePrediction.PredictedLabelIndex.


            // Show prediction result for the 3rd example.
            var nativePrediction = nativePredictions[2];
            // Console output:
            //   Our predicted label to this example is "AA" with probability 0.922597349.
            Console.WriteLine("Our predicted label to this example is {0} with probability {1}",
                nativeLabels[(int)nativePrediction.PredictedLabelIndex - 1],
                nativePrediction.Scores[(int)nativePrediction.PredictedLabelIndex - 1]);

            var expectedProbabilities = new float[] { 0.922597349f, 0.07508608f, 0.00221699756f, 9.95488E-05f };
            // Scores and nativeLabels are two parallel attributes; that is, Scores[i] is the probability of being nativeLabels[i].
            // Console output:
            //  The probability of being class "AA" is 0.922597349.
            //  The probability of being class "BB" is 0.07508608.
            //  The probability of being class "CC" is 0.00221699756.
            //  The probability of being class "DD" is 9.95488E-05.
            for (int i = 0; i < labelBuffer.Length; ++i)
                Console.WriteLine("The probability of being class {0} is {1}.", nativeLabels[i], nativePrediction.Scores[i]);
        }
    }
}
