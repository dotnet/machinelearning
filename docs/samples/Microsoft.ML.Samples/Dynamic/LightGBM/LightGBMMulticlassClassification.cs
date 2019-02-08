using System;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.SamplesUtils;

namespace Microsoft.ML.Samples.Dynamic.LightGBM
{
    class LightGbmMulticlassClassification
    {
        public static void LightGbmMulticlassClassificationExample()
        {
            // Create a general context for ML.NET operations. It can be used for exception tracking and logging,
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Create in-memory examples as C# native class.
            var examples = DatasetUtils.GenerateRandomMulticlassClassificationExamples(1000);

            // Convert native C# class to IDataView, a consumble format to ML.NET functions.
            var dataView = mlContext.Data.ReadFromEnumerable(examples);

            // Create a pipeline. 
            //  - Convert the string labels into key types.
            //  - Apply LightGbm multiclass trainer
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("LabelIndex", "Label")
                        .Append(mlContext.MulticlassClassification.Trainers.LightGbm(labelColumn: "LabelIndex"))
                        .Append(mlContext.Transforms.Conversion.MapValueToKey("PredictedLabelIndex", "PredictedLabel"))
                        .Append(mlContext.Transforms.CopyColumns("Scores", "Score"));

            // Split the static-typed data into training and test sets. Only training set is used in fitting
            // the created pipeline. Metrics are computed on the test.
            var (trainingData, testingData) = mlContext.MulticlassClassification.TrainTestSplit(dataView, testFraction: 0.5);

            // Train the model.
            var model = pipeline.Fit(trainingData);

            // Do prediction on the test set.
            var dataWithPredictions = model.Transform(testingData);

            // Evaluate the trained model is the test set.
            var metrics = mlContext.MulticlassClassification.Evaluate(dataWithPredictions, label: "LabelIndex");

            // Check if metrics are resonable.
            Console.WriteLine("Macro accuracy: {0}, Micro accuracy: {1}.", 0.863482146891263, 0.86309523809523814);

            // Convert prediction in ML.NET format to native C# class.
            var nativePredictions = mlContext.CreateEnumerable<DatasetUtils.MulticlassClassificationExample>(dataWithPredictions, false).ToList();

            // Get schema object out of the prediction. It contains metadata such as the mapping from predicted label index
            // (e.g., 1) to its actual label (e.g., "AA"). The call to "AsDynamic" converts our statically-typed pipeline into
            // a dynamically-typed one only for extracting metadata. In the future, metadata in statically-typed pipeline should
            // be accessible without dynamically-typed things.
            var schema = dataWithPredictions.Schema;

            // Retrieve the mapping from labels to label indexes.
            var labelBuffer = new VBuffer<ReadOnlyMemory<char>>();
            schema[nameof(DatasetUtils.MulticlassClassificationExample.PredictedLabelIndex)].Metadata.GetValue("KeyValues", ref labelBuffer);
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
