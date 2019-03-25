using System;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.SamplesUtils;
using Microsoft.ML.Trainers.LightGbm;

namespace Microsoft.ML.Samples.Dynamic.Trainers.MulticlassClassification
{
    public static class LightGbmWithOptions
    {
        // This example requires installation of additional nuget package <a href="https://www.nuget.org/packages/Microsoft.ML.LightGbm/">Microsoft.ML.LightGbm</a>.
        public static void Example()
        {
            // Create a general context for ML.NET operations. It can be used for exception tracking and logging,
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext(seed: 0);

            // Create a list of data examples.
            var examples = DatasetUtils.GenerateRandomMulticlassClassificationExamples(1000);

            // Convert the examples list to an IDataView object, which is consumable by ML.NET API.
            var dataView = mlContext.Data.LoadFromEnumerable(examples);

            //////////////////// Data Preview ////////////////////
            // Label    Features
            // AA       0.7262433,0.8173254,0.7680227,0.5581612,0.2060332,0.5588848,0.9060271,0.4421779,0.9775497,0.2737045
            // BB       0.4919063,0.6673147,0.8326591,0.6695119,1.182151,0.230367,1.06237,1.195347,0.8771811,0.5145918
            // CC       1.216908,1.248052,1.391902,0.4326252,1.099942,0.9262842,1.334019,1.08762,0.9468155,0.4811099
            // DD       0.7871246,1.053327,0.8971719,1.588544,1.242697,1.362964,0.6303943,0.9810045,0.9431419,1.557455

            // Create a pipeline. 
            //  - Convert the string labels into key types.
            //  - Apply LightGbm multiclass trainer with advanced options.
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("LabelIndex", "Label")
                        .Append(mlContext.MulticlassClassification.Trainers.LightGbm(new LightGbmMulticlassTrainer.Options
                        {
                            LabelColumnName = "LabelIndex",
                            FeatureColumnName = "Features",
                            Booster = new DartBooster.Options()
                            {
                                TreeDropFraction = 0.15,
                                XgboostDartMode = false
                            }
                        }))
                        .Append(mlContext.Transforms.Conversion.MapValueToKey("PredictedLabelIndex", "PredictedLabel"))
                        .Append(mlContext.Transforms.CopyColumns("Scores", "Score"));

            // Split the static-typed data into training and test sets. Only training set is used in fitting
            // the created pipeline. Metrics are computed on the test.
            var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.5);

            // Train the model.
            var model = pipeline.Fit(split.TrainSet);

            // Do prediction on the test set.
            var dataWithPredictions = model.Transform(split.TestSet);

            // Evaluate the trained model using the test set.
            var metrics = mlContext.MulticlassClassification.Evaluate(dataWithPredictions, labelColumnName: "LabelIndex");

            // Check if metrics are reasonable.
            Console.WriteLine($"Macro accuracy: {metrics.MacroAccuracy:F4}, Micro accuracy: {metrics.MicroAccuracy:F4}.");
            // Console output:
            //   Macro accuracy: 0.8619, Micro accuracy: 0.8611.

            // IDataView with predictions, to an IEnumerable<DatasetUtils.MulticlassClassificationExample>.
            var nativePredictions = mlContext.Data.CreateEnumerable<DatasetUtils.MulticlassClassificationExample>(dataWithPredictions, false).ToList();

            // Get schema object out of the prediction. It contains metadata such as the mapping from predicted label index
            // (e.g., 1) to its actual label (e.g., "AA").
            // The metadata can be used to get all the unique labels used during training.
            var labelBuffer = new VBuffer<ReadOnlyMemory<char>>();
            dataWithPredictions.Schema["PredictedLabelIndex"].GetKeyValues(ref labelBuffer);
            // nativeLabels is { "AA" , "BB", "CC", "DD" }
            var nativeLabels = labelBuffer.DenseValues().ToArray(); // nativeLabels[nativePrediction.PredictedLabelIndex - 1] is the original label indexed by nativePrediction.PredictedLabelIndex.


            // Show prediction result for the 3rd example.
            var nativePrediction = nativePredictions[2];
            // Console output:
            //   Our predicted label to this example is AA with probability 0.8986.
            Console.WriteLine($"Our predicted label to this example is {nativeLabels[(int)nativePrediction.PredictedLabelIndex - 1]} " +
                $"with probability {nativePrediction.Scores[(int)nativePrediction.PredictedLabelIndex - 1]:F4}.");

            // Scores and nativeLabels are two parallel attributes; that is, Scores[i] is the probability of being nativeLabels[i].
            // Console output:
            //  The probability of being class AA is 0.8986.
            //  The probability of being class BB is 0.0961.
            //  The probability of being class CC is 0.0050.
            //  The probability of being class DD is 0.0003.
            for (int i = 0; i < nativeLabels.Length; ++i)
                Console.WriteLine($"The probability of being class {nativeLabels[i]} is {nativePrediction.Scores[i]:F4}.");
        }
    }
}
