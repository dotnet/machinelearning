using System;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class FFMBinaryClassificationWithoutArguments
    {
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Download and featurize the dataset.
            var dataviews = SamplesUtils.DatasetUtils.LoadFeaturizedSentimentDataset(mlContext);
            var trainData = dataviews[0];
            var testData = dataviews[1];

            // ML.NET doesn't cache data set by default. Therefore, if one reads a data set from a file and accesses it many times, it can be slow due to
            // expensive featurization and disk operations. When the considered data can fit into memory, a solution is to cache the data in memory. Caching is especially
            // helpful when working with iterative algorithms which needs many data passes. Since SDCA is the case, we cache. Inserting a
            // cache step in a pipeline is also possible, please see the construction of pipeline below.
            trainData = mlContext.Data.Cache(trainData);

            // Step 2: Pipeline
            // Create the 'FieldAwareFactorizationMachine' binary classifier, setting the "Sentiment" column as the label of the dataset, and 
            // the "Features" column as the features column.
            var pipeline = mlContext.Transforms.CopyColumns("Label", "Sentiment")
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.BinaryClassification.Trainers.FieldAwareFactorizationMachine());

            // Fit the model.
            var model = pipeline.Fit(trainData);

            // Let's get the model parameters from the model.
            var modelParams = model.LastTransformer.Model;

            // Let's inspect the model parameters.
            var featureCount = modelParams.FeatureCount;
            var fieldCount = modelParams.FieldCount;
            var latentDim = modelParams.LatentDimension;
            var linearWeights = modelParams.GetLinearWeights();
            var latentWeights = modelParams.GetLatentWeights();

            Console.WriteLine("The feature count is: " + featureCount);
            Console.WriteLine("The number of fields is: " + fieldCount);
            Console.WriteLine("The latent dimension is: " + latentDim);
            Console.WriteLine("The linear weights of some of the features are: " + 
                string.Concat(Enumerable.Range(1, 10).Select(i => $"{linearWeights[i]:F4} ")));
            Console.WriteLine("The weights of some of the latent features are: " + 
                string.Concat(Enumerable.Range(1, 10).Select(i => $"{latentWeights[i]:F4} ")));

            // Expected Output:
            //  The feature count is: 9374
            //  The number of fields is: 1
            //  The latent dimension is: 20
            //  The linear weights of some of the features are: 0.0188 0.0000 -0.0048 -0.0184 0.0000 0.0031 0.0914 0.0112 -0.0152 0.0110
            //  The weights of some of the latent features are: 0.0631 0.0041 -0.0333 0.0694 0.1330 0.0790 0.1168 -0.0848 0.0431 0.0411

            // Evaluate how the model is doing on the test data.
            var dataWithPredictions = model.Transform(testData);

            var metrics = mlContext.BinaryClassification.Evaluate(dataWithPredictions, "Sentiment");
            SamplesUtils.ConsoleUtils.PrintMetrics(metrics);

            // Expected output:
            //  Accuracy: 0.61
            //  AUC: 0.72
            //  F1 Score: 0.59
            //  Negative Precision: 0.60
            //  Negative Recall: 0.67
            //  Positive Precision: 0.63
            //  Positive Recall: 0.56
            //  Log Loss: 1.21
            //  Log Loss Reduction: -21.20
            //  Entropy: 1.00
        }
    }
}
