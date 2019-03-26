using System;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class FFMBinaryClassificationWithOptions
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
            var pipeline = new EstimatorChain<ITransformer>().AppendCacheCheckpoint(mlContext)
                .Append(mlContext.BinaryClassification.Trainers.
                    FieldAwareFactorizationMachine(
                        new FieldAwareFactorizationMachineTrainer.Options
                        {
                            FeatureColumnName = "Features",
                            LabelColumnName = "Sentiment",
                            LearningRate = 0.1f,
                            NumberOfIterations = 10
                        }));

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

            //  The feature count is: 9374
            //  The number of fields is: 1
            //  The latent dimension is: 20
            //  The linear weights of some of the features are:  0.0410  0.0000 -0.0078 -0.0285  0.0000  0.0114  0.1313  0.0183 -0.0224  0.0166
            //  The weights of some of the latent features are: -0.0326  0.1127  0.0621  0.1446  0.2038  0.1608  0.2084  0.0141  0.2458 -0.0625

            // Evaluate how the model is doing on the test data.
            var dataWithPredictions = model.Transform(testData);

            var metrics = mlContext.BinaryClassification.Evaluate(dataWithPredictions, "Sentiment");
            SamplesUtils.ConsoleUtils.PrintMetrics(metrics);

            //  Accuracy: 0.78
            //  AUC: 0.81
            //  F1 Score: 0.78
            //  Negative Precision: 0.78
            //  Negative Recall: 0.78
            //  Positive Precision: 0.78
            //  Positive Recall: 0.78
        }
    }
}
