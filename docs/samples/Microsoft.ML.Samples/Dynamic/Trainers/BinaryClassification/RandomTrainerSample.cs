using System;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class RandomTrainer
    {
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext(seed: 1);

            // Download and featurize the dataset.
            var dataFiles = SamplesUtils.DatasetUtils.DownloadSentimentDataset();
            var trainFile = dataFiles[0];
            var testFile = dataFiles[1];

            // A preview of the data. 
            // Sentiment	SentimentText
            //      0	    " :Erm, thank you. "
            //      1	    ==You're cool==

            // Step 1: Read the data as an IDataView.
            // First, we define the reader: specify the data columns and where to find them in the text file.
            var reader = mlContext.Data.CreateTextLoader(
                columns: new[]
                    {
                        new TextLoader.Column("Sentiment", DataKind.Single, 0),
                        new TextLoader.Column("SentimentText", DataKind.String, 1)
                    },
                hasHeader: true
            );

            // Read the data
            var trainData = reader.Load(trainFile);

            // Step 2: Pipeline 
            // Featurize the text column through the FeaturizeText API. 
            // Then append a binary classifier, setting the "Label" column as the label of the dataset, and 
            // the "Features" column produced by FeaturizeText as the features column.
            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", "SentimentText")
                    .AppendCacheCheckpoint(mlContext)
                    .Append(mlContext.BinaryClassification.Trainers.Random());

            // Step 3: Train the pipeline
            var trainedPipeline = pipeline.Fit(trainData);

            // Step 4: Evaluate on the test set
            var transformedData = trainedPipeline.Transform(reader.Load(testFile));
            var evalMetrics = mlContext.BinaryClassification.Evaluate(transformedData, label: "Sentiment");
            SamplesUtils.ConsoleUtils.PrintMetrics(evalMetrics);

            // We expect an output probability closet to 0.5 as the Random trainer outputs a random prediction.
            // Regardless of the input features, the trainer will predict either positive or negative label with equal probability.
            // Expected output: (close to 0.5):

            //  Accuracy: 0.56
            //  AUC: 0.57
            //  F1 Score: 0.60
            //  Negative Precision: 0.57
            //  Negative Recall: 0.44
            //  Positive Precision: 0.55
            //  Positive Recall: 0.67
            //  LogLoss: 1.53
            //  LogLossReduction: -53.37
            //  Entropy: 1.00
        }
    }
}
