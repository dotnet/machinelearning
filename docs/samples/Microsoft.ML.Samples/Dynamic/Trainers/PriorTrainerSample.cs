using System;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public class PriorTrainer
    {
        public static void Example()
        {
            // Downloading the dataset from github.com/dotnet/machinelearning.
            // This will create a sentiment.tsv file in the filesystem.
            // You can open this file, if you want to see the data. 
            string dataFile = SamplesUtils.DatasetUtils.DownloadSentimentDataset();

            // A preview of the data. 
            // Sentiment	SentimentText
            //      0	    " :Erm, thank you. "
            //      1	    ==You're cool==

            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Step 1: Read the data as an IDataView.
            // First, we define the reader: specify the data columns and where to find them in the text file.
            var reader = mlContext.Data.CreateTextLoader(
                columns: new[]
                    {
                        new TextLoader.Column("Sentiment", DataKind.R4, 0),
                        new TextLoader.Column("SentimentText", DataKind.Text, 1)
                    },
                hasHeader: true
            );
            
            // Read the data
            var data = reader.Read(dataFile);

            // Split it between training and test data
            var trainTestData = mlContext.BinaryClassification.TrainTestSplit(data);

            // Step 2: Pipeline 
            // Featurize the text column through the FeaturizeText API. 
            // Then append a binary classifier, setting the "Label" column as the label of the dataset, and 
            // the "Features" column produced by FeaturizeText as the features column.
            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", "SentimentText")
                    .AppendCacheCheckpoint(mlContext) // Add a data-cache step within a pipeline.
                    .Append(mlContext.BinaryClassification.Trainers.Prior(labelColumn: "Sentiment"));

            // Step 3: Train the pipeline
            var trainedPipeline = pipeline.Fit(trainTestData.TrainSet);

            // Step 4: Evaluate on the test set
            var transformedData = trainedPipeline.Transform(trainTestData.TestSet);
            var evalMetrics = mlContext.BinaryClassification.Evaluate(transformedData, label: "Sentiment");

            // Step 5: Inspect the output
            Console.WriteLine("Accuracy: " + evalMetrics.Accuracy);

            // The Prior trainer outputs the proportion of a label in the dataset as the probability of that label.
            // In this case it means that there is a split of around 64%-36% of positive and negative labels in the dataset.
            // Expected output:
            // Accuracy: 0.647058823529412
        }
    }
}
