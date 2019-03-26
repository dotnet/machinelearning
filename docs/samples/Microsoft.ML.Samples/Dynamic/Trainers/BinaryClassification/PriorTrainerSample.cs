using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public class PriorTrainer
    {
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

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
            var loader = mlContext.Data.CreateTextLoader(
                columns: new[]
                    {
                        new TextLoader.Column("Sentiment", DataKind.Single, 0),
                        new TextLoader.Column("SentimentText", DataKind.String, 1)
                    },
                hasHeader: true
            );

            // Load the data
            var trainData = loader.Load(trainFile);

            // Step 2: Pipeline 
            // Featurize the text column through the FeaturizeText API. 
            // Then append a binary classifier, setting the "Label" column as the label of the dataset, and 
            // the "Features" column produced by FeaturizeText as the features column.
            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", "SentimentText")
                    .AppendCacheCheckpoint(mlContext) // Add a data-cache step within a pipeline.
                    .Append(mlContext.BinaryClassification.Trainers.Prior(labelColumnName: "Sentiment"));

            // Step 3: Train the pipeline
            var trainedPipeline = pipeline.Fit(trainData);

            // Step 4: Evaluate on the test set
            var transformedData = trainedPipeline.Transform(loader.Load(testFile));
            var evalMetrics = mlContext.BinaryClassification.Evaluate(transformedData, labelColumnName: "Sentiment");
            SamplesUtils.ConsoleUtils.PrintMetrics(evalMetrics);

            // The Prior trainer outputs the proportion of a label in the dataset as the probability of that label.
            // In this case 'Accuracy: 0.50' means that there is a split of around 50%-50% of positive and negative labels in the test dataset.
            // Expected output:

            //  Accuracy: 0.50
            //  AUC: 0.50
            //  F1 Score: 0.67
            //  Negative Precision: 0.00
            //  Negative Recall: 0.00
            //  Positive Precision: 0.50
            //  Positive Recall: 1.00
            //  LogLoss: 1.05
            //  LogLossReduction: -4.89
            //  Entropy: 1.00
        }
    }
}
