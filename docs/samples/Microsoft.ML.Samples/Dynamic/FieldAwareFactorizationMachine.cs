using System;
using Microsoft.ML.Data;
namespace Microsoft.ML.Samples.Dynamic
{
    public static class FFMBinaryClassification
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
                        new TextLoader.Column("Sentiment", DataKind.BL, 0),
                        new TextLoader.Column("SentimentText", DataKind.Text, 1)
                    },
                hasHeader: true
            );

            // Read the data
            var data = reader.Read(dataFile);

            // ML.NET doesn't cache data set by default. Therefore, if one reads a data set from a file and accesses it many times, it can be slow due to
            // expensive featurization and disk operations. When the considered data can fit into memory, a solution is to cache the data in memory. Caching is especially
            // helpful when working with iterative algorithms which needs many data passes. Since SDCA is the case, we cache. Inserting a
            // cache step in a pipeline is also possible, please see the construction of pipeline below.
            data = mlContext.Data.Cache(data);

            // Step 2: Pipeline 
            // Featurize the text column through the FeaturizeText API. 
            // Then append a binary classifier, setting the "Label" column as the label of the dataset, and 
            // the "Features" column produced by FeaturizeText as the features column.
            var pipeline = mlContext.Transforms.Text.FeaturizeText("SentimentText", "Features")
                    .AppendCacheCheckpoint(mlContext) // Add a data-cache step within a pipeline.
                    .Append(mlContext.BinaryClassification.Trainers.FieldAwareFactorizationMachine(labelColumn: "Sentiment", featureColumns: new[] { "Features" }));

            // Fit the model.
            var model = pipeline.Fit(data);

            // Let's get the model parameters from the model.
            var modelParams = model.LastTransformer.Model;

            // Let's inspect the model parameters.
            var featureCount = modelParams.GetFeatureCount();
            var fieldCount = modelParams.GetFieldCount();
            var latentDim = modelParams.GetLatentDim();
            var linearWeights = modelParams.GetLinearWeights();
            var latentWeights = modelParams.GetLatentWeights();

            Console.WriteLine("The feature count is: " + featureCount);
            Console.WriteLine("The number of fields is: " + fieldCount);
            Console.WriteLine("The latent dimension is: " + latentDim);
            Console.WriteLine("The lineear weights of the features are: " + string.Join(", ", linearWeights));
            Console.WriteLine("The weights of the latent features are: " + string.Join(", ", latentWeights));
        }
    }
}
