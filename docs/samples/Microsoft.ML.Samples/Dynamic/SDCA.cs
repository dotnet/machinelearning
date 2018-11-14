// the alignment of the usings with the methods is intentional so they can display on the same level in the docs site. 
        using Microsoft.ML.Runtime.Data;
        using System;
        using System.Linq;

namespace Microsoft.ML.Samples.Dynamic
{
    public partial class TrainerSamples
    {
        public static void SDCA_BinaryClassification()
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
            var reader = mlContext.Data.TextReader(new TextLoader.Arguments()
                {
                    Separator = "tab",
                    HasHeader = true,
                    Column = new[]
                    {
                        new TextLoader.Column("Sentiment", DataKind.BL, 0),
                        new TextLoader.Column("SentimentText", DataKind.Text, 1)
                    }
                });
            
            // Read the data
            var data = reader.Read(dataFile);

            // Step 2: Pipeline 
            // Featurize the text column through the FeaturizeText API. 
            // Then append a binary classifier, setting the "Label" column as the label of the dataset, and 
            // the "Features" column produced by FeaturizeText as the features column. 
            var pipeline = mlContext.Transforms.Text.FeaturizeText("SentimentText", "Features")
                    .Append(mlContext.BinaryClassification.Trainers.StochasticDualCoordinateAscent(labelColumn: "Sentiment", featureColumn: "Features", l2Const: 0.001f));

            // Step 3: Run Cross-Validation on this pipeline.
            var cvResults = mlContext.BinaryClassification.CrossValidate(data, pipeline, labelColumn: "Sentiment");

            var accuracies = cvResults.Select(r => r.metrics.Accuracy);
            Console.WriteLine(accuracies.Average());

            // If we wanted to specify more advanced parameters for the algorithm, 
            // we could do so by tweaking the 'advancedSetting'.
            var advancedPipeline = mlContext.Transforms.Text.FeaturizeText("SentimentText", "Features")
                                  .Append(mlContext.BinaryClassification.Trainers.StochasticDualCoordinateAscent
                                  (labelColumn: "Sentiment",
                                   featureColumn: "Features",
                                   advancedSettings: s=>
                                       {
                                           s.ConvergenceTolerance = 0.01f;   // The learning rate for adjusting bias from being regularized
                                           s.NumThreads = 2;            // Degree of lock-free parallelism 
                                       })
                                   );

            // Run Cross-Validation on this second pipeline.
            var cvResults_advancedPipeline = mlContext.BinaryClassification.CrossValidate(data, pipeline, labelColumn: "Sentiment", numFolds: 3);
            accuracies = cvResults_advancedPipeline.Select(r => r.metrics.Accuracy);
            Console.WriteLine(accuracies.Average());

        }
    }
}
