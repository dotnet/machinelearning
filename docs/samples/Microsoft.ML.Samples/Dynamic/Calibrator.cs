using Microsoft.ML.Calibrator;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using System;
using System.Linq;

namespace Microsoft.ML.Samples.Dynamic
{
    /// <summary>
    /// This example first trains a StochasticDualCoordinateAscentBinary Classifier and then convert its output to probability via training a calibrator.  
    /// </summary>
    public class CalibratorExample
    {
        public static void Calibration()
        {
            // Downloading the dataset from github.com/dotnet/machinelearning.
            // This will create a sentiment.tsv file in the filesystem.
            // The string, dataFile, is the path to the downloaded file.
            // You can open this file, if you want to see the data. 
            string dataFile = SamplesUtils.DatasetUtils.DownloadSentimentDataset();

            // A preview of the data. 
            // Sentiment	SentimentText
            //      0	    " :Erm, thank you. "
            //      1	    ==You're cool==

            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Create a text loader.
            var reader = mlContext.Data.CreateTextReader(new TextLoader.Arguments()
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

            // Split the dataset into two parts: one used for training, the other to train the calibrator
            var (trainData, calibratorTrainingData) = mlContext.BinaryClassification.TrainTestSplit(data, testFraction: 0.1);

            // Featurize the text column through the FeaturizeText API. 
            // Then append the StochasticDualCoordinateAscentBinary binary classifier, setting the "Label" column as the label of the dataset, and 
            // the "Features" column produced by FeaturizeText as the features column. 
            var pipeline = mlContext.Transforms.Text.FeaturizeText("SentimentText", "Features")
                    .Append(mlContext.BinaryClassification.Trainers.StochasticDualCoordinateAscent(
                        labelColumn: "Sentiment", 
                        featureColumn: "Features", 
                        l2Const: 0.001f, 
                        loss: new HingeLoss())); // By specifying loss: new HingeLoss(), StochasticDualCoordinateAscent will train a support vector machine (SVM).

            // Fit the pipeline, and get a transformer that knows how to score new data.  
            var transformer = pipeline.Fit(trainData);
            IPredictor model = transformer.LastTransformer.Model;

            // Let's score the new data. The score will give us a numerical estimation of the chance that the particular sample 
            // bears positive sentiment. This estimate is relative to the numbers obtained. 
            var scoredData = transformer.Transform(calibratorTrainingData);
            var scoredDataPreview = scoredData.Preview();

            PrintRowViewValues(scoredDataPreview);
            // Preview of scoredDataPreview.RowView
            //
            // Score - 0.458968
            // Score - 0.7022135
            // Score 1.138822
            // Score 0.4807112
            // Score 1.112813

            // Let's train a calibrator estimator on this scored dataset. The trained calibrator estimator produces a transformer
            // that can transform the scored data by adding a new column names "Probability". 
            var calibratorEstimator = new PlattCalibratorEstimator(mlContext, model, "Sentiment", "Features");
            var calibratorTransformer = calibratorEstimator.Fit(scoredData);

            // Transform the scored data with a calibrator transfomer by adding a new column names "Probability". 
            // This column is a calibrated version of the "Score" column, meaning its values are a valid probability value in the [0, 1] interval
            // representing the chance that the respective sample bears positive sentiment. 
            var finalData = calibratorTransformer.Transform(scoredData).Preview();

            PrintRowViewValues(finalData);

            //Preview of finalData.RowView
            // 
            // Score - 0.458968    Probability 0.4670409
            // Score - 0.7022135   Probability 0.3912723
            // Score 1.138822      Probability 0.8703266
            // Score 0.4807112     Probability 0.7437012
            // Score 1.112813      Probability 0.8665403

        }

        private static void PrintRowViewValues(Data.DataDebuggerPreview data)
        {
            var firstRows = data.RowView.Take(5);

            foreach(Data.DataDebuggerPreview.RowInfo row in firstRows)
            {
                foreach (var kvPair in row.Values)
                {
                    if (kvPair.Key.Equals("Score") || kvPair.Key.Equals("Probability"))
                        Console.Write($" {kvPair.Key} {kvPair.Value} ");
                }
                Console.WriteLine();
            }
        }
    }
}
