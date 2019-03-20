using System;
using System.Linq;

namespace Microsoft.ML.Samples.Dynamic.Trainers.BinaryClassification.Calibrators
{
    public static class Naive
    {
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            // Setting the seed to a fixed number in this example to make outputs deterministic.
            var mlContext = new MLContext(seed: 0);

            // Download and featurize the dataset.
            var data = SamplesUtils.DatasetUtils.LoadFeaturizedAdultDataset(mlContext);
            // Leave out 10% of data for testing.
            var trainTestData = mlContext.Data.TrainTestSplit(data, testFraction: 0.3);

            // Create data training pipeline for non calibrated trainer and train Naive calibrator on top of it.
            var pipeline = mlContext.BinaryClassification.Trainers.AveragedPerceptron();

            // Fit the pipeline, and get a transformer that knows how to score new data.  
            var transformer = pipeline.Fit(trainTestData.TrainSet);
            // Fit this pipeline to the training data.
            // Let's score the new data. The score will give us a numerical estimation of the chance that the particular sample 
            // bears positive sentiment. This estimate is relative to the numbers obtained. 
            var scoredData = transformer.Transform(trainTestData.TestSet);
            var scoredDataPreview = scoredData.Preview();

            PrintRowViewValues(scoredDataPreview);
            // Preview of scoredDataPreview.RowView
            // Score   4.18144
            // Score  -14.10248
            // Score   2.731951
            // Score  -2.554229
            // Score   5.36571

            // Let's train a calibrator estimator on this scored dataset. The trained calibrator estimator produces a transformer
            // that can transform the scored data by adding a new column names "Probability". 
            var calibratorEstimator = mlContext.BinaryClassification.Calibrators.Naive();
            var calibratorTransformer = calibratorEstimator.Fit(scoredData);

            // Transform the scored data with a calibrator transfomer by adding a new column names "Probability". 
            // This column is a calibrated version of the "Score" column, meaning its values are a valid probability value in the [0, 1] interval
            // representing the chance that the respective sample bears positive sentiment. 
            var finalData = calibratorTransformer.Transform(scoredData).Preview();
            PrintRowViewValues(finalData);
           // Score   4.18144   Probability 0.775
           // Score  -14.10248  Probability 0.01923077
           // Score   2.731951  Probability 0.7738096
           // Score  -2.554229  Probability 0.2011494
           // Score   5.36571   Probability 0.9117647
        }

        private static void PrintRowViewValues(Data.DataDebuggerPreview data)
        {
            var firstRows = data.RowView.Take(5);

            foreach (Data.DataDebuggerPreview.RowInfo row in firstRows)
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
