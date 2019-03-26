using System;
using System.Collections.Generic;
using static Microsoft.ML.DataOperationsCatalog;

namespace Microsoft.ML.Samples.Dynamic
{
    /// <summary>
    /// Sample class showing how to use CrossValidationSplit.
    /// </summary>
    public static class CrossValidationSplit
    {
        public static void Example()
        {
            // Creating the ML.Net IHostEnvironment object, needed for the pipeline.
            var mlContext = new MLContext();

            // Generate some data points.
            var examples = GenerateRandomDataPoints(10);

            // Convert the examples list to an IDataView object, which is consumable by ML.NET API.
            var dataview = mlContext.Data.LoadFromEnumerable(examples);

            // Cross validation splits your data randomly into set of "folds", and creates groups of Train and Test sets,
            // where for each group, one fold is the Test and the rest of the folds the Train.
            // So below, we specify Group column as the column containing the sampling keys.
            // If we pass that column to cross validation it would be used to break data into certain chunks.
            var folds = mlContext.Data.CrossValidationSplit(dataview, numberOfFolds: 3, samplingKeyColumnName: "Group");
            PrintPreviewRows(folds[0]);

            // The data in the Train split.
            // [Group, 1], [Features, 0.8173254]
            // [Group, 2], [Features, 0.7680227]
            // [Group, 1], [Features, 0.2060332]
            // [Group, 2], [Features, 0.5588848]
            // [Group, 1], [Features, 0.4421779]
            // [Group, 2], [Features, 0.9775497]
            // 
            // The data in the Test split.
            // [Group, 0], [Features, 0.7262433]
            // [Group, 0], [Features, 0.5581612]
            // [Group, 0], [Features, 0.9060271]
            // [Group, 0], [Features, 0.2737045]

            PrintPreviewRows(folds[1]);
            // The data in the Train split.
            // [Group, 0], [Features, 0.7262433]
            // [Group, 2], [Features, 0.7680227]
            // [Group, 0], [Features, 0.5581612]
            // [Group, 2], [Features, 0.5588848]
            // [Group, 0], [Features, 0.9060271]
            // [Group, 2], [Features, 0.9775497]
            // [Group, 0], [Features, 0.2737045]
            // 
            // The data in the Test split.
            // [Group, 1], [Features, 0.8173254]
            // [Group, 1], [Features, 0.2060332]
            // [Group, 1], [Features, 0.4421779]

            PrintPreviewRows(folds[2]);
            // The data in the Train split.
            // [Group, 0], [Features, 0.7262433]
            // [Group, 1], [Features, 0.8173254]
            // [Group, 0], [Features, 0.5581612]
            // [Group, 1], [Features, 0.2060332]
            // [Group, 0], [Features, 0.9060271]
            // [Group, 1], [Features, 0.4421779]
            // [Group, 0], [Features, 0.2737045]
            // 
            // The data in the Test split.
            // [Group, 2], [Features, 0.7680227]
            // [Group, 2], [Features, 0.5588848]
            // [Group, 2], [Features, 0.9775497]

            // Example of a split without specifying a sampling key column.
            folds = mlContext.Data.CrossValidationSplit(dataview, numberOfFolds: 3);
            PrintPreviewRows(folds[0]);
            // The data in the Train split.
            // [Group, 0], [Features, 0.7262433]
            // [Group, 1], [Features, 0.8173254]
            // [Group, 2], [Features, 0.7680227]
            // [Group, 0], [Features, 0.5581612]
            // [Group, 1], [Features, 0.2060332]
            // [Group, 1], [Features, 0.4421779]
            // [Group, 2], [Features, 0.9775497]
            // [Group, 0], [Features, 0.2737045]
            // 
            // The data in the Test split.
            // [Group, 2], [Features, 0.5588848]
            // [Group, 0], [Features, 0.9060271]

            PrintPreviewRows(folds[1]);
            // The data in the Train split.
            // [Group, 2], [Features, 0.7680227]
            // [Group, 0], [Features, 0.5581612]
            // [Group, 1], [Features, 0.2060332]
            // [Group, 2], [Features, 0.5588848]
            // [Group, 0], [Features, 0.9060271]
            // [Group, 1], [Features, 0.4421779]
            // 
            // The data in the Test split.
            // [Group, 0], [Features, 0.7262433]
            // [Group, 1], [Features, 0.8173254]
            // [Group, 2], [Features, 0.9775497]
            // [Group, 0], [Features, 0.2737045]

            PrintPreviewRows(folds[2]);
            // The data in the Train split.
            // [Group, 0], [Features, 0.7262433]
            // [Group, 1], [Features, 0.8173254]
            // [Group, 2], [Features, 0.5588848]
            // [Group, 0], [Features, 0.9060271]
            // [Group, 2], [Features, 0.9775497]
            // [Group, 0], [Features, 0.2737045]
            // 
            // The data in the Test split.
            // [Group, 2], [Features, 0.7680227]
            // [Group, 0], [Features, 0.5581612]
            // [Group, 1], [Features, 0.2060332]
            // [Group, 1], [Features, 0.4421779]
        }

        private static IEnumerable<DataPoint> GenerateRandomDataPoints(int count, int seed = 0)
        {
            var random = new Random(seed);
            for (int i = 0; i < count; i++)
            {
                yield return new DataPoint
                {
                    Group = i % 3,

                    // Create random features that are correlated with label.
                    Features = (float)random.NextDouble()
                };
            }
        }

        // Example with features and group column. A data set is a collection of such examples.
        private class DataPoint
        {
            public float Group { get; set; }

            public float Features { get; set; }
        }

        // print helper
        private static void PrintPreviewRows(TrainTestData split)
        {

            var trainDataPreview = split.TrainSet.Preview();
            var testDataPreview = split.TestSet.Preview();

            Console.WriteLine($"The data in the Train split.");
            foreach (var row in trainDataPreview.RowView)
                Console.WriteLine($"{row.Values[0]}, {row.Values[1]}");

            Console.WriteLine($"\nThe data in the Test split.");
            foreach (var row in testDataPreview.RowView)
                Console.WriteLine($"{row.Values[0]}, {row.Values[1]}");
        }
    }
}
