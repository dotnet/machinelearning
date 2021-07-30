using System;
using System.Collections.Generic;
using Microsoft.ML;

namespace Samples.Dynamic
{
    /// <summary>
    /// Sample class showing how to use TrainTestSplit.
    /// </summary>
    public static class TrainTestSplit
    {
        public static void Example()
        {
            // Creating the ML.Net IHostEnvironment object, needed for the pipeline.
            var mlContext = new MLContext();

            // Generate some data points.
            var examples = GenerateRandomDataPoints(10);

            // Convert the examples list to an IDataView object, which is consumable
            // by ML.NET API.
            var dataview = mlContext.Data.LoadFromEnumerable(examples);

            // Leave out 10% of the dataset for testing.For some types of problems,
            // for example for ranking or anomaly detection, we must ensure that the
            // split leaves the rows with the same value in a particular column, in
            // one of the splits. So below, we specify Group column as the column
            // containing the sampling keys. Notice how keeping the rows with the
            // same value in the Group column overrides the testFraction definition. 
            var split = mlContext.Data
                .TrainTestSplit(dataview, testFraction: 0.1,
                samplingKeyColumnName: "Group");

            var trainSet = mlContext.Data
                .CreateEnumerable<DataPoint>(split.TrainSet, reuseRowObject: false);

            var testSet = mlContext.Data
                .CreateEnumerable<DataPoint>(split.TestSet, reuseRowObject: false);

            PrintPreviewRows(trainSet, testSet);

            //  The data in the Train split.
            //  [Group, 1], [Features, 0.8173254]
            //  [Group, 1], [Features, 0.5581612]
            //  [Group, 1], [Features, 0.5588848]
            //  [Group, 1], [Features, 0.4421779]
            //  [Group, 1], [Features, 0.2737045]

            //  The data in the Test split.
            //  [Group, 0], [Features, 0.7262433]
            //  [Group, 0], [Features, 0.7680227]
            //  [Group, 0], [Features, 0.2060332]
            //  [Group, 0], [Features, 0.9060271]
            //  [Group, 0], [Features, 0.9775497]

            // Example of a split without specifying a sampling key column.
            split = mlContext.Data.TrainTestSplit(dataview, testFraction: 0.2);
            trainSet = mlContext.Data
                .CreateEnumerable<DataPoint>(split.TrainSet, reuseRowObject: false);

            testSet = mlContext.Data
                .CreateEnumerable<DataPoint>(split.TestSet, reuseRowObject: false);

            PrintPreviewRows(trainSet, testSet);

            // The data in the Train split.
            // [Group, 0], [Features, 0.7262433]
            // [Group, 1], [Features, 0.8173254]
            // [Group, 0], [Features, 0.7680227]
            // [Group, 1], [Features, 0.5581612]
            // [Group, 0], [Features, 0.2060332]
            // [Group, 1], [Features, 0.4421779]
            // [Group, 0], [Features, 0.9775497]
            // [Group, 1], [Features, 0.2737045]

            // The data in the Test split.
            // [Group, 1], [Features, 0.5588848]
            // [Group, 0], [Features, 0.9060271]

        }

        private static IEnumerable<DataPoint> GenerateRandomDataPoints(int count,
            int seed = 0)

        {
            var random = new Random(seed);
            for (int i = 0; i < count; i++)
            {
                yield return new DataPoint
                {
                    Group = i % 2,

                    // Create random features that are correlated with label.
                    Features = (float)random.NextDouble()
                };
            }
        }

        // Example with label and group column. A data set is a collection of such
        // examples.
        private class DataPoint
        {
            public float Group { get; set; }

            public float Features { get; set; }
        }

        // print helper
        private static void PrintPreviewRows(IEnumerable<DataPoint> trainSet,
            IEnumerable<DataPoint> testSet)

        {

            Console.WriteLine($"The data in the Train split.");
            foreach (var row in trainSet)
                Console.WriteLine($"{row.Group}, {row.Features}");

            Console.WriteLine($"\nThe data in the Test split.");
            foreach (var row in testSet)
                Console.WriteLine($"{row.Group}, {row.Features}");
        }
    }
}
