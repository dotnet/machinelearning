using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;

namespace Samples.Dynamic.Transforms.TreeFeaturization
{
    public static class FastTreeRegressionFeaturizationWithOptions
    {
        // This example requires installation of additional NuGet package
        // <a href="https://www.nuget.org/packages/Microsoft.ML.FastTree/">Microsoft.ML.FastTree</a>. 
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for
            // exception tracking and logging, as a catalog of available operations
            // and as the source of randomness. Setting the seed to a fixed number
            // in this example to make outputs deterministic.
            var mlContext = new MLContext(seed: 0);

            // Create a list of training data points.
            var dataPoints = GenerateRandomDataPoints(100).ToList();

            // Convert the list of data points to an IDataView object, which is
            // consumable by ML.NET API.
            var dataView = mlContext.Data.LoadFromEnumerable(dataPoints);

            // ML.NET doesn't cache data set by default. Therefore, if one reads a
            // data set from a file and accesses it many times, it can be slow due
            // to expensive featurization and disk operations. When the considered
            // data can fit into memory, a solution is to cache the data in memory.
            // Caching is especially helpful when working with iterative algorithms 
            // which needs many data passes.
            dataView = mlContext.Data.Cache(dataView);

            // Define input and output columns of tree-based featurizer.
            string labelColumnName = nameof(DataPoint.Label);
            string featureColumnName = nameof(DataPoint.Features);
            string treesColumnName = nameof(TransformedDataPoint.Trees);
            string leavesColumnName = nameof(TransformedDataPoint.Leaves);
            string pathsColumnName = nameof(TransformedDataPoint.Paths);

            // Define the configuration of the trainer used to train a tree-based
            // model.
            var trainerOptions = new FastTreeRegressionTrainer.Options
            {
                // Only use 80% of features to reduce over-fitting.
                FeatureFraction = 0.8,
                // Create a simpler model by penalizing usage of new features.
                FeatureFirstUsePenalty = 0.1,
                // Reduce the number of trees to 3.
                NumberOfTrees = 3,
                // Number of leaves per tree.
                NumberOfLeaves = 6,
                LabelColumnName = labelColumnName,
                FeatureColumnName = featureColumnName
            };

            // Define the tree-based featurizer's configuration.
            var options = new FastTreeRegressionFeaturizationEstimator.Options
            {
                InputColumnName = featureColumnName,
                TreesColumnName = treesColumnName,
                LeavesColumnName = leavesColumnName,
                PathsColumnName = pathsColumnName,
                TrainerOptions = trainerOptions
            };

            // Define the featurizer.
            var pipeline = mlContext.Transforms.FeaturizeByFastTreeRegression(
                options);

            // Train the model.
            var model = pipeline.Fit(dataView);

            // Create testing data. Use different random seed to make it different
            // from training data.
            var transformed = model.Transform(dataView);

            // Convert IDataView object to a list. Each element in the resulted list
            // corresponds to a row in the IDataView.
            var transformedDataPoints = mlContext.Data.CreateEnumerable<
                TransformedDataPoint>(transformed, false).ToList();

            // Print out the transformation of the first 3 data points.
            for (int i = 0; i < 3; ++i)
            {
                var dataPoint = dataPoints[i];
                var transformedDataPoint = transformedDataPoints[i];
                Console.WriteLine("The original feature vector [" + String.Join(",",
                    dataPoint.Features) + "] is transformed to three different " +
                    "tree-based feature vectors:");

                Console.WriteLine("  Trees' output values: [" + String.Join(",",
                    transformedDataPoint.Trees) + "].");

                Console.WriteLine("  Leave IDs' 0-1 representation: [" + String
                    .Join(",", transformedDataPoint.Leaves) + "].");

                Console.WriteLine("  Paths IDs' 0-1 representation: [" + String
                    .Join(",", transformedDataPoint.Paths) + "].");
            }

            // Expected output:
            //   The original feature vector [1.543569,1.494266,1.284405] is
            //   transformed to three different tree-based feature vectors:
            //     Trees' output values: [0.1507567,0.1372715,0.1019326].
            //     Leave IDs' 0-1 representation: [0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0].
            //     Paths IDs' 0-1 representation: [1,0,0,0,0,1,1,1,0,0,1,1,1,1,0].
            //   The original feature vector [0.764918,1.11206,0.648211] is
            //   transformed to three different tree-based feature vectors:
            //     Trees' output values: [0.07604675,0.08244576,0.03080027].
            //     Leave IDs' 0-1 representation: [0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1].
            //     Paths IDs' 0-1 representation: [1,1,1,0,0,1,1,0,0,0,1,0,0,0,1].
            //   The original feature vector [1.251254,1.269456,1.444864] is
            //   transformed to three different tree-based feature vectors:
            //     Trees' output values: [0.1507567,0.1090626,0.0731837].
            //     Leave IDs' 0-1 representation: [0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0].
            //     Paths IDs' 0-1 representation: [1,0,0,0,0,1,1,1,0,0,1,1,1,1,0].
        }

        private static IEnumerable<DataPoint> GenerateRandomDataPoints(int count,
            int seed = 0)
        {
            var random = new Random(seed);
            for (int i = 0; i < count; i++)
            {
                float label = (float)random.NextDouble();
                yield return new DataPoint
                {
                    Label = label,
                    // Create random features that are correlated with the label.
                    Features = Enumerable.Repeat(label, 3).Select(x => x +
                        (float)random.NextDouble()).ToArray()
                };
            }
        }

        // Example with label and 50 feature values. A data set is a collection of
        // such examples.
        private class DataPoint
        {
            public float Label { get; set; }
            [VectorType(3)]
            public float[] Features { get; set; }
        }

        // Class used to capture the output of tree-base featurization.
        private class TransformedDataPoint : DataPoint
        {
            // The i-th value is the output value of the i-th decision tree.
            public float[] Trees { get; set; }
            // The 0-1 encoding of leaves the input feature vector falls into.
            public float[] Leaves { get; set; }
            // The 0-1 encoding of paths the input feature vector reaches the
            // leaves.
            public float[] Paths { get; set; }
        }
    }
}

