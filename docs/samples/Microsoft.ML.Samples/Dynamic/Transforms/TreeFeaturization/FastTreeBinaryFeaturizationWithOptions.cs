using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;

namespace Samples.Dynamic.Transforms.TreeFeaturization
{
    public static class FastTreeBinaryFeaturizationWithOptions
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

            // Create a list of data points to be transformed.
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
            var trainerOptions = new FastTreeBinaryTrainer.Options
            {
                // Use L2Norm for early stopping.
                EarlyStoppingMetric = EarlyStoppingMetric.L2Norm,
                // Create a simpler model by penalizing usage of new features.
                FeatureFirstUsePenalty = 0.1,
                // Reduce the number of trees to 3.
                NumberOfTrees = 3,
                // Number of leaves per tree.
                NumberOfLeaves = 6,
                // Feature column name.
                FeatureColumnName = featureColumnName,
                // Label column name.
                LabelColumnName = labelColumnName
            };

            // Define the tree-based featurizer's configuration.
            var options = new FastTreeBinaryFeaturizationEstimator.Options
            {
                InputColumnName = featureColumnName,
                TreesColumnName = treesColumnName,
                LeavesColumnName = leavesColumnName,
                PathsColumnName = pathsColumnName,
                TrainerOptions = trainerOptions
            };

            // Define the featurizer.
            var pipeline = mlContext.Transforms.FeaturizeByFastTreeBinary(
                options);

            // Train the model.
            var model = pipeline.Fit(dataView);

            // Apply the trained transformer to the considered data set.
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
                Console.WriteLine("The original feature vector [" + String.Join(
                    ",", dataPoint.Features) + "] is transformed to three " +
                    "different tree-based feature vectors:");

                Console.WriteLine("  Trees' output values: [" + String.Join(",",
                    transformedDataPoint.Trees) + "].");

                Console.WriteLine("  Leave IDs' 0-1 representation: [" + String
                    .Join(",", transformedDataPoint.Leaves) + "].");

                Console.WriteLine("  Paths IDs' 0-1 representation: [" + String
                    .Join(",", transformedDataPoint.Paths) + "].");
            }

            // Expected output:
            //   The original feature vector [0.8173254,0.7680227,0.5581612] is
            //   transformed to three different tree-based feature vectors:
            //     Trees' output values: [0.5714286,0.4636412,0.535588].
            //     Leave IDs' 0-1 representation: [0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1].
            //     Paths IDs' 0-1 representation: [1,0,0,1,1,1,0,1,0,1,1,1,1,1,1].
            //   The original feature vector [0.5888848,0.9360271,0.4721779] is
            //   transformed to three different tree-based feature vectors:
            //     Trees' output values: [0.2352941,-0.1382389,0.535588].
            //     Leave IDs' 0-1 representation: [0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1].
            //     Paths IDs' 0-1 representation: [1,0,0,1,1,1,0,1,0,1,1,1,1,1,1].
            //   The original feature vector [0.2737045,0.2919063,0.4673147] is
            //   transformed to three different tree-based feature vectors:
            //     Trees' output values: [0.2352941,-0.1382389,-0.2184284].
            //     Leave IDs' 0-1 representation: [0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0].
            //     Paths IDs' 0-1 representation: [1,0,0,1,1,1,0,1,0,1,1,1,0,0,0].
        }

        private static IEnumerable<DataPoint> GenerateRandomDataPoints(int count,
            int seed = 0)
        {
            var random = new Random(seed);
            float randomFloat() => (float)random.NextDouble();
            for (int i = 0; i < count; i++)
            {
                var label = randomFloat() > 0.5f;
                yield return new DataPoint
                {
                    Label = label,
                    // Create random features that are correlated with the label.
                    // For data points with false label, the feature values are
                    // slightly increased by adding a constant.
                    Features = Enumerable.Repeat(label, 3).Select(x => x ?
                    randomFloat() : randomFloat() + 0.03f).ToArray()
                };
            }
        }

        // Example with label and 3 feature values. A data set is a collection of
        // such examples.
        private class DataPoint
        {
            public bool Label { get; set; }
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
