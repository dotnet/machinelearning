using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;

namespace Samples.Dynamic.Transforms.TreeFeaturization
{
    public static class PretrainedTreeEnsembleFeaturizationWithOptions
    {
        public static void Example()
        {
            // Create data set
            int dataPointCount = 200;
            // Create a new context for ML.NET operations. It can be used for
            // exception tracking and logging, as a catalog of available operations
            // and as the source of randomness. Setting the seed to a fixed number
            // in this example to make outputs deterministic.
            var mlContext = new MLContext(seed: 0);

            // Create a list of training data points.
            var dataPoints = GenerateRandomDataPoints(dataPointCount).ToList();

            // Convert the list of data points to an IDataView object, which is
            // consumable by ML.NET API.
            var dataView = mlContext.Data.LoadFromEnumerable(dataPoints);

            // Define input and output columns of tree-based featurizer.
            string labelColumnName = nameof(DataPoint.Label);
            string featureColumnName = nameof(DataPoint.Features);
            string treesColumnName = nameof(TransformedDataPoint.Trees);
            string leavesColumnName = nameof(TransformedDataPoint.Leaves);
            string pathsColumnName = nameof(TransformedDataPoint.Paths);

            // Define a tree model whose trees will be extracted to construct a tree
            // featurizer.
            var trainer = mlContext.BinaryClassification.Trainers.FastTree(
                new FastTreeBinaryTrainer.Options
                {
                    NumberOfThreads = 1,
                    NumberOfTrees = 1,
                    NumberOfLeaves = 4,
                    MinimumExampleCountPerLeaf = 1,
                    FeatureColumnName = featureColumnName,
                    LabelColumnName = labelColumnName
                });

            // Train the defined tree model.
            var model = trainer.Fit(dataView);
            var predicted = model.Transform(dataView);

            // Define the configuration of tree-based featurizer.
            var options = new PretrainedTreeFeaturizationEstimator.Options()
            {
                InputColumnName = featureColumnName,
                ModelParameters = model.Model.SubModel, // Pretrained tree model.
                TreesColumnName = treesColumnName,
                LeavesColumnName = leavesColumnName,
                PathsColumnName = pathsColumnName
            };

            // Fit the created featurizer. It doesn't perform actual training
            // because a pretrained model is provided.
            var treeFeaturizer = mlContext.Transforms
                .FeaturizeByPretrainTreeEnsemble(options).Fit(dataView);

            // Apply TreeEnsembleFeaturizer to the input data.
            var transformed = treeFeaturizer.Transform(dataView);

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
            //  The original feature vector[0.8173254, 0.7680227, 0.5581612] is
            //  transformed to three different tree - based feature vectors:
            //    Trees' output values: [0.4172185].
            //    Leave IDs' 0-1 representation: [1,0,0,0].
            //    Paths IDs' 0-1 representation: [1,1,1].
            //  The original feature vector[0.7588848, 1.106027, 0.6421779] is
            //   transformed to three different tree - based feature vectors:
            //    Trees' output values: [-1].
            //    Leave IDs' 0-1 representation: [0,0,1,0].
            //    Paths IDs' 0-1 representation: [1,1,0].
            //  The original feature vector[0.2737045, 0.2919063, 0.4673147] is
            //   transformed to three different tree - based feature vectors:
            //    Trees' output values: [0.4172185].
            //    Leave IDs' 0-1 representation: [1,0,0,0].
            //    Paths IDs' 0-1 representation: [1,1,1].
            //
            //   Note that the trained model contains only one tree.
            //
            //            Node 0
            //            /    \
            //           /    Leaf -2
            //         Node 1
            //         /    \
            //        /    Leaf -3
            //      Node 2
            //      /    \
            //     /    Leaf -4
            //   Leaf -1
            //
            //   Thus, if a data point reaches Leaf indexed by -1, its 0-1 path
            //   representation may be [1,1,1] because that data point
            //   went through all Node 0, Node 1, and Node 2.

        }

        private static IEnumerable<DataPoint> GenerateRandomDataPoints(int count,
            int seed = 0)
        {
            var random = new Random(seed);
            float randomFloat() => (float)random.NextDouble();
            for (int i = 0; i < count; i++)
            {
                var label = randomFloat() > 0.5;
                yield return new DataPoint
                {
                    Label = label,
                    // Create random features that are correlated with the label.
                    // For data points with false label, the feature values are
                    // slightly increased by adding a constant.
                    Features = Enumerable.Repeat(label, 3).Select(x => x ?
                    randomFloat() : randomFloat() + 0.2f).ToArray()
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
