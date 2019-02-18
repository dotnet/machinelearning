using System;
using System.Collections.Generic;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class FastTreeRegression
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext();

            // Get a small dataset as an IEnumerable and convert it to an IDataView.
            var data = SamplesUtils.DatasetUtils.GetInfertData();
            var trainData = ml.Data.ReadFromEnumerable(data);

            // Preview of the data.
            //
            // Age  Case  Education  Induced     Parity  PooledStratum  RowNum  ...
            // 26   1       0-5yrs      1         6         3             1  ...
            // 42   1       0-5yrs      1         1         1             2  ...
            // 39   1       0-5yrs      2         6         4             3  ...
            // 34   1       0-5yrs      2         4         2             4  ...
            // 35   1       6-11yrs     1         3         32            5  ...

            // A pipeline for concatenating the Parity and Induced columns together in the Features column.
            // We will train a FastTreeRegression model with 1 tree on these two columns to predict Age.
            string outputColumnName = "Features";
            var pipeline = ml.Transforms.Concatenate(outputColumnName, new[] { "Parity", "Induced" })
                .Append(ml.Regression.Trainers.FastTree(labelColumn: "Age", featureColumn: outputColumnName, numTrees: 1, numLeaves: 2, minDatapointsInLeaves: 1));

            var model = pipeline.Fit(trainData);

            // Get the trained model parameters.
            var modelParams = model.LastTransformer.Model;

            // Let's see where an example with Parity = 1 and Induced = 1 would end up in the single trained tree.
            var testRow = new VBuffer<float>(2, new[] { 1.0f, 1.0f });
            // Use the path object to pass to GetLeaf, which will populate path with the IDs of th nodes from root to leaf.
            List<int> path = default;
            // Get the ID of the leaf this example ends up in tree 0.
            var leafID = modelParams.GetLeaf(0, in testRow, ref path);
            // Get the leaf value for this leaf ID in tree 0.
            var leafValue = modelParams.GetLeafValue(0, leafID);
            Console.WriteLine("The leaf value in tree 0 is: " + leafValue);
        }
    }
}
