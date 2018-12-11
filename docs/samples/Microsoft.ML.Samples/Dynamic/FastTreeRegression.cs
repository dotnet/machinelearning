using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Samples.Dynamic
{
    public class FastTreeRegressionExample
    {
        public static void FastTreeRegression()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext();

            // Get a small dataset as an IEnumerable and convert it to an IDataView.
            var data = SamplesUtils.DatasetUtils.GetInfertData();
            var trainData = ml.CreateStreamingDataView(data);

            // Preview of the data.
            //
            // Age  Case  Education  Induced     Parity  PooledStratum  RowNum  ...
            // 26   1       0-5yrs      1         6         3             1  ...
            // 42   1       0-5yrs      1         1         1             2  ...
            // 39   1       0-5yrs      2         6         4             3  ...
            // 34   1       0-5yrs      2         4         2             4  ...
            // 35   1       6-11yrs     1         3         32            5  ...

            // A pipeline for concatenating the parity and induced columns together in the Features column and training a FastTreeRegression model on them.
            string outputColumnName = "Features";
            var pipeline = ml.Transforms.Concatenate(outputColumnName, new[] { "Parity", "Induced" })
                .Append(ml.Regression.Trainers.FastTree(labelColumn: "Age", featureColumn: outputColumnName, numTrees: 2, numLeaves: 2, minDatapointsInLeaves: 1));

            var model = pipeline.Fit(trainData);

            // Get the trained model parameters.
            var modelParams = model.LastTransformer.Model;

            // Get the leaf and the leaf value for a row of data with Parity = 1, Induced = 1 in the first tree.
            var testRow = new VBuffer<float>(2, new[] { 1.0f, 1.0f });
            List<int> path = default;
            var leaf = modelParams.GetLeaf(0, in testRow, ref path);
            var leafValue = modelParams.GetLeafValue(0, leaf);
            Console.WriteLine("The leaf value in tree 0 is: " + leafValue);                        
        }
    }
}
