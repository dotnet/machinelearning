using Microsoft.ML.Data;
using System;
using System.Linq;

namespace Microsoft.ML.Samples.Dynamic
{
    public class SDCARegressionExample
    {
        public static void SDCARegression()
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

            // A pipeline for concatenating the Parity and Induced columns together in the Features column.
            // We will train a FastTreeRegression model with 1 tree on these two columns to predict Age.
            string outputColumnName = "Features";
            var pipeline = ml.Transforms.Concatenate(outputColumnName, new[] { "Parity", "Induced" })
                .Append(ml.Regression.Trainers.StochasticDualCoordinateAscent(labelColumn: "Age", featureColumn: outputColumnName, maxIterations:2));

            var model = pipeline.Fit(trainData);

            // Get the trained model parameters.
            var modelParams = model.LastTransformer.Model;
            // Inspect the bias and model weights.
            Console.WriteLine("The bias term is: " + modelParams.Bias);
            Console.WriteLine("The feature weights are: " + string.Join(", ", modelParams.Weights.ToArray()));
        }
    }
}
