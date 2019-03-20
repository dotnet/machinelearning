using System;
using System.Linq;
using Microsoft.ML.Trainers;
using Microsoft.ML.SamplesUtils;

namespace Microsoft.ML.Samples.Dynamic.PermutationFeatureImportance
{
    public static class PfiHelper
    {
        public static IDataView GetHousingRegressionIDataView(MLContext mlContext, out string labelName, out string[] featureNames, bool binaryPrediction = false)
        {
            // Read the Housing regression dataset
            var data = DatasetUtils.LoadHousingRegressionDataset(mlContext);

            // Define the label column
            var labelColumn = "MedianHomeValue";

            if (binaryPrediction)
            {
                labelColumn = nameof(BinaryOutputRow.AboveAverage);
                data = mlContext.Transforms.CustomMapping(GreaterThanAverage, null).Fit(data).Transform(data);
                data = mlContext.Transforms.DropColumns("MedianHomeValue").Fit(data).Transform(data);
            }

            labelName = labelColumn;
            featureNames = data.Schema.AsEnumerable()
                .Select(column => column.Name) // Get the column names
                .Where(name => name != labelColumn) // Drop the Label
                .ToArray();

            return data;
        }

        // Define a class for all the input columns that we intend to consume.
        private class ContinuousInputRow
        {
            public float MedianHomeValue { get; set; }
        }

        // Define a class for all output columns that we intend to produce.
        private class BinaryOutputRow
        {
            public bool AboveAverage { get; set; }
        }

        // Define an Action to apply a custom mapping from one object to the other
        private readonly static Action<ContinuousInputRow, BinaryOutputRow> GreaterThanAverage = (input, output) 
            => output.AboveAverage = input.MedianHomeValue > 22.6;

        public static float[] GetLinearModelWeights(OlsModelParameters linearModel)
        {
            return linearModel.Weights.ToArray();
        }

        public static float[] GetLinearModelWeights(LinearBinaryModelParameters linearModel)
        {
            return linearModel.Weights.ToArray();
        }
    }
}
