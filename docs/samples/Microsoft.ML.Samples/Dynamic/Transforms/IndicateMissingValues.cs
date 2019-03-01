using System;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class IndicateMissingValues
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            // Create a TextLoader for the breast-cancer dataset, and load it into an IDataView.
            var loader = mlContext.Data.CreateTextLoader(new[]
            {
                new TextLoader.Column("Label", DataKind.Single, 0),
                new TextLoader.Column("Features1", DataKind.Single, 1, 5),
                new TextLoader.Column("Features2", DataKind.Single, 7, 9),
                new TextLoader.Column("Missing", DataKind.Single, 6)
            });
            var file = SamplesUtils.DatasetUtils.DownloadBreastCancerDataset();
            var data = loader.Load(new MultiFileSource(file));

            // Preview of the data. Some rows have a missing value in column 6.
            //
            // 0   5   1   1   1   2   1   3   1   1
            // 0   5   4   4   5   7   10  3   2   1
            // 0   3   1   1   1   2   2   3   1   1
            // 0   6   8   8   1   3   4   3   7   1
            // 0   4   1   1   3   2   1   3   1   1
            // ...
            // 1   8   4   5   1   2   ?   7   3   1

            // IndicateMissingValues is used to create a boolean column containing
            // 'true' where the value in the input column is NaN. This value can be used
            // to replace missing values with other values.
            // In this example, we replace the missing value with the vector (1, -1), and non-missing
            // values with the vector (0, x) (where x is the value in the input column).
            IEstimator<ITransformer> pipeline = mlContext.Transforms.IndicateMissingValues("MissingIndicator", "Missing");
            pipeline = pipeline.Append(mlContext.Transforms.CustomMapping<MissingValue, ReplacedMissingValue>(
                (m, r) => r.MissingReplaced = new float[2] { m.MissingIndicator ? 1 : 0, m.MissingIndicator ? -1 : m.Missing }, null));
            pipeline = pipeline.Append(mlContext.Transforms.Concatenate("Features", "Features1", "Features2", "MissingReplaced"));

            // Now we can transform the data and look at the output to confirm the behavior of IndicateMissingValues.
            // Don't forget that this operation doesn't actually evaluate data until we read the data below.
            var model = pipeline.Fit(data);
            var transformedData = model.Transform(data);

            // We can extract the newly created column as an IEnumerable of SampleDataTransformed, the class we define below.
            var rowEnumerable = mlContext.Data.CreateEnumerable<SampleDataTransformed>(transformedData, reuseRowObject: false);

            // And finally, we can write out the rows of the dataset, looking at the columns of interest.
            Console.WriteLine($"Missing, MissingIndicator, and MissingReplaced columns obtained post-transformation.");
            foreach (var row in rowEnumerable.Skip(20).Take(5))
            {
                Console.WriteLine($"Missing: {row.Missing} MissingIndicator: {row.MissingIndicator} MissingReplaced: ({row.MissingReplaced[0]}, {row.MissingReplaced[1]})");
            }

            // Expected output:
            // Missing, MissingIndicator, and MissingReplaced columns obtained post - transformation.
            // Missing: 10 MissingIndicator: False MissingReplaced: (0, 10)
            // Missing: 7 MissingIndicator: False MissingReplaced: (0, 7)
            // Missing: 1 MissingIndicator: False MissingReplaced: (0, 1)
            // Missing: NaN MissingIndicator: True MissingReplaced: (1, -1)
            // Missing: 1 MissingIndicator: False MissingReplaced: (0, 1)
        }

        private class MissingValue
        {
            public float Missing { get; set; }
            public bool MissingIndicator { get; set; }
        }

        private class ReplacedMissingValue
        {
            [VectorType(2)]
            public float[] MissingReplaced { get; set; }
        }

        private class SampleDataTransformed
        {
            public float Missing { get; set; }
            public bool MissingIndicator { get; set; }
            public float[] MissingReplaced { get; set; }
        }
    }
}
