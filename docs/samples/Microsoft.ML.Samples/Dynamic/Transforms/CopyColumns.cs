using System;
using System.Collections.Generic;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class CopyColumns
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable and them read it as ML.NET's data type.
            IEnumerable<SamplesUtils.DatasetUtils.SampleInfertData> data = SamplesUtils.DatasetUtils.GetInfertData();
            var trainData = mlContext.Data.LoadFromEnumerable(data);

            // Preview of the data.
            //
            // Age    Case  Education  induced     parity  pooled.stratum  row_num  ...
            // 26.0   1.0   0-5yrs      1.0         6.0       3.0      1.0  ...
            // 42.0   1.0   0-5yrs      1.0         1.0       1.0      2.0  ...
            // 39.0   1.0   0-5yrs      2.0         6.0       4.0      3.0  ...
            // 34.0   1.0   0-5yrs      2.0         4.0       2.0      4.0  ...
            // 35.0   1.0   6-11yrs     1.0         3.0       32.0     5.0  ...

            // CopyColumns is commonly used to rename columns.
            // For example, if you want to train towards Age, and your learner expects a "Label" column, you can
            // use CopyColumns to rename Age to Label. Technically, the Age columns still exists, but it won't be
            // materialized unless you actually need it somewhere (e.g. if you were to save the transformed data
            // without explicitly dropping the column). This is a general property of IDataView's lazy evaluation.
            string labelColumnName = "Label";
            var pipeline = mlContext.Transforms.CopyColumns(labelColumnName, "Age") as IEstimator<ITransformer>;

            // You also may want to copy a column to perform some hand-featurization using built-in transforms or
            // a CustomMapping transform. For example, we could make an indicator variable if a feature, such as Parity
            // goes above some threshold. We simply copy the Parity column to a new column, then pass it through a custom function.
            Action<InputRow, OutputRow> mapping = (input, output) =>output.CustomValue = input.CustomValue > 4 ? 1 : 0;
            pipeline = pipeline.Append(mlContext.Transforms.CopyColumns("CustomValue", "Parity"))
                .Append(mlContext.Transforms.CustomMapping(mapping, null));

            // Now we can transform the data and look at the output to confirm the behavior of CopyColumns.
            // Don't forget that this operation doesn't actually evaluate data until we read the data below.
            var transformedData = pipeline.Fit(trainData).Transform(trainData);

            // We can extract the newly created column as an IEnumerable of SampleInfertDataTransformed, the class we define below.
            var rowEnumerable = mlContext.Data.CreateEnumerable<SampleInfertDataTransformed>(transformedData, reuseRowObject: false);

            // And finally, we can write out the rows of the dataset, looking at the columns of interest.
            Console.WriteLine($"Label, Parity, and CustomValue columns obtained post-transformation.");
            foreach (var row in rowEnumerable)
            {
                Console.WriteLine($"Label: {row.Label} Parity: {row.Parity} CustomValue: {row.CustomValue}");
            }

            // Expected output:
            //  Label, Parity, and CustomValue columns obtained post-transformation.
            //  Label: 26 Parity: 6 CustomValue: 1
            //  Label: 42 Parity: 1 CustomValue: 0
            //  Label: 39 Parity: 6 CustomValue: 1
            //  Label: 34 Parity: 4 CustomValue: 0
            //  Label: 35 Parity: 3 CustomValue: 0
        }

        private class SampleInfertDataTransformed
        {
            public float Label { get; set; }
            public float Parity { get; set; }
            public float CustomValue { get; set; }
        }

        private class OutputRow
        {
            public float CustomValue { get; set; }
        }

        private class InputRow
        {
            public float CustomValue { get; set; }
        }
    }
}
