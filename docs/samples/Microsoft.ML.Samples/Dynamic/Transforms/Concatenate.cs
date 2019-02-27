using System;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class ConcatTransform
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable and them read it as ML.NET's data type.
            var data = SamplesUtils.DatasetUtils.GetInfertData();
            var trainData = mlContext.Data.LoadFromEnumerable(data);

            // Preview of the data.
            //
            // Age    Case  Education  induced     parity  pooled.stratum  row_num  ...
            // 26.0   1.0   0-5yrs      1.0         6.0       3.0      1.0  ...
            // 42.0   1.0   0-5yrs      1.0         1.0       1.0      2.0  ...
            // 39.0   1.0   0-5yrs      2.0         6.0       4.0      3.0  ...
            // 34.0   1.0   0-5yrs      2.0         4.0       2.0      4.0  ...
            // 35.0   1.0   6-11yrs     1.0         3.0       32.0     5.0  ...

            // A pipeline for concatenating the Age, Parity and Induced columns together into a vector that will be the Features column.
            // Concatenation is necessary because learners take **feature vectors** as inputs.
            //   e.g. var regressionTrainer = mlContext.Regression.Trainers.FastTree(labelColumn: "Label", featureColumn: "Features");
            string outputColumnName = "Features";
            var pipeline = mlContext.Transforms.Concatenate(outputColumnName, new[] { "Age", "Parity", "Induced" });

            // The transformed data.
            var transformedData = pipeline.Fit(trainData).Transform(trainData);

            // Now let's take a look at what this concatenation did.
            // We can extract the newly created column as an IEnumerable of SampleInfertDataWithFeatures, the class we define above.
            var featuresColumn = mlContext.Data.CreateEnumerable<SampleInfertDataWithFeatures>(transformedData, reuseRowObject: false);

            // And we can write out a few rows
            Console.WriteLine($"{outputColumnName} column obtained post-transformation.");
            foreach (var featureRow in featuresColumn)
            {
                foreach (var value in featureRow.Features.GetValues())
                    Console.Write($"{value} ");
                Console.WriteLine("");
            }

            // Expected output:
            // Features column obtained post-transformation.
            //
            // 26 6 1
            // 42 1 1
            // 39 6 2
            // 34 4 2
            // 35 3 1
        }

        private class SampleInfertDataWithFeatures
        {
            public VBuffer<float> Features { get; set; }
        }
    }
}
