using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Normalizers;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class NormalizerTransform
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext();

            // Get a small dataset as an IEnumerable and convert it to an IDataView.
            IEnumerable<SamplesUtils.DatasetUtils.SampleInfertData> data = SamplesUtils.DatasetUtils.GetInfertData();
            var trainData = ml.Data.ReadFromEnumerable(data);

            // Preview of the data.
            //
            // Age  Case  Education  Induced     Parity  PooledStratum  RowNum  ...
            // 26   1       0-5yrs      1         6         3             1  ...
            // 42   1       0-5yrs      1         1         1             2  ...
            // 39   1       0-5yrs      2         6         4             3  ...
            // 34   1       0-5yrs      2         4         2             4  ...
            // 35   1       6-11yrs     1         3         32            5  ...

            // A pipeline for normalizing the Induced column.
            var pipeline = ml.Transforms.Normalize("Induced");
            // The transformed (normalized according to Normalizer.NormalizerMode.MinMax) data.
            var transformer = pipeline.Fit(trainData);

            var modelParams = transformer.Columns
                                         .First(x => x.Name == "Induced")
                                         .ModelParameters as NormalizingTransformer.AffineNormalizerModelParameters<float>;

            Console.WriteLine($"The normalization parameters are: Scale = {modelParams.Scale} and Offset = {modelParams.Offset}");
            //Preview 
            //
            //The normalization parameters are: Scale = 0.5 and Offset = 0"

            var transformedData = transformer.Transform(trainData);

            // Getting the data of the newly created column, so we can preview it.
            var normalizedColumn = transformedData.GetColumn<float>(ml, "Induced");

            // A small printing utility.
            Action<string, IEnumerable<float>> printHelper = (colName, column) =>
            {
                Console.WriteLine($"{colName} column obtained post-transformation.");
                foreach (var row in column)
                    Console.WriteLine($"{row} ");
            };

            printHelper("Induced", normalizedColumn);

            // Induced column obtained post-transformation.
            //
            // 0.5
            // 0.5
            // 1
            // 1
            // 0.5

            // Composing a different pipeline if we wanted to normalize more than one column at a time. 
            // Using log scale as the normalization mode. 
            var multiColPipeline = ml.Transforms.Normalize(NormalizingEstimator.NormalizerMode.LogMeanVariance, new SimpleColumnInfo[] { ("LogInduced", "Induced"), ("LogSpontaneous", "Spontaneous") });
            // The transformed data.
            var multiColtransformer = multiColPipeline.Fit(trainData);
            var multiColtransformedData = multiColtransformer.Transform(trainData);

            // Getting the newly created columns. 
            var normalizedInduced = multiColtransformedData.GetColumn<float>(ml, "LogInduced");
            var normalizedSpont = multiColtransformedData.GetColumn<float>(ml, "LogSpontaneous");

            printHelper("LogInduced", normalizedInduced);

            // LogInduced column obtained post-transformation.
            //
            // 0.2071445
            // 0.2071445
            // 0.889631
            // 0.889631
            // 0.2071445

            printHelper("LogSpontaneous", normalizedSpont);

            // LogSpontaneous column obtained post-transformation.
            //
            // 0.8413026
            // 0
            // 0
            // 0
            // 0.1586974
            
            // Inspect the weights of normalizing the columns
            var multiColModelParams = multiColtransformer.Columns
                .First(x=> x.Name == "LogInduced")
                .ModelParameters as NormalizingTransformer.CdfNormalizerModelParameters<float>;

            Console.WriteLine($"The normalization parameters are: Mean = {multiColModelParams.Mean} and Stddev = {multiColModelParams.Stddev}");
        }
    }
}
