using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace Samples.Dynamic
{
    public static class CountTargetEncoding
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();
            var samples = new List<DataPoint>()
            {
                new DataPoint(){ Category = "a", Label = 0 },
                new DataPoint(){ Category = "a", Label = 1 },
                new DataPoint(){ Category = "a", Label = 0 },
                new DataPoint(){ Category = "a", Label = 0 },
                new DataPoint(){ Category = "b", Label = 1 },
                new DataPoint(){ Category = "b", Label = 2 },
                new DataPoint(){ Category = "b", Label = 2 },
                new DataPoint(){ Category = "b", Label = 1 },
                new DataPoint(){ Category = "c", Label = 0 },
                new DataPoint(){ Category = "c", Label = 0 },
                new DataPoint(){ Category = "c", Label = 0 },
                new DataPoint(){ Category = "c", Label = 0 },
                new DataPoint(){ Category = "d", Label = 0 },
                new DataPoint(){ Category = "d", Label = 1 },
                new DataPoint(){ Category = "d", Label = 2 },
                new DataPoint(){ Category = "d", Label = 3 },
            };

            // Convert training data to IDataView, the general data type used in
            // ML.NET.
            var data = mlContext.Data.LoadFromEnumerable(samples);

            // Define the CountTargetEncoding estimator.
            var count = mlContext.Transforms.CountTargetEncode("Features", "Category");

            // Now we can transform the data and look at the output to confirm the
            // behavior of the estimator. This operation doesn't actually evaluate
            // data until we read the data below.
            var transformer = count.Fit(data);
            var transformedData = transformer.Transform(data);
            var column = transformedData.GetColumn<float[]>("Features").ToArray();
            foreach (var row in column)
                Console.WriteLine(string.Join(", ", row.Select(x => x.ToString("f4"))));
            // Expected output:
            // 3.0000, 1.0000, 0.0000, 0.0000, 0.8473, -1.0986, -3.2452, -4.3694, 0.0000
            // 3.0000, 1.0000, 0.0000, 0.0000, 0.8473, -1.0986, -3.2452, -4.3694, 0.0000
            // 3.0000, 1.0000, 0.0000, 0.0000, 0.8473, -1.0986, -3.2452, -4.3694, 0.0000
            // 3.0000, 1.0000, 0.0000, 0.0000, 0.8473, -1.0986, -3.2452, -4.3694, 0.0000
            // 0.0000, 2.0000, 2.0000, 0.0000, -2.1972, -0.2007, -0.2513, -4.3694, 0.0000
            // 0.0000, 2.0000, 2.0000, 0.0000, -2.1972, -0.2007, -0.2513, -4.3694, 0.0000
            // 0.0000, 2.0000, 2.0000, 0.0000, -2.1972, -0.2007, -0.2513, -4.3694, 0.0000
            // 0.0000, 2.0000, 2.0000, 0.0000, -2.1972, -0.2007, -0.2513, -4.3694, 0.0000
            // 4.0000, 0.0000, 0.0000, 0.0000, 2.1972, -2.9444, -3.2452, -4.3694, 0.0000
            // 4.0000, 0.0000, 0.0000, 0.0000, 2.1972, -2.9444, -3.2452, -4.3694, 0.0000
            // 4.0000, 0.0000, 0.0000, 0.0000, 2.1972, -2.9444, -3.2452, -4.3694, 0.0000
            // 4.0000, 0.0000, 0.0000, 0.0000, 2.1972, -2.9444, -3.2452, -4.3694, 0.0000
            // 1.0000, 1.0000, 1.0000, 1.0000, -0.8473, -1.0986, -1.1664, -1.3099, 0.0000
            // 1.0000, 1.0000, 1.0000, 1.0000, -0.8473, -1.0986, -1.1664, -1.3099, 0.0000
            // 1.0000, 1.0000, 1.0000, 1.0000, -0.8473, -1.0986, -1.1664, -1.3099, 0.0000
            // 1.0000, 1.0000, 1.0000, 1.0000, -0.8473, -1.0986, -1.1664, -1.3099, 0.0000

            // The count tables can be saved and be retrained later with additional data.
            mlContext.Model.Save(transformer, data.Schema, "CountTargetEncoding.zip");
            var loadedTransformer = mlContext.Model.Load("CountTargetEncoding.zip", out _);
             count = mlContext.Transforms.CountTargetEncode("Features", loadedTransformer as CountTargetEncodingTransformer, "Category");

            var moreSamples = new List<DataPoint>()
            {
                new DataPoint(){ Category = "a", Label = 3 },
                new DataPoint(){ Category = "a", Label = 3 },
                new DataPoint(){ Category = "b", Label = 2 },
                new DataPoint(){ Category = "c", Label = 1 },
                new DataPoint(){ Category = "c", Label = 1 },
                new DataPoint(){ Category = "d", Label = 0 },
                new DataPoint(){ Category = "e", Label = 3 },
                new DataPoint(){ Category = "d", Label = 4 },
            };
            var moreData = mlContext.Data.LoadFromEnumerable(moreSamples);
            transformer = count.Fit(moreData);
            transformedData = transformer.Transform(moreData);
            column = transformedData.GetColumn<float[]>("Features").ToArray();
            foreach (var row in column)
                Console.WriteLine(string.Join(", ", row.Select(x => x.ToString("f4"))));

            // Expected output:
            // 3.0000, 1.0000, 0.0000, 2.0000, 0.0000, -0.2151, -1.5261, -4.0073, -0.6665, -4.0073, 0.0000
            // 3.0000, 1.0000, 0.0000, 2.0000, 0.0000, -0.2151, -1.5261, -4.0073, -0.6665, -4.0073, 0.0000
            // 0.0000, 2.0000, 3.0000, 0.0000, 0.0000, -3.8501, -0.5108, 0.0834, -2.7081, -3.8501, 0.0000
            // 4.0000, 2.0000, 0.0000, 0.0000, 0.0000, 0.3610, -0.7472, -4.0073, -2.8717, -4.0073, 0.0000
            // 4.0000, 2.0000, 0.0000, 0.0000, 0.0000, 0.3610, -0.7472, -4.0073, -2.8717, -4.0073, 0.0000
            // 2.0000, 1.0000, 1.0000, 1.0000, 1.0000, -0.8303, -1.5261, -1.6529, -1.4088, -1.6529, 0.0000
            // 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, -2.7081, -1.9459, -2.7081, 0.7885, -2.7081, 0.0000
            // 2.0000, 1.0000, 1.0000, 1.0000, 1.0000, -0.8303, -1.5261, -1.6529, -1.4088, -1.6529, 0.0000
        }

        private class DataPoint
        {
            public string Category;
            public float Label;
        }
    }
}
