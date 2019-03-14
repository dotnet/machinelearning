using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using static Microsoft.ML.Transforms.OneHotEncodingEstimator;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class OneHotEncoding
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable.
            var samples = new List<DataPoint>()
            {
                new DataPoint(){ Label = 0, Education = "0-5yrs" },
                new DataPoint(){ Label = 1, Education = "0-5yrs" },
                new DataPoint(){ Label = 45, Education = "6-11yrs" },
                new DataPoint(){ Label = 50, Education = "6-11yrs" },
                new DataPoint(){ Label = 50, Education = "11-15yrs" },
            };

            // Convert training data to IDataView.
            var trainData = mlContext.Data.LoadFromEnumerable(samples);

            // A pipeline for one hot encoding the Education column.
            var bagPipeline = mlContext.Transforms.Categorical.OneHotEncoding("EducationOneHotEncoded", "Education", OutputKind.Bag);
            // Fit to data.
            var bagTransformer = bagPipeline.Fit(trainData);

            // Get transformed data
            var bagTransformedData = bagTransformer.Transform(trainData);
            // Getting the data of the newly created column, so we can preview it.
            var bagEncodedColumn = bagTransformedData.GetColumn<float[]>("EducationOneHotEncoded");

            var keyPipeline = mlContext.Transforms.Categorical.OneHotEncoding("EducationOneHotEncoded", "Education", OutputKind.Key);
            // Fit to data.
            var keyTransformer = keyPipeline.Fit(trainData);

            // Get transformed data
            var keyTransformedData = keyTransformer.Transform(trainData);
            // Getting the data of the newly created column, so we can preview it.
            var keyEncodedColumn = keyTransformedData.GetColumn<uint>("EducationOneHotEncoded");

            Console.WriteLine("One Hot Encoding based on the bagging strategy.");
            foreach (var row in bagEncodedColumn)
            {
                for (var i = 0; i < row.Length; i++)
                    Console.Write($"{row[i]} ");
            }

            // data column obtained post-transformation.
            // Since there are only two categories in the Education column of the trainData, the output vector
            // for one hot will have two slots.
            //
            // 0 0 0
            // 0 0 0
            // 0 0 1
            // 0 0 1
            // 0 1 0

            Console.WriteLine("One Hot Encoding with key type output.");
            foreach (var element in keyEncodedColumn)
                Console.WriteLine(element);

            // 1
            // 1
            // 2
            // 2
            // 3

        }

        private class DataPoint
        {
            public float Label { get; set; }

            public string Education { get; set; }
        }
    }
}