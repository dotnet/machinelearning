﻿using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic
{
    public sealed class VectorWhitenWithOptions
    {
        /// This example requires installation of additional nuget package
        /// <a href="https://www.nuget.org/packages/Microsoft.ML.Mkl.Components/">Microsoft.ML.Mkl.Components</a>.
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var ml = new MLContext();

            // Get a small dataset as an IEnumerable and convert it to an IDataView.
            var data = GetVectorOfNumbersData();
            var trainData = ml.Data.LoadFromEnumerable(data);

            // Preview of the data.
            //
            // Features
            // 0   1   2   3   4   5   6   7   8   9
            // 1   2   3   4   5   6   7   8   9   0  
            // 2   3   4   5   6   7   8   9   0   1
            // 3   4   5   6   7   8   9   0   1   2
            // 4   5   6   7   8   9   0   1   2   3
            // 5   6   7   8   9   0   1   2   3   4
            // 6   7   8   9   0   1   2   3   4   5

            // A small printing utility.
            Action<string, IEnumerable<VBuffer<float>>> printHelper = (colName,
                column) =>
            {
                Console.WriteLine($"{colName} column obtained" +
                    $"post-transformation.");

                foreach (var row in column)
                    Console.WriteLine(string.Join(" ", row.DenseValues().Select(x =>
                        x.ToString("f3")))+" ");
            };


            // A pipeline to project Features column into white noise vector.
            var whiteningPipeline = ml.Transforms.VectorWhiten(nameof(
                SampleVectorOfNumbersData.Features), kind: Microsoft.ML.Transforms
                .WhiteningKind.PrincipalComponentAnalysis, rank: 4);

            // The transformed (projected) data.
            var transformedData = whiteningPipeline.Fit(trainData).Transform(
                trainData);

            // Getting the data of the newly created column, so we can preview it.
            var whitening = transformedData.GetColumn<VBuffer<float>>(
                transformedData.Schema[nameof(SampleVectorOfNumbersData.Features)]);

            printHelper(nameof(SampleVectorOfNumbersData.Features), whitening);

            // Features column obtained post-transformation.
            // -0.979  0.867  1.449  1.236
            // -1.030  1.012  0.426 -0.902
            // -1.047  0.677 -0.946 -1.060
            // -1.029  0.019 -1.502  1.108
            // -0.972 -1.338 -0.028  0.614
            // -0.938 -1.405  0.752 -0.967
        }

        private class SampleVectorOfNumbersData
        {
            [VectorType(10)]
            public float[] Features { get; set; }
        }

        /// <summary>
        /// Returns a few rows of the infertility dataset.
        /// </summary>
        private static IEnumerable<SampleVectorOfNumbersData> 
            GetVectorOfNumbersData()
        {
            var data = new List<SampleVectorOfNumbersData>();
            data.Add(new SampleVectorOfNumbersData { Features = new float[10] { 0,
                1, 2, 3, 4, 5, 6, 7, 8, 9 } });

            data.Add(new SampleVectorOfNumbersData { Features = new float[10] { 1,
                2, 3, 4, 5, 6, 7, 8, 9, 0 } });

            data.Add(new SampleVectorOfNumbersData
            {
                Features = new float[10] { 2, 3, 4, 5, 6, 7, 8, 9, 0, 1 }
            });
            data.Add(new SampleVectorOfNumbersData
            {
                Features = new float[10] { 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, }
            });
            data.Add(new SampleVectorOfNumbersData
            {
                Features = new float[10] { 5, 6, 7, 8, 9, 0, 1, 2, 3, 4 }
            });
            data.Add(new SampleVectorOfNumbersData
            {
                Features = new float[10] { 6, 7, 8, 9, 0, 1, 2, 3, 4, 5 }
            });
            return data;
        }
    }
}
