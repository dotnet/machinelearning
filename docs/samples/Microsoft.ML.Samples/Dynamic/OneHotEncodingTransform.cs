using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using static Microsoft.ML.Transforms.OneHotEncodingTransformer;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class OneHotEncodingTransform
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext();

            // Get a small dataset as an IEnumerable and convert it to an IDataView.
            IEnumerable<SamplesUtils.DatasetUtils.SampleInfertData> data = SamplesUtils.DatasetUtils.GetInfertData();
            var trainData = ml.Data.LoadFromEnumerable(data);

            // Preview of the data.
            //
            // Age  Case  Education  Induced     Parity  PooledStratum  RowNum  ...
            // 26   1       0-5yrs      1         6         3             1  ...
            // 42   1       0-5yrs      1         1         1             2  ...
            // 39   1       0-5yrs      2         6         4             3  ...
            // 34   1       0-5yrs      2         4         2             4  ...
            // 35   1       6-11yrs     1         3         32            5  ...

            // A pipeline for one hot encoding the Education column.
            var pipeline = ml.Transforms.Categorical.OneHotEncoding("EducationOneHotEncoded", "Education", OutputKind.Bag);
            // Fit to data.
            var transformer = pipeline.Fit(trainData);

            // Get transformed data
            var transformedData = transformer.Transform(trainData);

            // Getting the data of the newly created column, so we can preview it.
            var encodedColumn = transformedData.GetColumn<float[]>(ml, "EducationOneHotEncoded");

            // A small printing utility.
            Action<string, IEnumerable<float[]>> printHelper = (colName, column) =>
            {
                foreach (var row in column)
                {
                    for (var i = 0; i < row.Length; i++)
                        Console.Write($"{row[i]} ");
                    Console.WriteLine();
                }
            };

            printHelper("Education", encodedColumn);

            // data column obtained post-transformation.
            // 1 0 0 0 ...
            // 1 0 0 0 ... 
            // 1 0 0 0 ...
            // 1 0 0 0 ... 
            // 0 1 0 0 ...
            // ....
        }
    }
}
