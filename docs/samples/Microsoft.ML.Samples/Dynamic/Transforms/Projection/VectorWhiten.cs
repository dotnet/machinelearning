using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public sealed class VectorWhiten
    {
        
        /// This example requires installation of additional nuget package <a href="https://www.nuget.org/packages/Microsoft.ML.Mkl.Components/">Microsoft.ML.Mkl.Components</a>.
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext();

            // Get a small dataset as an IEnumerable and convert it to an IDataView.
            var data = SamplesUtils.DatasetUtils.GetVectorOfNumbersData();
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
            Action<string, IEnumerable<VBuffer<float>>> printHelper = (colName, column) =>
            {
                Console.WriteLine($"{colName} column obtained post-transformation.");
                foreach (var row in column)
                    Console.WriteLine($"{string.Join(" ", row.DenseValues().Select(x => x.ToString("f3")))} ");
            };

            // A pipeline to project Features column into white noise vector.
            var whiteningPipeline = ml.Transforms.VectorWhiten(nameof(SamplesUtils.DatasetUtils.SampleVectorOfNumbersData.Features),
                kind: Transforms.WhiteningKind.ZeroPhaseComponentAnalysis);
            // The transformed (projected) data.
            var transformedData = whiteningPipeline.Fit(trainData).Transform(trainData);
            // Getting the data of the newly created column, so we can preview it.
            var whitening = transformedData.GetColumn<VBuffer<float>>(transformedData.Schema[nameof(SamplesUtils.DatasetUtils.SampleVectorOfNumbersData.Features)]);

            printHelper(nameof(SamplesUtils.DatasetUtils.SampleVectorOfNumbersData.Features), whitening);

            // Features column obtained post-transformation.
            //
            //-0.394 -0.318 -0.243 -0.168  0.209  0.358  0.433  0.589  0.873  2.047
            //-0.034  0.030  0.094  0.159  0.298  0.427  0.492  0.760  1.855 -1.197
            // 0.099  0.161  0.223  0.286  0.412  0.603  0.665  1.797 -1.265 -0.172
            // 0.211  0.277  0.344  0.410  0.606  1.267  1.333 -1.340 -0.205  0.065
            // 0.454  0.523  0.593  0.664  1.886 -0.757 -0.687 -0.022  0.176  0.310
            // 0.863  0.938  1.016  1.093 -1.326 -0.096 -0.019  0.189  0.330  0.483
        }
    }
}
