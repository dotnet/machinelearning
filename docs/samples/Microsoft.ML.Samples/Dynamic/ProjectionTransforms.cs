using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class ProjectionTransforms
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext();

            // Get a small dataset as an IEnumerable and convert it to an IDataView.
            IEnumerable<SamplesUtils.DatasetUtils.SampleVectorOfNumbersData> data = SamplesUtils.DatasetUtils.GetVectorOfNumbersData();
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
                    Console.WriteLine($"{string.Join(" ",row.DenseValues().Select(x=>x.ToString("f3")))} ");
            };

            // A pipeline to project Features column into Random fourier space.
            var rffPipeline = ml.Transforms.ApproximatedKernelMap(nameof(SamplesUtils.DatasetUtils.SampleVectorOfNumbersData.Features), rank: 4);
            // The transformed (projected) data.
            var transformedData = rffPipeline.Fit(trainData).Transform(trainData);
            // Getting the data of the newly created column, so we can preview it.
            var randomFourier = transformedData.GetColumn<VBuffer<float>>(transformedData.Schema[nameof(SamplesUtils.DatasetUtils.SampleVectorOfNumbersData.Features)]);

            printHelper(nameof(SamplesUtils.DatasetUtils.SampleVectorOfNumbersData.Features), randomFourier);

            // Features column obtained post-transformation.
            //
            //0.634 0.628 -0.705 -0.337
            //0.704 0.683 -0.555 -0.422
            //0.407 0.542 -0.707 -0.616
            //0.473 0.331 -0.400 -0.699
            //0.181 0.361 -0.335 -0.157
            //0.165 0.117 -0.547  0.014

            // A pipeline to project Features column into L-p normalized vector.
            var lpNormalizePipeline = ml.Transforms.NormalizeLpNorm(nameof(SamplesUtils.DatasetUtils.SampleVectorOfNumbersData.Features), norm: Transforms.LpNormNormalizingEstimatorBase.NormFunction.L1);
            // The transformed (projected) data.
            transformedData = lpNormalizePipeline.Fit(trainData).Transform(trainData);
            // Getting the data of the newly created column, so we can preview it.
            var lpNormalize= transformedData.GetColumn<VBuffer<float>>(transformedData.Schema[nameof(SamplesUtils.DatasetUtils.SampleVectorOfNumbersData.Features)]);

            printHelper(nameof(SamplesUtils.DatasetUtils.SampleVectorOfNumbersData.Features), lpNormalize);

            // Features column obtained post-transformation.
            //
            // 0.000 0.022 0.044 0.067 0.089 0.111 0.133 0.156 0.178 0.200
            // 0.022 0.044 0.067 0.089 0.111 0.133 0.156 0.178 0.200 0.000
            // 0.044 0.067 0.089 0.111 0.133 0.156 0.178 0.200 0.000 0.022
            // 0.067 0.089 0.111 0.133 0.156 0.178 0.200 0.000 0.022 0.044
            // 0.111 0.133 0.156 0.178 0.200 0.000 0.022 0.044 0.067 0.089
            // 0.133 0.156 0.178 0.200 0.000 0.022 0.044 0.067 0.089 0.111

            // A pipeline to project Features column into L-p normalized vector.
            var gcNormalizePipeline = ml.Transforms.NormalizeGlobalContrast(nameof(SamplesUtils.DatasetUtils.SampleVectorOfNumbersData.Features), ensureZeroMean:false);
            // The transformed (projected) data.
            transformedData = gcNormalizePipeline.Fit(trainData).Transform(trainData);
            // Getting the data of the newly created column, so we can preview it.
            var gcNormalize = transformedData.GetColumn<VBuffer<float>>(transformedData.Schema[nameof(SamplesUtils.DatasetUtils.SampleVectorOfNumbersData.Features)]);

            printHelper(nameof(SamplesUtils.DatasetUtils.SampleVectorOfNumbersData.Features), gcNormalize);

            // Features column obtained post-transformation.
            //
            // 0.000 0.059 0.118 0.178 0.237 0.296 0.355 0.415 0.474 0.533
            // 0.059 0.118 0.178 0.237 0.296 0.355 0.415 0.474 0.533 0.000
            // 0.118 0.178 0.237 0.296 0.355 0.415 0.474 0.533 0.000 0.059
            // 0.178 0.237 0.296 0.355 0.415 0.474 0.533 0.000 0.059 0.118
            // 0.296 0.355 0.415 0.474 0.533 0.000 0.059 0.118 0.178 0.237
            // 0.355 0.415 0.474 0.533 0.000 0.059 0.118 0.178 0.237 0.296
        }
    }
}
