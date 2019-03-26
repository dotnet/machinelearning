// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.


using Microsoft.ML.Data;

namespace Microsoft.ML.Functional.Tests.Datasets
{
    /// <summary>
    /// A class describing the TrivialMatrixFactorization test dataset.
    /// </summary>
    internal sealed class TrivialMatrixFactorization
    {
        [LoadColumn(0)]
        public float Label { get; set; }

        [LoadColumn(1)]
        public uint MatrixColumnIndex { get; set; }

        [LoadColumn(2)]
        public uint MatrixRowIndex { get; set; }

        public static IDataView LoadAndFeaturizeFromTextFile(MLContext mlContext, string filePath, bool hasHeader, char separatorChar)
        {
            // Load the data from a textfile.
            var data = mlContext.Data.LoadFromTextFile<TrivialMatrixFactorization>(filePath, hasHeader: hasHeader, separatorChar: separatorChar);

            // Describe a pipeline to translate the uints to keys.
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("MatrixColumnIndex")
                .Append(mlContext.Transforms.Conversion.MapValueToKey("MatrixRowIndex"));

            // Transform the data.
            var transformedData = pipeline.Fit(data).Transform(data);

            return transformedData;
        }
    }
}
