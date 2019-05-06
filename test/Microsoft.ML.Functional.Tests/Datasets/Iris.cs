// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.


using System;
using Microsoft.ML.Data;

namespace Microsoft.ML.Functional.Tests.Datasets
{
    /// <summary>
    /// A class for the Iris test dataset.
    /// </summary>
    internal sealed class Iris
    {
        [LoadColumn(0)]
        public float Label { get; set; }

        [LoadColumn(1)]
        public float SepalLength { get; set; }

        [LoadColumn(2)]
        public float SepalWidth { get; set; }

        [LoadColumn(4)]
        public float PetalLength { get; set; }

        [LoadColumn(5)]
        public float PetalWidth { get; set; }

        /// <summary>
        /// The list of columns commonly used as features.
        /// </summary>
        public static readonly string[] Features = new string[] { "SepalLength", "SepalWidth", "PetalLength", "PetalWidth" };

        public static IDataView LoadAsRankingProblem(MLContext mlContext, string filePath, bool hasHeader, char separatorChar, int seed = 1)
        {
            // Load the Iris data.
            var data = mlContext.Data.LoadFromTextFile<Iris>(filePath, hasHeader: hasHeader, separatorChar: separatorChar);

            // Create a function that generates a random groupId.
            var rng = new Random(seed);
            Action<Iris, IrisWithGroup> generateGroupId = (input, output) =>
            {
                output.Label = input.Label;
                // The standard set used in tests has 150 rows
                output.GroupId = rng.Next(0, 30);
                output.PetalLength = input.PetalLength;
                output.PetalWidth = input.PetalWidth;
                output.SepalLength = input.SepalLength;
                output.SepalWidth = input.SepalWidth;
            };

            // Describe a pipeline that generates a groupId and converts it to a key.
            var pipeline = mlContext.Transforms.CustomMapping(generateGroupId, null)
                .Append(mlContext.Transforms.Conversion.MapValueToKey("GroupId"));

            // Transform the data
            var transformedData = pipeline.Fit(data).Transform(data);

            return transformedData;
        }
    }

    /// <summary>
    /// A class for the Iris dataset with a GroupId column.
    /// </summary>
    internal sealed class IrisWithGroup
    {
        public float Label { get; set; }
        public int GroupId { get; set; }
        public float SepalLength { get; set; }
        public float SepalWidth { get; set; }
        public float PetalLength { get; set; }
        public float PetalWidth { get; set; }
    }

    /// <summary>
    /// A class for the Iris dataset with an extra float column.
    /// </summary>
    internal sealed class IrisWithOneExtraColumn
    {
        public float Label { get; set; }
        public float SepalLength { get; set; }
        public float SepalWidth { get; set; }
        public float PetalLength { get; set; }
        public float PetalWidth { get; set; }
        public float Float1 { get; set; }
    }

    /// <summary>
    /// A class for the Iris dataset with two extra float columns.
    /// </summary>
    internal sealed class IrisWithTwoExtraColumns
    {
        public float Label { get; set; }
        public float SepalLength { get; set; }
        public float SepalWidth { get; set; }
        public float PetalLength { get; set; }
        public float PetalWidth { get; set; }
        public float Float1 { get; set; }
        public float Float2 { get; set; }
    }
}
