// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Data.StaticPipe;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.TestFramework;
using System;
using System.Collections.Generic;
using System.Text;
using Xunit;
using Xunit.Abstractions;
using System.Linq;

namespace Microsoft.ML.Tests.Scenarios.Api.CookbookSamples
{
    /// <summary>
    /// Samples that are written as part of 'ML.NET Cookbook' are also added here as tests.
    /// These tests don't actually test anything, other than the fact that the code compiles and
    /// doesn't throw when it is executed.
    /// </summary>
    public sealed class CookbookSamples : BaseTestClass
    {
        public CookbookSamples(ITestOutputHelper output) : base(output)
        {
        }

        private void IntermediateData(string dataPath)
        {
            // Create a new environment for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var env = new LocalEnvironment();

            // Create the reader: define the data columns and where to find them in the text file.
            var reader = TextLoader.CreateReader(env, ctx => (
                    // A boolean column depicting the 'target label'.
                    IsOver50K: ctx.LoadBool(0),
                    // Three text columns.
                    Workclass: ctx.LoadText(1),
                    Education: ctx.LoadText(2),
                    MaritalStatus: ctx.LoadText(3)),
                hasHeader: true);

            // Start creating our processing pipeline. For now, let's just concatenate all the text columns
            // together into one.
            var dataPipeline = reader.MakeNewEstimator()
                .Append(row =>
                    (
                        row.IsOver50K,
                        AllFeatures: row.Workclass.ConcatWith(row.Education, row.MaritalStatus)
                    ));

            // Let's verify that the data has been read correctly. 
            // First, we read the data file.
            var data = reader.Read(new MultiFileSource(dataPath));

            // Fit our data pipeline and transform data with it.
            var transformedData = dataPipeline.Fit(data).Transform(data);

            // Extract the 'AllFeatures' column.
            // This will give the entire dataset: make sure to only take several row
            // in case the dataset is huge.
            var featureColumns = transformedData.GetColumn(r => r.AllFeatures)
                .Take(20).ToArray();

            // The same extension method also applies to the dynamic-typed data, except you have to
            // specify the column name and type:
            var dynamicData = transformedData.AsDynamic;
            var sameFeatureColumns = dynamicData.GetColumn<string[]>(env, "AllFeatures")
                .Take(20).ToArray();
        }

        [Fact]
        public void InspectIntermediateDataGetColumn()
            => IntermediateData(GetDataPath("adult.tiny.with-schema.txt"));

    }
}
