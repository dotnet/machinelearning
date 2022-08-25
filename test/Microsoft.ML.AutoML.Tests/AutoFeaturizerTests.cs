// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using System.Text.Json;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;
using ApprovalTests;
using ApprovalTests.Namers;
using ApprovalTests.Reporters;
using System.Text.Json.Serialization;

namespace Microsoft.ML.AutoML.Test
{
    public class AutoFeaturizerTests : BaseTestClass
    {
        private readonly JsonSerializerOptions _jsonSerializerOptions;

        public AutoFeaturizerTests(ITestOutputHelper output)
            : base(output)
        {
            _jsonSerializerOptions = new JsonSerializerOptions()
            {
                WriteIndented = true,
                Converters =
                {
                    new JsonStringEnumConverter(), new DoubleToDecimalConverter(), new FloatToDecimalConverter(),
                },
            };

            if (Environment.GetEnvironmentVariable("HELIX_CORRELATION_ID") != null)
            {
                Approvals.UseAssemblyLocationForApprovedFiles();
            }
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void AutoFeaturizer_uci_adult_test()
        {
            var context = new MLContext(1);
            var dataset = DatasetUtil.GetUciAdultDataView();
            var pipeline = context.Auto().Featurizer(dataset, outputColumnName: "OutputFeature", excludeColumns: new[] { "Label" });

            Approvals.Verify(JsonSerializer.Serialize(pipeline, _jsonSerializerOptions));
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void AutoFeaturizer_iris_test()
        {
            var context = new MLContext(1);
            var dataset = DatasetUtil.GetIrisDataView();
            var pipeline = context.Auto().Featurizer(dataset, excludeColumns: new[] { "Label" });

            Approvals.Verify(JsonSerializer.Serialize(pipeline, _jsonSerializerOptions));
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void AutoFeaturizer_newspaperchurn_test()
        {
            var context = new MLContext(1);
            var dataset = DatasetUtil.GetNewspaperChurnDataView();
            var pipeline = context.Auto().Featurizer(dataset, excludeColumns: new[] { DatasetUtil.NewspaperChurnLabel });

            Approvals.Verify(JsonSerializer.Serialize(pipeline, _jsonSerializerOptions));
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void AutoFeaturizer_creditapproval_test()
        {
            // this test verify if auto featurizer can convert vector<bool> column to vector<numeric>.
            var context = new MLContext(1);
            var dataset = DatasetUtil.GetCreditApprovalDataView();
            var pipeline = context.Auto().Featurizer(dataset, excludeColumns: new[] { "A16" });
            Approvals.Verify(JsonSerializer.Serialize(pipeline, _jsonSerializerOptions));
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void ImagePathFeaturizerTest()
        {
            var context = new MLContext(1);
            var pipeline = context.Auto().ImagePathFeaturizer("imagePath", "imagePath");

            Approvals.Verify(JsonSerializer.Serialize(pipeline, _jsonSerializerOptions));
        }


        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void AutoFeaturizer_image_test()
        {
            var context = new MLContext(1);
            var datasetPath = DatasetUtil.GetFlowersDataset();
            var columnInference = context.Auto().InferColumns(datasetPath, "Label");
            var textLoader = context.Data.CreateTextLoader(columnInference.TextLoaderOptions);
            var trainData = textLoader.Load(datasetPath);
            var pipeline = context.Auto().Featurizer(trainData, columnInference.ColumnInformation);

            Approvals.Verify(JsonSerializer.Serialize(pipeline, _jsonSerializerOptions));
        }
    }
}
