// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.AutoML.Test
{

    public class EstimatorExtensionTests : BaseTestClass
    {
        public EstimatorExtensionTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void EstimatorExtensionInstanceTests()
        {
            var context = new MLContext(1);
            var pipelineNode = new PipelineNode()
            {
                InColumns = new string[] { "Input" },
                OutColumns = new string[] { "Output" },
                Properties = new Dictionary<string, object> { }
            };

            var estimatorNames = Enum.GetValues(typeof(EstimatorName)).Cast<EstimatorName>();
            foreach (var estimatorName in estimatorNames)
            {
                var extension = EstimatorExtensionCatalog.GetExtension(estimatorName);
                var instance = extension.CreateInstance(context, pipelineNode);
                Assert.NotNull(instance);
            }
        }

        [Fact]
        public void EstimatorExtensionStaticTests()
        {
            var context = new MLContext(1);
            var inCol = "Input";
            var outCol = "Output";
            var inCols = new string[] { inCol };
            var outCols = new string[] { outCol };
            Assert.NotNull(ColumnConcatenatingExtension.CreateSuggestedTransform(context, inCols, outCol));
            Assert.NotNull(ColumnCopyingExtension.CreateSuggestedTransform(context, inCol, outCol));
            Assert.NotNull(MissingValueIndicatingExtension.CreateSuggestedTransform(context, inCols, outCols));
            Assert.NotNull(MissingValueReplacingExtension.CreateSuggestedTransform(context, inCols, outCols));
            Assert.NotNull(NormalizingExtension.CreateSuggestedTransform(context, inCol, outCol));
            Assert.NotNull(OneHotEncodingExtension.CreateSuggestedTransform(context, inCols, outCols));
            Assert.NotNull(OneHotHashEncodingExtension.CreateSuggestedTransform(context, inCols, outCols));
            Assert.NotNull(TextFeaturizingExtension.CreateSuggestedTransform(context, inCol, outCol));
            Assert.NotNull(TypeConvertingExtension.CreateSuggestedTransform(context, inCols, outCols));
            Assert.NotNull(ValueToKeyMappingExtension.CreateSuggestedTransform(context, inCol, outCol));
        }
    }
}
