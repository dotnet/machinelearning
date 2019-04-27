// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Microsoft.ML.AutoML.Test
{
    [TestClass]
    public class EstimatorExtensionTests
    {
        [TestMethod]
        public void EstimatorExtensionInstanceTests()
        {
            var context = new MLContext();
            var pipelineNode = new PipelineNode()
            {
                InColumns = new string[] { "Input" },
                OutColumns = new string[] { "Output" }
            };

            var estimatorNames = Enum.GetValues(typeof(EstimatorName)).Cast<EstimatorName>();
            foreach (var estimatorName in estimatorNames)
            {
                var extension = EstimatorExtensionCatalog.GetExtension(estimatorName);
                var instance = extension.CreateInstance(context, pipelineNode);
                Assert.IsNotNull(instance);
            }
        }

        [TestMethod]
        public void EstimatorExtensionStaticTests()
        {
            var context = new MLContext();
            var inCol = "Input";
            var outCol = "Output";
            var inCols = new string[] { inCol };
            var outCols = new string[] { outCol };
            Assert.IsNotNull(ColumnConcatenatingExtension.CreateSuggestedTransform(context, inCols, outCol));
            Assert.IsNotNull(ColumnCopyingExtension.CreateSuggestedTransform(context, inCol, outCol));
            Assert.IsNotNull(MissingValueIndicatingExtension.CreateSuggestedTransform(context, inCols, outCols));
            Assert.IsNotNull(MissingValueReplacingExtension.CreateSuggestedTransform(context, inCols, outCols));
            Assert.IsNotNull(NormalizingExtension.CreateSuggestedTransform(context, inCol, outCol));
            Assert.IsNotNull(OneHotEncodingExtension.CreateSuggestedTransform(context, inCols, outCols));
            Assert.IsNotNull(OneHotHashEncodingExtension.CreateSuggestedTransform(context, inCols, outCols));
            Assert.IsNotNull(TextFeaturizingExtension.CreateSuggestedTransform(context, inCol, outCol));
            Assert.IsNotNull(TypeConvertingExtension.CreateSuggestedTransform(context, inCols, outCols));
            Assert.IsNotNull(ValueToKeyMappingExtension.CreateSuggestedTransform(context, inCol, outCol));
        }
    }
}
